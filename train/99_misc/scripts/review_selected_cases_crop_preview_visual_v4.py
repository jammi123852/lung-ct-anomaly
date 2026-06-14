"""
review_selected_cases_crop_preview_visual_v4.py

목적:
reference bank v4 selected cases crop preview(20 PNG, 4 montage) 육안 검토 결과를
report-only 로 정리한다. 등급(GOOD_FOR_DEMO / USABLE_WITH_CAUTION / NOT_RECOMMENDED)은
사람이 montage 를 직접 보고 내린 판정을 REVIEW 딕셔너리에 코드화한 것이다.

이번 단계 금지:
  PNG 재생성 / CT load / ROI load / model / feature / contribution /
  stage2_holdout 접근 / 기존 artifact 수정 / 진단·병변·암·혈관 원인 단정 /
  same-z matched 등 금지 표현.
허용:
  기존 PNG/metadata read-only 존재 확인 + review report 생성(신규 root).
"""

import os
import sys
import csv
import json
import pathlib
from datetime import date

# ============================================================
# GUARD FLAGS (전부 False; 이번 단계는 어떤 load/생성도 없음)
# ============================================================
ALLOW_CT_LOAD             = False
ALLOW_ROI_LOAD            = False
ALLOW_PNG_WRITE           = False   # review 단계: PNG 재생성 금지
ALLOW_MODEL_FORWARD       = False
ALLOW_FEATURE_EXTRACTION  = False
ALLOW_CONTRIBUTION_RECALC = False
ALLOW_STAGE2_HOLDOUT      = False
ALLOW_FULL300             = False

# ============================================================
# PATHS
# ============================================================
REPO_ROOT = pathlib.Path("/home/jinhy/project/lung-ct-anomaly")
PREVIEW_ROOT = REPO_ROOT / (
    "outputs/position-aware-padim-v1/visualizations/"
    "reference_bank_v4_selected_cases_crop_preview")
OUTPUT_ROOT = REPO_ROOT / (
    "outputs/position-aware-padim-v1/reports/"
    "reference_bank_v4_selected_cases_crop_preview_visual_review")

EXPECTED_CASES = ["LUNG1-052__c3", "LUNG1-320__c2", "LUNG1-041__c3", "MSD_lung_059__c2"]
CAUTION_TEXT = "same-cell comparison, not same-z matching"

FORBIDDEN_PHRASES = [
    "same-z matched", "same z matched", "z-matched", "z matched",
    "identical z", "동일 z 위치", "같은 z 위치",
    "diagnostic heatmap", "grad-cam", "pixel attribution",
    "병변 원인", "암 위치", "혈관 때문에", "diagnostic conclusion",
]


def _abort(msg, code=2):
    print(f"\nBLOCKED: {msg}", file=sys.stderr)
    sys.exit(code)


def scan_forbidden(blob):
    low = str(blob).lower()
    return [p for p in FORBIDDEN_PHRASES if p.lower() in low]


# ============================================================
# 육안 검토 결과 (사람이 montage 4개를 직접 보고 판정)
# ============================================================
REVIEW = {
    "LUNG1-052__c3": {
        "position_similarity_rating": "moderate-good",
        "fov_consistency_rating": "good",
        "reference_quality_rating": "mixed (ref1/ref3 clean normal; ref2 vessel-dominated)",
        "candidate_reference_difference_rating": "clear",
        "final_visual_verdict": "GOOD_FOR_DEMO",
        "recommended_use": "S5 demo card (primary); prefer ref1/ref3 as comparison",
        "notes": "candidate shows a distinct clustered dense opacity vs clean normal "
                 "parenchyma in ref1/ref3; difference is explainable as a different local "
                 "pattern vs position-matched normal; candidate slightly peripheral "
                 "(pleural curve visible); ref2 is vessel-dominated -> deprioritize",
        "refs": {
            "normal_ref_1": ("good", "good", "keep",
                             "clean normal lung interior; good comparison baseline"),
            "normal_ref_2": ("vessel-dominated", "moderate", "deprioritize",
                             "large branching vessel dominates the crop; weaker as a "
                             "parenchyma baseline"),
            "normal_ref_3": ("good", "good", "keep",
                             "clean normal parenchyma with small vessels; best comparison ref"),
        },
    },
    "LUNG1-320__c2": {
        "position_similarity_rating": "good",
        "fov_consistency_rating": "good",
        "reference_quality_rating": "acceptable (peripheral, boundary/pleura-adjacent)",
        "candidate_reference_difference_rating": "weak",
        "final_visual_verdict": "USABLE_WITH_CAUTION",
        "recommended_use": "appendix / supporting material only",
        "notes": "candidate region is largely aerated lung; candidate-vs-normal visual "
                 "difference is subtle; boundary/pleura-adjacent region; position match and "
                 "window are consistent but explanatory power is limited -> keep caution",
        "refs": {
            "normal_ref_1": ("acceptable", "good", "keep",
                             "dark aerated lung with peripheral bright structure; OK"),
            "normal_ref_2": ("acceptable", "good", "keep",
                             "similar peripheral aerated lung; OK"),
            "normal_ref_3": ("acceptable", "good", "keep",
                             "peripheral aerated lung with vessels; OK"),
        },
    },
    "LUNG1-041__c3": {
        "position_similarity_rating": "moderate",
        "fov_consistency_rating": "good",
        "reference_quality_rating": "poor (vessel/large-structure dominated)",
        "candidate_reference_difference_rating": "strong but structural-risk",
        "final_visual_verdict": "NOT_RECOMMENDED",
        "recommended_use": "demo NOT recommended",
        "notes": "all three references are dominated by large branching vessels / junctions; "
                 "the candidate-vs-reference difference can be misread as a structural "
                 "difference rather than a parenchymal one -> high misinterpretation risk; "
                 "not suitable for a persuasive demo crop",
        "refs": {
            "normal_ref_1": ("vessel-dominated", "moderate", "exclude",
                             "vessel junction dominates; not a clean parenchyma baseline"),
            "normal_ref_2": ("vessel-dominated", "moderate", "exclude",
                             "large horizontal vessel dominates the crop"),
            "normal_ref_3": ("vessel-dominated", "moderate", "exclude",
                             "prominent branching vessel cluster; structural-difference risk"),
        },
    },
    "MSD_lung_059__c2": {
        "position_similarity_rating": "good",
        "fov_consistency_rating": "good (display slightly compressed)",
        "reference_quality_rating": "acceptable",
        "candidate_reference_difference_rating": "moderate",
        "final_visual_verdict": "USABLE_WITH_CAUTION",
        "recommended_use": "appendix; MSD source must be stated; no diagnostic interpretation",
        "notes": "MSD source (record only, no diagnostic interpretation); candidate shows a "
                 "busier reticular/nodular parenchyma vs cleaner normal references; apex "
                 "region (Z0) with z difference -> keep MSD-source and z-direction caution",
        "refs": {
            "normal_ref_1": ("acceptable", "good", "keep",
                             "normal lung with vessels; OK baseline"),
            "normal_ref_2": ("acceptable", "good", "keep",
                             "normal parenchyma with vessels; OK"),
            "normal_ref_3": ("acceptable", "good", "keep",
                             "normal lung with vessels; OK"),
        },
    },
}

VERDICT_RANK = {"GOOD_FOR_DEMO": 1, "USABLE_WITH_CAUTION": 2, "NOT_RECOMMENDED": 3}


def png_names(case_id):
    return {
        "candidate": f"{case_id}_candidate.png",
        "normal_ref_1": f"{case_id}_normal_ref_1.png",
        "normal_ref_2": f"{case_id}_normal_ref_2.png",
        "normal_ref_3": f"{case_id}_normal_ref_3.png",
        "montage": f"{case_id}_reference_preview_montage.png",
    }


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 64)
    print("selected cases crop preview VISUAL REVIEW (report-only)")
    print(f"date: {date.today()}")
    print("=" * 64)

    # guard self-check
    if ALLOW_CT_LOAD or ALLOW_ROI_LOAD or ALLOW_PNG_WRITE or ALLOW_MODEL_FORWARD \
       or ALLOW_FEATURE_EXTRACTION or ALLOW_CONTRIBUTION_RECALC or ALLOW_STAGE2_HOLDOUT:
        _abort("a forbidden guard is True; review stage must keep all False")

    if not PREVIEW_ROOT.exists():
        _abort(f"preview root not found: {PREVIEW_ROOT}")

    # metadata read-only (caution/msd_source 확인)
    case_meta = {}
    for case_id in EXPECTED_CASES:
        mp = PREVIEW_ROOT / f"{case_id}_preview_metadata.json"
        case_meta[case_id] = json.loads(mp.read_text()) if mp.exists() else {}

    # file existence + caution check (read-only)
    case_rows, ref_rows, errors = [], [], []
    for case_id in EXPECTED_CASES:
        names = png_names(case_id)
        exists = {k: (PREVIEW_ROOT / v).exists() for k, v in names.items()}
        rv = REVIEW[case_id]
        meta = case_meta.get(case_id, {})
        caution_ok = (meta.get("caution_text") == CAUTION_TEXT)
        # forbidden scan over this case review text + metadata
        blob = json.dumps(rv, ensure_ascii=False) + json.dumps(meta, ensure_ascii=False)
        fb_hits = scan_forbidden(blob)
        fb_ok = (len(fb_hits) == 0)
        if fb_hits:
            errors.append({"case_id": case_id, "stage": "forbidden_scan",
                           "error_type": "FORBIDDEN_WORDING", "detail": str(fb_hits)})

        case_rows.append({
            "case_id": case_id,
            "montage_exists": exists["montage"],
            "candidate_exists": exists["candidate"],
            "normal_ref_1_exists": exists["normal_ref_1"],
            "normal_ref_2_exists": exists["normal_ref_2"],
            "normal_ref_3_exists": exists["normal_ref_3"],
            "position_similarity_rating": rv["position_similarity_rating"],
            "fov_consistency_rating": rv["fov_consistency_rating"],
            "reference_quality_rating": rv["reference_quality_rating"],
            "candidate_reference_difference_rating": rv["candidate_reference_difference_rating"],
            "label_caution_ok": caution_ok,
            "forbidden_wording_ok": fb_ok,
            "final_visual_verdict": rv["final_visual_verdict"],
            "recommended_use": rv["recommended_use"],
            "notes": rv["notes"],
        })
        for role, (vq, ps, keep, reason) in rv["refs"].items():
            ref_rows.append({
                "case_id": case_id, "reference_role": role,
                "visual_quality": vq, "position_similarity": ps,
                "keep_or_exclude": keep, "reason": reason,
            })

    # missing montage check
    missing_montage = [r["case_id"] for r in case_rows if not r["montage_exists"]]
    caution_problems = [r["case_id"] for r in case_rows if not r["label_caution_ok"]]
    fb_problems = [r["case_id"] for r in case_rows if not r["forbidden_wording_ok"]]

    # verdict tally
    verdicts = {r["case_id"]: r["final_visual_verdict"] for r in case_rows}
    n_good = sum(1 for v in verdicts.values() if v == "GOOD_FOR_DEMO")
    n_caution = sum(1 for v in verdicts.values() if v == "USABLE_WITH_CAUTION")
    n_not = sum(1 for v in verdicts.values() if v == "NOT_RECOMMENDED")

    if missing_montage or caution_problems or fb_problems:
        verdict = "NEEDS_FIX"
    elif n_good >= 1:
        verdict = "PASS"
    else:
        verdict = "PARTIAL_PASS"

    # ---- write outputs ----
    if (OUTPUT_ROOT / "DONE.json").exists():
        _abort("existing DONE.json at OUTPUT_ROOT. Archive before re-review.")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # 3. case_visual_review_table_v4.csv
    CASE_COLS = ["case_id", "montage_exists", "candidate_exists",
                 "normal_ref_1_exists", "normal_ref_2_exists", "normal_ref_3_exists",
                 "position_similarity_rating", "fov_consistency_rating",
                 "reference_quality_rating", "candidate_reference_difference_rating",
                 "label_caution_ok", "forbidden_wording_ok",
                 "final_visual_verdict", "recommended_use", "notes"]
    with open(OUTPUT_ROOT / "case_visual_review_table_v4.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CASE_COLS)
        w.writeheader()
        for r in case_rows:
            w.writerow(r)

    # 4. reference_quality_notes_v4.csv
    REF_COLS = ["case_id", "reference_role", "visual_quality",
                "position_similarity", "keep_or_exclude", "reason"]
    with open(OUTPUT_ROOT / "reference_quality_notes_v4.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REF_COLS)
        w.writeheader()
        for r in ref_rows:
            w.writerow(r)

    # 5. recommended_demo_cases_v4.csv
    REC_COLS = ["rank", "case_id", "final_visual_verdict", "why_recommended",
                "caution_to_keep", "suggested_next_use"]
    recommended = sorted(
        [r for r in case_rows if r["final_visual_verdict"] in
         ("GOOD_FOR_DEMO", "USABLE_WITH_CAUTION")],
        key=lambda r: (VERDICT_RANK[r["final_visual_verdict"]], r["case_id"]))
    with open(OUTPUT_ROOT / "recommended_demo_cases_v4.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REC_COLS)
        w.writeheader()
        for i, r in enumerate(recommended, 1):
            w.writerow({
                "rank": i, "case_id": r["case_id"],
                "final_visual_verdict": r["final_visual_verdict"],
                "why_recommended": r["notes"],
                "caution_to_keep": CAUTION_TEXT + "; side=image x-coord (NOT anatomical)"
                + ("; MSD source" if r["case_id"] == "MSD_lung_059__c2" else ""),
                "suggested_next_use": r["recommended_use"],
            })

    # 6. excluded_or_caution_cases_v4.csv
    EXC_COLS = ["case_id", "verdict", "reason", "possible_fix", "use_policy"]
    with open(OUTPUT_ROOT / "excluded_or_caution_cases_v4.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EXC_COLS)
        w.writeheader()
        for r in case_rows:
            if r["final_visual_verdict"] == "GOOD_FOR_DEMO":
                continue
            if r["final_visual_verdict"] == "NOT_RECOMMENDED":
                fix = ("re-retrieve references with less vessel-dominated crops, or "
                       "pick a different candidate slice within the same cell")
                pol = "exclude from demo; research-use auxiliary only"
            else:
                fix = ("use ref1/ref3-style clean parenchyma references; state "
                       "z-direction limitation and (if MSD) source")
                pol = "appendix / supporting material only; keep caution"
            w.writerow({
                "case_id": r["case_id"], "verdict": r["final_visual_verdict"],
                "reason": r["notes"], "possible_fix": fix, "use_policy": pol,
            })

    # 7. errors.csv
    with open(OUTPUT_ROOT / "errors.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "stage", "error_type", "detail"])
        for e in errors:
            w.writerow([e["case_id"], e["stage"], e["error_type"], e["detail"]])

    # 2. json report
    report = {
        "date": str(date.today()), "verdict": verdict, "report_only": True,
        "n_cases": len(case_rows),
        "verdict_counts": {"GOOD_FOR_DEMO": n_good,
                           "USABLE_WITH_CAUTION": n_caution,
                           "NOT_RECOMMENDED": n_not},
        "case_verdicts": verdicts,
        "safety": {
            "ct_load": 0, "roi_load": 0, "png_regeneration": 0,
            "model_forward": 0, "feature_extraction": 0, "contribution_recalc": 0,
            "stage2_holdout": 0, "existing_artifact_modified": 0,
            "forbidden_wording": len(errors),
        },
        "review_basis": "human visual inspection of 4 montage PNGs (read-only)",
    }
    (OUTPUT_ROOT / "selected_cases_crop_preview_visual_review.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))

    # 1. md report
    md = [
        "# selected cases crop preview — VISUAL REVIEW (v4)",
        f"date: {date.today()}",
        f"verdict: **{verdict}**",
        "",
        "## scope",
        "- report-only human visual review of 4 montage PNGs (read-only)",
        "- no PNG regeneration / CT / ROI / model / feature / contribution / stage2",
        "",
        "## case visual verdicts",
        "| case_id | position | fov | ref_quality | diff | caution_ok | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in case_rows:
        md.append(f"| {r['case_id']} | {r['position_similarity_rating']} | "
                  f"{r['fov_consistency_rating']} | {r['reference_quality_rating']} | "
                  f"{r['candidate_reference_difference_rating']} | "
                  f"{r['label_caution_ok']} | {r['final_visual_verdict']} |")
    md += ["", "## GOOD_FOR_DEMO"]
    g = [r for r in case_rows if r["final_visual_verdict"] == "GOOD_FOR_DEMO"]
    md += ([f"- {r['case_id']}: {r['notes']}" for r in g] or ["- (none)"])
    md += ["", "## USABLE_WITH_CAUTION"]
    u = [r for r in case_rows if r["final_visual_verdict"] == "USABLE_WITH_CAUTION"]
    md += ([f"- {r['case_id']}: {r['notes']}" for r in u] or ["- (none)"])
    md += ["", "## NOT_RECOMMENDED"]
    nn = [r for r in case_rows if r["final_visual_verdict"] == "NOT_RECOMMENDED"]
    md += ([f"- {r['case_id']}: {r['notes']}" for r in nn] or ["- (none)"])
    md += ["", "## reference quality notes"]
    for r in ref_rows:
        md.append(f"- {r['case_id']} {r['reference_role']}: {r['visual_quality']} "
                  f"/ pos={r['position_similarity']} / {r['keep_or_exclude']} "
                  f"-- {r['reason']}")
    md += [
        "",
        "## safety",
        "- CT load: 0 / ROI load: 0 / PNG regeneration: 0",
        "- model: 0 / feature: 0 / contribution: 0 / stage2_holdout: 0",
        "- existing artifact modified: 0 (writes only to new review root)",
        f"- forbidden wording: {len(errors)}",
        "",
        "## notes",
        f"- montage caution verified present: {CAUTION_TEXT}",
        "- side by image x-coordinate (NOT anatomical); MSD_lung_059 is MSD source "
        "(recorded only, no diagnostic interpretation)",
        "",
    ]
    body = "\n".join(md)
    body_hits = scan_forbidden(body)
    if body_hits:
        _abort(f"forbidden phrase in review md body: {body_hits}")
    (OUTPUT_ROOT / "selected_cases_crop_preview_visual_review.md").write_text(body)

    # 8. DONE.json
    done = {
        "status": "DONE", "verdict": verdict, "date": str(date.today()),
        "report_only": True,
        "verdict_counts": {"GOOD_FOR_DEMO": n_good,
                           "USABLE_WITH_CAUTION": n_caution,
                           "NOT_RECOMMENDED": n_not},
        "errors": len(errors),
        "outputs": [
            "selected_cases_crop_preview_visual_review.md",
            "selected_cases_crop_preview_visual_review.json",
            "case_visual_review_table_v4.csv",
            "reference_quality_notes_v4.csv",
            "recommended_demo_cases_v4.csv",
            "excluded_or_caution_cases_v4.csv",
            "errors.csv",
            "DONE.json",
        ],
    }
    (OUTPUT_ROOT / "DONE.json").write_text(json.dumps(done, indent=2))

    print(f"\nVERDICT: {verdict}")
    print(f"  GOOD_FOR_DEMO={n_good} USABLE_WITH_CAUTION={n_caution} NOT_RECOMMENDED={n_not}")
    print(f"  missing montage: {missing_montage} | caution problems: {caution_problems}")
    print(f"  forbidden wording errors: {len(errors)}")
    print(f"  outputs -> {OUTPUT_ROOT}")

    if verdict == "NEEDS_FIX":
        sys.exit(1)


if __name__ == "__main__":
    main()
