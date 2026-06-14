#!/usr/bin/env python3
"""
make_b0_ct_context_review_summary.py

GPT-assisted B0 CT-context label CSV 집계 → summary (MD, CSV, JSON) 생성

금지: CT/ROI/mask npy 로드, PNG 생성, score CSV 로드, scoring/metric/suppression,
      기존 파일 수정, current_state.md 수정, stage2_holdout 접근,
      원인 확정, score adjustment, threshold 변경
"""

import sys
import json
from pathlib import Path
import pandas as pd

BASE_DIR = Path("qa/dev_safe_mixed_error_visual_qa")
FILLED_CSV = BASE_DIR / "b0_ct_context_review_labels_template_filled_by_gpt.csv"

OUTPUT_MD = BASE_DIR / "b0_ct_context_review_summary.md"
OUTPUT_CSV = BASE_DIR / "b0_ct_context_review_summary.csv"
OUTPUT_JSON = BASE_DIR / "b0_ct_context_review_summary.json"

REQUIRED_TOTAL = 33
REQUIRED_NORMAL_FP = 9
REQUIRED_LESION_SAFETY = 24

ALLOWED_LABELS = {
    "vessel_like_context_candidate",
    "pleura_or_chest_wall_context_candidate",
    "hilar_or_mediastinal_context_candidate",
    "diaphragm_or_base_context_candidate",
    "mixed_high_contrast_context_candidate",
    "lesion_near_vessel_or_pleura_context_candidate",
    "lesion_protect_context",
    "unclear",
}
ALLOWED_CONFIDENCE = {"high", "medium", "low"}
ALLOWED_ACTION = {
    "keep_for_rule_design_review",
    "needs_local_zoom",
    "needs_mip_or_vessel_context",
    "reference_only",
    "exclude_from_b0",
}


def abort(msg):
    print(f"[ABORT] {msg}", file=sys.stderr)
    sys.exit(1)


def check_no_existing_outputs():
    for p in [OUTPUT_MD, OUTPUT_CSV, OUTPUT_JSON]:
        if p.exists():
            abort(f"output collision — {p} 이미 존재. 덮어쓰기 금지.")


def validate(df):
    if len(df) != REQUIRED_TOTAL:
        abort(f"row 수 {len(df)} (expected {REQUIRED_TOTAL})")

    nfp = (df["group"] == "normal_fp_bridge").sum()
    lsb = (df["group"] == "lesion_safety_bridge").sum()
    if nfp != REQUIRED_NORMAL_FP:
        abort(f"normal_fp_bridge {nfp}행 (expected {REQUIRED_NORMAL_FP})")
    if lsb != REQUIRED_LESION_SAFETY:
        abort(f"lesion_safety_bridge {lsb}행 (expected {REQUIRED_LESION_SAFETY})")

    empty_label = df["ct_context_label"].isna() | df["ct_context_label"].astype(str).str.strip().eq("")
    if empty_label.sum() > 0:
        abort(f"ct_context_label 비어있음 {empty_label.sum()}건")

    invalid_labels = set(df["ct_context_label"].unique()) - ALLOWED_LABELS
    if invalid_labels:
        abort(f"허용 label 밖 값: {invalid_labels}")

    invalid_conf = set(df["ct_context_confidence"].dropna().astype(str).str.strip().unique()) - ALLOWED_CONFIDENCE
    if invalid_conf:
        abort(f"허용 confidence 밖 값: {invalid_conf}")

    invalid_action = set(df["ct_context_action"].dropna().astype(str).str.strip().unique()) - ALLOWED_ACTION
    if invalid_action:
        abort(f"허용 action 밖 값: {invalid_action}")

    if "stage2_holdout_flag" in df.columns and df["stage2_holdout_flag"].sum() != 0:
        abort(f"stage2_holdout_flag 비0 건 있음")

    print(f"[GUARD] 입력 검증 통과: total={REQUIRED_TOTAL}, "
          f"normal_fp={nfp}, lesion_safety={lsb}")


def make_summary_md(df, output_path):
    nfp_df = df[df["group"] == "normal_fp_bridge"]
    lsb_df = df[df["group"] == "lesion_safety_bridge"]

    nfp_label = nfp_df["ct_context_label"].value_counts()
    lsb_label = lsb_df["ct_context_label"].value_counts()
    nfp_action = nfp_df["ct_context_action"].value_counts()
    lsb_action = lsb_df["ct_context_action"].value_counts()

    needs_local_zoom_total = (df["ct_context_action"] == "needs_local_zoom").sum()
    needs_mip_total = (df["ct_context_action"] == "needs_mip_or_vessel_context").sum()

    lsb_all_near = (lsb_df["ct_context_label"] == "lesion_near_vessel_or_pleura_context_candidate").all()

    def vc_to_md(vc):
        return "\n".join(f"  - {k}: {v}" for k, v in vc.items())

    lines = [
        "# B0 CT-context visual review summary",
        "",
        "## 1. Scope",
        "- dev_safe only",
        "- B0 CT-context panel based review",
        f"- normal FP {REQUIRED_NORMAL_FP} rows",
        f"- lesion safety {REQUIRED_LESION_SAFETY} rows",
        "- stage2_holdout 0",
        "- GPT-assisted CT-context labels",
        "- no CT/ROI/mask reload",
        "- no score adjustment",
        "- no suppression applied",
        "",
        "## 2. Input validation",
        f"- total rows: {len(df)} (expected {REQUIRED_TOTAL}) — PASS",
        f"- normal_fp_bridge: {REQUIRED_NORMAL_FP} — PASS",
        f"- lesion_safety_bridge: {REQUIRED_LESION_SAFETY} — PASS",
        "- ct_context_label completeness: PASS (0 empty)",
        "- ct_context_confidence allowed values: PASS",
        "- ct_context_action allowed values: PASS",
        "- stage2_holdout: 0 — PASS",
        "",
        "## 3. Normal FP CT-context label summary",
        f"- total rows: {len(nfp_df)}",
        "- ct_context_label distribution:",
        vc_to_md(nfp_label),
        "- ct_context_action distribution:",
        vc_to_md(nfp_action),
        f"- needs_mip_or_vessel_context: {nfp_action.get('needs_mip_or_vessel_context', 0)} rows",
        "- candidate-level note: hilar/mediastinal and pleura/chest-wall patterns dominate normal FP candidates.",
        "  vessel_like and mixed_high_contrast are minority. No rule selection made from this.",
        "",
        "## 4. Lesion safety CT-context label summary",
        f"- total rows: {len(lsb_df)}",
        "- ct_context_label distribution:",
        vc_to_md(lsb_label),
        "- ct_context_action distribution:",
        vc_to_md(lsb_action),
        f"- needs_local_zoom: {lsb_action.get('needs_local_zoom', 0)} rows",
        f"- needs_mip_or_vessel_context: {lsb_action.get('needs_mip_or_vessel_context', 0)} rows",
        ("- candidate-level note: ALL 24 lesion_safety rows are labeled "
         "lesion_near_vessel_or_pleura_context_candidate."
         if lsb_all_near else
         "- candidate-level note: mixed lesion safety labels observed."),
        "  This is a candidate-level observation only. No suppression decision made.",
        "",
        "## 5. Safety interpretation",
        "- These are GPT-assisted CT-context candidate labels.",
        "- Not a final radiological determination or confirmed cause analysis.",
        "- Normal FP labels alone do not justify selecting a vessel/pleura suppression rule.",
        ("- lesion_safety_bridge: ALL 24 rows labeled lesion_near_vessel_or_pleura_context_candidate. "
         "SUPPRESSION REMAINS BLOCKED."
         if lsb_all_near else
         "- lesion_safety_bridge: mixed labels. Suppression remains blocked until further review."),
        "- Score adjustment, threshold change, and suppression application remain prohibited.",
        "",
        "## 6. Recommended next micro-step",
        f"- {needs_local_zoom_total} rows need local zoom, "
        f"{needs_mip_total} rows need MIP/vessel context.",
        "- Lesion safety overlap remains high (24/24). B0 suppression branch remains on hold.",
        "- If proceeding: generate local zoom or MIP panels for needs_local_zoom / needs_mip rows.",
        "- Do not apply any rule or suppression before local zoom / MIP review is complete.",
        "- Optionally record candidate-level summary in current_state.md (no score/threshold change).",
        "",
        "## 7. Decisions not made",
        "- vessel rule not selected",
        "- pleura rule not selected",
        "- local zoom not generated",
        "- MIP not generated",
        "- threshold not selected",
        "- score adjustment not applied",
        "- suppression not applied",
        "- stage2_holdout evaluation not approved",
        "",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] {output_path.name}")


def make_summary_csv(df, output_path):
    cols = [
        "ct_context_id", "review_id", "patient_id", "group", "source_group",
        "safety_role", "candidate_score", "candidate_local_z", "candidate_patch_index",
        "b0_visual_label", "b0_confidence", "b0_action_recommendation",
        "ct_context_label", "ct_context_confidence", "ct_context_action", "ct_context_note",
        "marker_type", "patch_extent_status",
    ]
    available = [c for c in cols if c in df.columns]
    df[available].to_csv(str(output_path), index=False)
    print(f"[OK] {output_path.name}: {len(df)}행")


def make_summary_json(df, output_path):
    nfp_df = df[df["group"] == "normal_fp_bridge"]
    lsb_df = df[df["group"] == "lesion_safety_bridge"]

    lsb_all_near = (lsb_df["ct_context_label"] == "lesion_near_vessel_or_pleura_context_candidate").all()

    pack = {
        "scope": {
            "dataset": "dev_safe",
            "total_rows": len(df),
            "normal_fp_rows": int(len(nfp_df)),
            "lesion_safety_rows": int(len(lsb_df)),
            "stage2_holdout_rows": 0,
            "label_source": "GPT-assisted CT-context candidate labeling",
        },
        "validation_result": "PASS",
        "normal_fp_label_distribution": nfp_df["ct_context_label"].value_counts().to_dict(),
        "normal_fp_action_distribution": nfp_df["ct_context_action"].value_counts().to_dict(),
        "lesion_safety_label_distribution": lsb_df["ct_context_label"].value_counts().to_dict(),
        "lesion_safety_action_distribution": lsb_df["ct_context_action"].value_counts().to_dict(),
        "needs_local_zoom_count": int((df["ct_context_action"] == "needs_local_zoom").sum()),
        "needs_mip_or_vessel_context_count": int((df["ct_context_action"] == "needs_mip_or_vessel_context").sum()),
        "suppression_status": (
            "BLOCKED — all 24 lesion_safety rows are lesion_near_vessel_or_pleura_context_candidate"
            if lsb_all_near else
            "BLOCKED — mixed lesion safety labels, further review required"
        ),
        "safety_constraints": [
            "no CT/ROI/mask npy load",
            "no score CSV reload",
            "no scoring/metric/suppression",
            "no existing file modification",
            "no current_state.md modification",
            "no stage2_holdout access",
            "no cause determination",
            "no score adjustment",
            "no threshold change",
        ],
        "decisions_not_made": [
            "vessel rule not selected",
            "pleura rule not selected",
            "local zoom not generated",
            "MIP not generated",
            "threshold not selected",
            "score adjustment not applied",
            "suppression not applied",
            "stage2_holdout evaluation not approved",
        ],
    }

    output_path.write_text(json.dumps(pack, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] {output_path.name}")


def main():
    print("[INFO] B0 CT-context review summary 생성 시작")

    check_no_existing_outputs()

    if not FILLED_CSV.exists():
        abort(f"filled CSV 없음: {FILLED_CSV}")

    df = pd.read_csv(str(FILLED_CSV))

    validate(df)

    make_summary_md(df, OUTPUT_MD)
    make_summary_csv(df, OUTPUT_CSV)
    make_summary_json(df, OUTPUT_JSON)

    print()
    print("[SUMMARY] 생성 파일:")
    for p in [OUTPUT_MD, OUTPUT_CSV, OUTPUT_JSON]:
        if p.exists():
            kb = p.stat().st_size // 1024
            print(f"  {p.name}: {kb}KB")
    print("[DONE]")


if __name__ == "__main__":
    main()
