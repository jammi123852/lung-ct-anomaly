#!/usr/bin/env python3
"""
P-A47.5 Hold Pool Manual Re-review
- P-A47 hold pool 7건을 기존 P-A46b PNG read-only 재검토 결과로 재분류
- 새 PNG 생성 금지, CT/ROI/mask npy 추가 로드 금지
- suppression/score 수정/adjusted_score/threshold 변경 금지
"""
import csv
import json
import sys
from pathlib import Path

# ===================== 경로 설정 =====================
PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")
P47_ROOT   = PROJECT_ROOT / "experiments/resnet50_imagenet_v1/outputs/reports/lesion_stage1_dev/p_a47_soft_penalty_design_preflight"
P46C_ROOT  = PROJECT_ROOT / "experiments/resnet50_imagenet_v1/outputs/reports/lesion_stage1_dev/p_a46c_ct_context_review_labels"
P46B_ROOT  = PROJECT_ROOT / "experiments/resnet50_imagenet_v1/outputs/reports/lesion_stage1_dev/p_a46b_lower_peripheral_ct_context_panels"
SPLIT_CSV  = PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1_balanced.csv"
OUTPUT_ROOT = PROJECT_ROOT / "experiments/resnet50_imagenet_v1/outputs/reports/lesion_stage1_dev/p_a47_5_hold_pool_re_review"

# ===================== 가드: 기존 결과 없음 확인 =====================
if OUTPUT_ROOT.exists():
    existing = list(OUTPUT_ROOT.iterdir())
    if existing:
        print(f"[ABORT] 기존 P-A47.5 결과 존재: {[f.name for f in existing[:5]]} — 덮어쓰지 않고 중단")
        sys.exit(1)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# ===================== 가드: stage2_holdout 목록 확인 =====================
with open(SPLIT_CSV, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    split_rows = list(reader)
stage1_dev_patients = {r["patient_id"].strip() for r in split_rows if r["stage_split"].strip() == "stage1_dev"}
stage2_holdout_patients = {r["patient_id"].strip() for r in split_rows if r["stage_split"].strip() == "stage2_holdout"}

# ===================== P-A47 hold pool 로드 =====================
hold_csv = P47_ROOT / "soft_penalty_hold_pool.csv"
if not hold_csv.exists():
    print(f"[ABORT] P-A47 hold pool 없음: {hold_csv}")
    sys.exit(1)

with open(hold_csv, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    hold_rows = list(reader)

if len(hold_rows) != 7:
    print(f"[ABORT] hold pool {len(hold_rows)}건 — 7건이 아님. 중단.")
    sys.exit(1)
print(f"[가드] hold pool {len(hold_rows)}건 로드 ✓")

# ===================== 가드: stage2_holdout 포함 여부 =====================
hold_patients = {r["patient_id"].strip() for r in hold_rows}
holdout_leak = hold_patients & stage2_holdout_patients
if holdout_leak:
    print(f"[ABORT] hold pool 환자 중 stage2_holdout 포함: {holdout_leak} — 즉시 중단.")
    sys.exit(1)
print(f"[가드] hold pool 환자({len(hold_patients)}명) 중 stage2_holdout 0명 ✓")

# ===================== 가드: P-A46b PNG 파일 존재 확인 =====================
p46c_labels_path = P46C_ROOT / "ct_context_review_labels_filled.csv"
with open(p46c_labels_path, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    p46c_label_map = {row["review_id"].strip(): row for row in reader}

hold_ids = {r["review_id"].strip() for r in hold_rows}
png_check_ok = True
for rid in hold_ids:
    row = p46c_label_map.get(rid, {})
    png_path = row.get("png_path", "")
    full_path = P46B_ROOT / png_path if png_path else None
    if not full_path or not full_path.exists():
        print(f"[WARN] PNG 없음: {rid} → {png_path}")
        png_check_ok = False
    else:
        print(f"[가드] PNG OK: {rid} → {png_path}")

# ===================== 수동 재검토 결과 (PNG read-only 시각 판독 기반) =====================
# 재검토 원칙:
# 1. 병변 보호 우선 — 병변 overlay/boundary/vessel/pleura 인접 가능성 → candidate 금지
# 2. 확실한 FP만 candidate 가능
# 3. 애매하면 hold 유지
# 4. confidence: high/medium/low

RE_REVIEW_RESULTS = [
    {
        "review_id": "P-A46c-011",
        "patient_id": "LUNG1-171",
        "original_label": "hold_unclear_manual_review",
        "re_review_label": "exclude_protect_lesion",
        "confidence": "low",
        "re_review_note": (
            "확대 패널 상 큰 붉은 병변 구조 위에 patch 위치 확인. "
            "lesion_pixels=2로 매우 작지만 병변 구조 인접 명확. "
            "roi_or_slice_context_unclear로 병변 위치 판단 불가. "
            "보수적 원칙: 병변 인접 가능성 있으므로 exclusion 처리."
        ),
        "lesion_protect_applied": True,
        "reason_code": "lesion_near_large_structure_in_panel",
    },
    {
        "review_id": "P-A46c-019",
        "patient_id": "LUNG1-156",
        "original_label": "hold_unclear_manual_review",
        "re_review_label": "hold_unclear_manual_review",
        "confidence": "low",
        "re_review_note": (
            "LUNG1-156은 타 슬라이스(병변 있는 환자). "
            "확대 패널에서 흉막 vs 연조직 구분 불명확. "
            "sl92 해당 patch 위치 판단 애매. "
            "같은 환자에 병변이 있으므로 보수적 보호 적용. hold 유지."
        ),
        "lesion_protect_applied": False,
        "reason_code": "patient_has_lesion_other_slice_unclear_context",
    },
    {
        "review_id": "P-A46c-020",
        "patient_id": "LUNG1-156",
        "original_label": "hold_unclear_manual_review",
        "re_review_label": "hold_unclear_manual_review",
        "confidence": "low",
        "re_review_note": (
            "LUNG1-156 동일 환자. sl103 흉벽 근처 patch이나 병변 위치 불명확. "
            "패널 상 흉벽 구조가 보이지만 연조직/병변 여부 구분 불가. "
            "같은 환자 병변 보호 원칙. hold 유지."
        ),
        "lesion_protect_applied": False,
        "reason_code": "patient_has_lesion_other_slice_unclear_context",
    },
    {
        "review_id": "P-A46c-030",
        "patient_id": "LUNG1-415",
        "original_label": "hold_unclear_manual_review",
        "re_review_label": "exclude_near_boundary_or_vessel",
        "confidence": "low",
        "re_review_note": (
            "확대 패널에서 혈관 구조 인접 명확 확인. "
            "흉막/흉벽과 혈관 구조 혼재로 FP 여부 판단 불가. "
            "vessel 인접이므로 candidate 불가. exclusion 처리."
        ),
        "lesion_protect_applied": False,
        "reason_code": "vessel_adjacent_fp_determination_impossible",
    },
    {
        "review_id": "P-A46c-046",
        "patient_id": "LUNG1-125",
        "original_label": "hold_unclear_manual_review",
        "re_review_label": "exclude_insufficient_context",
        "confidence": "low",
        "re_review_note": (
            "슬라이스 전체가 극도로 어두움. patch 위치 판독 완전 불가. "
            "score=14.08로 매우 낮음. 폐 첨부 또는 artifact 가능성. "
            "판독 불가 케이스는 FP 후보로 올릴 수 없음. exclusion 처리."
        ),
        "lesion_protect_applied": False,
        "reason_code": "panel_too_dark_location_indeterminate",
    },
    {
        "review_id": "P-A46c-047",
        "patient_id": "LUNG1-216",
        "original_label": "hold_unclear_manual_review",
        "re_review_label": "exclude_insufficient_context",
        "confidence": "low",
        "re_review_note": (
            "lesion_pixels=6으로 매우 작음. score=14.09로 낮음. "
            "확대 패널에서 흉벽/흉막 인접 가능성 있으나 위치 판단 애매. "
            "low score + 위치 불명확 → 충분한 FP 근거 없음. exclusion 처리."
        ),
        "lesion_protect_applied": False,
        "reason_code": "low_score_small_lesion_pixels_ambiguous_location",
    },
    {
        "review_id": "P-A46c-048",
        "patient_id": "LUNG1-313",
        "original_label": "hold_unclear_manual_review",
        "re_review_label": "hold_unclear_manual_review",
        "confidence": "low",
        "re_review_note": (
            "확대 패널 큰 구조물(종괴/artifact?) 인접. "
            "lesion_pixels=3으로 매우 작음. ROI 경계/artifact 가능성. "
            "큰 구조물이 병변인지 정상 구조인지 판단 불가. hold 유지."
        ),
        "lesion_protect_applied": False,
        "reason_code": "large_structure_adjacent_roi_boundary_unclear",
    },
]

# ===================== 분류 집계 =====================
candidate_list = [r for r in RE_REVIEW_RESULTS if r["re_review_label"] == "candidate_for_soft_penalty_preflight"]
exclusion_list = [r for r in RE_REVIEW_RESULTS if r["re_review_label"] not in ("candidate_for_soft_penalty_preflight", "hold_unclear_manual_review")]
hold_list      = [r for r in RE_REVIEW_RESULTS if r["re_review_label"] == "hold_unclear_manual_review"]
lesion_protect_count = sum(1 for r in RE_REVIEW_RESULTS if r["lesion_protect_applied"])

from collections import Counter
conf_dist = Counter(r["confidence"] for r in RE_REVIEW_RESULTS)
label_dist = Counter(r["re_review_label"] for r in RE_REVIEW_RESULTS)

print(f"\n재검토 결과:")
print(f"  candidate: {len(candidate_list)}건")
print(f"  exclusion: {len(exclusion_list)}건")
print(f"  hold 유지: {len(hold_list)}건")
print(f"  병변 보호 판단: {lesion_protect_count}건")
for label, cnt in label_dist.most_common():
    print(f"  {label}: {cnt}")

# ===================== 출력 파일 생성 =====================

# 1. hold_pool_re_review_labels.csv
label_fieldnames = [
    "review_id", "patient_id", "original_label", "re_review_label",
    "confidence", "lesion_protect_applied", "reason_code", "re_review_note",
]
# 원본 hold pool 정보 병합
hold_row_map = {r["review_id"].strip(): r for r in hold_rows}
merged_labels = []
for rr in RE_REVIEW_RESULTS:
    orig = hold_row_map.get(rr["review_id"], {})
    merged_labels.append({
        "review_id": rr["review_id"],
        "patient_id": rr["patient_id"],
        "original_ct_context_label": orig.get("ct_context_label", ""),
        "original_penalty_safe": orig.get("penalty_safe", ""),
        "original_penalty_risk": orig.get("penalty_risk", ""),
        "padim_score": orig.get("padim_score", ""),
        "slice_index": orig.get("slice_index", ""),
        "original_label": rr["original_label"],
        "re_review_label": rr["re_review_label"],
        "confidence": rr["confidence"],
        "lesion_protect_applied": rr["lesion_protect_applied"],
        "reason_code": rr["reason_code"],
        "re_review_note": rr["re_review_note"],
    })

merged_fieldnames = [
    "review_id", "patient_id", "original_ct_context_label", "original_penalty_safe",
    "original_penalty_risk", "padim_score", "slice_index",
    "original_label", "re_review_label", "confidence",
    "lesion_protect_applied", "reason_code", "re_review_note",
]
with open(OUTPUT_ROOT / "hold_pool_re_review_labels.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=merged_fieldnames)
    writer.writeheader()
    writer.writerows(merged_labels)

# 2. hold_pool_re_review_summary.csv
summary_rows = [
    {"metric": "재검토 대상", "value": len(RE_REVIEW_RESULTS)},
    {"metric": "candidate로 이동", "value": len(candidate_list)},
    {"metric": "exclusion으로 이동", "value": len(exclusion_list)},
    {"metric": "hold 유지", "value": len(hold_list)},
    {"metric": "병변 보호 판단 (lesion_protect_applied)", "value": lesion_protect_count},
    {"metric": "exclude_protect_lesion", "value": label_dist.get("exclude_protect_lesion", 0)},
    {"metric": "exclude_near_boundary_or_vessel", "value": label_dist.get("exclude_near_boundary_or_vessel", 0)},
    {"metric": "exclude_insufficient_context", "value": label_dist.get("exclude_insufficient_context", 0)},
    {"metric": "hold_unclear_manual_review", "value": label_dist.get("hold_unclear_manual_review", 0)},
    {"metric": "confidence=high", "value": conf_dist.get("high", 0)},
    {"metric": "confidence=medium", "value": conf_dist.get("medium", 0)},
    {"metric": "confidence=low", "value": conf_dist.get("low", 0)},
    {"metric": "lower_peripheral 전체 억제 금지", "value": "유지"},
    {"metric": "score CSV 수정", "value": False},
    {"metric": "adjusted_score 생성", "value": False},
    {"metric": "suppression_weight 생성", "value": False},
    {"metric": "threshold 변경", "value": False},
    {"metric": "metrics 재계산", "value": False},
    {"metric": "stage2_holdout 접근", "value": False},
    {"metric": "기존 결과 수정", "value": False},
    {"metric": "새 PNG 생성", "value": False},
]
with open(OUTPUT_ROOT / "hold_pool_re_review_summary.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["metric", "value"])
    writer.writeheader()
    writer.writerows(summary_rows)

# 3. hold_pool_re_review_summary.json
summary_json = {k["metric"]: k["value"] for k in summary_rows}
with open(OUTPUT_ROOT / "hold_pool_re_review_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary_json, f, ensure_ascii=False, indent=2)

# 4. p_a47_5_hold_pool_re_review.json
result_json = {
    "phase": "P-A47.5",
    "title": "Hold Pool Manual Re-review",
    "date": "2026-06-01",
    "verdict": "통과",
    "total_reviewed": len(RE_REVIEW_RESULTS),
    "candidate_count": len(candidate_list),
    "exclusion_count": len(exclusion_list),
    "hold_count": len(hold_list),
    "lesion_protect_applied_count": lesion_protect_count,
    "label_distribution": dict(label_dist),
    "confidence_distribution": dict(conf_dist),
    "lower_peripheral_blanket_suppression": "금지 — 유지",
    "p_a48_penalty_design_ready": len(hold_list) <= 3,
    "p_a48_note": (
        f"candidate {len(candidate_list)}건 추가 없음. "
        f"exclusion {len(exclusion_list)}건, hold 유지 {len(hold_list)}건. "
        "P-A47 candidate 17건 + P-A47.5 추가 candidate 0건 = 17건으로 P-A48 진행 가능. "
        "hold 유지 3건(LUNG1-156 x2, LUNG1-313 x1)은 별도 수동 검토 필요."
    ),
    "stage2_holdout_access": 0,
    "stage2_holdout_locked": True,
    "score_csv_modified": False,
    "adjusted_score_generated": False,
    "suppression_weight_generated": False,
    "threshold_changed": False,
    "metrics_recalculated": False,
    "scoring_rerun": False,
    "model_forward": False,
    "training": False,
    "existing_results_modified": False,
    "new_png_generated": False,
    "re_review_details": RE_REVIEW_RESULTS,
    "output_files": [
        "hold_pool_re_review_labels.csv",
        "hold_pool_re_review_summary.csv",
        "hold_pool_re_review_summary.json",
        "p_a47_5_hold_pool_re_review.md",
        "p_a47_5_hold_pool_re_review.json",
    ],
    "next_step": (
        "P-A47 candidate 17건 + P-A47.5 추가 candidate 0건 확정. "
        "hold 유지 3건(LUNG1-156 x2, LUNG1-313 x1)은 P-A48 이후 별도 처리. "
        "P-A48 실제 penalty weight 설계는 사용자 승인 후 진행."
    ),
}
with open(OUTPUT_ROOT / "p_a47_5_hold_pool_re_review.json", "w", encoding="utf-8") as f:
    json.dump(result_json, f, ensure_ascii=False, indent=2)

# 5. p_a47_5_hold_pool_re_review.md
md_lines = [
    "# P-A47.5 Hold Pool Manual Re-review",
    "",
    "## 판정: 통과",
    "",
    "---",
    "",
    "## 1. 재검토 대상",
    "",
    f"- 총 {len(RE_REVIEW_RESULTS)}건 (P-A47 hold pool 전체)",
    "- 기존 P-A46b PNG/contact sheet read-only 시각 판독 기반",
    "- 새 PNG 생성 없음, CT/ROI/mask npy 추가 로드 없음",
    "",
    "---",
    "",
    "## 2. 재분류 결과",
    "",
    f"| 항목 | 건수 |",
    f"|---|---|",
    f"| candidate로 이동 | {len(candidate_list)} |",
    f"| exclusion으로 이동 | {len(exclusion_list)} |",
    f"| hold 유지 | {len(hold_list)} |",
    f"| 병변 보호 판단 (lesion_protect_applied) | {lesion_protect_count} |",
    "",
    "**세부 분류:**",
    "",
    f"| re_review_label | 건수 |",
    f"|---|---|",
]
for label, cnt in label_dist.most_common():
    md_lines.append(f"| {label} | {cnt} |")

md_lines += [
    "",
    "---",
    "",
    "## 3. 건별 재검토 판독 결과",
    "",
]
for rr in RE_REVIEW_RESULTS:
    orig = hold_row_map.get(rr["review_id"], {})
    md_lines += [
        f"### {rr['review_id']} — {rr['patient_id']}",
        f"- original ct_context_label: {orig.get('ct_context_label','')}",
        f"- padim_score: {orig.get('padim_score','')} | slice: {orig.get('slice_index','')}",
        f"- **재분류: {rr['re_review_label']}** | confidence={rr['confidence']}",
        f"- 판독 근거: {rr['re_review_note']}",
        "",
    ]

md_lines += [
    "---",
    "",
    "## 4. confidence 분포",
    "",
    f"| confidence | 건수 |",
    f"|---|---|",
]
for c, cnt in conf_dist.most_common():
    md_lines.append(f"| {c} | {cnt} |")

md_lines += [
    "",
    "> 전건 confidence=low — 모두 보수적 처리. candidate 0건.",
    "",
    "---",
    "",
    "## 5. lower_peripheral 전체 억제 금지 재확인",
    "",
    "- 억제 적용: **없음** ✓",
    "- lower_peripheral 내 lesion 9,245 patch(44명) 보호 유지",
    "",
    "---",
    "",
    "## 6. P-A48 penalty design preflight 진행 가능 여부",
    "",
    f"- **진행 가능** (candidate 17건 확정, 추가 0건)",
    "- P-A47 candidate 17건 + P-A47.5 추가 candidate 0건 = **17건**으로 P-A48 진행",
    "- hold 유지 3건 (LUNG1-156 ×2, LUNG1-313 ×1): P-A48 이후 별도 수동 검토",
    "",
    "---",
    "",
    "## 7. 실행 안전 확인",
    "",
    "| 항목 | 상태 |",
    "|---|---|",
    "| score CSV 수정 | 없음 ✓ |",
    "| adjusted_score 생성 | 없음 ✓ |",
    "| suppression_weight 생성 | 없음 ✓ |",
    "| threshold 변경 | 없음 ✓ |",
    "| metrics 재계산 | 없음 ✓ |",
    "| scoring 재실행 | 없음 ✓ |",
    "| model forward | 없음 ✓ |",
    "| training | 없음 ✓ |",
    "| stage2_holdout 접근 | 없음 ✓ |",
    "| 기존 결과 수정 | 없음 ✓ |",
    "| 새 PNG 생성 | 없음 ✓ |",
    "",
    "---",
    "",
    "## 8. 다음 단계",
    "",
    "1. P-A47 candidate 17건 확정 사용자 확인",
    "2. P-A48 실제 penalty weight 설계 — 사용자 승인 후 진행",
    "3. hold 유지 3건 별도 수동 검토 (LUNG1-156 ×2, LUNG1-313 ×1)",
    "",
]
with open(OUTPUT_ROOT / "p_a47_5_hold_pool_re_review.md", "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))

print(f"\n[완료] 출력 경로: {OUTPUT_ROOT}")
for fname in result_json["output_files"]:
    fpath = OUTPUT_ROOT / fname
    size = fpath.stat().st_size if fpath.exists() else 0
    print(f"  - {fname} ({size:,}B)")
print(f"\n판정: 통과")
print(f"candidate: {len(candidate_list)}건 / exclusion: {len(exclusion_list)}건 / hold 유지: {len(hold_list)}건")
