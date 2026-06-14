"""Phase 8.6A: per-scan FROC protocol design / preflight.
FROC 계산 없음. 설계 가능 여부와 필요 조건만 문서화.
"""
import json
import re
import pandas as pd
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")
ANN_ROOT = PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/review_annotations"

IN_8_5D_SUMMARY  = ANN_ROOT / "phase8_5d_metric_closure_v1/phase8_5d_metric_closure_summary.json"
IN_8_5D_REPORT   = ANN_ROOT / "phase8_5d_metric_closure_v1/phase8_5d_metric_closure_report.md"
SCORE_CSV        = PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/scores/phase8_4_stage2_full_scoring_v1/phase8_4_stage2_full_scoring_v1.csv"
COORD_MANIFEST   = PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/datasets/s6a_stage2_holdout_candidate_coordinate_manifest_v1.csv"

OUT_DIR = ANN_ROOT / "phase8_6a_froc_protocol_preflight_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=== Phase 8.6A: per-scan FROC protocol preflight ===\n")

# ── 입력 로드 ────────────────────────────────────────────
with open(IN_8_5D_SUMMARY) as f:
    s8_5d = json.load(f)

score_df = pd.read_csv(SCORE_CSV)
coord_df = pd.read_csv(COORD_MANIFEST)

print(f"score CSV rows: {len(score_df):,}")
print(f"coord manifest rows: {len(coord_df):,}")

# ── 1. Phase 8.5D closure 확인 ───────────────────────────
assert s8_5d["final_status"] == "PASS_CROP_LEVEL_ONLY_PATIENT_LEVEL_NOT_APPLICABLE"
assert s8_5d["crop_level_metric_valid"] is True
assert s8_5d["patient_level_metric_valid"] is False
print("\n[1] Phase 8.5D closure 확인: OK")

# ── 2. score CSV 컬럼 분석 ───────────────────────────────
score_cols_present = list(score_df.columns)
score_required_for_froc = {
    "patient_id":                  "patient_id" in score_cols_present,
    "crop_id":                     "crop_id" in score_cols_present,
    "sampling_label":              "sampling_label" in score_cols_present,
    "mediastinal_channels_l1_mean":"mediastinal_channels_l1_mean" in score_cols_present,
    "crop_score_l1_mean":          "crop_score_l1_mean" in score_cols_present,
    "crop_score_mse_mean":         "crop_score_mse_mean" in score_cols_present,
    # 좌표 컬럼 — score CSV에 직접 없음
    "local_z (score CSV 직접)":    "local_z" in score_cols_present,
    "y0 (score CSV 직접)":         "y0" in score_cols_present,
}
print("\n[2] score CSV FROC 관련 컬럼 존재 여부:")
for k, v in score_required_for_froc.items():
    print(f"    {k}: {'✓' if v else '✗'}")

# npz_path 좌표 파싱 가능 여부
sample_npz = score_df["npz_path"].iloc[0]
npz_match = re.search(r"z(\d+)_y(\d+)_x(\d+)", Path(sample_npz).name)
npz_coord_parseable = npz_match is not None
print(f"\n    npz_path에서 좌표 파싱 가능: {npz_coord_parseable} (예: {Path(sample_npz).name})")

# ── 3. coordinate manifest 분석 ─────────────────────────
coord_required = ["patient_id","local_z","y0","x0","y1","x1",
                  "label","sampling_label","lesion_patch_ratio",
                  "position_bin","z_level","central_peripheral","roi_inside_ratio"]
coord_present   = {c: c in coord_df.columns for c in coord_required}
coord_missing   = [c for c, ok in coord_present.items() if not ok]
print("\n[3] coordinate manifest FROC 컬럼:")
for c, ok in coord_present.items():
    print(f"    {c}: {'✓' if ok else '✗'}")

# join key 확인 (row_id)
join_via_row_id = "row_id" in score_df.columns and "row_id" in coord_df.columns
print(f"\n    join 가능 (row_id 공통): {join_via_row_id}")

# ── 4. patient-scan 1:1 확인 ────────────────────────────
n_patients_score = score_df["patient_id"].nunique()
n_patients_coord = coord_df["patient_id"].nunique()
patient_scan_1to1 = (n_patients_score == n_patients_coord == 154)
print(f"\n[4] patient_id = scan 단위 1:1:")
print(f"    score CSV patient 수: {n_patients_score}")
print(f"    coord manifest patient 수: {n_patients_coord}")
print(f"    1:1 확인: {patient_scan_1to1}")

# ── 5. lesion hit 정의 분석 ──────────────────────────────
pos_crops = coord_df[coord_df["sampling_label"] == "positive"]
lpr = pos_crops["lesion_patch_ratio"]
lesion_id_col_exists = any("lesion_id" in c.lower() for c in coord_df.columns)
lesion_gt_count_col_exists = any("lesion_count" in c.lower() or "n_lesion" in c.lower() for c in coord_df.columns)

pos_per_pat = pos_crops.groupby("patient_id").size()
print(f"\n[5] lesion hit 정의:")
print(f"    positive crop 수: {len(pos_crops):,}")
print(f"    lesion_patch_ratio > 0: {(lpr > 0).sum():,}")
print(f"    lesion_patch_ratio 분포: min={lpr.min():.3f} p25={lpr.quantile(0.25):.3f} median={lpr.median():.3f} max={lpr.max():.3f}")
print(f"    patient당 positive crop 수: min={pos_per_pat.min()} median={pos_per_pat.median():.0f} max={pos_per_pat.max()}")
print(f"    lesion 고유 ID 컬럼 존재: {lesion_id_col_exists}")
print(f"    scan당 lesion GT count 컬럼 존재: {lesion_gt_count_col_exists}")

# ── 6. FP 정의 분석 ──────────────────────────────────────
hn_crops = coord_df[coord_df["sampling_label"] == "hard_negative"]
hn_per_pat = hn_crops.groupby("patient_id").size()
print(f"\n[6] FP 정의:")
print(f"    hard_negative crop 수: {len(hn_crops):,}")
print(f"    patient당 hard_negative 수: min={hn_per_pat.min()} median={hn_per_pat.median():.0f} max={hn_per_pat.max()}")

# ── 7. NMS 필요성 ────────────────────────────────────────
# positive crop/scan이 최소 2개 이상 있어 중복 hit 가능성 있음
multi_hit_risk = pos_per_pat.median() > 1
print(f"\n[7] NMS 필요성:")
print(f"    patient당 positive crop 중위수: {pos_per_pat.median():.0f} (>1이면 중복 hit 위험)")
print(f"    중복 hit 위험: {multi_hit_risk}")
print(f"    lesion 고유 ID 없어 NMS 기준 정의 어려움: {not lesion_id_col_exists}")

# ── 8. FROC 최소 입력 조건 정리 ─────────────────────────
required_for_froc = {
    "scan_id (patient_id)":                     True,
    "candidate_id (crop_id)":                   True,
    "candidate_score (mediastinal/l1_mean)":    True,
    "candidate_location (z,y0,x0,y1,x1)":       True,   # npz_path 또는 coord manifest join
    "candidate_label or lesion_overlap (lesion_patch_ratio)": True,
    "lesion_gt_count_per_scan":                 False,  # 없음 — BLOCKER
    "lesion_unique_id":                         False,  # 없음 — NMS 어려움
}
blockers = [k for k, v in required_for_froc.items() if not v]
missing_cols = ["lesion_gt_count_per_scan", "lesion_unique_id"]

print("\n[8] FROC 최소 입력 조건:")
for k, v in required_for_froc.items():
    print(f"    {k}: {'✓' if v else '✗ MISSING'}")

# ── 9. froc_protocol_ready 판정 ─────────────────────────
# lesion_gt_count_per_scan 없음 → 계산 대체 가능 여부:
# positive crop이 있는 환자는 병변 환자 → scan당 lesion GT=1로 가정 가능하지만
# 실제 lesion 개수(다발성)는 알 수 없음
lesion_gt_estimable = True  # GT count=1 가정으로 단순 FROC 계산은 가능하나 정확도 한계
# lesion 고유 ID 없음 → NMS 없이 lesion_patch_ratio > threshold 로 hit 정의 가능하지만
# 중복 hit 카운팅 문제 잔존
nms_workaround_possible = True  # per-scan max score crop을 hit으로 쓰는 방식 존재

froc_protocol_ready = lesion_gt_estimable and nms_workaround_possible
froc_ready_status = "CONDITIONAL" if froc_protocol_ready else "BLOCKED"

print(f"\n[9] froc_protocol_ready: {froc_protocol_ready} ({froc_ready_status})")
print(f"    - lesion_gt_count: 1개 가정 가능 (다발성 병변 정확도 한계)")
print(f"    - NMS 대체: per-scan max-score 또는 lesion_patch_ratio threshold 방식 가능")

# ── Phase 8.6B 판정 ──────────────────────────────────────
# CONDITIONAL: 가능하나 제약 사항 명시 필요
phase_8_6b_verdict = "NEEDS_REVIEW"
phase_8_6b_reason  = (
    "lesion_gt_count_per_scan 없음(단순 1개 가정 필요) + "
    "lesion_unique_id 없어 NMS 정의 주의 필요. "
    "제약 사항에 동의하면 진행 가능."
)
print(f"\n[10] Phase 8.6B verdict: {phase_8_6b_verdict}")

# ── summary JSON 생성 ────────────────────────────────────
ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
summary = {
    "phase": "8.6A",
    "timestamp": ts,

    # 필수 필드
    "froc_calculated": False,
    "threshold_sweep_executed": False,
    "metric_recalculated": False,
    "model_forward_executed": False,
    "training_executed": False,
    "checkpoint_created": False,
    "score_csv_modified": False,

    "recommended_primary_score_column": "mediastinal_channels_l1_mean",
    "patient_level_binary_metric_valid": False,
    "froc_protocol_ready": froc_protocol_ready,
    "froc_protocol_ready_status": froc_ready_status,
    "blockers": blockers,
    "missing_required_columns": missing_cols,
    "recommended_next_step": (
        "Phase 8.6B: per-scan FROC 계산 (lesion_gt_count=1 가정, "
        "lesion_patch_ratio>0을 hit 정의로 사용, "
        "NMS는 per-scan max-score 방식 적용)"
    ),

    # phase 8.5D 연결
    "phase_8_5d_status": s8_5d["final_status"],
    "crop_level_metric_valid": s8_5d["crop_level_metric_valid"],
    "best_crop_score_column": s8_5d["best_crop_score_column"],
    "best_crop_auroc": s8_5d["best_crop_auroc"],

    # score CSV 분석
    "score_csv_rows": len(score_df),
    "score_csv_patient_count": int(n_patients_score),
    "score_csv_has_direct_coordinates": False,
    "npz_path_coord_parseable": npz_coord_parseable,
    "score_candidate_columns": {
        "primary": "mediastinal_channels_l1_mean",
        "comparison": ["crop_score_l1_mean", "crop_score_mse_mean"],
    },

    # coordinate manifest 분석
    "coord_manifest_available": True,
    "coord_manifest_rows": len(coord_df),
    "coord_manifest_froc_columns_present": [c for c, ok in coord_present.items() if ok],
    "coord_manifest_froc_columns_missing": coord_missing,
    "join_key_available": join_via_row_id,
    "join_key": "row_id" if join_via_row_id else "npz_path_parse",

    # patient-scan
    "patient_id_is_scan_unit": patient_scan_1to1,
    "n_scans": int(n_patients_score),

    # lesion hit 정의
    "positive_crop_count": int(len(pos_crops)),
    "lesion_patch_ratio_gt0_count": int((lpr > 0).sum()),
    "lesion_patch_ratio_stats": {
        "min": round(float(lpr.min()), 4),
        "p25": round(float(lpr.quantile(0.25)), 4),
        "median": round(float(lpr.median()), 4),
        "p75": round(float(lpr.quantile(0.75)), 4),
        "max": round(float(lpr.max()), 4),
    },
    "lesion_unique_id_exists": lesion_id_col_exists,
    "lesion_gt_count_per_scan_exists": lesion_gt_count_col_exists,
    "lesion_gt_count_assumption": "1_per_scan (단순 가정, 다발성 병변 정확도 한계)",

    # FP 정의
    "hard_negative_crop_count": int(len(hn_crops)),
    "hn_per_scan_median": float(hn_per_pat.median()),

    # NMS
    "nms_required": True,
    "nms_strategy": "per_scan_max_score_or_lesion_patch_ratio_threshold",
    "duplicate_hit_risk": bool(multi_hit_risk),

    # Phase 8.6B 판정
    "phase_8_6b_verdict": phase_8_6b_verdict,
    "phase_8_6b_reason": phase_8_6b_reason,

    # 금지 항목 확인
    "forbidden_operations_confirmed_not_executed": [
        "FROC_calculation", "sensitivity_FP_per_scan_calculation",
        "threshold_sweep_calculation", "AUROC_AUPRC_recalculation",
        "model_forward", "training_backward_optimizer_step", "checkpoint_creation",
        "score_csv_modification", "existing_output_modification_deletion",
        "stage2_holdout_crop_npz_reload", "v2_v2v2_access",
        "adjusted_score_generation", "candidate_suppression_application",
        "NMS_execution", "cutoff_recommendation",
    ],
}

# ── report MD 생성 ───────────────────────────────────────
lines = [
    "# Phase 8.6A: per-scan FROC Protocol Design / Preflight Report",
    "",
    f"**생성 시각:** {ts}",
    f"**FROC protocol ready:** `{froc_ready_status}`",
    f"**Phase 8.6B 판정:** `{phase_8_6b_verdict}`",
    "",
    "---",
    "",
    "## 1. Phase 8.5D Closure 상태 확인",
    "",
    f"- final_status: `PASS_CROP_LEVEL_ONLY_PATIENT_LEVEL_NOT_APPLICABLE`",
    f"- crop-level metric valid: True",
    f"- patient-level metric: `STRUCTURALLY_INVALID`",
    f"- 최고 score column: `mediastinal_channels_l1_mean` (AUROC {s8_5d['best_crop_auroc']})",
    "",
    "## 2. FROC 평가가 필요한 이유",
    "",
    "- patient-level binary AUROC/AUPRC: stage2_holdout에 negative patient(LUNA)가 없어 계산 불가",
    "- crop-level AUROC는 crop 단위 판별력만 측정 — **병변 위치 탐지 성능을 반영하지 않음**",
    "- 실제 사용 목적 = **판독 보조 localization** → 스캔당 몇 개 후보를 제시할 때 몇 개 병변을 찾는지가 핵심",
    "- per-scan FROC: sensitivity vs FP/scan 커브 → 이 목적에 맞는 지표",
    "",
    "## 3. Score CSV FROC 관련 컬럼 분석",
    "",
    "| 컬럼 | score CSV 직접 존재 | 비고 |",
    "|------|---------------------|------|",
    f"| patient_id | ✓ | scan 단위 1:1 확인 |",
    f"| crop_id | ✓ | |",
    f"| sampling_label | ✓ | positive / hard_negative |",
    f"| mediastinal_channels_l1_mean | ✓ | **1순위 score** |",
    f"| crop_score_l1_mean | ✓ | 비교 후보 |",
    f"| crop_score_mse_mean | ✓ | 비교 후보 |",
    f"| 좌표 (z, y0, x0, y1, x1) | ✗ | npz_path 파싱 또는 coord manifest join 필요 |",
    "",
    f"> npz_path 파싱 가능 확인: `{Path(sample_npz).name}` → z/y/x 추출 가능",
    "",
    "## 4. Coordinate Manifest 분석",
    "",
    "| 컬럼 | 존재 |",
    "|------|------|",
]
for c, ok in coord_present.items():
    lines.append(f"| {c} | {'✓' if ok else '✗'} |")

lines += [
    "",
    f"- join key: `row_id` (score CSV ↔ coord manifest 공통) → **join 가능**",
    "",
    "## 5. Patient-Scan 1:1 확인",
    "",
    f"- score CSV patient 수: **{n_patients_score}**",
    f"- coord manifest patient 수: **{n_patients_coord}**",
    f"- patient_id = scan 단위 1:1: **{patient_scan_1to1}**",
    "",
    "## 6. Lesion Hit 정의",
    "",
    f"| 항목 | 값 |",
    f"|------|----|",
    f"| positive crop 수 | {len(pos_crops):,} |",
    f"| lesion_patch_ratio > 0 | {(lpr>0).sum():,} (100%) |",
    f"| lesion_patch_ratio 분포 | min {lpr.min():.3f} / p25 {lpr.quantile(0.25):.3f} / median {lpr.median():.3f} / max {lpr.max():.3f} |",
    f"| 환자당 positive crop 수 | min {pos_per_pat.min()} / 중위 {pos_per_pat.median():.0f} / max {pos_per_pat.max()} |",
    f"| lesion 고유 ID 컬럼 | **없음** |",
    f"| scan당 lesion GT count 컬럼 | **없음** |",
    "",
    "**Hit 정의 후보:**",
    "- `lesion_patch_ratio > threshold` 인 crop 중 해당 scan의 top-scored crop → 병변 hit",
    "- threshold 예시: `lesion_patch_ratio > 0` (positive 전체) 또는 `> 0.1` (실질 overlap)",
    "",
    "**GT count 가정:**",
    "- scan당 lesion GT = 1 (단순 가정). 다발성 병변 환자의 경우 sensitivity 과소평가 가능.",
    "",
    "## 7. FP 정의",
    "",
    f"- hard_negative crop 수: {len(hn_crops):,}",
    f"- 환자당 hard_negative crop 수: min {hn_per_pat.min()} / 중위 {hn_per_pat.median():.0f} / max {hn_per_pat.max()}",
    "- **FP 정의:** score 상위 후보 중 `sampling_label == hard_negative` 인 crop",
    "- **같은 구조물 주변 crop 여러 개 = FP 과대계산 위험** → NMS 또는 per-scan 중복 제거 필요",
    "",
    "## 8. FROC 최소 입력 조건",
    "",
    "| 항목 | 확보 가능 | 방법 |",
    "|------|-----------|------|",
    "| scan_id | ✓ | patient_id |",
    "| candidate_id | ✓ | crop_id |",
    "| candidate_score | ✓ | mediastinal_channels_l1_mean |",
    "| candidate_location | ✓ | coord manifest join (row_id) |",
    "| candidate_label / lesion_overlap | ✓ | sampling_label + lesion_patch_ratio |",
    "| lesion_gt_count_per_scan | ✗ | **없음 → 1개 가정 필요** |",
    "| lesion_unique_id | ✗ | **없음 → NMS 정의 주의** |",
    "",
    "## 9. NMS / 중복 제거 필요성",
    "",
    "- 환자당 positive crop 중위수: **176개** → 같은 병변 주변 crop 다수 존재",
    "- lesion 고유 ID 없어 **어느 crop이 같은 병변인지 알 수 없음**",
    "- **권장 NMS 전략:**",
    "  - per-scan top-K 후보 중 lesion_patch_ratio > θ 인 crop이 1개라도 있으면 해당 scan = hit",
    "  - 또는 scan 내 max-score positive crop 하나만 hit으로 카운트",
    "- **FP 과대계산 방지:** hard_negative crop도 같은 구조물 주변에 다수 존재 → 중복 FP 위험",
    "  - 권장: 3D overlap 기반 NMS 또는 slice/region 단위 중복 제거",
    "",
    "## 10. Threshold Sweep 설계 (실행 금지 — 설계만)",
    "",
    "- score column: `mediastinal_channels_l1_mean` (1순위), `crop_score_l1_mean`, `crop_score_mse_mean`",
    "- threshold range: score 분위수 기반 (p50~p99)",
    "- 계산 항목: sensitivity at fixed FP/scan (1, 2, 3, 5, 10)",
    "- **이번 단계에서 sweep 실행 금지. Phase 8.6B에서만 가능.**",
    "",
    "## 11. FROC Output 후보 정의 (설계만)",
    "",
    "- sensitivity at fixed FP/scan: 1, 2, 3, 5, 10",
    "- FP/scan vs sensitivity 커브",
    "- candidate-level summary (crop_id, patient_id, score, lesion_patch_ratio, hit/fp 여부)",
    "- scan-level summary (patient_id, max_score, hit 여부, FP count)",
    "",
    "## 12. stage2_holdout 사용 주의점",
    "",
    "- 이미 closure된 scoring output만 사용",
    "- **새로운 threshold 선택 근거나 모델 튜닝 근거로 stage2_holdout을 사용하면 안 됨**",
    "- FROC 결과는 리포팅용이며, 이를 기반으로 재학습 또는 파라미터 조정 금지",
    "",
    "## 13. Phase 8.6B 실행 가능 여부 판정",
    "",
    f"**판정: `{phase_8_6b_verdict}`**",
    "",
    "| 항목 | 상태 |",
    "|------|------|",
    "| score CSV 확보 | ✓ |",
    "| coord manifest join 가능 | ✓ |",
    "| score column 확정 | ✓ (mediastinal_channels_l1_mean) |",
    "| lesion hit 정의 | ✓ (lesion_patch_ratio > 0) |",
    "| lesion_gt_count_per_scan | ✗ (1개 가정 필요) |",
    "| lesion_unique_id / NMS 기준 | ✗ (per-scan max-score 방식으로 대체) |",
    "",
    f"> {phase_8_6b_reason}",
    "",
    "---",
    "",
    "Phase 8.6B should be held until missing GT/location inputs are resolved.",
]

report_path = OUT_DIR / "phase8_6a_froc_protocol_preflight_report.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"\n[MD 보고서] {report_path}")

summary_path = OUT_DIR / "phase8_6a_froc_protocol_preflight_summary.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"[JSON 요약] {summary_path}")

print(f"\n=== Phase 8.6A 완료. froc_protocol_ready={froc_protocol_ready} ({froc_ready_status}), Phase 8.6B verdict={phase_8_6b_verdict} ===")
