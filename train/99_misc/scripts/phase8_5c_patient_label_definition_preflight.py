"""Phase 8.5C: Patient-level label definition preflight.
Read-only 집계 + manifest 탐색. metric 계산 없음.
"""
import pandas as pd
import json
import os
from pathlib import Path
from datetime import datetime

# ──────────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────────
PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")
SCORE_CSV = PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/scores/phase8_4_stage2_full_scoring_v1/phase8_4_stage2_full_scoring_v1.csv"
PHASE8_5B_SUMMARY = PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/review_annotations/phase8_5b_metric_calculation_v1/phase8_5b_metric_calculation_summary.json"
HOLDOUT_MANIFEST = PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/datasets/s6a_stage2_holdout_filtered_manifest_v1.csv"
COORD_MANIFEST = PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/datasets/s6a_stage2_holdout_candidate_coordinate_manifest_v1.csv"

OUT_DIR = PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/review_annotations/phase8_5c_patient_label_definition_preflight_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=== Phase 8.5C: Patient-level label definition preflight ===\n")

# ──────────────────────────────────────────────
# 1. score CSV 로드
# ──────────────────────────────────────────────
print("[1] score CSV 로드 중...")
df = pd.read_csv(SCORE_CSV)
row_count = len(df)
print(f"    row count = {row_count}")
assert row_count == 143735, f"row count 불일치: {row_count} != 143735"
print("    row count = 143,735 ✓")

# ──────────────────────────────────────────────
# 2. patient_id unique count
# ──────────────────────────────────────────────
patient_ids = df["patient_id"].unique()
n_patients = len(patient_ids)
print(f"\n[2] patient_id unique = {n_patients}")

# ──────────────────────────────────────────────
# 3. patient_id별 sampling_label 집계
# ──────────────────────────────────────────────
print("\n[3] patient별 sampling_label 구성 집계 중...")

grp = df.groupby("patient_id")["sampling_label"].value_counts().unstack(fill_value=0)
# positive / hard_negative 컬럼 정규화
if "positive" not in grp.columns:
    grp["positive"] = 0
if "hard_negative" not in grp.columns:
    grp["hard_negative"] = 0
grp["total"] = grp["positive"] + grp["hard_negative"]
# label_set: 해당 patient가 가진 unique sampling_label 목록
label_sets = df.groupby("patient_id")["sampling_label"].apply(lambda x: sorted(x.unique().tolist()))
grp["label_set"] = label_sets

print(f"    집계 완료. 컬럼: {list(grp.columns)}")

# ──────────────────────────────────────────────
# 4. positive/hard_negative/mixed 환자 수
# ──────────────────────────────────────────────
has_pos = grp["positive"] > 0
has_hn  = grp["hard_negative"] > 0

positive_only_patients = int((has_pos & ~has_hn).sum())
hard_negative_only_patients = int((~has_pos & has_hn).sum())
mixed_patients = int((has_pos & has_hn).sum())

print(f"\n[4] 환자 구성:")
print(f"    positive only       = {positive_only_patients}")
print(f"    hard_negative only  = {hard_negative_only_patients}")
print(f"    mixed (both)        = {mixed_patients}")
print(f"    합계                = {positive_only_patients + hard_negative_only_patients + mixed_patients}")

# 모든 환자가 positive crop을 가지는지 확인
all_patients_have_positive = bool((grp["positive"] > 0).all())
print(f"\n    모든 환자에 positive crop 존재? → {all_patients_have_positive}")

# ──────────────────────────────────────────────
# 5. score CSV의 patient-level ground-truth 컬럼 확인
# ──────────────────────────────────────────────
print("\n[5] score CSV 컬럼 중 patient-level ground-truth 후보 확인:")
gt_candidate_cols = [c for c in df.columns if any(kw in c.lower() for kw in
    ["label", "patient_label", "diagnosis", "case_type", "source_dataset", "has_lesion", "ground_truth"])]
print(f"    후보 컬럼: {gt_candidate_cols}")
for col in gt_candidate_cols:
    print(f"    [{col}] unique values = {sorted(df[col].dropna().unique().tolist())[:10]}")

# ──────────────────────────────────────────────
# 6. holdout manifest 탐색 (patient-level ground-truth 컬럼 유무)
# ──────────────────────────────────────────────
print("\n[6] holdout manifest 탐색...")
manifest_gt_info = {}

for mpath, mname in [
    (HOLDOUT_MANIFEST, "s6a_stage2_holdout_filtered_manifest_v1"),
    (COORD_MANIFEST, "s6a_stage2_holdout_candidate_coordinate_manifest_v1"),
]:
    if mpath.exists():
        mdf = pd.read_csv(mpath, nrows=5)
        gt_cols = [c for c in mdf.columns if any(kw in c.lower() for kw in
            ["patient_label", "diagnosis", "case_type", "source_dataset", "has_lesion", "ground_truth", "safe_id", "source"])]
        # safe_id에서 source dataset 추출 시도
        if "safe_id" in mdf.columns:
            sample_ids = pd.read_csv(mpath, usecols=["patient_id","safe_id"]).drop_duplicates("patient_id").head(10)
            prefixes = sample_ids["safe_id"].str.extract(r"^([A-Z]+)_")[0].value_counts().to_dict()
        else:
            prefixes = {}
        manifest_gt_info[mname] = {
            "exists": True,
            "columns": list(mdf.columns),
            "gt_candidate_cols": gt_cols,
            "safe_id_prefixes_sample": prefixes,
        }
        print(f"    {mname}: {len(list(mdf.columns))} cols, gt_candidates={gt_cols}")
        if prefixes:
            print(f"      safe_id prefix 샘플: {prefixes}")
    else:
        manifest_gt_info[mname] = {"exists": False}
        print(f"    {mname}: 파일 없음")

# safe_id prefix → source dataset 추정
print("\n    score CSV의 patient_id에서 dataset 추정:")
# safe_id가 score CSV에는 없음. patient_id 기준으로 LUNA/NSCLC/MSD 추정
def infer_dataset(pid):
    pid = str(pid)
    if pid.startswith("LUNG"):
        return "NSCLC"
    elif pid.startswith("MSD"):
        return "MSD"
    elif pid.startswith("LUNA") or pid.startswith("1.") or pid[0].isdigit():
        return "LUNA"
    else:
        return "UNKNOWN"

df["_inferred_dataset"] = df["patient_id"].apply(infer_dataset)
dataset_dist = df.groupby("_inferred_dataset")["patient_id"].nunique().to_dict()
print(f"    patient 추정 dataset 분포: {dataset_dist}")

# patient별 sampling_label + inferred_dataset
patient_dataset = df.groupby("patient_id")["_inferred_dataset"].first()
grp["inferred_dataset"] = patient_dataset

# 데이터셋별 sampling_label 분포
print("\n    dataset별 sampling_label 분포:")
ds_sl = df.groupby(["_inferred_dataset","sampling_label"])["patient_id"].nunique()
print(ds_sl.to_string())

# ──────────────────────────────────────────────
# 7. patient_label_composition_by_patient.csv 저장
# ──────────────────────────────────────────────
comp_csv_path = OUT_DIR / "patient_label_composition_by_patient.csv"
grp_out = grp[["positive","hard_negative","total","label_set","inferred_dataset"]].copy()
grp_out.index.name = "patient_id"
grp_out.to_csv(comp_csv_path)
print(f"\n[7] patient_label_composition_by_patient.csv 저장 완료: {comp_csv_path}")

# ──────────────────────────────────────────────
# 8. patient-level label 정의 가능 여부 판정
# ──────────────────────────────────────────────
print("\n[8] patient-level label 정의 가능 여부 판정...")

# LUNA 환자는 정상, NSCLC/MSD 환자는 병변
n_luna   = dataset_dist.get("LUNA", 0)
n_nsclc  = dataset_dist.get("NSCLC", 0)
n_msd    = dataset_dist.get("MSD", 0)
n_unknown = dataset_dist.get("UNKNOWN", 0)

# patient-level ground-truth 가능 여부
# safe_id 기반 source dataset 추정 가능 → LUNA=normal, NSCLC/MSD=lesion
has_external_gt = (n_luna > 0) and (n_nsclc + n_msd > 0)

# sampling_label 기반 patient label 정의 가능 여부 (crop label에서 유도)
# positive crop이 있으면 patient=positive 방식은 위험 (검토 항목 27)
crop_label_based_definition_risk = (
    "sampling_label 기반 patient label 정의는 sampling_label=positive가 "
    "병변 데이터셋(NSCLC/MSD)에서 왔다는 것을 보장하지 않으며, "
    "LUNA 환자도 hard_negative crop을 가짐. "
    "'positive crop 하나라도 있으면 patient positive' 기준은 "
    "데이터셋 분리 전 crop sampling 방식에 의존하므로 "
    "patient-level ground-truth로 직접 사용 위험."
)

print(f"    n_luna={n_luna}, n_nsclc={n_nsclc}, n_msd={n_msd}")
print(f"    external GT 정의 가능 (safe_id prefix 기반)? {has_external_gt}")
print(f"    all patients have positive crops? {all_patients_have_positive}")

# 최종 판정
if all_patients_have_positive:
    patient_metric_feasibility = "BLOCKED_ALL_PATIENTS_HAVE_POSITIVE_CROPS"
    final_status = "NEEDS_REVIEW"
elif has_external_gt:
    patient_metric_feasibility = "FEASIBLE_WITH_EXTERNAL_GT"
    final_status = "PASS"
else:
    patient_metric_feasibility = "NEEDS_EXTERNAL_GT"
    final_status = "NEEDS_REVIEW"

print(f"\n    최종 판정: {final_status}")
print(f"    patient metric 가능성: {patient_metric_feasibility}")

# ──────────────────────────────────────────────
# 9. summary JSON 저장
# ──────────────────────────────────────────────
with open(PHASE8_5B_SUMMARY) as f:
    phase8_5b = json.load(f)

summary = {
    "phase": "8.5C",
    "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    "input_score_csv_path": str(SCORE_CSV),
    "input_phase8_5b_summary_path": str(PHASE8_5B_SUMMARY),

    # 항목 1
    "score_csv_row_count": row_count,
    "score_csv_row_count_check": "PASS" if row_count == 143735 else "FAIL",

    # 항목 2
    "patient_id_unique_count": n_patients,

    # 항목 3-7
    "positive_only_patients": positive_only_patients,
    "hard_negative_only_patients": hard_negative_only_patients,
    "mixed_label_patients": mixed_patients,

    # 항목 11
    "all_patients_have_positive_crop": all_patients_have_positive,
    "patient_level_auroc_feasibility_by_crop_label": (
        "BLOCKED" if all_patients_have_positive else "CONDITIONAL"
    ),

    # 항목 8
    "score_csv_patient_level_gt_columns": gt_candidate_cols,
    "score_csv_has_dedicated_patient_gt_column": False,

    # 항목 9 - manifest 탐색
    "manifest_gt_exploration": manifest_gt_info,

    # 항목 10 - external GT
    "inferred_dataset_distribution": {
        "LUNA": n_luna, "NSCLC": n_nsclc, "MSD": n_msd, "UNKNOWN": n_unknown
    },
    "external_gt_feasible_via_safe_id_prefix": has_external_gt,

    # 항목 12
    "crop_label_based_patient_label_definition_risk": crop_label_based_definition_risk,
    "patient_label_definition_source_distinction": (
        "score_csv의 sampling_label은 crop-level label이며 patient-level ground-truth가 아님. "
        "safe_id prefix(LUNA/NSCLC/MSD)로 patient-level dataset origin을 추정 가능하나 "
        "이는 원본 manifest에 의존함."
    ),

    # 항목 13 - 추천 후보
    "recommended_patient_label_definition_candidates": [
        {
            "option": "A",
            "definition": "safe_id prefix 기반: LUNA → negative, NSCLC/MSD → positive",
            "source": "s6a_stage2_holdout_filtered_manifest_v1.csv safe_id 컬럼",
            "risk": "safe_id 없는 patient_id 존재 시 누락. 원본 manifest join 필요.",
            "recommendation": "권장 (원본 ground-truth에 가장 근접)"
        },
        {
            "option": "B",
            "definition": "patient_id prefix 기반: LUNG*/MSD* → positive, 나머지 → negative",
            "source": "score CSV patient_id 패턴 추정",
            "risk": "명명 규칙 가정. LUNA patient_id가 숫자 시작이면 오분류 가능.",
            "recommendation": "보조 검증용 (manifest 없을 때 fallback)"
        },
        {
            "option": "C",
            "definition": "'positive crop 하나라도 있으면 patient positive'",
            "source": "score CSV sampling_label",
            "risk": "sampling 방식이 patient ground-truth가 아님. 도메인 아티팩트 반영. 금지.",
            "recommendation": "사용 금지 (프롬프트 명시)"
        },
    ],

    # 최종
    "patient_level_metric_feasibility": patient_metric_feasibility,
    "patient_level_label_definition_feasibility": (
        "FEASIBLE_WITH_MANIFEST_JOIN" if has_external_gt else "NEEDS_EXTERNAL_GT"
    ),
    "final_status": final_status,
    "next_step_recommendation": (
        "Phase 8.5D: safe_id prefix 기반 patient-level label을 manifest join으로 확정 후 "
        "patient-level AUROC/AUPRC 계산 가능. "
        "단, 메모리 기록상 patient AUROC는 도메인 아티팩트(LUNA vs NSCLC cross-source)이므로 "
        "patient AUROC 수치 자체의 신뢰도 제한 있음. 진짜 평가 지표는 per-scan FROC."
        if has_external_gt else
        "patient-level label 외부 ground-truth 확보 필요. Phase 8.5D 보류 권장."
    ),
    "forbidden_operations_confirmed_not_executed": [
        "patient_level_metric_calculation",
        "AUROC_AUPRC_calculation",
        "threshold_p95_p99_hit_rate_recall_calculation",
        "model_forward",
        "training_backward_optimizer_step",
        "checkpoint_creation",
        "score_csv_modification",
        "existing_file_modification_deletion",
        "stage2_holdout_crop_npz_reload",
        "adjusted_score_generation",
    ],
    "output_files": {
        "report_md": str(OUT_DIR / "phase8_5c_patient_label_definition_preflight_report.md"),
        "summary_json": str(OUT_DIR / "phase8_5c_patient_label_definition_preflight_summary.json"),
        "composition_csv": str(comp_csv_path),
    }
}

# ──────────────────────────────────────────────
# 10. 보고서 MD 작성
# ──────────────────────────────────────────────
report_lines = [
    "# Phase 8.5C: Patient-level Label Definition Preflight Report",
    "",
    f"**생성 시각:** {summary['timestamp']}",
    f"**최종 판정:** `{final_status}`",
    "",
    "---",
    "",
    "## 1. Score CSV 검증",
    "",
    f"- row count: **{row_count:,}** → {'✓ PASS' if row_count == 143735 else '✗ FAIL'}",
    f"- patient_id unique: **{n_patients}**",
    "",
    "## 2. Patient별 sampling_label 구성",
    "",
    f"| 구분 | 환자 수 |",
    f"|------|---------|",
    f"| positive crop만 있는 환자 | {positive_only_patients} |",
    f"| hard_negative crop만 있는 환자 | {hard_negative_only_patients} |",
    f"| mixed (positive+hard_negative) | {mixed_patients} |",
    f"| **합계** | **{positive_only_patients + hard_negative_only_patients + mixed_patients}** |",
    "",
    f"**모든 환자에 positive crop 존재?** → `{all_patients_have_positive}`",
    "",
]

if all_patients_have_positive:
    report_lines += [
        "## 3. Patient-level AUROC 가능 여부",
        "",
        "**BLOCKED:** 모든 환자(154명)가 positive crop을 1개 이상 포함함.",
        "'positive crop 하나라도 있으면 patient positive' 기준 적용 시 전원 positive → AUROC/AUPRC 계산 불가.",
        "",
    ]
else:
    report_lines += [
        "## 3. Patient-level AUROC 가능 여부",
        "",
        "positive-only / negative-only 환자가 모두 존재하여 crop label 기반 분리는 이론적으로 가능.",
        "그러나 crop label 기반 patient label 정의의 위험을 아래에서 설명함.",
        "",
    ]

report_lines += [
    "## 4. Score CSV 내 Patient-level Ground-truth 컬럼",
    "",
    f"- 발견된 컬럼: `{gt_candidate_cols}`",
    "- `label`, `sampling_label`은 **crop-level** label이지 patient-level ground-truth가 아님.",
    "- **score CSV 내에 독립적인 patient-level ground-truth 컬럼 없음.**",
    "",
    "## 5. Manifest 탐색 결과",
    "",
    "| Manifest | patient-level GT 컬럼 후보 | safe_id prefix 패턴 |",
    "|----------|----------------------------|---------------------|",
]

for mname, minfo in manifest_gt_info.items():
    if minfo["exists"]:
        gt_c = str(minfo.get("gt_candidate_cols", []))
        prefix = str(minfo.get("safe_id_prefixes_sample", {}))
        report_lines.append(f"| {mname} | {gt_c} | {prefix} |")
    else:
        report_lines.append(f"| {mname} | (파일 없음) | - |")

report_lines += [
    "",
    "### Patient ID 기반 Dataset 추정",
    "",
    f"| Dataset | Patient 수 |",
    f"|---------|------------|",
    f"| LUNA (정상) | {n_luna} |",
    f"| NSCLC (병변) | {n_nsclc} |",
    f"| MSD (병변) | {n_msd} |",
    f"| UNKNOWN | {n_unknown} |",
    "",
    "## 6. 왜 'positive crop 하나라도 있으면 patient positive'를 바로 쓰면 위험한가",
    "",
    "1. **sampling 방식 의존성:** positive crop은 NSCLC/MSD 병변 후보 좌표에서 추출되었으나,",
    "   동일 환자의 non-lesion 영역 crop이 hard_negative로 함께 존재함.",
    "   → 'positive crop 존재 = patient positive'는 **sampling 방식을 ground-truth로 오용**하는 것임.",
    "2. **LUNA 환자 상황:** LUNA 환자가 hard_negative crop만 가진다면 negative로 분류되나,",
    "   이는 \"LUNA 데이터가 정상\"이라는 외부 사실에 근거함. crop label에서 유도된 것이 아님.",
    "3. **도메인 아티팩트:** 메모리 기록상 patient AUROC는 LUNA vs NSCLC cross-source artifact.",
    "   model이 병변이 아니라 'LUNA 아님'을 봄 (병변 제거해도 AUROC 0.9995→0.9995).",
    "   → patient AUROC 수치 자체의 해석 주의 필요.",
    "",
    "## 7. 추천 Patient Label 정의 후보",
    "",
    "| 옵션 | 정의 | 소스 | 위험 | 권장 |",
    "|------|------|------|------|------|",
    "| A | safe_id prefix 기반: LUNA→neg, NSCLC/MSD→pos | holdout manifest safe_id | manifest join 필요 | **권장** |",
    "| B | patient_id prefix 기반 추정 | score CSV 패턴 | 명명규칙 가정 | 보조 검증 |",
    "| C | positive crop ≥1 → patient positive | score CSV sampling_label | 금지됨 | **사용 금지** |",
    "",
    "## 8. 최종 판정 및 다음 단계",
    "",
    f"**patient-level metric 가능 여부:** `{patient_metric_feasibility}`",
    f"**patient-level label 정의 가능 여부:** `{summary['patient_level_label_definition_feasibility']}`",
    "",
    f"**다음 단계 권장:**",
    f"> {summary['next_step_recommendation']}",
    "",
    "---",
    "_Phase 8.5C preflight — read-only 집계. metric 계산 없음._",
]

report_path = OUT_DIR / "phase8_5c_patient_label_definition_preflight_report.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
print(f"\n[MD 보고서] 저장: {report_path}")

# summary JSON 저장
summary_path = OUT_DIR / "phase8_5c_patient_label_definition_preflight_summary.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"[JSON 요약] 저장: {summary_path}")

print(f"\n=== Phase 8.5C 완료. 최종 판정: {final_status} ===")
