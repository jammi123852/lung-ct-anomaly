"""
S4_patient_balanced 후보와 D2_grid4x4_all_suspicious_slices 후보를 합친
union manifest를 생성하는 스크립트 (manifest-only, crop 없음)

입력:
  - rule_c4_training_sampling_manifest_dryrun.csv  (S4 후보)
  - rule_d_stage1_dev_candidate_manifest_dryrun.csv (D2 후보)
  - lesion_stage_split_v1_balanced.csv             (stage split)
  - stage1_dev_v1v2_vs_v2v2_candidate_diagnostic.csv (lesion_size_bin)

출력:
  - s4_plus_d2_union_stage1_dev_candidate_manifest_dryrun.csv
  - s4_plus_d2_union_candidate_summary.csv
  - s4_plus_d2_union_candidate_summary.json
  - s4_plus_d2_union_candidate_summary.md
  - lung1_156_special_failure_note.md
"""

import sys
import json
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# 경로 설정
# ---------------------------------------------------------------------------
BASE = Path("/home/jinhy/project/lung-ct-anomaly/outputs/second-stage-lesion-refiner-v1")

INPUT_S4    = BASE / "candidates/rule_c4_training_sampling_manifest_dryrun.csv"
INPUT_D     = BASE / "candidates/rule_d_stage1_dev_candidate_manifest_dryrun.csv"
INPUT_SPLIT = BASE / "splits/lesion_stage_split_v1_balanced.csv"
INPUT_DIAG  = BASE / "reports/stage1_dev_v1v2_vs_v2v2_candidate_diagnostic.csv"

OUT_MANIFEST = BASE / "candidates/s4_plus_d2_union_stage1_dev_candidate_manifest_dryrun.csv"
OUT_SUM_CSV  = BASE / "reports/s4_plus_d2_union_candidate_summary.csv"
OUT_SUM_JSON = BASE / "reports/s4_plus_d2_union_candidate_summary.json"
OUT_SUM_MD   = BASE / "reports/s4_plus_d2_union_candidate_summary.md"
OUT_NOTE_MD  = BASE / "reports/lung1_156_special_failure_note.md"

DEDUP_COLS   = ["patient_id", "local_z", "y0", "x0", "y1", "x1"]
UNION_COLS   = ["patient_id", "local_z", "y0", "x0", "y1", "x1",
                "sampling_label", "stage_split", "source"]

STAGE1_DEV_EXPECTED = 154

# ---------------------------------------------------------------------------
# Guard: 출력 파일 존재 여부 확인
# ---------------------------------------------------------------------------
for out_path in [OUT_MANIFEST, OUT_SUM_CSV, OUT_SUM_JSON, OUT_SUM_MD, OUT_NOTE_MD]:
    if out_path.exists():
        print(f"[ABORT] 출력 파일이 이미 존재합니다: {out_path}")
        print("기존 파일을 덮어쓰지 않습니다. 수동 삭제 후 재실행하세요.")
        sys.exit(1)

# ---------------------------------------------------------------------------
# 입력 파일 로드
# ---------------------------------------------------------------------------
print("[INFO] 입력 파일 로드 중...")
df_s4_all = pd.read_csv(INPUT_S4)
df_d_all  = pd.read_csv(INPUT_D)
df_split  = pd.read_csv(INPUT_SPLIT)
df_diag   = pd.read_csv(INPUT_DIAG)

# ---------------------------------------------------------------------------
# 필터링
# ---------------------------------------------------------------------------
df_s4 = df_s4_all[df_s4_all["sampling_rule"] == "S4_patient_balanced"].copy()
df_d2 = df_d_all[df_d_all["rule_d_variant"] == "D2_grid4x4_all_suspicious_slices"].copy()

s4_count = len(df_s4)
d2_count = len(df_d2)

print(f"[INFO] S4_patient_balanced 후보 수: {s4_count}")
print(f"[INFO] D2_grid4x4_all_suspicious_slices 후보 수: {d2_count}")

# Guard: S4 후보 수 == 0
if s4_count == 0:
    print("[ABORT] S4 후보 수가 0입니다.")
    sys.exit(1)

# Guard: D2 후보 수 == 0
if d2_count == 0:
    print("[ABORT] D2 후보 수가 0입니다.")
    sys.exit(1)

# Guard: local_z 컬럼 확인
for name, df in [("S4", df_s4), ("D2", df_d2)]:
    if "local_z" not in df.columns:
        print(f"[ABORT] {name} 데이터에 local_z 컬럼이 없습니다.")
        sys.exit(1)

# ---------------------------------------------------------------------------
# stage2_holdout 환자 목록 추출
# ---------------------------------------------------------------------------
holdout_patients = set(
    df_split[df_split["stage_split"] == "stage2_holdout"]["patient_id"].tolist()
)
stage1_dev_patients = set(
    df_split[df_split["stage_split"] == "stage1_dev"]["patient_id"].tolist()
)

# Guard: stage2_holdout 환자 포함 여부 확인
s4_holdout_overlap = set(df_s4["patient_id"].unique()) & holdout_patients
d2_holdout_overlap = set(df_d2["patient_id"].unique()) & holdout_patients

if s4_holdout_overlap:
    print(f"[ABORT] S4 후보에 stage2_holdout 환자가 포함되어 있습니다: {s4_holdout_overlap}")
    sys.exit(1)

if d2_holdout_overlap:
    print(f"[ABORT] D2 후보에 stage2_holdout 환자가 포함되어 있습니다: {d2_holdout_overlap}")
    sys.exit(1)

print("[INFO] stage2_holdout 봉인 확인: 이상 없음")

# ---------------------------------------------------------------------------
# Union 컬럼 준비
# ---------------------------------------------------------------------------
# S4: source = "S4"
df_s4_union = df_s4[["patient_id", "local_z", "y0", "x0", "y1", "x1",
                      "sampling_label", "stage_split"]].copy()
df_s4_union["source"] = "S4"

# D2: stage_split 컬럼 없으므로 "stage1_dev"로 채움
df_d2_union = df_d2[["patient_id", "local_z", "y0", "x0", "y1", "x1",
                      "sampling_label"]].copy()
df_d2_union["stage_split"] = "stage1_dev"
df_d2_union["source"] = "D2"

# 컬럼 순서 정렬
df_s4_union = df_s4_union[UNION_COLS]
df_d2_union = df_d2_union[UNION_COLS]

# ---------------------------------------------------------------------------
# concat (S4 먼저 → keep="first" 시 S4 우선)
# ---------------------------------------------------------------------------
df_union_pre = pd.concat([df_s4_union, df_d2_union], ignore_index=True)
union_before_dedup = len(df_union_pre)
print(f"[INFO] concat 후 행 수 (중복 제거 전): {union_before_dedup}")

# ---------------------------------------------------------------------------
# 중복 제거 (patient_id, local_z, y0, x0, y1, x1 기준, S4 우선)
# ---------------------------------------------------------------------------
df_union = df_union_pre.drop_duplicates(subset=DEDUP_COLS, keep="first").reset_index(drop=True)
dedup_removed = union_before_dedup - len(df_union)
union_final_count = len(df_union)

print(f"[INFO] 중복 제거 수: {dedup_removed}")
print(f"[INFO] 최종 union 후보 수: {union_final_count}")

# Guard: 중복 제거 후 후보 수 == 0
if union_final_count == 0:
    print("[ABORT] 중복 제거 후 후보 수가 0입니다.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# stage1_dev 154명 확인 (경고만, 중단 안 함)
# ---------------------------------------------------------------------------
union_patients = set(df_union["patient_id"].unique())
stage1_dev_in_union = union_patients & stage1_dev_patients
if len(stage1_dev_in_union) != STAGE1_DEV_EXPECTED:
    print(f"[WARN] stage1_dev 환자 수가 {STAGE1_DEV_EXPECTED}명이 아닙니다: "
          f"{len(stage1_dev_in_union)}명 (계속 진행)")

# ---------------------------------------------------------------------------
# 지표 계산
# ---------------------------------------------------------------------------
positive_count = int((df_union["sampling_label"] == "positive").sum())
hard_negative_count = int((df_union["sampling_label"] == "hard_negative").sum())
positive_ratio = positive_count / union_final_count if union_final_count > 0 else 0.0
hard_negative_ratio = hard_negative_count / union_final_count if union_final_count > 0 else 0.0

# nohit 환자: positive 후보 없는 환자
pos_patients = set(df_union[df_union["sampling_label"] == "positive"]["patient_id"].unique())
nohit_patients = sorted(stage1_dev_patients - pos_patients)
nohit_patient_count = len(nohit_patients)

# LUNG1-415 / LUNG1-156 특수 케이스
lung1_415_hit = "LUNG1-415" in pos_patients
lung1_156_miss = "LUNG1-156" not in pos_patients

# lesion slice hit rate
lesion_slice_hit_rate = (
    len(stage1_dev_patients & pos_patients) / len(stage1_dev_patients)
    if stage1_dev_patients else 0.0
)

# lesion_size_bin별 positive 있는 환자 비율
df_diag_s1 = df_diag[df_diag["stage_split"] == "stage1_dev"][["patient_id", "lesion_size_bin"]].drop_duplicates()
lesion_size_hit_rate = {}
for size_bin, grp in df_diag_s1.groupby("lesion_size_bin"):
    patients_in_bin = set(grp["patient_id"].tolist())
    hit_in_bin = patients_in_bin & pos_patients
    lesion_size_hit_rate[size_bin] = round(
        len(hit_in_bin) / len(patients_in_bin), 4
    ) if patients_in_bin else 0.0

# per-patient 후보 수 통계
per_patient_counts = df_union.groupby("patient_id").size()
per_patient_min    = int(per_patient_counts.min())
per_patient_median = float(per_patient_counts.median())
per_patient_mean   = float(round(per_patient_counts.mean(), 2))
per_patient_max    = int(per_patient_counts.max())
max_patient_candidate_count = per_patient_max
max_patient_id = per_patient_counts.idxmax()

# group별 positive 비율 (split 파일의 group 컬럼 join)
df_group = df_split[df_split["stage_split"] == "stage1_dev"][["patient_id", "group"]].drop_duplicates()
df_union_g = df_union.merge(df_group, on="patient_id", how="left")
nsclc_df  = df_union_g[df_union_g["group"] == "NSCLC"]
msd_df    = df_union_g[df_union_g["group"].str.startswith("MSD", na=False)]
nsclc_positive_ratio = (
    float(round((nsclc_df["sampling_label"] == "positive").sum() / len(nsclc_df), 4))
    if len(nsclc_df) > 0 else 0.0
)
msd_positive_ratio = (
    float(round((msd_df["sampling_label"] == "positive").sum() / len(msd_df), 4))
    if len(msd_df) > 0 else 0.0
)

# ---------------------------------------------------------------------------
# Union manifest 저장
# ---------------------------------------------------------------------------
OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
df_union.to_csv(OUT_MANIFEST, index=False)
print(f"[OK] Union manifest 저장: {OUT_MANIFEST}")

# ---------------------------------------------------------------------------
# Summary 구성
# ---------------------------------------------------------------------------
summary = {
    "s4_count": s4_count,
    "d2_count": d2_count,
    "union_before_dedup": union_before_dedup,
    "dedup_removed": dedup_removed,
    "union_final_count": union_final_count,
    "positive_count": positive_count,
    "hard_negative_count": hard_negative_count,
    "positive_ratio": round(positive_ratio, 4),
    "hard_negative_ratio": round(hard_negative_ratio, 4),
    "nohit_patient_count": nohit_patient_count,
    "nohit_patient_list": nohit_patients,
    "lung1_415_hit": lung1_415_hit,
    "lung1_156_miss": lung1_156_miss,
    "lesion_slice_hit_rate": round(lesion_slice_hit_rate, 4),
    "lesion_size_hit_rate": lesion_size_hit_rate,
    "per_patient_min": per_patient_min,
    "per_patient_median": per_patient_median,
    "per_patient_mean": per_patient_mean,
    "per_patient_max": per_patient_max,
    "max_patient_candidate_count": max_patient_candidate_count,
    "max_patient_id": max_patient_id,
    "nsclc_positive_ratio": nsclc_positive_ratio,
    "msd_positive_ratio": msd_positive_ratio,
    "stage2_holdout_sealed": True,
}

# Summary CSV
OUT_SUM_CSV.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame([summary]).to_csv(OUT_SUM_CSV, index=False)
print(f"[OK] Summary CSV 저장: {OUT_SUM_CSV}")

# Summary JSON (nohit_patient_list를 문자열로 직렬화)
summary_json = dict(summary)
OUT_SUM_JSON.write_text(json.dumps(summary_json, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[OK] Summary JSON 저장: {OUT_SUM_JSON}")

# ---------------------------------------------------------------------------
# Summary MD
# ---------------------------------------------------------------------------
tiny_small_hit = {
    k: v for k, v in lesion_size_hit_rate.items()
    if "tiny" in k or "small" in k
}

md_lines = [
    "# S4 + D2 Union Candidate Summary",
    "",
    "## 결정 사항",
    "",
    "- D2_grid4x4_all 채택 (실제 variant명: `D2_grid4x4_all_suspicious_slices`)",
    "- D1/D3/D4/D5는 기존 파일에서 삭제하지 않고 분석에서만 제외",
    "- S4 + D2 union을 다음 후보군으로 사용",
    "- LUNG1-415는 D2에서 hit으로 회수됨",
    "- LUNG1-156은 label-free 방식으로는 현재 회수 불가",
    "- LUNG1-156 병변 local_z를 직접 후보에 넣는 것은 label leakage라 금지",
    "- stage2_holdout은 계속 봉인",
    "",
    "## 수치 요약",
    "",
    f"| 항목 | 값 |",
    f"|------|-----|",
    f"| S4 후보 수 | {s4_count:,} |",
    f"| D2 후보 수 | {d2_count:,} |",
    f"| concat 후 (중복 제거 전) | {union_before_dedup:,} |",
    f"| 중복 제거 수 | {dedup_removed:,} |",
    f"| 최종 union 후보 수 | {union_final_count:,} |",
    f"| positive 수 | {positive_count:,} |",
    f"| hard_negative 수 | {hard_negative_count:,} |",
    f"| positive 비율 | {positive_ratio:.4f} |",
    f"| hard_negative 비율 | {hard_negative_ratio:.4f} |",
    f"| no-hit 환자 수 | {nohit_patient_count} |",
    f"| lesion slice hit rate | {lesion_slice_hit_rate:.4f} |",
    f"| LUNG1-415 hit | {lung1_415_hit} |",
    f"| LUNG1-156 miss | {lung1_156_miss} |",
    f"| 환자별 후보 min | {per_patient_min} |",
    f"| 환자별 후보 median | {per_patient_median} |",
    f"| 환자별 후보 mean | {per_patient_mean} |",
    f"| 환자별 후보 max | {per_patient_max} (환자: {max_patient_id}) |",
    f"| NSCLC positive 비율 | {nsclc_positive_ratio:.4f} |",
    f"| MSD positive 비율 | {msd_positive_ratio:.4f} |",
    f"| stage2_holdout 봉인 | True |",
    "",
    "## lesion_size_bin별 positive 환자 비율",
    "",
    "| size_bin | hit_rate |",
    "|----------|----------|",
]
for bin_name, rate in sorted(lesion_size_hit_rate.items()):
    md_lines.append(f"| {bin_name} | {rate:.4f} |")

md_lines += [
    "",
    "## no-hit 환자 목록",
    "",
    ", ".join(nohit_patients) if nohit_patients else "(없음)",
    "",
    "## 다음 단계",
    "",
    "- union manifest를 기반으로 crop 생성 단계로 진행 가능",
    "- 입력 파일: `s4_plus_d2_union_stage1_dev_candidate_manifest_dryrun.csv`",
]

OUT_SUM_MD.write_text("\n".join(md_lines), encoding="utf-8")
print(f"[OK] Summary MD 저장: {OUT_SUM_MD}")

# ---------------------------------------------------------------------------
# LUNG1-156 특수 실패 케이스 노트
# ---------------------------------------------------------------------------
note_lines = [
    "# LUNG1-156 특수 실패 케이스 노트",
    "",
    "## 요약",
    "",
    "LUNG1-156은 S4 + D2 union 후보군에서도 positive 후보가 없는 특수 실패 케이스이다.",
    "",
    "## 원인 분석",
    "",
    "- LUNG1-156은 S4 suspicious slice 목록에 병변 local_z가 포함되지 않음",
    "- Rule D grid 확장은 suspicious slice 기반이므로 LUNG1-156을 회수하지 못함",
    "- 병변이 작고 흐릿한 특수 실패 케이스로 기록",
    "",
    "## 금지 사항",
    "",
    "- 병변 local_z를 직접 후보에 넣는 예외 처리는 label leakage라 금지",
    "",
    "## 후속 보완 방향",
    "",
    "- label-free slice-level fallback 방식으로만 검토 가능",
    "- 또는 별도 tiny-lesion strategy (병변 크기별 특화 방법론) 수립 후 검토",
    "- 현재 단계에서는 수용 가능한 한계로 기록하고 다음 단계로 진행",
]

OUT_NOTE_MD.write_text("\n".join(note_lines), encoding="utf-8")
print(f"[OK] LUNG1-156 note 저장: {OUT_NOTE_MD}")

# ---------------------------------------------------------------------------
# 최종 결과 출력
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("[DONE] S4 + D2 Union Manifest 생성 완료")
print("=" * 60)
print(f"  S4 후보 수            : {s4_count:,}")
print(f"  D2 후보 수            : {d2_count:,}")
print(f"  concat 후 (dedup 전)  : {union_before_dedup:,}")
print(f"  중복 제거 수           : {dedup_removed:,}")
print(f"  최종 union 후보 수    : {union_final_count:,}")
print(f"  positive 비율         : {positive_ratio:.4f}")
print(f"  hard_negative 비율    : {hard_negative_ratio:.4f}")
print(f"  no-hit 환자 수        : {nohit_patient_count}")
print(f"  LUNG1-415 hit         : {lung1_415_hit}")
print(f"  LUNG1-156 miss        : {lung1_156_miss}")
print(f"  lesion slice hit rate : {lesion_slice_hit_rate:.4f}")
print(f"  stage2_holdout 봉인   : True")
