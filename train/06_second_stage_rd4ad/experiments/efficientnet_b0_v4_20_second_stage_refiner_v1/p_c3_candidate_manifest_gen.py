"""
P-C3 Candidate Manifest Generation
EfficientNet-B0 v4_20 ROI branch

금지:
- crop 생성 금지
- 2차학습 금지
- model forward 금지
- scoring 재실행 금지
- stage2_holdout 접근 금지
"""

import pandas as pd
import numpy as np
import json
import datetime
import sys
from pathlib import Path

# ── 경로 설정 ───────────────────────────────────────────────────────────────
BASE = Path("/home/jinhy/project/lung-ct-anomaly")
WORKSPACE = BASE / "experiments/efficientnet_b0_v4_20_second_stage_refiner_v1"
SCORE_DIR = BASE / "experiments/efficientnet_b0_imagenet_chestwall_removed_roi_v1/outputs/scores/lesion_stage1_dev_by_patient"
THRESHOLD_JSON = BASE / "experiments/efficientnet_b0_imagenet_chestwall_removed_roi_v1/outputs/evaluation/normal_val_thresholds/normal_val_threshold.json"
SPLIT_CSV = BASE / "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1_balanced.csv"
P_C2_DIR = WORKSPACE / "outputs/reports/p_c2_candidate_extraction_preflight"
OUTPUT_DIR = WORKSPACE / "outputs/candidates/p_c3_candidate_manifest"

# [보완1] 덮어쓰기 방지: 이미 존재하면 즉시 중단
if OUTPUT_DIR.exists():
    print(f"[ERROR] 출력 경로가 이미 존재합니다. 덮어쓰기 방지로 중단합니다: {OUTPUT_DIR}")
    print("  → 재실행하려면 해당 폴더를 먼저 삭제하거나 이름을 변경하세요.")
    sys.exit(1)
OUTPUT_DIR.mkdir(parents=True)

# ── 상수 ────────────────────────────────────────────────────────────────────
BRANCH = "efficientnet_b0_v4_20_roi"
NO_HIT_PATIENTS = {"LUNG1-086", "LUNG1-386", "MSD_lung_096"}
TINY_LESION_PATIENTS = {"LUNG1-156", "LUNG1-192", "LUNG1-311", "LUNG1-386"}
RISK6_PATIENTS = {"LUNG1-386", "LUNG1-156", "LUNG1-028", "LUNG1-306", "LUNG1-421", "LUNG1-295"}
DOMINANT_BINS = {"middle_peripheral", "lower_peripheral"}
TARGET_POS_HN_RATIO = 3   # positive : hard_negative = 1:3
FALLBACK_TOP_K = 20       # no-hit 환자당 fallback positive 최대 수
RAND_SEED = 42

errors = []
t_start = datetime.datetime.now()

# ── 1. threshold 로드 ────────────────────────────────────────────────────────
print("[1] threshold JSON 로드")
with open(THRESHOLD_JSON) as f:
    thr = json.load(f)
P95 = thr["threshold_p95"]   # 13.231265
P99 = thr["threshold_p99"]   # 15.472385
assert abs(P95 - 13.231265) < 1e-4, f"p95 mismatch: {P95}"
assert abs(P99 - 15.472385) < 1e-4, f"p99 mismatch: {P99}"
print(f"  p95={P95:.6f}  p99={P99:.6f}")

# ── 2. split CSV 로드 ────────────────────────────────────────────────────────
print("[2] split CSV 로드")
split_df = pd.read_csv(SPLIT_CSV)
stage1_dev_set = set(split_df[split_df["stage_split"] == "stage1_dev"]["patient_id"])
stage2_holdout_set = set(split_df[split_df["stage_split"] == "stage2_holdout"]["patient_id"])
split_map = split_df.set_index("patient_id")["stage_split"].to_dict()
group_map = split_df.set_index("patient_id")["group"].to_dict()
safe_id_map = split_df.set_index("patient_id")["safe_id"].to_dict()
print(f"  stage1_dev={len(stage1_dev_set)}  stage2_holdout={len(stage2_holdout_set)}")

# ── 3. score CSV 로드 (stage1_dev only, stage2_holdout 차단) ─────────────────
print("[3] score CSV 로드")

# [보완2] 필수 source 컬럼 정의
REQUIRED_SOURCE_COLS = [
    "patient_id", "slice_index", "local_z",
    "y0", "x0", "y1", "x1",
    "position_bin", "z_level", "padim_score",
    "has_lesion_patch", "lesion_pixels",
]
COORD_COLS = ["slice_index", "local_z", "y0", "x0", "y1", "x1"]

dfs = []
loaded_patients = []
for csv_file in sorted(SCORE_DIR.glob("*.csv")):
    pid = csv_file.stem
    if pid in stage2_holdout_set:
        errors.append({"patient_id": pid, "stage": "load", "error": "BLOCKED: stage2_holdout"})
        continue
    if pid not in stage1_dev_set:
        errors.append({"patient_id": pid, "stage": "load", "error": "not_in_stage1_dev_skipped"})
        continue
    try:
        df = pd.read_csv(csv_file)
        # [보완2] 필수 컬럼 누락 시 즉시 중단 (None 채워서 통과 금지)
        missing_src = [c for c in REQUIRED_SOURCE_COLS if c not in df.columns]
        if missing_src:
            print(f"[ERROR] {pid}: 필수 source 컬럼 누락 → {missing_src}")
            sys.exit(1)
        df["_patient_id_source"] = pid
        dfs.append(df)
        loaded_patients.append(pid)
    except SystemExit:
        raise
    except Exception as e:
        errors.append({"patient_id": pid, "stage": "load", "error": str(e)})

all_df = pd.concat(dfs, ignore_index=True)
n_loaded = len(loaded_patients)
print(f"  로드 완료: {n_loaded}명  총 patch={len(all_df):,}")

# stage2_holdout contamination 검사
if "patient_id" in all_df.columns:
    contamination = all_df["patient_id"].isin(stage2_holdout_set).sum()
else:
    contamination = 0
assert contamination == 0, f"stage2_holdout contamination: {contamination}"
print(f"  stage2_holdout contamination={contamination}  PASS")

# ── 4. 전처리 ───────────────────────────────────────────────────────────────
print("[4] 전처리")
all_df["has_lesion_patch"] = all_df["has_lesion_patch"].astype(int).astype(bool)
all_df["is_positive"] = (all_df["has_lesion_patch"] == True) | (all_df["lesion_pixels"] > 0)

# ── 5. Rule A / D 적용 ──────────────────────────────────────────────────────
print("[5] Rule A / D 적용")
rule_a_mask = all_df["padim_score"] > P95
rule_d_mask = all_df["padim_score"] > P99

rule_a_df = all_df[rule_a_mask].copy()
rule_a_pos = rule_a_df[rule_a_df["is_positive"]].copy()
rule_a_hn = rule_a_df[~rule_a_df["is_positive"]].copy()

rule_d_df = rule_a_df[rule_d_mask].copy()
rule_d_pos = rule_d_df[rule_d_df["is_positive"]].copy()
rule_d_hn = rule_d_df[~rule_d_df["is_positive"]].copy()

print(f"  Rule A total={len(rule_a_df):,}  pos={len(rule_a_pos):,}  hn={len(rule_a_hn):,}")
print(f"  Rule D total={len(rule_d_df):,}  pos={len(rule_d_pos):,}  hn={len(rule_d_hn):,}")

# ── 6. positive candidate 구성 ───────────────────────────────────────────────
print("[6] positive candidate 구성")

def make_pos_row(df, rule, sampling_reason, fallback_below_p95=False, no_hit=False):
    out = df.copy()
    out["candidate_label"] = "positive"
    out["candidate_rule"] = rule
    out["sampling_reason"] = sampling_reason
    out["fallback_positive_below_p95"] = fallback_below_p95
    out["no_hit_patient"] = no_hit
    return out

# Rule A positive 전부 포함
pos_rule_a = make_pos_row(rule_a_pos, "rule_a", "rule_a_positive")

# Rule D positive: 이미 rule_a_pos에 포함되어 있으므로 rule 표시만 업데이트
pos_rule_a.loc[pos_rule_a["padim_score"] > P99, "candidate_rule"] = "rule_d_p99"

print(f"  Rule A positive={len(pos_rule_a):,}  (rule_d 포함)")

# ── 7. no-hit 3명 fallback positive ─────────────────────────────────────────
print("[7] no-hit 3명 fallback positive")
fallback_summary_rows = []
fallback_dfs = []

for pid in sorted(NO_HIT_PATIENTS):
    pat_df = all_df[all_df["patient_id"] == pid].copy()
    pat_lesion = pat_df[pat_df["is_positive"] == True].copy()
    n_lesion = len(pat_lesion)

    if n_lesion == 0:
        fallback_summary_rows.append({
            "patient_id": pid,
            "status": "no_positive_patch_available",
            "n_total_patches": len(pat_df),
            "n_lesion_patches": 0,
            "n_fallback_selected": 0,
            "fallback_score_min": None,
            "fallback_score_max": None,
            "no_positive_patch_available": True,
            "fallback_positive_below_p95": False,
        })
        errors.append({"patient_id": pid, "stage": "fallback", "error": "no_lesion_patch_available"})
    else:
        selected = pat_lesion.nlargest(min(FALLBACK_TOP_K, n_lesion), "padim_score")
        fb_df = make_pos_row(
            selected,
            rule="fallback_positive_below_p95",
            sampling_reason="no_hit_patient_fallback",
            fallback_below_p95=True,
            no_hit=True,
        )
        fallback_dfs.append(fb_df)
        fallback_summary_rows.append({
            "patient_id": pid,
            "status": "fallback_added",
            "n_total_patches": len(pat_df),
            "n_lesion_patches": n_lesion,
            "n_fallback_selected": len(selected),
            "fallback_score_min": float(selected["padim_score"].min()),
            "fallback_score_max": float(selected["padim_score"].max()),
            "no_positive_patch_available": False,
            "fallback_positive_below_p95": True,
        })
        print(f"  {pid}: fallback {len(selected)}개  score_max={float(selected['padim_score'].max()):.4f}")

# 전체 positive 합치기
all_pos_parts = [pos_rule_a] + fallback_dfs
all_positives = pd.concat(all_pos_parts, ignore_index=True)

# no_hit, tiny_lesion, risk6 flag 설정
all_positives["no_hit_patient"] = all_positives.apply(
    lambda r: True if r["patient_id"] in NO_HIT_PATIENTS else r.get("no_hit_patient", False), axis=1
)
all_positives["tiny_lesion_flag"] = all_positives["patient_id"].isin(TINY_LESION_PATIENTS)
all_positives["p_b3_risk6_flag"] = all_positives["patient_id"].isin(RISK6_PATIENTS)

n_pos_total = len(all_positives)
n_pos_fallback = sum(len(d) for d in fallback_dfs)
n_pos_rule_a = len(pos_rule_a)
print(f"  총 positive={n_pos_total:,}  (rule_a={n_pos_rule_a:,}  fallback={n_pos_fallback})")

# ── 8. hard_negative sampling ────────────────────────────────────────────────
print("[8] hard_negative sampling  (목표 1:3)")

n_hn_target = n_pos_total * TARGET_POS_HN_RATIO
print(f"  목표 hard_negative={n_hn_target:,}")

# Rule A only hn (p95~p99)
rule_a_only_hn = rule_a_hn[rule_a_hn["padim_score"] <= P99].copy()

# --- (A) Rule D hn: 전체의 절반 할당 ---
n_rule_d_budget = n_hn_target // 2  # ~52,912
print(f"  [A] Rule D hn budget={n_rule_d_budget:,}  available={len(rule_d_hn):,}")

# patient × position_bin 균형 샘플링
n_patients_with_d_hn = rule_d_hn["patient_id"].nunique()
cap_per_patient_d = max(10, n_rule_d_budget // max(1, n_patients_with_d_hn))

d_hn_sampled_parts = []
for pid, grp in rule_d_hn.groupby("patient_id"):
    n_select = min(cap_per_patient_d, len(grp))
    # position_bin 균형: 각 bin에서 proportional
    bin_parts = []
    bins_in_grp = grp["position_bin"].unique()
    n_per_bin = max(1, n_select // len(bins_in_grp))
    for b, bg in grp.groupby("position_bin"):
        bin_parts.append(bg.nlargest(min(n_per_bin, len(bg)), "padim_score"))
    selected = pd.concat(bin_parts).nlargest(n_select, "padim_score")
    d_hn_sampled_parts.append(selected)

rule_d_hn_selected = pd.concat(d_hn_sampled_parts, ignore_index=True)
# budget 초과 시 trim (score 기준)
if len(rule_d_hn_selected) > n_rule_d_budget:
    rule_d_hn_selected = rule_d_hn_selected.nlargest(n_rule_d_budget, "padim_score")
print(f"  [A] Rule D hn 선택={len(rule_d_hn_selected):,}")

# --- (B) Rule A only hn: 나머지 예산, position_bin diversity 강화 ---
n_a_budget = n_hn_target - len(rule_d_hn_selected)
print(f"  [B] Rule A only hn budget={n_a_budget:,}  available={len(rule_a_only_hn):,}")

# 6개 bin에 균등 배분 → dominant bin (middle_peri, lower_peri) 억제
ALL_BINS = rule_a_only_hn["position_bin"].unique().tolist()
n_bins = len(ALL_BINS)
# non-dominant bin에 1.5x weight, dominant bin에 0.7x weight
bin_weights = {}
total_weight = 0
for b in ALL_BINS:
    w = 0.7 if b in DOMINANT_BINS else 1.5
    bin_weights[b] = w
    total_weight += w

a_hn_bin_parts = []
for b in ALL_BINS:
    n_bin = int(n_a_budget * bin_weights[b] / total_weight)
    bin_df = rule_a_only_hn[rule_a_only_hn["position_bin"] == b]
    if len(bin_df) == 0:
        continue
    n_patients_in_bin = bin_df["patient_id"].nunique()
    cap_pat = max(1, n_bin // max(1, n_patients_in_bin))
    # patient × z_level 다양성
    pat_parts = []
    for pid, pg in bin_df.groupby("patient_id"):
        n_pat_select = min(cap_pat, len(pg))
        # z_level별 균등
        z_parts = []
        z_levels = pg["z_level"].unique()
        n_per_z = max(1, n_pat_select // len(z_levels))
        for zl, zg in pg.groupby("z_level"):
            z_parts.append(zg.nlargest(min(n_per_z, len(zg)), "padim_score"))
        pat_selected = pd.concat(z_parts).nlargest(n_pat_select, "padim_score")
        pat_parts.append(pat_selected)
    bin_selected = pd.concat(pat_parts, ignore_index=True)
    if len(bin_selected) > n_bin:
        bin_selected = bin_selected.nlargest(n_bin, "padim_score")
    a_hn_bin_parts.append(bin_selected)

rule_a_only_hn_selected = pd.concat(a_hn_bin_parts, ignore_index=True)
print(f"  [B] Rule A only hn 선택={len(rule_a_only_hn_selected):,}")

# --- (C) Rule C: patient coverage 보장 (각 stage1_dev 환자 최소 1개 hn) ---
# rule_d + rule_a_only 합친 후 누락 환자 보완
combined_hn = pd.concat([rule_d_hn_selected, rule_a_only_hn_selected], ignore_index=True)
covered_patients = set(combined_hn["patient_id"].unique())
missing_patients = stage1_dev_set - covered_patients

rule_c_parts = []
if missing_patients:
    print(f"  [C] Rule C coverage 보완 환자={len(missing_patients)}명")
    for pid in missing_patients:
        pid_hn = rule_a_hn[rule_a_hn["patient_id"] == pid]
        if len(pid_hn) > 0:
            rule_c_parts.append(pid_hn.nlargest(min(5, len(pid_hn)), "padim_score"))

if rule_c_parts:
    rule_c_selected = pd.concat(rule_c_parts, ignore_index=True)
    combined_hn = pd.concat([combined_hn, rule_c_selected], ignore_index=True)
    print(f"  [C] Rule C 추가={len(rule_c_selected)}개")
else:
    print(f"  [C] Rule C: 누락 환자 없음 → 모두 커버됨")

# 중복 제거
combined_hn = combined_hn.drop_duplicates(
    subset=["patient_id", "slice_index", "y0", "x0"]
)

# 최종 target에 맞게 trim
if len(combined_hn) > n_hn_target:
    combined_hn = combined_hn.nlargest(n_hn_target, "padim_score")

# hard_negative labeling
combined_hn["candidate_label"] = "hard_negative"
combined_hn["is_positive"] = False
combined_hn["fallback_positive_below_p95"] = False
combined_hn["no_hit_patient"] = combined_hn["patient_id"].isin(NO_HIT_PATIENTS)
combined_hn["tiny_lesion_flag"] = combined_hn["patient_id"].isin(TINY_LESION_PATIENTS)
combined_hn["p_b3_risk6_flag"] = combined_hn["patient_id"].isin(RISK6_PATIENTS)

# candidate_rule 설정 (우선순위: rule_d > rule_b_diversity > rule_c > rule_a)
def assign_hn_rule(row):
    if row["padim_score"] > P99:
        return "rule_d_p99"
    elif row["position_bin"] not in DOMINANT_BINS:
        return "rule_b_diversity"
    else:
        return "rule_a"

combined_hn["candidate_rule"] = combined_hn.apply(assign_hn_rule, axis=1)
combined_hn["sampling_reason"] = "hard_negative_sampled"

# Rule C 환자에게 rule_c_patient_coverage 표시
if rule_c_parts:
    rule_c_pids = set(pd.concat(rule_c_parts)["patient_id"])
    combined_hn.loc[
        combined_hn["patient_id"].isin(rule_c_pids) & (combined_hn["candidate_rule"] == "rule_a"),
        "candidate_rule"
    ] = "rule_c_patient_coverage"

n_hn_final = len(combined_hn)
print(f"  최종 hard_negative={n_hn_final:,}")

# ── 9. manifest 합치기 ───────────────────────────────────────────────────────
print("[9] manifest 합치기")
manifest = pd.concat([all_positives, combined_hn], ignore_index=True)

# 파생 컬럼 추가
manifest["threshold_p95"] = P95
manifest["threshold_p99"] = P99
manifest["rule_a_p95_positive"] = manifest["padim_score"] > P95
manifest["rule_d_p99_positive"] = manifest["padim_score"] > P99
manifest["source_branch"] = BRANCH
manifest["source_score_csv"] = manifest["patient_id"].apply(
    lambda p: f"lesion_stage1_dev_by_patient/{p}.csv"
)
manifest["split"] = manifest["patient_id"].map(split_map)
manifest["group"] = manifest["patient_id"].map(group_map)
manifest["safe_id"] = manifest["patient_id"].map(safe_id_map)

# stage2_holdout 최종 검사
contamination_final = manifest["patient_id"].isin(stage2_holdout_set).sum()
assert contamination_final == 0, f"stage2_holdout contamination (final): {contamination_final}"

# candidate_id 부여
manifest = manifest.reset_index(drop=True)
manifest["candidate_id"] = manifest.index.map(lambda i: f"C{i:07d}")

# ── 10. 컬럼 순서 정렬 ──────────────────────────────────────────────────────
REQUIRED_COLS = [
    "candidate_id", "patient_id", "safe_id", "group", "split",
    "slice_index", "local_z", "y0", "x0", "y1", "x1",
    "position_bin", "z_level", "padim_score",
    "threshold_p95", "threshold_p99",
    "rule_a_p95_positive", "rule_d_p99_positive",
    "has_lesion_patch", "lesion_pixels",
    "candidate_label", "candidate_rule", "sampling_reason",
    "fallback_positive_below_p95", "no_hit_patient",
    "tiny_lesion_flag", "p_b3_risk6_flag",
    "source_branch", "source_score_csv",
]
# [보완2] 필수 컬럼 누락은 None 채우지 않고 즉시 중단
missing_cols = [c for c in REQUIRED_COLS if c not in manifest.columns]
if missing_cols:
    print(f"[ERROR] manifest 필수 컬럼 누락: {missing_cols}")
    sys.exit(1)
manifest = manifest[REQUIRED_COLS]

# [보완2] 저장 전 crop 좌표 컬럼 null count 검증
MANIFEST_COORD_COLS = ["slice_index", "local_z", "y0", "x0", "y1", "x1"]
coord_null_counts = {c: int(manifest[c].isnull().sum()) for c in MANIFEST_COORD_COLS}
coord_null_total = sum(coord_null_counts.values())
if coord_null_total > 0:
    print(f"[ERROR] 좌표 컬럼에 null 존재 → {coord_null_counts}")
    sys.exit(1)
print(f"  좌표 null 검증 PASS: {coord_null_counts}")

# ── 11. 검증 ────────────────────────────────────────────────────────────────
print("[11] 검증")
n_pos = (manifest["candidate_label"] == "positive").sum()
n_hn = (manifest["candidate_label"] == "hard_negative").sum()
ratio = n_hn / n_pos if n_pos > 0 else 0
print(f"  positive={n_pos:,}  hard_negative={n_hn:,}  ratio=1:{ratio:.2f}")

# stage2_holdout 검사
assert manifest["patient_id"].isin(stage2_holdout_set).sum() == 0, "FAIL: stage2_holdout"

# 필수 컬럼 누락 검사
assert all(c in manifest.columns for c in REQUIRED_COLS), "FAIL: missing required columns"

# stage1_dev 154명 확인
assert manifest["split"].nunique() == 1 and manifest["split"].iloc[0] == "stage1_dev", \
    f"FAIL: split check {manifest['split'].unique()}"

# [보완3] LUNG1-386: positive 존재뿐 아니라 fallback_positive_below_p95=True row 필수
lung386_pos = manifest[
    (manifest["patient_id"] == "LUNG1-386") & (manifest["candidate_label"] == "positive")
]
lung386_has_fallback_flag = bool(lung386_pos["fallback_positive_below_p95"].any())
lung386_fallback_ok = (len(lung386_pos) > 0) and lung386_has_fallback_flag
if not lung386_fallback_ok:
    print(f"  [WARN] LUNG1-386: positive={len(lung386_pos)}  fallback_flag={lung386_has_fallback_flag}  → 판정 실패/부분통과 처리")

# no-hit 3명 fallback 확인
for pid in NO_HIT_PATIENTS:
    pid_pos = manifest[
        (manifest["patient_id"] == pid) & (manifest["candidate_label"] == "positive")
    ]
    print(f"  {pid}: positive={len(pid_pos)}  fallback_below_p95={pid_pos['fallback_positive_below_p95'].any()}")

# tiny lesion 4명 확인
for pid in TINY_LESION_PATIENTS:
    pid_pos = manifest[
        (manifest["patient_id"] == pid) & (manifest["candidate_label"] == "positive")
    ]
    print(f"  tiny[{pid}]: positive={len(pid_pos)}")

# risk6 6명 확인
for pid in RISK6_PATIENTS:
    pid_rows = manifest[manifest["patient_id"] == pid]
    pid_pos = pid_rows[pid_rows["candidate_label"] == "positive"]
    pid_hn = pid_rows[pid_rows["candidate_label"] == "hard_negative"]
    print(f"  risk6[{pid}]: pos={len(pid_pos)}  hn={len(pid_hn)}")

# ── 12. 저장 ────────────────────────────────────────────────────────────────
print("[12] 저장")
manifest.to_csv(OUTPUT_DIR / "p_c3_candidate_manifest.csv", index=False)
print(f"  manifest saved: {len(manifest):,} rows")

# ── 13. summary CSV 생성 ────────────────────────────────────────────────────
print("[13] summary 생성")

# patient balance
patient_summary = []
for pid in sorted(stage1_dev_set):
    pm = manifest[manifest["patient_id"] == pid]
    pm_pos = pm[pm["candidate_label"] == "positive"]
    pm_hn = pm[pm["candidate_label"] == "hard_negative"]
    patient_summary.append({
        "patient_id": pid,
        "group": group_map.get(pid, ""),
        "n_positive": len(pm_pos),
        "n_hard_negative": len(pm_hn),
        "n_total": len(pm),
        "has_fallback": pm["fallback_positive_below_p95"].any(),
        "no_hit_patient": pid in NO_HIT_PATIENTS,
        "tiny_lesion_flag": pid in TINY_LESION_PATIENTS,
        "p_b3_risk6_flag": pid in RISK6_PATIENTS,
        "pos_score_max": float(pm_pos["padim_score"].max()) if len(pm_pos) > 0 else None,
        "hn_score_max": float(pm_hn["padim_score"].max()) if len(pm_hn) > 0 else None,
    })
pd.DataFrame(patient_summary).to_csv(OUTPUT_DIR / "p_c3_patient_balance_summary.csv", index=False)

# position_bin balance
pos_bin = manifest.groupby(["position_bin", "candidate_label"]).size().unstack(fill_value=0).reset_index()
pos_bin.to_csv(OUTPUT_DIR / "p_c3_position_bin_balance_summary.csv", index=False)

# pos/hn balance
balance_df = pd.DataFrame([
    {"metric": "n_positive", "value": int(n_pos)},
    {"metric": "n_hard_negative", "value": int(n_hn)},
    {"metric": "pos_hn_ratio", "value": round(ratio, 4)},
    {"metric": "target_ratio", "value": TARGET_POS_HN_RATIO},
    {"metric": "ratio_in_target_1_2_to_1_3", "value": str(2.0 <= ratio <= 3.0)},
    {"metric": "n_fallback_positive", "value": int(n_pos_fallback)},
    {"metric": "n_rule_d_hn", "value": int((manifest["candidate_rule"] == "rule_d_p99").sum())},
    {"metric": "n_rule_b_diversity_hn", "value": int((manifest["candidate_rule"] == "rule_b_diversity").sum())},
    {"metric": "n_rule_c_coverage_hn", "value": int((manifest["candidate_rule"] == "rule_c_patient_coverage").sum())},
    {"metric": "stage2_holdout_contamination", "value": 0},
    {"metric": "n_patients_covered", "value": int(manifest["patient_id"].nunique())},
])
balance_df.to_csv(OUTPUT_DIR / "p_c3_positive_hard_negative_balance.csv", index=False)

# no-hit fallback summary
pd.DataFrame(fallback_summary_rows).to_csv(OUTPUT_DIR / "p_c3_no_hit_fallback_summary.csv", index=False)

# risk6 + tiny lesion summary
risk_tiny_rows = []
for pid in sorted(RISK6_PATIENTS | TINY_LESION_PATIENTS):
    pm = manifest[manifest["patient_id"] == pid]
    pm_pos = pm[pm["candidate_label"] == "positive"]
    pm_hn = pm[pm["candidate_label"] == "hard_negative"]
    risk_tiny_rows.append({
        "patient_id": pid,
        "is_risk6": pid in RISK6_PATIENTS,
        "is_tiny_lesion": pid in TINY_LESION_PATIENTS,
        "is_no_hit": pid in NO_HIT_PATIENTS,
        "n_positive": len(pm_pos),
        "n_hard_negative": len(pm_hn),
        "n_fallback": int(pm["fallback_positive_below_p95"].sum()),
        "positive_preserved": len(pm_pos) > 0,
    })
pd.DataFrame(risk_tiny_rows).to_csv(OUTPUT_DIR / "p_c3_risk6_tiny_lesion_summary.csv", index=False)

# errors
pd.DataFrame(errors).to_csv(OUTPUT_DIR / "p_c3_errors.csv", index=False)

# ── 14. 판정 ────────────────────────────────────────────────────────────────
ratio_ok = 2.0 <= ratio <= 3.0
n_covered = manifest["patient_id"].nunique()
all_stage1_covered = n_covered == 154
lung386_ok = lung386_fallback_ok
contamination_ok = contamination_final == 0

if contamination_ok and all_stage1_covered and lung386_ok and ratio_ok:
    verdict = "통과"
elif contamination_ok and lung386_ok:
    verdict = "부분통과"
else:
    verdict = "실패"

elapsed = (datetime.datetime.now() - t_start).total_seconds()

# ── 15. summary JSON ─────────────────────────────────────────────────────────
summary = {
    "step": "P-C3",
    "verdict": verdict,
    "created": datetime.datetime.now().isoformat(),
    "elapsed_seconds": round(elapsed, 1),
    "source_branch": BRANCH,
    "threshold": {"p95": P95, "p99": P99},
    "input_validation": {
        "n_csv_files_loaded": n_loaded,
        "stage1_dev_154_confirmed": bool(all_stage1_covered),
        "stage2_holdout_contamination": int(contamination_final),
        "n_patients_in_manifest": int(n_covered),
    },
    "candidate_counts": {
        "n_total": int(len(manifest)),
        "n_positive": int(n_pos),
        "n_hard_negative": int(n_hn),
        "pos_hn_ratio": round(float(ratio), 4),
        "target_ratio_min": 2.0,
        "target_ratio_max": 3.0,
        "ratio_in_target": bool(ratio_ok),
        "n_fallback_positive": int(n_pos_fallback),
        "n_rule_a_positive": int(n_pos_rule_a),
        "n_rule_d_hn": int((manifest["candidate_rule"] == "rule_d_p99").sum()),
        "n_rule_b_diversity_hn": int((manifest["candidate_rule"] == "rule_b_diversity").sum()),
        "n_rule_c_coverage_hn": int((manifest["candidate_rule"] == "rule_c_patient_coverage").sum()),
    },
    "fallback": {
        "no_hit_patients": sorted(NO_HIT_PATIENTS),
        "lung1_386_fallback_ok": bool(lung386_fallback_ok),
        "lung1_386_has_fallback_flag": bool(lung386_has_fallback_flag),
        "fallback_summary": [
            {k: (bool(v) if isinstance(v, (bool, np.bool_)) else
                 float(v) if isinstance(v, (float, np.floating)) else v)
             for k, v in row.items()}
            for row in fallback_summary_rows
        ],
    },
    "tiny_lesion_patients": list(TINY_LESION_PATIENTS),
    "risk6_patients": list(RISK6_PATIENTS),
    "guardrails": {
        "crop_generated": False,
        "training_executed": False,
        "model_forward": False,
        "scoring_rerun": False,
        "threshold_recalculated": False,
        "metrics_recalculated": False,
        "stage2_holdout_accessed": False,
        "p_a80b_executed": False,
        "existing_results_modified": False,
    },
    "n_errors": len(errors),
    "next_step": {
        "primary": "P-C4 crop generation preflight (사용자 승인 필요)",
        "condition": "manifest quality review 완료 후",
    },
}
with open(OUTPUT_DIR / "p_c3_candidate_manifest_summary.json", "w") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

# ── 16. report MD 생성 ──────────────────────────────────────────────────────
pos_bin_dist = manifest.groupby("position_bin")["candidate_label"].value_counts().unstack(fill_value=0)
rule_dist = manifest["candidate_rule"].value_counts()

report = f"""# P-C3 Candidate Manifest Generation Report

**판정: {verdict}**
생성일: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  소요: {elapsed:.1f}초

## 1. 입력 검증

| 항목 | 결과 |
|------|------|
| score CSV 수 | {n_loaded} |
| stage1_dev 154명 확인 | {"OK" if all_stage1_covered else "FAIL"} |
| stage2_holdout contamination | {contamination_final} |
| threshold p95 | {P95:.6f} |
| threshold p99 | {P99:.6f} |

## 2. 후보 수 요약

| 구분 | 수 |
|------|----|
| 총 candidate | {len(manifest):,} |
| positive | {n_pos:,} |
| hard_negative | {n_hn:,} |
| positive:hard_negative | 1:{ratio:.2f} |
| 목표 1:2~1:3 달성 | {"YES" if ratio_ok else "NO (부분통과)"} |
| Rule A positive (원본) | {n_pos_rule_a:,} |
| fallback positive | {n_pos_fallback} |

## 3. Rule 분포

{rule_dist.to_string()}

## 4. position_bin 분포 (candidate_label별)

{pos_bin_dist.to_string()}

## 5. no-hit 3명 fallback 결과

| 환자 | status | n_fallback | score_max |
|------|--------|------------|-----------|
"""
for row in fallback_summary_rows:
    report += f"| {row['patient_id']} | {row['status']} | {row['n_fallback_selected']} | {row.get('fallback_score_max', 'N/A')} |\n"

report += f"""
**LUNG1-386 fallback positive 포함: {"SUCCESS" if lung386_ok else "FAIL"}**

## 6. tiny lesion 4명 보존

| 환자 | positive 수 | fallback |
|------|------------|---------|
"""
for pid in sorted(TINY_LESION_PATIENTS):
    pm = manifest[(manifest["patient_id"] == pid) & (manifest["candidate_label"] == "positive")]
    report += f"| {pid} | {len(pm)} | {pm['fallback_positive_below_p95'].any()} |\n"

report += f"""
## 7. P-B3 risk6 6명 보존

| 환자 | positive | hard_negative | fallback |
|------|---------|--------------|---------|
"""
for pid in sorted(RISK6_PATIENTS):
    pm = manifest[manifest["patient_id"] == pid]
    pm_pos = pm[pm["candidate_label"] == "positive"]
    pm_hn = pm[pm["candidate_label"] == "hard_negative"]
    report += f"| {pid} | {len(pm_pos)} | {len(pm_hn)} | {pm['fallback_positive_below_p95'].any()} |\n"

report += f"""
## 8. 가드레일 확인

- crop 생성: 없음
- 2차학습: 없음
- model forward: 없음
- scoring 재실행: 없음
- stage2_holdout 접근: 없음 (contamination=0)
- 기존 결과 수정: 없음

## 9. 오류

n_errors={len(errors)}

## 10. 다음 단계

- **P-C4**: crop generation preflight (사용자 승인 필요)
- 또는 manifest quality review (patient별 coverage 육안 확인)
"""

with open(OUTPUT_DIR / "p_c3_candidate_manifest_report.md", "w") as f:
    f.write(report)

print(f"\n===== P-C3 완료 =====")
print(f"판정: {verdict}")
print(f"총 candidate={len(manifest):,}  positive={n_pos:,}  hard_negative={n_hn:,}  비율=1:{ratio:.2f}")
print(f"LUNG1-386 fallback={'SUCCESS' if lung386_ok else 'FAIL'}")
print(f"소요={elapsed:.1f}초")
print(f"출력: {OUTPUT_DIR}")
