"""
P-C9: EfficientNet-B0 v4_20 second-stage full crop artifact validation + mask warning audit
read-only 분석 전용 — crop 생성/수정/학습/forward 금지
"""

import json, csv, datetime, sys
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# 경로 상수
# ============================================================
BASE      = Path("/home/jinhy/project/lung-ct-anomaly")
WORKSPACE = BASE / "experiments/efficientnet_b0_v4_20_second_stage_refiner_v1"

P8_REPORT = WORKSPACE / "outputs/reports/p_c8_full_crop_generation"
CROP_DIR  = WORKSPACE / "outputs/crops/p_c8_full_crops"
OUT_DIR   = WORKSPACE / "outputs/reports/p_c9_full_crop_artifact_validation"

DONE_JSON     = P8_REPORT / "DONE.json"
SUMMARY_JSON  = P8_REPORT / "p_c8_full_crop_generation_summary.json"
LABELS_CSV    = P8_REPORT / "p_c8_full_crop_labels.csv"
INTEGRITY_CSV = P8_REPORT / "p_c8_full_crop_integrity.csv"
WARN_CSV      = P8_REPORT / "p_c8_mask_warning_summary.csv"
MANIFEST_CSV  = WORKSPACE / "outputs/candidates/p_c3_candidate_manifest/p_c3_candidate_manifest.csv"
SPLIT_CSV     = BASE / "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1_balanced.csv"

OUT_MD       = OUT_DIR / "p_c9_full_crop_artifact_validation.md"
OUT_JSON     = OUT_DIR / "p_c9_full_crop_artifact_validation.json"
OUT_ID_CSV   = OUT_DIR / "p_c9_candidate_id_consistency.csv"
OUT_LABEL    = OUT_DIR / "p_c9_label_distribution.csv"
OUT_WARN_TYPE= OUT_DIR / "p_c9_mask_warning_by_type.csv"
OUT_WARN_PAT = OUT_DIR / "p_c9_mask_warning_by_patient.csv"
OUT_WARN_RULE= OUT_DIR / "p_c9_mask_warning_by_rule.csv"
OUT_WARN_BIN = OUT_DIR / "p_c9_mask_warning_by_position_bin.csv"
OUT_OPT_CMP  = OUT_DIR / "p_c9_option_b_c_lite_c_full_comparison.csv"
OUT_TRAIN_REC= OUT_DIR / "p_c9_training_manifest_recommendation.csv"
OUT_ERRORS   = OUT_DIR / "p_c9_errors.csv"

EXPECTED_TOTAL = 114381
NOW_STR = datetime.datetime.now().isoformat(timespec="seconds")

errors = []

def log_error(msg):
    errors.append(msg)
    print(f"  [ERROR] {msg}")

# ============================================================
# 1. DONE.json 확인
# ============================================================
print("[1] DONE.json 확인...")
if not DONE_JSON.exists():
    log_error("DONE.json 없음")
    done_ok = False
    done_data = {}
else:
    with open(DONE_JSON) as f:
        done_data = json.load(f)
    done_ok = done_data.get("done", False)
    print(f"  done={done_ok}, generated={done_data.get('generated')}, errors={done_data.get('n_errors')}")

# ============================================================
# 2. summary JSON 확인
# ============================================================
print("[2] summary JSON 확인...")
with open(SUMMARY_JSON) as f:
    summary = json.load(f)
print(f"  verdict={summary['verdict']}, generated={summary['generated']}, n_errors={summary['n_errors']}")

# ============================================================
# 3. CSV 로드
# ============================================================
print("[3] CSV 로드...")
df_labels    = pd.read_csv(LABELS_CSV)
df_integrity = pd.read_csv(INTEGRITY_CSV)
df_warn      = pd.read_csv(WARN_CSV)
df_manifest  = pd.read_csv(MANIFEST_CSV)
df_split     = pd.read_csv(SPLIT_CSV)

print(f"  labels={len(df_labels)}, integrity={len(df_integrity)}, warn={len(df_warn)}, manifest={len(df_manifest)}")

# ============================================================
# 4. count 정합 확인
# ============================================================
print("[4] count 정합 확인...")
crop_count = len(list(CROP_DIR.glob("*.npz")))
label_count = len(df_labels)
integ_count = len(df_integrity)
count_ok = (crop_count == EXPECTED_TOTAL and label_count == EXPECTED_TOTAL and integ_count == EXPECTED_TOTAL)
if not count_ok:
    log_error(f"count mismatch: npz={crop_count}, labels={label_count}, integrity={integ_count}, expected={EXPECTED_TOTAL}")
print(f"  npz={crop_count}, labels={label_count}, integrity={integ_count}, expected={EXPECTED_TOTAL} → {'OK' if count_ok else 'FAIL'}")

# ============================================================
# 5. candidate_id set 일치 확인
# ============================================================
print("[5] candidate_id set 일치 확인...")
manifest_ids  = set(df_manifest["candidate_id"])
labels_ids    = set(df_labels["candidate_id"])
integrity_ids = set(df_integrity["candidate_id"])
npz_ids       = set(f.stem for f in CROP_DIR.glob("*.npz"))

only_manifest  = manifest_ids  - labels_ids
only_labels    = labels_ids    - manifest_ids
label_vs_integ = labels_ids.symmetric_difference(integrity_ids)
label_vs_npz   = labels_ids.symmetric_difference(npz_ids)

id_ok = (len(only_manifest) == 0 and len(only_labels) == 0
         and len(label_vs_integ) == 0 and len(label_vs_npz) == 0)
print(f"  manifest-labels diff={len(only_manifest)}, labels-manifest diff={len(only_labels)}")
print(f"  labels⊕integrity={len(label_vs_integ)}, labels⊕npz={len(label_vs_npz)} → {'OK' if id_ok else 'FAIL'}")
if not id_ok:
    log_error(f"candidate_id set mismatch")

# ID consistency CSV
id_rows = [
    {"set_pair": "manifest vs labels", "only_left": len(only_manifest), "only_right": len(only_labels), "ok": len(only_manifest)==0 and len(only_labels)==0},
    {"set_pair": "labels vs integrity", "only_left": 0, "only_right": 0, "ok": len(label_vs_integ)==0},
    {"set_pair": "labels vs npz", "only_left": 0, "only_right": 0, "ok": len(label_vs_npz)==0},
]
pd.DataFrame(id_rows).to_csv(OUT_ID_CSV, index=False)

# ============================================================
# 6. stage2_holdout contamination 확인
# ============================================================
print("[6] stage2_holdout contamination 확인...")
holdout_ids = set(df_split[df_split["stage_split"] == "stage2_holdout"]["patient_id"])
cont = df_labels[df_labels["patient_id"].isin(holdout_ids)]
holdout_cont = len(cont)
if holdout_cont > 0:
    log_error(f"stage2_holdout contamination={holdout_cont}")
print(f"  holdout_contamination={holdout_cont} → {'OK' if holdout_cont==0 else 'FAIL'}")

# ============================================================
# 7. errors=0 확인
# ============================================================
print("[7] errors 확인...")
error_count = summary.get("n_errors", -1)
print(f"  n_errors={error_count} → {'OK' if error_count==0 else 'FAIL'}")
if error_count != 0:
    log_error(f"n_errors={error_count}")

# ============================================================
# 8. crop shape 확인 (integrity 기준 전수)
# ============================================================
print("[8] crop shape 확인 (integrity 전수)...")
expected_shape_str = "(3, 96, 96)"
bad_shape = df_integrity[df_integrity["crop_shape"] != expected_shape_str]
shape_ok = len(bad_shape) == 0
print(f"  bad_shape={len(bad_shape)} / {len(df_integrity)} → {'OK' if shape_ok else 'FAIL'}")
if not shape_ok:
    log_error(f"crop_shape 불일치 {len(bad_shape)}건")

# ============================================================
# 9. ct NaN/Inf 확인
# ============================================================
print("[9] ct NaN/Inf 확인 (integrity)...")
nan_total = pd.to_numeric(df_integrity["ct_nan"], errors="coerce").fillna(0).sum()
inf_total = pd.to_numeric(df_integrity["ct_inf"], errors="coerce").fillna(0).sum()
nan_ok = (nan_total == 0 and inf_total == 0)
print(f"  ct_nan_sum={nan_total}, ct_inf_sum={inf_total} → {'OK' if nan_ok else 'FAIL'}")
if not nan_ok:
    log_error(f"ct NaN={nan_total}, Inf={inf_total}")

# ============================================================
# 10. roi/mask binary valid 확인
# ============================================================
print("[10] roi/mask binary valid 확인...")
def bool_col(df, col):
    return df[col].map(lambda x: str(x).strip().lower() in ("true", "1"))

roi_invalid  = (~bool_col(df_integrity, "roi_binary_valid")).sum()
mask_invalid = (~bool_col(df_integrity, "mask_binary_valid")).sum()
binary_ok = (roi_invalid == 0 and mask_invalid == 0)
print(f"  roi_invalid={roi_invalid}, mask_invalid={mask_invalid} → {'OK' if binary_ok else 'FAIL'}")
if not binary_ok:
    log_error(f"binary invalid: roi={roi_invalid}, mask={mask_invalid}")

# ============================================================
# 11. label 분포
# ============================================================
print("[11] label 분포...")
label_dist = df_labels["candidate_label"].value_counts()
n_pos = int(label_dist.get("positive", 0))
n_hn  = int(label_dist.get("hard_negative", 0))
print(f"  positive={n_pos:,}, hard_negative={n_hn:,}, total={n_pos+n_hn:,}")

label_dist_df = label_dist.reset_index()
label_dist_df.columns = ["candidate_label", "count"]
label_dist_df["ratio"] = label_dist_df["count"] / len(df_labels)
label_dist_df.to_csv(OUT_LABEL, index=False)

# ============================================================
# 12. mask warning 분석 (labels CSV 기준)
# ============================================================
print("[12] mask warning 분석...")

# bool 변환 헬퍼
def to_bool(series):
    return series.map(lambda x: str(x).strip().lower() in ("true", "1"))

df_labels["_mnw"] = to_bool(df_labels["mask_nonzero_warning"])
df_labels["_cnw"] = to_bool(df_labels["center_mask_nonzero"])
df_labels["_anw"] = to_bool(df_labels["adjacent_mask_nonzero"])

hn = df_labels[df_labels["candidate_label"] == "hard_negative"].copy()
pos = df_labels[df_labels["candidate_label"] == "positive"].copy()

hn_total = len(hn)
pos_total = len(pos)

# hard_negative warning 분류
hn_mnw        = int(hn["_mnw"].sum())   # mask_nonzero_warning=True (전체 warning)
hn_cnw        = int(hn["_cnw"].sum())   # center_mask_nonzero=True
hn_adj_only   = int((hn["_mnw"] & ~hn["_cnw"]).sum())   # adjacent only (center는 clean)
hn_both       = int((hn["_cnw"] & hn["_anw"]).sum())    # center + adjacent 모두
hn_center_only= int((hn["_cnw"] & ~hn["_anw"]).sum())   # center만 (adjacent는 clean)

# positive 중 mask 없는 케이스
pos_no_mask   = int((~pos["_mnw"]).sum())
pos_mask_any  = int(pos["_mnw"].sum())

print(f"  HN total={hn_total:,}")
print(f"  HN mask_nonzero_warning={hn_mnw:,} ({hn_mnw/hn_total*100:.1f}%)")
print(f"  HN center_mask_nonzero={hn_cnw:,} ({hn_cnw/hn_total*100:.1f}%)")
print(f"  HN adjacent_only={hn_adj_only:,} ({hn_adj_only/hn_total*100:.1f}%)")
print(f"  HN center_only={hn_center_only:,}")
print(f"  HN center+adjacent_both={hn_both:,}")
print(f"  POS no_mask={pos_no_mask:,} ({pos_no_mask/pos_total*100:.1f}%)")
print(f"  POS mask_any={pos_mask_any:,} ({pos_mask_any/pos_total*100:.1f}%)")

warn_type_rows = [
    {"warning_type": "hn_total",           "count": hn_total,      "pct_of_hn": 100.0},
    {"warning_type": "hn_mask_nonzero_warning", "count": hn_mnw,   "pct_of_hn": round(hn_mnw/hn_total*100,2)},
    {"warning_type": "hn_center_mask_nonzero",  "count": hn_cnw,   "pct_of_hn": round(hn_cnw/hn_total*100,2)},
    {"warning_type": "hn_adjacent_only",        "count": hn_adj_only, "pct_of_hn": round(hn_adj_only/hn_total*100,2)},
    {"warning_type": "hn_center_only",          "count": hn_center_only, "pct_of_hn": round(hn_center_only/hn_total*100,2)},
    {"warning_type": "hn_center_and_adjacent",  "count": hn_both,  "pct_of_hn": round(hn_both/hn_total*100,2)},
    {"warning_type": "pos_total",               "count": pos_total, "pct_of_hn": None},
    {"warning_type": "pos_no_mask",             "count": pos_no_mask, "pct_of_hn": None},
    {"warning_type": "pos_mask_any",            "count": pos_mask_any, "pct_of_hn": None},
]
pd.DataFrame(warn_type_rows).to_csv(OUT_WARN_TYPE, index=False)

# ============================================================
# 13. patient별 warning 분포
# ============================================================
print("[13] patient별 warning 분포...")
hn_warn = hn[hn["_mnw"]]
pat_warn = (hn_warn.groupby("patient_id")
            .agg(hn_warn_count=("candidate_id","count"))
            .reset_index())
pat_total = (hn.groupby("patient_id")
             .agg(hn_total_count=("candidate_id","count"))
             .reset_index())
pat_df = pat_total.merge(pat_warn, on="patient_id", how="left").fillna(0)
pat_df["hn_warn_count"] = pat_df["hn_warn_count"].astype(int)
pat_df["warn_ratio"] = pat_df["hn_warn_count"] / pat_df["hn_total_count"]
pat_df = pat_df.sort_values("hn_warn_count", ascending=False)
pat_df.to_csv(OUT_WARN_PAT, index=False)
top3 = pat_df.head(3)[["patient_id","hn_warn_count","warn_ratio"]].to_dict("records")
print(f"  top3 patients: {top3}")

# ============================================================
# 14. rule별 warning 분포
# ============================================================
print("[14] candidate_rule별 warning 분포...")
rule_warn = (hn.groupby("candidate_rule")
             .agg(hn_total=("candidate_id","count"),
                  hn_warn=("_mnw","sum"),
                  hn_center=("_cnw","sum"),
                  hn_adj_only=("_anw",lambda x: (x & ~hn.loc[x.index,"_cnw"]).sum()))
             .reset_index())
rule_warn["warn_ratio"] = rule_warn["hn_warn"] / rule_warn["hn_total"]
rule_warn["center_ratio"] = rule_warn["hn_center"] / rule_warn["hn_total"]
rule_warn = rule_warn.sort_values("hn_warn", ascending=False)
rule_warn.to_csv(OUT_WARN_RULE, index=False)
print(f"  rules: {rule_warn[['candidate_rule','hn_total','hn_warn','warn_ratio']].to_dict('records')}")

# ============================================================
# 15. position_bin별 warning 분포
# ============================================================
print("[15] position_bin별 warning 분포...")
manifest_pos = df_manifest[["candidate_id", "position_bin", "z_level"]].copy()
hn_with_pos = hn.merge(manifest_pos, on="candidate_id", how="left")

bin_warn = (hn_with_pos.groupby("position_bin")
            .agg(hn_total=("candidate_id","count"),
                 hn_warn=("_mnw","sum"),
                 hn_center=("_cnw","sum"))
            .reset_index())
bin_warn["warn_ratio"] = bin_warn["hn_warn"] / bin_warn["hn_total"]
bin_warn["center_ratio"] = bin_warn["hn_center"] / bin_warn["hn_total"]
bin_warn = bin_warn.sort_values("hn_warn", ascending=False)
bin_warn.to_csv(OUT_WARN_BIN, index=False)
print(f"  position_bins:\n{bin_warn.to_string(index=False)}")

# ============================================================
# 16. 특수 flag별 warning 영향
# ============================================================
print("[16] 특수 flag별 warning 분포...")
for flag in ["no_hit_patient", "tiny_lesion_flag", "p_b3_risk6_flag", "fallback_positive_below_p95"]:
    hn_flag = hn[to_bool(hn[flag])] if flag in hn.columns else pd.DataFrame()
    cnt = len(hn_flag)
    warn_cnt = int(hn_flag["_mnw"].sum()) if cnt > 0 else 0
    print(f"  {flag}: hn={cnt}, hn_warn={warn_cnt} ({warn_cnt/cnt*100:.1f}%" if cnt > 0 else f"  {flag}: hn={cnt}")

# ============================================================
# 17. Option B / C-lite / C-full 비교
# ============================================================
print("[17] Option B/C-lite/C-full 비교...")

# Option B: 현재 그대로 (40,006 warning 포함)
opt_b_pos = pos_total
opt_b_hn  = hn_total
opt_b_ratio = opt_b_pos / opt_b_hn if opt_b_hn > 0 else 0

# Option C-lite: HN 중 center_mask_nonzero=True 제거
hn_clite = hn[~hn["_cnw"]]
opt_clite_hn  = len(hn_clite)
opt_clite_ratio = opt_b_pos / opt_clite_hn if opt_clite_hn > 0 else 0
removed_clite = hn_total - opt_clite_hn

# Option C-full: HN 중 mask_nonzero_warning=True 전체 제거
hn_cfull = hn[~hn["_mnw"]]
opt_cfull_hn  = len(hn_cfull)
opt_cfull_ratio = opt_b_pos / opt_cfull_hn if opt_cfull_hn > 0 else 0
removed_cfull = hn_total - opt_cfull_hn

print(f"  Option B:      pos={opt_b_pos:,}, hn={opt_b_hn:,}, ratio={opt_b_ratio:.3f}")
print(f"  Option C-lite: pos={opt_b_pos:,}, hn={opt_clite_hn:,}, removed={removed_clite:,}, ratio={opt_clite_ratio:.3f}")
print(f"  Option C-full: pos={opt_b_pos:,}, hn={opt_cfull_hn:,}, removed={removed_cfull:,}, ratio={opt_cfull_ratio:.3f}")

opt_rows = [
    {"option": "B",      "pos": opt_b_pos,    "hn": opt_b_hn,     "removed_hn": 0,
     "total": opt_b_pos+opt_b_hn,    "pos_hn_ratio": round(opt_b_ratio,4),
     "description": "현재 그대로, warning flag만 보존"},
    {"option": "C-lite", "pos": opt_b_pos,    "hn": opt_clite_hn, "removed_hn": removed_clite,
     "total": opt_b_pos+opt_clite_hn, "pos_hn_ratio": round(opt_clite_ratio,4),
     "description": "center_mask_nonzero=True HN 제거"},
    {"option": "C-full", "pos": opt_b_pos,    "hn": opt_cfull_hn, "removed_hn": removed_cfull,
     "total": opt_b_pos+opt_cfull_hn, "pos_hn_ratio": round(opt_cfull_ratio,4),
     "description": "mask_nonzero_warning=True HN 전체 제거"},
]
opt_df = pd.DataFrame(opt_rows)
opt_df.to_csv(OUT_OPT_CMP, index=False)

# ============================================================
# 18. training manifest recommendation
# ============================================================
print("[18] training manifest recommendation 생성...")
# 추천 option 판단 로직
center_ratio = hn_cnw / hn_total if hn_total > 0 else 0
adj_only_ratio = hn_adj_only / hn_total if hn_total > 0 else 0

if center_ratio > 0.3:
    recommended_option = "C-lite"
    recommendation_reason = f"center_mask_nonzero 비율={center_ratio:.1%} > 30% → HN label 오염 위험"
elif center_ratio > 0.1:
    recommended_option = "C-lite"
    recommendation_reason = f"center_mask_nonzero 비율={center_ratio:.1%} > 10% → C-lite 적용 권장"
else:
    recommended_option = "B"
    recommendation_reason = f"center_mask_nonzero 비율={center_ratio:.1%} 낮음 → Option B 유지 가능"

rec_rows = [
    {"option": "B",      "use_for_training": recommended_option == "B",
     "pos": opt_b_pos,    "hn": opt_b_hn,     "total": opt_b_pos+opt_b_hn,
     "pos_hn_ratio": round(opt_b_ratio,4),    "note": "현재 labels CSV 그대로 사용"},
    {"option": "C-lite", "use_for_training": recommended_option == "C-lite",
     "pos": opt_b_pos,    "hn": opt_clite_hn, "total": opt_b_pos+opt_clite_hn,
     "pos_hn_ratio": round(opt_clite_ratio,4), "note": "center_mask_nonzero=True HN 제외 manifest 생성 필요"},
    {"option": "C-full", "use_for_training": False,
     "pos": opt_b_pos,    "hn": opt_cfull_hn, "total": opt_b_pos+opt_cfull_hn,
     "pos_hn_ratio": round(opt_cfull_ratio,4), "note": "과도한 HN 제거, 비추천"},
]
pd.DataFrame(rec_rows).to_csv(OUT_TRAIN_REC, index=False)

# ============================================================
# 19. errors CSV (이번 validation 과정 에러)
# ============================================================
err_df = pd.DataFrame([{"error": e} for e in errors]) if errors else pd.DataFrame(columns=["error"])
err_df.to_csv(OUT_ERRORS, index=False)

# ============================================================
# 20. 최종 판정
# ============================================================
print("\n[20] 최종 판정...")
blockers = [e for e in errors]
verdict = "통과" if len(blockers) == 0 else "실패"

# Option B 유지 가능 여부
opt_b_safe = center_ratio < 0.20
opt_b_note = "유지 가능" if opt_b_safe else f"위험 (center_mask_nonzero {center_ratio:.1%})"

adjacent_safe = adj_only_ratio / (hn_mnw/hn_total) > 0.5 if hn_mnw > 0 else True
adjacent_note = f"adjacent_only={adj_only_ratio:.1%} (전체 warning 중 비중 높으면 유지 가능)"

print(f"  판정: {verdict}")
print(f"  Option B 유지: {opt_b_note}")
print(f"  추천 option: {recommended_option} — {recommendation_reason}")
print(f"  blockers: {blockers if blockers else '없음'}")

# ============================================================
# 21. JSON 저장
# ============================================================
result = {
    "step": "P-C9",
    "verdict": verdict,
    "created": NOW_STR,
    "p_c8_done": done_ok,
    "count_validation": {
        "crop_npz": crop_count, "labels_csv": label_count,
        "integrity_csv": integ_count, "expected": EXPECTED_TOTAL, "ok": count_ok,
    },
    "id_consistency": {
        "manifest_vs_labels_diff": len(only_manifest) + len(only_labels),
        "labels_vs_integrity_diff": len(label_vs_integ),
        "labels_vs_npz_diff": len(label_vs_npz),
        "ok": id_ok,
    },
    "stage2_holdout_contamination": holdout_cont,
    "errors_in_p_c8": error_count,
    "crop_shape_ok": shape_ok,
    "ct_nan_inf_ok": nan_ok,
    "binary_valid_ok": binary_ok,
    "label_distribution": {
        "positive": n_pos, "hard_negative": n_hn, "total": n_pos + n_hn,
    },
    "mask_warning_analysis": {
        "hn_total":           hn_total,
        "hn_mask_nonzero_warning": hn_mnw,   "hn_mnw_ratio": round(hn_mnw/hn_total*100,2),
        "hn_center_mask_nonzero":  hn_cnw,   "hn_cnw_ratio": round(hn_cnw/hn_total*100,2),
        "hn_adjacent_only":        hn_adj_only, "hn_adj_only_ratio": round(hn_adj_only/hn_total*100,2),
        "hn_center_and_adjacent":  hn_both,
        "hn_center_only":          hn_center_only,
        "pos_no_mask":             pos_no_mask, "pos_no_mask_ratio": round(pos_no_mask/pos_total*100,2),
        "pos_mask_any":            pos_mask_any,
    },
    "option_comparison": {
        "B":      {"pos": opt_b_pos,   "hn": opt_b_hn,     "ratio": round(opt_b_ratio,4)},
        "C-lite": {"pos": opt_b_pos,   "hn": opt_clite_hn, "removed": removed_clite, "ratio": round(opt_clite_ratio,4)},
        "C-full": {"pos": opt_b_pos,   "hn": opt_cfull_hn, "removed": removed_cfull, "ratio": round(opt_cfull_ratio,4)},
    },
    "label_policy_judgment": {
        "option_b_safe": opt_b_safe,
        "option_b_note": opt_b_note,
        "recommended_option": recommended_option,
        "recommendation_reason": recommendation_reason,
        "adjacent_only_safe": adjacent_safe,
        "adjacent_note": adjacent_note,
    },
    "next_step_recommendation": (
        "P-C10 relabeled/filtered training manifest 생성 (Option C-lite 적용)"
        if recommended_option == "C-lite" else
        "P-C10 training preflight (Option B 그대로 사용)"
    ),
    "guardrails": {
        "crop_generated": False, "crop_modified": False,
        "training_executed": False, "model_forward": False,
        "scoring_rerun": False, "stage2_holdout_accessed": False,
        "existing_results_modified": False,
    },
    "blockers": blockers,
    "validation_errors": len(blockers),
}

with open(OUT_JSON, "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False, default=str)

# ============================================================
# 22. Markdown 보고서
# ============================================================
adj_only_of_warn_pct = hn_adj_only / hn_mnw * 100 if hn_mnw > 0 else 0
center_of_warn_pct   = hn_cnw / hn_mnw * 100 if hn_mnw > 0 else 0

md = f"""# P-C9 Full Crop Artifact Validation + Mask Warning Audit

**판정: {verdict}**
생성일시: {NOW_STR}

---

## 1. P-C8 DONE 확인

| 항목 | 값 |
|------|----|
| DONE.json 존재 | {DONE_JSON.exists()} |
| done=true | {done_ok} |
| generated | {done_data.get('generated', '?'):,} |
| n_errors (P-C8) | {done_data.get('n_errors', '?')} |

## 2. Count 정합

| 항목 | count | 기대값 | OK |
|------|-------|--------|-----|
| crop npz | {crop_count:,} | {EXPECTED_TOTAL:,} | {crop_count==EXPECTED_TOTAL} |
| labels CSV | {label_count:,} | {EXPECTED_TOTAL:,} | {label_count==EXPECTED_TOTAL} |
| integrity CSV | {integ_count:,} | {EXPECTED_TOTAL:,} | {integ_count==EXPECTED_TOTAL} |

## 3. candidate_id set 일치

| 비교 쌍 | diff | OK |
|---------|------|----|
| manifest vs labels | {len(only_manifest)+len(only_labels)} | {len(only_manifest)==0 and len(only_labels)==0} |
| labels vs integrity | {len(label_vs_integ)} | {len(label_vs_integ)==0} |
| labels vs npz | {len(label_vs_npz)} | {len(label_vs_npz)==0} |

## 4. 데이터 품질 확인

| 항목 | 결과 | OK |
|------|------|----|
| stage2_holdout contamination | {holdout_cont} | {holdout_cont==0} |
| P-C8 n_errors | {error_count} | {error_count==0} |
| crop shape (3,96,96) | bad={len(bad_shape)} | {shape_ok} |
| ct NaN sum | {nan_total} | {nan_total==0} |
| ct Inf sum | {inf_total} | {inf_total==0} |
| roi binary valid | invalid={roi_invalid} | {roi_invalid==0} |
| mask binary valid | invalid={mask_invalid} | {mask_invalid==0} |

## 5. label 분포

| label | count | ratio |
|-------|-------|-------|
| positive | {n_pos:,} | {n_pos/(n_pos+n_hn):.3f} |
| hard_negative | {n_hn:,} | {n_hn/(n_pos+n_hn):.3f} |
| **total** | **{n_pos+n_hn:,}** | **1.000** |

## 6. Mask Warning 분석 (hard_negative 기준)

| warning type | count | % of HN |
|---|---|---|
| HN total | {hn_total:,} | 100% |
| mask_nonzero_warning | {hn_mnw:,} | {hn_mnw/hn_total*100:.1f}% |
| center_mask_nonzero | {hn_cnw:,} | {hn_cnw/hn_total*100:.1f}% |
| adjacent_only (center clean) | {hn_adj_only:,} | {hn_adj_only/hn_total*100:.1f}% |
| center_only (adjacent clean) | {hn_center_only:,} | {hn_center_only/hn_total*100:.1f}% |
| center + adjacent 모두 | {hn_both:,} | {hn_both/hn_total*100:.1f}% |

**경보 비중 내역** (warning 40,006건 중):
- center 포함: {hn_cnw:,}건 ({center_of_warn_pct:.1f}%)
- adjacent only: {hn_adj_only:,}건 ({adj_only_of_warn_pct:.1f}%)

positive 중 mask 없는 케이스: {pos_no_mask:,} / {pos_total:,} ({pos_no_mask/pos_total*100:.1f}%)

## 7. patient별 warning (상위 10명)

"""

def df_to_md_table(df):
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep    = "| " + " | ".join("---" for _ in cols) + " |"
    rows   = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join([header, sep] + rows)

pat_top10 = pat_df.head(10)[["patient_id","hn_total_count","hn_warn_count","warn_ratio"]].copy()
pat_top10["warn_ratio"] = pat_top10["warn_ratio"].map(lambda x: f"{x:.3f}")
md += df_to_md_table(pat_top10)
md += "\n\n"

md += f"""## 8. candidate_rule별 warning

"""
rule_show = rule_warn[["candidate_rule","hn_total","hn_warn","warn_ratio","hn_center","center_ratio"]].copy()
rule_show["warn_ratio"]   = rule_show["warn_ratio"].map(lambda x: f"{x:.3f}")
rule_show["center_ratio"] = rule_show["center_ratio"].map(lambda x: f"{x:.3f}")
md += df_to_md_table(rule_show)
md += "\n\n"

md += f"""## 9. position_bin별 warning

"""
bin_show = bin_warn[["position_bin","hn_total","hn_warn","warn_ratio","hn_center","center_ratio"]].copy()
bin_show["warn_ratio"]   = bin_show["warn_ratio"].map(lambda x: f"{x:.3f}")
bin_show["center_ratio"] = bin_show["center_ratio"].map(lambda x: f"{x:.3f}")
md += df_to_md_table(bin_show)
md += "\n\n"

md += f"""## 10. Option B / C-lite / C-full 비교

| Option | pos | HN | removed HN | total | pos:HN ratio | 설명 |
|--------|-----|----|-----------:|-------|:-------------|------|
| B | {opt_b_pos:,} | {opt_b_hn:,} | 0 | {opt_b_pos+opt_b_hn:,} | 1:{1/opt_b_ratio:.2f} | 현재 그대로 |
| C-lite | {opt_b_pos:,} | {opt_clite_hn:,} | {removed_clite:,} | {opt_b_pos+opt_clite_hn:,} | 1:{1/opt_clite_ratio:.2f} | center_mask_nonzero HN 제거 |
| C-full | {opt_b_pos:,} | {opt_cfull_hn:,} | {removed_cfull:,} | {opt_b_pos+opt_cfull_hn:,} | 1:{1/opt_cfull_ratio:.2f} | mask_nonzero_warning 전체 제거 |

## 11. Label Policy 판단

| 판단 항목 | 결과 |
|-----------|------|
| Option B 유지 가능 여부 | **{opt_b_note}** |
| center_mask_nonzero 비율 | {center_ratio:.1%} |
| adjacent_only 비율 | {adj_only_ratio:.1%} |
| adjacent_only 안전 여부 | {adjacent_safe} |
| **추천 option** | **{recommended_option}** |
| 추천 이유 | {recommendation_reason} |

### Option A 비사용 이유
hard_negative를 mask 기준으로 positive 재라벨링하면 label 기준이 흔들리므로 제외.

## 12. 결론 및 다음 단계

"""

if recommended_option == "C-lite":
    md += f"""**추천: Option C-lite 적용**

- hard_negative 중 `center_mask_nonzero=True`인 {hn_cnw:,}건 제외
- 남은 hard_negative: {opt_clite_hn:,}건 (pos:hn = 1:{1/opt_clite_ratio:.2f})
- adjacent_only {hn_adj_only:,}건은 2.5D crop 특성상 유지 가능

**다음 단계: P-C10 relabeled/filtered training manifest 생성 (Option C-lite)**
"""
else:
    md += f"""**추천: Option B 유지**

- center_mask_nonzero 비율 {center_ratio:.1%}로 낮음 → HN label 오염 낮음
- warning flag를 DataLoader에서 참조해 선택적으로 필터링 가능

**다음 단계: P-C10 training preflight (Option B)**
"""

md += f"""
## 13. Guardrails 확인

| 항목 | 확인 |
|------|------|
| crop 생성/수정 | False |
| 2차학습 | False |
| model forward | False |
| scoring 재실행 | False |
| stage2_holdout 접근 | False |
| 기존 결과 수정 | False |

## 14. Blockers

{str(blockers) if blockers else "없음 — P-C10 진행 가능"}
"""

with open(OUT_MD, "w") as f:
    f.write(md)

print(f"\n[완료] 판정: {verdict}")
print(f"  출력 디렉토리: {OUT_DIR}")
print(f"  추천 option: {recommended_option}")
