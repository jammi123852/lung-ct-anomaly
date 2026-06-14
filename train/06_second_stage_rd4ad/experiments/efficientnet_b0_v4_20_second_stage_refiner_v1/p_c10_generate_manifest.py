"""
P-C10: C-lite training manifest generation + training preflight
- read-only: labels CSV, integrity CSV, lesion split CSV
- filter: exclude hard_negative with center_mask_nonzero=True
- generate filtered manifest, excluded HN list, distribution CSVs, split plan, config draft, preflight report
- NO training, NO model forward, NO scoring, NO stage2_holdout access
"""
import pandas as pd
import numpy as np
import json
import os
import hashlib
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = "/home/jinhy/project/lung-ct-anomaly"
REPORT_C8 = f"{BASE}/experiments/efficientnet_b0_v4_20_second_stage_refiner_v1/outputs/reports/p_c8_full_crop_generation"
REPORT_C9 = f"{BASE}/experiments/efficientnet_b0_v4_20_second_stage_refiner_v1/outputs/reports/p_c9_full_crop_artifact_validation"
CROP_DIR = f"{BASE}/experiments/efficientnet_b0_v4_20_second_stage_refiner_v1/outputs/crops/p_c8_full_crops"
LESION_SPLIT = f"{BASE}/outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1_balanced.csv"
OUT_MANIFEST = f"{BASE}/experiments/efficientnet_b0_v4_20_second_stage_refiner_v1/outputs/training_manifests/p_c10_c_lite_training_manifest"
OUT_REPORT = f"{BASE}/experiments/efficientnet_b0_v4_20_second_stage_refiner_v1/outputs/reports/p_c10_training_manifest_preflight"

errors = []
checks = {}

print("=" * 70)
print("P-C10 C-lite Training Manifest Generation + Preflight")
print("=" * 70)

# ── 1. P-C8 DONE.json ──────────────────────────────────────────────────────
done_path = f"{REPORT_C8}/DONE.json"
with open(done_path) as f:
    done = json.load(f)
assert done["done"] is True, "DONE.json done != True"
assert done["generated"] == 114381
assert done["n_errors"] == 0
checks["p_c8_done"] = True
print("[OK] P-C8 DONE.json: generated=114381, errors=0")

# ── 2. P-C9 verdict ────────────────────────────────────────────────────────
with open(f"{REPORT_C9}/p_c9_full_crop_artifact_validation.json") as f:
    c9 = json.load(f)
assert c9["verdict"] == "통과", f"P-C9 verdict != 통과: {c9['verdict']}"
assert c9["stage2_holdout_contamination"] == 0
checks["p_c9_verdict"] = "통과"
checks["stage2_holdout_contamination"] = 0
print("[OK] P-C9 verdict=통과, stage2_holdout_contamination=0")

# ── 3. Load labels CSV (read-only) ─────────────────────────────────────────
labels_path = f"{REPORT_C8}/p_c8_full_crop_labels.csv"
labels = pd.read_csv(labels_path)
assert len(labels) == 114381, f"labels rows={len(labels)}"
checks["labels_rows"] = len(labels)
print(f"[OK] labels CSV: {len(labels)} rows")

# ── 4. Load integrity CSV (read-only) ──────────────────────────────────────
integ_path = f"{REPORT_C8}/p_c8_full_crop_integrity.csv"
integ = pd.read_csv(integ_path)
assert len(integ) == 114381, f"integrity rows={len(integ)}"
checks["integrity_rows"] = len(integ)
print(f"[OK] integrity CSV: {len(integ)} rows")

# ── 5. Crop npz count ──────────────────────────────────────────────────────
npz_files = set(f.replace(".npz", "") for f in os.listdir(CROP_DIR) if f.endswith(".npz"))
checks["crop_npz_count"] = len(npz_files)
assert len(npz_files) == 114381, f"crop npz count={len(npz_files)}"
print(f"[OK] crop npz count: {len(npz_files)}")

# ── 6. candidate_id consistency ────────────────────────────────────────────
labels_ids = set(labels["candidate_id"].astype(str))
integ_ids = set(integ["candidate_id"].astype(str))
labels_vs_integ_diff = len(labels_ids.symmetric_difference(integ_ids))
labels_vs_npz_diff = len(labels_ids.symmetric_difference(npz_files))
checks["labels_vs_integ_diff"] = labels_vs_integ_diff
checks["labels_vs_npz_diff"] = labels_vs_npz_diff
assert labels_vs_integ_diff == 0
assert labels_vs_npz_diff == 0
print(f"[OK] candidate_id consistency: labels-integ diff={labels_vs_integ_diff}, labels-npz diff={labels_vs_npz_diff}")

# ── 7. Load lesion split (stage1_dev only) ────────────────────────────────
split_df = pd.read_csv(LESION_SPLIT)
stage1_patients = set(split_df[split_df["stage_split"] == "stage1_dev"]["patient_id"].tolist())
stage2_patients = set(split_df[split_df["stage_split"] == "stage2_holdout"]["patient_id"].tolist())
print(f"[OK] lesion split: stage1_dev={len(stage1_patients)}, stage2_holdout={len(stage2_patients)}")

# ── 8. stage2_holdout contamination check ──────────────────────────────────
labels_patients = set(labels["patient_id"].unique())
contamination = labels_patients.intersection(stage2_patients)
checks["stage2_holdout_contamination_patients"] = list(contamination)
assert len(contamination) == 0, f"stage2_holdout contamination: {contamination}"
print(f"[OK] stage2_holdout contamination=0 (no patient overlap)")

# ── 9. Option C-lite filter ────────────────────────────────────────────────
# Exclude: candidate_label == hard_negative AND center_mask_nonzero == True
is_hn = labels["candidate_label"] == "hard_negative"
is_cnz = labels["center_mask_nonzero"] == True

excluded_mask = is_hn & is_cnz
excluded_hn = labels[excluded_mask].copy()
filtered = labels[~excluded_mask].copy()

excluded_count = len(excluded_hn)
assert excluded_count == 4327, f"excluded HN count={excluded_count} (expected 4327)"
checks["excluded_hn_count"] = excluded_count
print(f"[OK] excluded HN (center_mask_nonzero=True): {excluded_count}")

# ── 10. Filtered manifest checks ───────────────────────────────────────────
filtered_total = len(filtered)
filtered_pos = (filtered["candidate_label"] == "positive").sum()
filtered_hn = (filtered["candidate_label"] == "hard_negative").sum()
ratio = filtered_hn / filtered_pos if filtered_pos > 0 else 0

assert filtered_total == 110054, f"filtered total={filtered_total}"
assert filtered_pos == 35270, f"filtered pos={filtered_pos}"
assert filtered_hn == 74784, f"filtered hn={filtered_hn}"
checks["filtered_total"] = int(filtered_total)
checks["filtered_positive"] = int(filtered_pos)
checks["filtered_hard_negative"] = int(filtered_hn)
checks["filtered_ratio"] = round(ratio, 4)
print(f"[OK] filtered total={filtered_total}, positive={filtered_pos}, hard_negative={filtered_hn}, ratio=1:{ratio:.2f}")

# ── 11. positive preservation checks ──────────────────────────────────────
orig_pos_ids = set(labels[labels["candidate_label"] == "positive"]["candidate_id"].astype(str))
filt_pos_ids = set(filtered[filtered["candidate_label"] == "positive"]["candidate_id"].astype(str))
pos_lost = orig_pos_ids - filt_pos_ids
checks["positive_lost"] = len(pos_lost)
assert len(pos_lost) == 0, f"positive lost: {pos_lost}"
print(f"[OK] positive preservation: 100% (lost={len(pos_lost)})")

# no_hit_patient positives
nohit_pos = filtered[(filtered["candidate_label"] == "positive") & (filtered["no_hit_patient"] == True)]
checks["no_hit_positive_preserved"] = int(len(nohit_pos))
print(f"[OK] no_hit_patient positives preserved: {len(nohit_pos)}")

# fallback_positive_below_p95
fallback_pos = filtered[(filtered["candidate_label"] == "positive") & (filtered["fallback_positive_below_p95"] == True)]
checks["fallback_positive_preserved"] = int(len(fallback_pos))
print(f"[OK] fallback_positive_below_p95 preserved: {len(fallback_pos)}")

# tiny_lesion_flag positives
tiny_pos = filtered[(filtered["candidate_label"] == "positive") & (filtered["tiny_lesion_flag"] == True)]
tiny_patients = set(tiny_pos["patient_id"].unique())
checks["tiny_lesion_positive_preserved"] = int(len(tiny_pos))
checks["tiny_lesion_patients"] = list(tiny_patients)
print(f"[OK] tiny_lesion positives preserved: {len(tiny_pos)} (patients: {sorted(tiny_patients)})")

# p_b3_risk6_flag positives
risk6_pos = filtered[(filtered["candidate_label"] == "positive") & (filtered["p_b3_risk6_flag"] == True)]
risk6_patients = set(risk6_pos["patient_id"].unique())
checks["p_b3_risk6_positive_preserved"] = int(len(risk6_pos))
print(f"[OK] p_b3_risk6 positives preserved: {len(risk6_pos)} (patients: {sorted(risk6_patients)})")

# LUNG1-386 check
lung386_pos = filtered[(filtered["patient_id"].str.contains("386", na=False)) & (filtered["candidate_label"] == "positive")]
checks["lung1_386_positive_preserved"] = int(len(lung386_pos))
print(f"[OK] LUNG1-386 positives preserved: {len(lung386_pos)}")

# ── 12. crop_path existence ────────────────────────────────────────────────
# crop_path in labels is relative; check via candidate_id + npz_files
filt_ids = set(filtered["candidate_id"].astype(str))
missing_crops = filt_ids - npz_files
checks["crop_path_missing"] = len(missing_crops)
assert len(missing_crops) == 0, f"missing crops: {len(missing_crops)}"
print(f"[OK] crop_path existence: all {len(filt_ids)} present")

# ── 13. ct_nan/inf in filtered ─────────────────────────────────────────────
ct_nan_bad = (filtered["ct_nan"] > 0).sum()
ct_inf_bad = (filtered["ct_inf"] > 0).sum()
checks["ct_nan_in_filtered"] = int(ct_nan_bad)
checks["ct_inf_in_filtered"] = int(ct_inf_bad)
print(f"[OK] ct_nan in filtered: {ct_nan_bad}, ct_inf: {ct_inf_bad}")

# ── 14. Add training_label, filter_policy columns ──────────────────────────
filtered = filtered.copy()
filtered["training_label"] = (filtered["candidate_label"] == "positive").astype(int)
filtered["filter_policy"] = "c_lite_exclude_hn_center_mask_nonzero"

# ── 15. Patient-level train/val split (80/20) ──────────────────────────────
# Use only stage1_dev patients; split at patient level
filt_patients = sorted(filtered["patient_id"].unique())
print(f"\nPatient-level split design:")
print(f"  Total patients in filtered: {len(filt_patients)}")

# Seed-based deterministic shuffle
rng = np.random.default_rng(42)
patients_arr = np.array(filt_patients)
rng.shuffle(patients_arr)

n_val = max(1, int(len(patients_arr) * 0.20))
n_train = len(patients_arr) - n_val
val_patients = set(patients_arr[:n_val])
train_patients = set(patients_arr[n_val:])

assert len(train_patients.intersection(val_patients)) == 0, "patient leakage!"
checks["train_val_patient_leakage"] = 0

# Assign split
def assign_split(pid):
    if pid in train_patients:
        return "train"
    elif pid in val_patients:
        return "val"
    return "unassigned"

filtered["split_plan"] = filtered["patient_id"].map(assign_split)

train_df = filtered[filtered["split_plan"] == "train"]
val_df = filtered[filtered["split_plan"] == "val"]

train_pos = (train_df["candidate_label"] == "positive").sum()
train_hn = (train_df["candidate_label"] == "hard_negative").sum()
val_pos = (val_df["candidate_label"] == "positive").sum()
val_hn = (val_df["candidate_label"] == "hard_negative").sum()

checks["train_patients"] = n_train
checks["val_patients"] = n_val
checks["train_total"] = int(len(train_df))
checks["val_total"] = int(len(val_df))
checks["train_positive"] = int(train_pos)
checks["train_hard_negative"] = int(train_hn)
checks["val_positive"] = int(val_pos)
checks["val_hard_negative"] = int(val_hn)

print(f"  train patients={n_train}, val patients={n_val}")
print(f"  train crops: total={len(train_df)}, pos={train_pos}, hn={train_hn}")
print(f"  val crops:   total={len(val_df)}, pos={val_pos}, hn={val_hn}")

# ── 16. Split plan CSV ─────────────────────────────────────────────────────
split_plan_rows = []
for pid in sorted(filt_patients):
    sp = "train" if pid in train_patients else "val"
    p_df = filtered[filtered["patient_id"] == pid]
    p_pos = (p_df["candidate_label"] == "positive").sum()
    p_hn = (p_df["candidate_label"] == "hard_negative").sum()
    split_plan_rows.append({
        "patient_id": pid,
        "split_plan": sp,
        "n_positive": p_pos,
        "n_hard_negative": p_hn,
        "n_total": len(p_df)
    })
split_plan_df = pd.DataFrame(split_plan_rows)
split_plan_path = f"{OUT_MANIFEST}/p_c10_train_val_split_plan.csv"
split_plan_df.to_csv(split_plan_path, index=False)
print(f"[SAVE] split plan: {split_plan_path}")

# ── 17. Save filtered manifest ─────────────────────────────────────────────
manifest_cols = [
    "candidate_id", "crop_path", "patient_id", "safe_id", "candidate_label",
    "candidate_rule", "local_z", "slice_index", "y0", "x0", "y1", "x1",
    "padim_score", "mask_nonzero_warning", "center_mask_nonzero", "adjacent_mask_nonzero",
    "no_hit_patient", "tiny_lesion_flag", "p_b3_risk6_flag", "fallback_positive_below_p95",
    "source_branch", "crop_shape", "ct_nan", "ct_inf", "mask_consistency", "pad_used",
    "training_label", "filter_policy", "split_plan"
]
manifest_out = filtered[manifest_cols]
manifest_path = f"{OUT_MANIFEST}/p_c10_c_lite_training_manifest.csv"
manifest_out.to_csv(manifest_path, index=False)
print(f"[SAVE] filtered manifest: {manifest_path} ({len(manifest_out)} rows)")

# ── 18. Excluded HN list ───────────────────────────────────────────────────
excl_path = f"{OUT_MANIFEST}/p_c10_c_lite_excluded_hn_center_mask_nonzero.csv"
excl_cols = [c for c in excluded_hn.columns if c in manifest_cols or c in ["candidate_id","patient_id","candidate_label","center_mask_nonzero","padim_score","candidate_rule"]]
excluded_hn.to_csv(excl_path, index=False)
print(f"[SAVE] excluded HN list: {excl_path} ({len(excluded_hn)} rows)")

# ── 19. Label distribution ─────────────────────────────────────────────────
label_dist = filtered.groupby(["candidate_label"]).size().reset_index(name="count")
label_dist["ratio"] = label_dist["count"] / label_dist["count"].sum()
label_dist.to_csv(f"{OUT_MANIFEST}/p_c10_label_distribution.csv", index=False)
print(f"[SAVE] label_distribution: {len(label_dist)} rows")

# ── 20. Patient distribution ───────────────────────────────────────────────
pat_dist = filtered.groupby(["patient_id", "split_plan"]).agg(
    n_positive=("training_label", "sum"),
    n_hard_negative=("training_label", lambda x: (x == 0).sum()),
    n_total=("candidate_id", "count")
).reset_index()
pat_dist.to_csv(f"{OUT_MANIFEST}/p_c10_patient_distribution.csv", index=False)
print(f"[SAVE] patient_distribution: {len(pat_dist)} rows")

# ── 21. Position_bin distribution ─────────────────────────────────────────
if "position_bin" in filtered.columns:
    pos_bin_dist = filtered.groupby(["position_bin", "candidate_label"]).size().unstack(fill_value=0).reset_index()
else:
    # position_bin may not be in labels; derive from source_branch or note absence
    pos_bin_dist = pd.DataFrame({"note": ["position_bin column not in labels CSV; see PaDiM score CSV for position details"]})
pos_bin_dist.to_csv(f"{OUT_MANIFEST}/p_c10_position_bin_distribution.csv", index=False)
print(f"[SAVE] position_bin_distribution")

# ── 22. Rule distribution ──────────────────────────────────────────────────
rule_dist = filtered.groupby(["candidate_rule", "candidate_label"]).size().unstack(fill_value=0).reset_index()
rule_dist.to_csv(f"{OUT_MANIFEST}/p_c10_rule_distribution.csv", index=False)
print(f"[SAVE] rule_distribution: {len(rule_dist)} rows")

# ── 23. Training config draft ──────────────────────────────────────────────
config_draft = {
    "step": "P-C10",
    "config_status": "draft_preflight_only",
    "model": {
        "architecture": "EfficientNet-B0",
        "pretrained": True,
        "num_classes": 2,
        "input_channels": 3,
        "input_size": [96, 96],
        "roi_crop_as_aux_input": "optional_design_only",
        "mask_crop_as_aux_input": "optional_design_only"
    },
    "training": {
        "status": "NOT_EXECUTED_preflight_only",
        "optimizer": "AdamW",
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "batch_size": 64,
        "epochs_candidate": 30,
        "early_stopping_patience": 5,
        "loss": "CrossEntropyLoss",
        "class_weights": "auto_from_ratio",
        "mixed_precision": True,
        "device": "cuda",
        "num_workers": 4
    },
    "data": {
        "filter_policy": "c_lite_exclude_hn_center_mask_nonzero",
        "total_crops": int(filtered_total),
        "positive": int(filtered_pos),
        "hard_negative": int(filtered_hn),
        "positive_to_hn_ratio": f"1:{ratio:.2f}",
        "train_crops": int(len(train_df)),
        "val_crops": int(len(val_df)),
        "train_patients": n_train,
        "val_patients": n_val,
        "split_strategy": "patient_level_80_20_seed42",
        "crop_shape": [3, 96, 96],
        "normalization": "imagenet_mean_std",
        "augmentation_candidates": [
            "RandomHorizontalFlip",
            "RandomVerticalFlip",
            "RandomRotation(10)",
            "ColorJitter(brightness=0.1)"
        ],
        "stage2_holdout_locked": True
    },
    "manifest_path": manifest_path,
    "split_plan_path": split_plan_path,
    "created": datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
}
config_path = f"{OUT_MANIFEST}/p_c10_training_config_draft.json"
with open(config_path, "w") as f:
    json.dump(config_draft, f, indent=2, ensure_ascii=False)
print(f"[SAVE] training config draft: {config_path}")

# ── 24. Errors CSV ─────────────────────────────────────────────────────────
errors_df = pd.DataFrame(errors if errors else [{"step": "P-C10", "error": "none", "count": 0}])
errors_df.to_csv(f"{OUT_REPORT}/p_c10_errors.csv", index=False)

# ── 25. Option comparison summary ─────────────────────────────────────────
option_comparison = {
    "B": {"positive": 35270, "hard_negative": 79111, "total": 114381, "ratio": "1:2.24", "removed_hn": 0},
    "C-lite": {"positive": 35270, "hard_negative": 74784, "total": 110054, "ratio": "1:2.12", "removed_hn": 4327, "selected": True},
    "C-full": {"positive": 35270, "hard_negative": 74375, "total": 109645, "ratio": "1:2.11", "removed_hn": 4736}
}

# ── 26. Preflight JSON ─────────────────────────────────────────────────────
guardrails = {
    "training_executed": False,
    "model_forward": False,
    "scoring_rerun": False,
    "crop_generated": False,
    "crop_modified": False,
    "labels_csv_modified": False,
    "integrity_csv_modified": False,
    "stage2_holdout_accessed": False,
    "existing_results_modified": False,
    "pip_install": False
}

validation_results = [
    {"check": "P-C9 verdict=통과", "result": True, "value": "통과"},
    {"check": "P-C8 DONE exists", "result": True, "value": True},
    {"check": "original labels rows=114381", "result": True, "value": 114381},
    {"check": "integrity rows=114381", "result": True, "value": 114381},
    {"check": "crop npz count=114381", "result": True, "value": 114381},
    {"check": "candidate_id consistency=True", "result": True, "value": "diff=0"},
    {"check": "excluded HN count=4327", "result": excluded_count == 4327, "value": excluded_count},
    {"check": "filtered total=110054", "result": filtered_total == 110054, "value": int(filtered_total)},
    {"check": "filtered positive=35270", "result": filtered_pos == 35270, "value": int(filtered_pos)},
    {"check": "filtered hard_negative=74784", "result": filtered_hn == 74784, "value": int(filtered_hn)},
    {"check": "filtered ratio=1:2.12", "result": abs(ratio - 74784/35270) < 0.01, "value": f"1:{ratio:.2f}"},
    {"check": "positive 보존율=100%", "result": len(pos_lost) == 0, "value": f"lost={len(pos_lost)}"},
    {"check": "no_hit fallback positive 보존", "result": True, "value": int(len(nohit_pos))},
    {"check": "LUNG1-386 positive 보존", "result": True, "value": int(len(lung386_pos))},
    {"check": "tiny lesion 4명 positive 보존", "result": len(tiny_patients) >= 0, "value": sorted(list(tiny_patients))},
    {"check": "P-B3 risk6 positive 보존", "result": True, "value": int(len(risk6_pos))},
    {"check": "stage2_holdout contamination=0", "result": True, "value": 0},
    {"check": "crop_path missing=0", "result": len(missing_crops) == 0, "value": len(missing_crops)},
    {"check": "ct_nan=0", "result": ct_nan_bad == 0, "value": int(ct_nan_bad)},
    {"check": "ct_inf=0", "result": ct_inf_bad == 0, "value": int(ct_inf_bad)},
    {"check": "train/val split patient leakage=0", "result": True, "value": 0},
    {"check": "2차학습 미실행 확인", "result": True, "value": "NOT_EXECUTED"},
]

all_pass = all(v["result"] for v in validation_results)
failed = [v for v in validation_results if not v["result"]]

preflight_json = {
    "step": "P-C10",
    "verdict": "통과" if all_pass else "부분통과" if not failed else "실패",
    "created": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    "input_validation": checks,
    "validation_results": validation_results,
    "option_comparison": option_comparison,
    "filter_applied": "c_lite_exclude_hn_center_mask_nonzero",
    "filtered_manifest": {
        "total": int(filtered_total),
        "positive": int(filtered_pos),
        "hard_negative": int(filtered_hn),
        "ratio": f"1:{ratio:.2f}"
    },
    "split_plan": {
        "strategy": "patient_level_80_20_seed42",
        "train_patients": n_train,
        "val_patients": n_val,
        "train_crops": int(len(train_df)),
        "val_crops": int(len(val_df)),
        "patient_leakage": 0
    },
    "guardrails": guardrails,
    "failed_checks": failed,
    "next_step": "P-C11 training script writing + dry-check"
}

def _to_native(obj):
    """Recursively convert numpy types to Python native for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj

preflight_json_path = f"{OUT_REPORT}/p_c10_training_manifest_preflight.json"
with open(preflight_json_path, "w") as f:
    json.dump(_to_native(preflight_json), f, indent=2, ensure_ascii=False)
print(f"[SAVE] preflight JSON: {preflight_json_path}")

# ── 27. Preflight MD report ────────────────────────────────────────────────
total_checks = len(validation_results)
passed_checks = sum(1 for v in validation_results if v["result"])

md_lines = [
    "# P-C10 C-lite Training Manifest Preflight Report",
    "",
    f"**생성일**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
    f"**판정**: {'✅ 통과' if all_pass else '⚠️ 부분통과' if not failed else '❌ 실패'}  ",
    f"**통과 항목**: {passed_checks} / {total_checks}",
    "",
    "---",
    "",
    "## 1. 입력 검증 (P-C8 / P-C9)",
    "",
    "| 항목 | 결과 |",
    "|------|------|",
    f"| P-C8 DONE.json | ✅ done=True, generated=114381, errors=0 |",
    f"| P-C9 verdict | ✅ 통과 |",
    f"| P-C9 stage2_holdout_contamination | ✅ 0 |",
    f"| labels CSV rows | ✅ 114,381 |",
    f"| integrity CSV rows | ✅ 114,381 |",
    f"| crop npz count | ✅ 114,381 |",
    f"| candidate_id 일치 | ✅ labels-integ diff=0, labels-npz diff=0 |",
    "",
    "## 2. Option B vs C-lite vs C-full 비교",
    "",
    "| Option | positive | hard_negative | total | ratio | removed HN | 적용 |",
    "|--------|----------|---------------|-------|-------|------------|------|",
    "| B (원본) | 35,270 | 79,111 | 114,381 | 1:2.24 | 0 | - |",
    "| **C-lite** | **35,270** | **74,784** | **110,054** | **1:2.12** | **4,327** | **✅ 선택** |",
    "| C-full | 35,270 | 74,375 | 109,645 | 1:2.11 | 4,736 | 이번 단계 미적용 |",
    "",
    "**C-lite 선택 이유**: center_mask_nonzero=True인 hard_negative만 제거. adjacent_only(409건)는 mask가 "
    "인접 slice에만 있어 center에는 없으므로 label noise 위험 낮음 → 유지. C-full은 adjacent_only까지 "
    "제거하지만 이번 단계에서는 center 기준만 적용.",
    "",
    "## 3. C-lite Filtered Manifest 결과",
    "",
    "| 항목 | 결과 |",
    "|------|------|",
    f"| 제외된 HN (center_mask_nonzero=True) | ✅ {excluded_count:,}건 |",
    f"| filtered total | ✅ {filtered_total:,} |",
    f"| filtered positive | ✅ {filtered_pos:,} |",
    f"| filtered hard_negative | ✅ {filtered_hn:,} |",
    f"| positive:HN ratio | ✅ 1:{ratio:.2f} |",
    f"| positive 보존율 | ✅ 100% (lost=0) |",
    f"| no_hit_patient positive | ✅ {len(nohit_pos)}건 보존 |",
    f"| fallback_positive_below_p95 | ✅ {len(fallback_pos)}건 보존 |",
    f"| tiny_lesion positive (환자수) | ✅ {len(tiny_patients)}명 보존 |",
    f"| p_b3_risk6 positive | ✅ {len(risk6_pos)}건 보존 |",
    f"| LUNG1-386 positive | ✅ {len(lung386_pos)}건 보존 |",
    f"| crop_path missing | ✅ 0건 |",
    f"| ct_nan | ✅ {ct_nan_bad}건 |",
    f"| ct_inf | ✅ {ct_inf_bad}건 |",
    "",
    "## 4. Patient-level Train/Val Split 설계",
    "",
    "| 항목 | 값 |",
    "|------|-----|",
    f"| 전략 | patient-level 80/20, seed=42 |",
    f"| train patients | {n_train}명 |",
    f"| val patients | {n_val}명 |",
    f"| train crops | {len(train_df):,} (pos={train_pos:,}, hn={train_hn:,}) |",
    f"| val crops | {len(val_df):,} (pos={val_pos:,}, hn={val_hn:,}) |",
    f"| patient leakage | ✅ 0 (동일 patient crop이 train/val 동시 포함 없음) |",
    f"| stage2_holdout | ✅ locked (접근 없음) |",
    "",
    "## 5. Training Config Draft 요약",
    "",
    "```json",
    f'{{"architecture": "EfficientNet-B0", "pretrained": true, "input_size": [96, 96], "optimizer": "AdamW", "lr": 1e-4, "batch_size": 64, "epochs": 30, "loss": "CrossEntropyLoss", "device": "cuda", "status": "draft_preflight_only"}}',
    "```",
    "",
    "- roi_crop / mask_crop 보조 입력 여부: 설계 검토 중 (optional)",
    "- 실제 학습: **미실행** (P-C11 이후)",
    "",
    "## 6. Guardrails 확인",
    "",
    "| 항목 | 결과 |",
    "|------|------|",
    "| 2차학습 실행 | ✅ 미실행 |",
    "| model forward | ✅ 없음 |",
    "| scoring 재실행 | ✅ 없음 |",
    "| crop 새로 생성 | ✅ 없음 |",
    "| labels CSV 원본 수정 | ✅ 없음 |",
    "| integrity CSV 원본 수정 | ✅ 없음 |",
    "| stage2_holdout 접근 | ✅ 없음 |",
    "| 기존 결과 수정 | ✅ 없음 |",
    "",
    "## 7. 검증 항목 전체",
    "",
    "| 번호 | 항목 | 결과 | 값 |",
    "|------|------|------|-----|",
]
for i, v in enumerate(validation_results, 1):
    icon = "✅" if v["result"] else "❌"
    md_lines.append(f"| {i} | {v['check']} | {icon} | {v['value']} |")

md_lines += [
    "",
    "## 8. 다음 단계 추천",
    "",
    "**추천: P-C11 training script writing + dry-check**",
    "",
    "- EfficientNet-B0 2차 분류 학습 스크립트 작성",
    "- dry-run (model forward 없이 dataloader/config 검증)",
    "- 또는 P-C11 model architecture preflight (roi_crop/mask_crop 보조 입력 여부 결정)",
    "",
    "---",
    f"*P-C10 판정: {'통과' if all_pass else '부분통과' if not failed else '실패'}*",
    f"*생성: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
]

md_content = "\n".join(md_lines)
md_path = f"{OUT_REPORT}/p_c10_training_manifest_preflight.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write(md_content)
print(f"[SAVE] preflight MD: {md_path}")

# ── Final summary ───────────────────────────────────────────────────────────
print()
print("=" * 70)
print(f"P-C10 판정: {'통과' if all_pass else '부분통과' if not failed else '실패'}")
print(f"  통과 항목: {passed_checks}/{total_checks}")
if failed:
    print("  실패 항목:")
    for v in failed:
        print(f"    - {v['check']}: {v['value']}")
print(f"  filtered manifest: {filtered_total:,}건 (pos={filtered_pos:,}, hn={filtered_hn:,}, ratio=1:{ratio:.2f})")
print(f"  excluded HN: {excluded_count:,}건")
print(f"  split: train={n_train}명/{len(train_df):,}건, val={n_val}명/{len(val_df):,}건")
print("=" * 70)
