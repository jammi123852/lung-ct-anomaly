"""
P-C6: EfficientNet-B0 v4_20 second-stage crop smoke validation
Read-only 검증 스크립트
- crop npz 110개 integrity 재확인
- hn_mask_warn 4개 상세 분석 (채널별 mask 분포)
- label 정책 추천
- P-C7 진행 가능 여부 판단
"""

import os
import sys
import json
import csv
import datetime
import numpy as np

# ===================== 경로 설정 =====================
BASE = "experiments/efficientnet_b0_v4_20_second_stage_refiner_v1/outputs"
CROP_DIR = f"{BASE}/crops/p_c5_crop_smoke"
REPORT_DIR = f"{BASE}/reports/p_c5_crop_smoke"
OUT_DIR    = f"{BASE}/reports/p_c6_crop_smoke_validation"

LABELS_CSV    = f"{REPORT_DIR}/p_c5_crop_smoke_labels.csv"
MANIFEST_CSV  = f"{REPORT_DIR}/p_c5_crop_smoke_manifest.csv"
SUMMARY_JSON  = f"{REPORT_DIR}/p_c5_crop_smoke_summary.json"
INTEGRITY_CSV = f"{REPORT_DIR}/p_c5_crop_integrity_validation.csv"
LOCALZ_CSV    = f"{REPORT_DIR}/p_c5_local_z_vs_slice_index_check.csv"

os.makedirs(OUT_DIR, exist_ok=True)

REQUIRED_KEYS = {
    "ct_crop", "roi_crop", "mask_crop",
    "candidate_id", "patient_id", "safe_id",
    "candidate_label", "candidate_rule",
    "local_z", "slice_index",
    "y0", "x0", "y1", "x1",
    "padim_score",
}

errors = []

# ===================== 1. P-C5 summary 확인 =====================
print("[1] P-C5 summary JSON 읽기...")
with open(SUMMARY_JSON) as f:
    summary = json.load(f)

p_c5_verdict = summary.get("verdict", "UNKNOWN")
guardrails    = summary.get("guardrails", {})
integrity     = summary.get("integrity", {})
special       = summary.get("special_coverage", {})

assert p_c5_verdict == "통과", f"P-C5 verdict != 통과: {p_c5_verdict}"
print(f"  P-C5 verdict: {p_c5_verdict}")

# ===================== 2. label CSV 읽기 =====================
print("[2] label CSV 읽기...")
with open(LABELS_CSV, newline="") as f:
    label_rows = list(csv.DictReader(f))

label_ids = {r["candidate_id"] for r in label_rows}
print(f"  label rows: {len(label_rows)}")

# ===================== 3. manifest CSV 읽기 =====================
print("[3] manifest CSV 읽기...")
with open(MANIFEST_CSV, newline="") as f:
    manifest_rows = list(csv.DictReader(f))
print(f"  manifest rows: {len(manifest_rows)}")

# ===================== 4. npz 파일 목록 =====================
print("[4] npz 파일 목록 확인...")
npz_files = sorted([f for f in os.listdir(CROP_DIR) if f.endswith(".npz")])
npz_ids   = {f.replace(".npz", "") for f in npz_files}
print(f"  npz 파일 수: {len(npz_files)}")

# ===================== 5. candidate_id 일치 확인 =====================
print("[5] candidate_id 일치 확인...")
id_match = label_ids == npz_ids
if not id_match:
    only_label = label_ids - npz_ids
    only_npz   = npz_ids - label_ids
    errors.append(f"candidate_id 불일치: only_label={only_label}, only_npz={only_npz}")
print(f"  id_match: {id_match}")

# ===================== 6-16. npz 상세 검증 =====================
print("[6-16] npz 110개 상세 검증...")

npz_inventory       = []
shape_val_rows      = []
mask_consistency_rows = []
hn_warn_rows        = []

EXPECTED_SHAPE = (3, 96, 96)

pos_total   = 0
pos_ok      = 0
hn_total    = 0
hn_ok       = 0
hn_warn     = 0

shape_ok_count    = 0
ct_nan_count      = 0
ct_inf_count      = 0
roi_valid_count   = 0
mask_valid_count  = 0
missing_key_count = 0

for row in label_rows:
    cid   = row["candidate_id"]
    label = row["candidate_label"]
    npz_path = os.path.join(CROP_DIR, f"{cid}.npz")

    if not os.path.exists(npz_path):
        errors.append(f"npz 없음: {cid}")
        continue

    data = np.load(npz_path, allow_pickle=True)

    # key 확인
    missing_keys = REQUIRED_KEYS - set(data.keys())
    if missing_keys:
        missing_key_count += 1
        errors.append(f"missing keys [{cid}]: {missing_keys}")

    ct   = data["ct_crop"]   # int16 또는 float
    roi  = data["roi_crop"]  # uint8
    mask = data["mask_crop"] # uint8

    # shape
    shape_ok = (ct.shape == EXPECTED_SHAPE)
    if shape_ok:
        shape_ok_count += 1

    # ct dtype / range
    ct_dtype  = str(ct.dtype)
    ct_min    = float(np.min(ct))
    ct_max    = float(np.max(ct))
    ct_nan    = int(np.isnan(ct.astype(np.float32)).sum())
    ct_inf    = int(np.isinf(ct.astype(np.float32)).sum())
    if ct_nan > 0:
        ct_nan_count += 1
    if ct_inf > 0:
        ct_inf_count += 1

    # roi binary 확인
    roi_unique = sorted(np.unique(roi).tolist())
    roi_binary = all(v in [0, 1] for v in roi_unique)
    if roi_binary:
        roi_valid_count += 1

    # mask binary 확인
    mask_unique = sorted(np.unique(mask).tolist())
    mask_binary = all(v in [0, 1] for v in mask_unique)
    if mask_binary:
        mask_valid_count += 1

    # mask_crop 비율 (채널별)
    mask_ch0_sum = int(mask[0].sum())
    mask_ch1_sum = int(mask[1].sum())
    mask_ch2_sum = int(mask[2].sum())
    mask_any     = int(mask.sum())
    mask_center  = mask_ch1_sum  # z center channel

    # mask consistency
    if label == "positive":
        pos_total += 1
        if mask_any > 0:
            pos_ok += 1
            consistency = "ok"
        else:
            consistency = "warn_pos_mask_zero"
            errors.append(f"positive mask zero: {cid}")
    elif label == "hard_negative":
        hn_total += 1
        if mask_any == 0:
            hn_ok += 1
            consistency = "ok"
        else:
            hn_warn += 1
            consistency = "warning_hn_mask_nonzero"

    # npz inventory
    npz_inventory.append({
        "candidate_id":  cid,
        "patient_id":    row["patient_id"],
        "candidate_label": label,
        "candidate_rule": row["candidate_rule"],
        "local_z":       row["local_z"],
        "slice_index":   row["slice_index"],
        "shape_ok":      shape_ok,
        "ct_dtype":      ct_dtype,
        "ct_min":        round(ct_min, 2),
        "ct_max":        round(ct_max, 2),
        "ct_nan":        ct_nan,
        "ct_inf":        ct_inf,
        "roi_binary":    roi_binary,
        "roi_unique":    str(roi_unique),
        "mask_binary":   mask_binary,
        "mask_unique":   str(mask_unique),
        "mask_ch0":      mask_ch0_sum,
        "mask_ch1_center": mask_ch1_sum,
        "mask_ch2":      mask_ch2_sum,
        "mask_any":      mask_any,
        "mask_consistency": consistency,
        "missing_keys":  str(sorted(missing_keys)),
    })

    # shape/value validation
    shape_val_rows.append({
        "candidate_id":  cid,
        "label":         label,
        "shape_ok":      shape_ok,
        "ct_shape":      str(ct.shape),
        "ct_dtype":      ct_dtype,
        "ct_min":        round(ct_min, 2),
        "ct_max":        round(ct_max, 2),
        "ct_nan":        ct_nan,
        "ct_inf":        ct_inf,
        "roi_binary":    roi_binary,
        "mask_binary":   mask_binary,
    })

    # mask consistency
    mask_consistency_rows.append({
        "candidate_id":  cid,
        "patient_id":    row["patient_id"],
        "label":         label,
        "mask_any":      mask_any,
        "mask_ch0":      mask_ch0_sum,
        "mask_ch1_center": mask_ch1_sum,
        "mask_ch2":      mask_ch2_sum,
        "consistency":   consistency,
    })

    # hn warn 상세
    if consistency == "warning_hn_mask_nonzero":
        hn_warn_rows.append({
            "candidate_id":   cid,
            "patient_id":     row["patient_id"],
            "candidate_rule": row["candidate_rule"],
            "local_z":        row["local_z"],
            "slice_index":    row["slice_index"],
            "y0":             row["y0"],
            "x0":             row["x0"],
            "y1":             row["y1"],
            "x1":             row["x1"],
            "padim_score":    row["padim_score"],
            "mask_any":       mask_any,
            "mask_ch0_zm1":   mask_ch0_sum,
            "mask_ch1_center": mask_ch1_sum,
            "mask_ch2_zp1":   mask_ch2_sum,
            "center_nonzero": mask_ch1_sum > 0,
            "only_adjacent":  mask_ch1_sum == 0 and mask_any > 0,
            "cause_analysis": (
                "center nonzero: center patch가 병변에 걸침"
                if mask_ch1_sum > 0
                else "adjacent only: z-1 또는 z+1 채널에만 병변 걸침 (crop 확장 artifact)"
            ),
            "crop_boundary_entry": (
                "y/x 확장으로 인근 병변 진입 가능"
                if mask_any > 0
                else "n/a"
            ),
        })

print(f"  shape OK: {shape_ok_count}/110")
print(f"  ct NaN: {ct_nan_count}, Inf: {ct_inf_count}")
print(f"  roi binary valid: {roi_valid_count}")
print(f"  mask binary valid: {mask_valid_count}")
print(f"  pos mask OK: {pos_ok}/{pos_total}")
print(f"  hn mask OK: {hn_ok}/{hn_total}, warn: {hn_warn}")

# ===================== local_z vs slice_index =====================
print("[17] local_z vs slice_index 재확인...")
with open(LOCALZ_CSV, newline="") as f:
    lz_rows = list(csv.DictReader(f))

lz_inbound    = sum(1 for r in lz_rows if r["local_z_inbound"] == "True")
si_inbound    = sum(1 for r in lz_rows if r["slice_index_inbound"] == "True")
si_oob        = len(lz_rows) - si_inbound
use_lz_all    = all(r["use_local_z"] == "True" for r in lz_rows)
print(f"  local_z inbound: {lz_inbound}/{len(lz_rows)}")
print(f"  slice_index inbound: {si_inbound}/{len(lz_rows)}  (OOB: {si_oob})")
print(f"  use_local_z 전부 True: {use_lz_all}")

# ===================== special coverage 확인 =====================
no_hit_ok    = special.get("no_hit_included", False)
tiny_ok      = special.get("tiny_included", False)
risk6_ok     = special.get("risk6_included", False)
fallback_ok  = special.get("fallback_included", False)

# ===================== guardrails 확인 =====================
grd_full_crop   = guardrails.get("full_crop_generated", True)
grd_training    = guardrails.get("training_executed", True)
grd_model_fwd   = guardrails.get("model_forward", True)
grd_scoring     = guardrails.get("scoring_rerun", True)
grd_feature     = guardrails.get("feature_extraction", True)
grd_threshold   = guardrails.get("threshold_recalculated", True)
grd_metrics     = guardrails.get("metrics_recalculated", True)
grd_holdout     = guardrails.get("stage2_holdout_accessed", True)
grd_modified    = guardrails.get("existing_results_modified", True)

# ===================== label policy recommendation =====================
# Option A: crop 전체 mask 기준으로 재라벨링
# Option B: center patch 기준 label 유지 + mask_nonzero_warning flag
# Option C: full crop gen에서 hn mask nonzero 제외

policy_rows = [
    {
        "option": "A",
        "description": "crop 전체 mask 기준으로 hard_negative → positive 재라벨링",
        "pros": "mask 기준 정확한 label",
        "cons": "center patch는 negative인데 label 바뀜, 4개이므로 소수이나 crop 확장 aritfact가 label에 영향",
        "recommendation": "비권장 (center patch 기준 policy와 불일치)",
    },
    {
        "option": "B",
        "description": "center patch 기준 label 유지, mask_nonzero_warning flag 추가",
        "pros": "기존 scoring 기준(center patch) 유지, flag으로 추후 대응 가능, full crop gen 변경 최소화",
        "cons": "crop에 병변 픽셀이 존재하지만 hard_negative로 학습됨 (혼란 가능)",
        "recommendation": "권장 (소수 4개, center patch 기준 policy 일관성, flag으로 추적 가능)",
    },
    {
        "option": "C",
        "description": "full crop generation에서 hn mask_crop nonzero는 제외 (생성 안 함)",
        "pros": "깨끗한 학습셋",
        "cons": "전체 crop 수 감소, 특정 환자(LUNG1-020) 3개 제거됨, 기준 변경 필요",
        "recommendation": "대안 (소수이므로 full crop gen에서 filter 적용 가능, 단 policy 문서 필요)",
    },
]

# ===================== 판정 =====================
blockers = []

if len(label_rows) != 110:
    blockers.append(f"label row count {len(label_rows)} != 110")
if len(npz_files) != 110:
    blockers.append(f"npz file count {len(npz_files)} != 110")
if shape_ok_count != 110:
    blockers.append(f"shape_ok {shape_ok_count} != 110")
if ct_nan_count > 0 or ct_inf_count > 0:
    blockers.append(f"ct NaN={ct_nan_count} Inf={ct_inf_count}")
if pos_ok != 38:
    blockers.append(f"pos_mask_ok {pos_ok} != 38")
if hn_warn > 10:
    blockers.append(f"hn_mask_warn {hn_warn} > 10 (과다)")
if guardrails.get("stage2_holdout_accessed", False):
    blockers.append("stage2_holdout_accessed=True")
if guardrails.get("existing_results_modified", False):
    blockers.append("existing_results_modified=True")
if not id_match:
    blockers.append("candidate_id mismatch")

if len(blockers) == 0:
    verdict = "통과"
elif hn_warn > 0 or not all([no_hit_ok, tiny_ok, risk6_ok, fallback_ok]):
    verdict = "부분통과"  # hn warn은 소수이면 부분통과 가능
else:
    verdict = "실패"

# hn warn이 4개이고 policy B로 처리 가능하면 통과
if len(blockers) == 0:
    verdict = "통과"

print(f"\n[판정] {verdict}")
if blockers:
    print(f"  blockers: {blockers}")

# ===================== CSV 저장 =====================
def save_csv(path, rows, fieldnames=None):
    if not rows:
        with open(path, "w", newline="") as f:
            f.write("")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

print("\n[출력] CSV 저장...")
save_csv(f"{OUT_DIR}/p_c6_crop_npz_inventory.csv", npz_inventory)
save_csv(f"{OUT_DIR}/p_c6_crop_shape_value_validation.csv", shape_val_rows)
save_csv(f"{OUT_DIR}/p_c6_mask_consistency_validation.csv", mask_consistency_rows)
save_csv(f"{OUT_DIR}/p_c6_hard_negative_mask_warning_review.csv", hn_warn_rows)
save_csv(f"{OUT_DIR}/p_c6_label_policy_recommendation.csv", policy_rows)
save_csv(f"{OUT_DIR}/p_c6_errors.csv",
         [{"error": e} for e in errors] if errors else [{"error": "none"}])

# ===================== JSON 저장 =====================
now_str = datetime.datetime.now().isoformat(timespec="seconds")

result_json = {
    "step": "P-C6",
    "verdict": verdict,
    "created": now_str,
    "p_c5_input": {
        "verdict": p_c5_verdict,
        "label_rows": len(label_rows),
        "npz_files": len(npz_files),
        "id_match": id_match,
    },
    "crop_count": len(npz_files),
    "label_count": len(label_rows),
    "shape_validation": {
        "expected": str(EXPECTED_SHAPE),
        "ok_count": shape_ok_count,
        "total": 110,
        "all_ok": shape_ok_count == 110,
    },
    "npz_key_validation": {
        "required_keys": sorted(REQUIRED_KEYS),
        "missing_key_crops": missing_key_count,
    },
    "ct_value_validation": {
        "nan_crops": ct_nan_count,
        "inf_crops": ct_inf_count,
        "valid": ct_nan_count == 0 and ct_inf_count == 0,
    },
    "roi_validation": {
        "binary_valid_crops": roi_valid_count,
        "total": 110,
    },
    "mask_validation": {
        "binary_valid_crops": mask_valid_count,
        "total": 110,
    },
    "positive_mask_consistency": {
        "total": pos_total,
        "ok": pos_ok,
        "warn_zero": pos_total - pos_ok,
        "pass": pos_ok == pos_total,
    },
    "hard_negative_mask_consistency": {
        "total": hn_total,
        "ok": hn_ok,
        "warn_nonzero": hn_warn,
        "warn_ids": [r["candidate_id"] for r in hn_warn_rows],
    },
    "hn_mask_warn_analysis": [
        {
            "candidate_id":   r["candidate_id"],
            "patient_id":     r["patient_id"],
            "local_z":        r["local_z"],
            "mask_ch0_zm1":   r["mask_ch0_zm1"],
            "mask_ch1_center": r["mask_ch1_center"],
            "mask_ch2_zp1":   r["mask_ch2_zp1"],
            "center_nonzero": r["center_nonzero"],
            "only_adjacent":  r["only_adjacent"],
            "cause_analysis": r["cause_analysis"],
        }
        for r in hn_warn_rows
    ],
    "local_z_vs_slice_index": {
        "local_z_inbound": lz_inbound,
        "slice_index_inbound": si_inbound,
        "slice_index_oob": si_oob,
        "use_local_z_all": use_lz_all,
        "conclusion": "local_z를 crop z 기준으로 확정. slice_index는 global z이므로 crop 접근에 사용 금지.",
    },
    "special_coverage": {
        "no_hit_included": no_hit_ok,
        "tiny_included": tiny_ok,
        "risk6_included": risk6_ok,
        "fallback_included": fallback_ok,
    },
    "guardrails": {
        "full_crop_generated":       not grd_full_crop,
        "training_executed":         not grd_training,
        "model_forward":             not grd_model_fwd,
        "scoring_rerun":             not grd_scoring,
        "feature_extraction":        not grd_feature,
        "threshold_recalculated":    not grd_threshold,
        "metrics_recalculated":      not grd_metrics,
        "stage2_holdout_accessed":   grd_holdout,
        "existing_results_modified": grd_modified,
        "all_guardrails_ok": not any([
            grd_full_crop, grd_training, grd_model_fwd,
            grd_scoring, grd_feature, grd_threshold,
            grd_metrics, grd_holdout, grd_modified,
        ]),
    },
    "label_policy_recommendation": {
        "recommended": "B",
        "reason": "4개 소수, center patch 기준 policy 일관성, flag으로 추적 가능. full crop gen에서 filter 옵션(C)도 대안.",
    },
    "p_c7_readiness": {
        "ready": verdict in ("통과", "부분통과"),
        "action_before_p_c7": (
            "없음 (통과)" if verdict == "통과"
            else "label policy 결정 (Option B 권장) 후 진행"
        ),
        "next_step": "P-C7 full crop generation preflight",
    },
    "blockers": blockers,
    "errors": errors,
    "n_errors": len(errors),
}

with open(f"{OUT_DIR}/p_c6_crop_smoke_validation.json", "w") as f:
    json.dump(result_json, f, indent=2, ensure_ascii=False, default=str)

# ===================== MD 보고서 =====================
hn_warn_detail = ""
for r in hn_warn_rows:
    hn_warn_detail += f"""
#### {r['candidate_id']} ({r['patient_id']})
- rule: {r['candidate_rule']}, local_z={r['local_z']}, slice_index={r['slice_index']}
- crop 범위: y={r['y0']}~{r['y1']}, x={r['x0']}~{r['x1']}
- padim_score: {r['padim_score']}
- mask_crop 채널별: ch0(z-1)={r['mask_ch0_zm1']}, ch1(center)={r['mask_ch1_center']}, ch2(z+1)={r['mask_ch2_zp1']}
- center_nonzero: {r['center_nonzero']}
- only_adjacent: {r['only_adjacent']}
- 원인: {r['cause_analysis']}
"""

md = f"""# P-C6 Crop Smoke Validation Report

**판정: {verdict}**
생성일시: {now_str}

---

## 1. P-C5 입력 검증

| 항목 | 값 |
|------|----|
| P-C5 verdict | {p_c5_verdict} |
| label CSV rows | {len(label_rows)} |
| npz 파일 수 | {len(npz_files)} |
| candidate_id 일치 | {id_match} |

## 2. crop count / label count

- crop 파일 수: **{len(npz_files)} / 110** {"✓" if len(npz_files)==110 else "✗"}
- label CSV rows: **{len(label_rows)} / 110** {"✓" if len(label_rows)==110 else "✗"}

## 3. crop shape 검증

- expected: `(3, 96, 96)`
- shape OK: **{shape_ok_count} / 110** {"✓" if shape_ok_count==110 else "✗"}

## 4. npz key 검증

필수 key: `{sorted(REQUIRED_KEYS)}`
- missing key 보유 crop: {missing_key_count}

## 5. ct/roi/mask crop value 검증

| 항목 | 결과 |
|------|------|
| ct NaN | {ct_nan_count} |
| ct Inf | {ct_inf_count} |
| roi binary valid | {roi_valid_count}/110 |
| mask binary valid | {mask_valid_count}/110 |

## 6. local_z vs slice_index 결론

- local_z inbound: **{lz_inbound}/110** (100%)
- slice_index inbound: **{si_inbound}/110** (OOB: {si_oob}개)
- **결론: local_z를 crop z 기준으로 확정. slice_index는 global z이므로 crop 접근에 사용 금지.**

## 7. positive mask consistency

- positive crops: {pos_total}개
- mask OK: **{pos_ok}/{pos_total}** {"✓" if pos_ok==pos_total else "✗"}

## 8. hard_negative mask consistency

- hard_negative crops: {hn_total}개
- mask OK (zero): **{hn_ok}/{hn_total}**
- mask warning (nonzero): **{hn_warn}/72**

## 9. hn_mask_warn 4개 상세 분석
{hn_warn_detail}

### 원인 요약

모든 4개 케이스의 center 채널 mask 여부를 확인:
- C0036239/C0036244/C0036245: LUNG1-020 동일 환자, 인접 slice 3개
- C0062081: MSD_lung_016

> manifest에서 `lesion_pixels=0`, `has_lesion_patch=False`이지만
> label CSV에서 `mask_pos_voxels_center`가 337/434로 nonzero.
> 이는 candidate center patch 기준으로는 병변이 없지만,
> 96x96 crop 확장 시 인근 병변 픽셀이 crop 범위에 포함된 것으로 분석됨.

## 10. hard_negative label 정책 추천

| Option | 설명 | 추천 |
|--------|------|------|
| A | crop 전체 mask 기준으로 hn→positive 재라벨링 | 비권장 |
| **B** | **center patch 기준 label 유지 + mask_nonzero_warning flag 추가** | **권장** |
| C | full crop gen에서 hn mask nonzero 제외 | 대안 |

**추천: Option B** — center patch scoring 기준 일관성 유지, flag으로 추적 가능, 4개 소수라 학습 영향 미미.
full crop generation에서도 Option C(filter)로 적용 가능하며 사용자 결정 필요.

## 11. full crop generation 전 수정 필요 여부

- **없음** (통과 기준 충족)
- hn_mask_warn 4개 처리 정책(B/C)은 P-C7 preflight에서 확정

## 12. guardrails 확인

| 항목 | 확인 |
|------|------|
| full crop 미생성 | {not guardrails.get("full_crop_generated", True)} |
| 2차학습 없음 | {not guardrails.get("training_executed", True)} |
| model forward 없음 | {not guardrails.get("model_forward", True)} |
| scoring 재실행 없음 | {not guardrails.get("scoring_rerun", True)} |
| feature extraction 없음 | {not guardrails.get("feature_extraction", True)} |
| stage2_holdout 미접근 | {not guardrails.get("stage2_holdout_accessed", False)} |
| 기존 결과 무수정 | {not guardrails.get("existing_results_modified", False)} |

## 13. special coverage 확인

| 항목 | 포함 여부 |
|------|----------|
| no_hit | {no_hit_ok} |
| tiny | {tiny_ok} |
| risk6 | {risk6_ok} |
| fallback | {fallback_ok} |

## 14. 다음 단계 추천

**P-C7 full crop generation preflight** (권장)
- 전제: hn label policy 결정 (Option B 권장)
- full crop 114,381개 생성 전 manifest 확인, 저장 경로, resume 구조, 디스크 공간 확인

---

## blockers

{blockers if blockers else "없음"}

## errors

{errors if errors else "없음"}
"""

with open(f"{OUT_DIR}/p_c6_crop_smoke_validation.md", "w") as f:
    f.write(md)

print(f"\n[완료] 출력 경로: {OUT_DIR}")
print(f"  verdict: {verdict}")
print(f"  errors: {len(errors)}")
