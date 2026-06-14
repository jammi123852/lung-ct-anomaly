"""
P-C7: EfficientNet-B0 v4_20 full crop generation preflight
Read-only preflight: full crop 생성 전 경로·용량·policy·resume 설계 확정
"""

import os, json, csv, datetime
import numpy as np

# ===================== 경로 =====================
BASE    = "experiments/efficientnet_b0_v4_20_second_stage_refiner_v1/outputs"
REPORTS = f"{BASE}/reports"

P_C3_MANIFEST   = f"{BASE}/candidates/p_c3_candidate_manifest/p_c3_candidate_manifest.csv"
P_C3_SUMMARY    = f"{BASE}/candidates/p_c3_candidate_manifest/p_c3_candidate_manifest_summary.json"
P_C4_JSON       = f"{REPORTS}/p_c4_crop_generation_preflight/p_c4_crop_generation_preflight.json"
P_C4_STORAGE    = f"{REPORTS}/p_c4_crop_generation_preflight/p_c4_storage_estimate.csv"
P_C5_JSON       = f"{REPORTS}/p_c5_crop_smoke/p_c5_crop_smoke_summary.json"
P_C6_JSON       = f"{REPORTS}/p_c6_crop_smoke_validation/p_c6_crop_smoke_validation.json"

# planned full crop output
FULL_CROP_DIR      = f"{BASE}/crops/p_c8_full_crops"
FULL_REPORT_DIR    = f"{REPORTS}/p_c8_full_crop_generation"
FULL_LABELS_CSV    = f"{FULL_REPORT_DIR}/p_c8_full_crop_labels.csv"
FULL_MANIFEST_CSV  = f"{FULL_REPORT_DIR}/p_c8_full_crop_manifest.csv"
DONE_MARKER        = f"{FULL_REPORT_DIR}/DONE.json"

OUT_DIR = f"{REPORTS}/p_c7_full_crop_generation_preflight"
os.makedirs(OUT_DIR, exist_ok=True)

errors = []

# ===================== 1. P-C6/P-C5/P-C4/P-C3 입력 검증 =====================
print("[1] 입력 검증...")

with open(P_C6_JSON) as f: p_c6 = json.load(f)
with open(P_C5_JSON) as f: p_c5 = json.load(f)
with open(P_C4_JSON) as f: p_c4 = json.load(f)
with open(P_C3_SUMMARY) as f: p_c3 = json.load(f)

p_c6_verdict = p_c6["verdict"]
p_c5_verdict = p_c5["verdict"]
p_c4_verdict = p_c4["verdict"]
p_c3_verdict = p_c3["verdict"]

for step, v in [("P-C6", p_c6_verdict), ("P-C5", p_c5_verdict), ("P-C3", p_c3_verdict)]:
    if v != "통과":
        errors.append(f"{step} verdict={v} (통과 아님)")
print(f"  P-C6={p_c6_verdict}, P-C5={p_c5_verdict}, P-C4={p_c4_verdict}, P-C3={p_c3_verdict}")

# ===================== 2. P-C3 manifest 통계 =====================
print("[2] P-C3 manifest 통계...")

n_total    = p_c3["candidate_counts"]["n_total"]
n_positive = p_c3["candidate_counts"]["n_positive"]
n_hn       = p_c3["candidate_counts"]["n_hard_negative"]
holdout_contamination = p_c3["input_validation"]["stage2_holdout_contamination"]

# manifest 행 수 직접 확인
with open(P_C3_MANIFEST) as f:
    manifest_rows = sum(1 for _ in f) - 1  # 헤더 제외

if manifest_rows != n_total:
    errors.append(f"manifest row mismatch: file={manifest_rows} vs summary={n_total}")
if holdout_contamination != 0:
    errors.append(f"stage2_holdout_contamination={holdout_contamination}")

print(f"  total={n_total}, pos={n_positive}, hn={n_hn}, holdout_contamination={holdout_contamination}")
print(f"  manifest file rows={manifest_rows}")

# ===================== 3. output path collision 확인 =====================
print("[3] output path collision 확인...")

full_crop_exists   = os.path.exists(FULL_CROP_DIR)
full_report_exists = os.path.exists(FULL_REPORT_DIR)
labels_exists      = os.path.exists(FULL_LABELS_CSV)
done_exists        = os.path.exists(DONE_MARKER)

if full_crop_exists:
    existing_npz = [f for f in os.listdir(FULL_CROP_DIR) if f.endswith(".npz")]
    existing_crop_count = len(existing_npz)
else:
    existing_crop_count = 0

collision = full_crop_exists and existing_crop_count > 0
print(f"  crop dir exists={full_crop_exists}, existing crops={existing_crop_count}")
print(f"  report dir exists={full_report_exists}, labels exists={labels_exists}, DONE={done_exists}")

# ===================== 4. disk free space =====================
print("[4] disk space 확인...")

import shutil
disk = shutil.disk_usage(BASE if os.path.exists(BASE) else ".")
disk_free_gb  = disk.free  / (1024**3)
disk_total_gb = disk.total / (1024**3)
disk_used_gb  = disk.used  / (1024**3)
print(f"  total={disk_total_gb:.1f}GB, used={disk_used_gb:.1f}GB, free={disk_free_gb:.1f}GB")

# ===================== 5. storage estimate =====================
print("[5] storage estimate 계산...")

CROP_SIZE  = 96
N_CHANNELS = 3
N_CROPS    = n_total  # 114381

# raw (npz 압축 전)
bytes_per_ct_int16   = CROP_SIZE * CROP_SIZE * N_CHANNELS * 2   # int16
bytes_per_roi_uint8  = CROP_SIZE * CROP_SIZE * N_CHANNELS * 1   # uint8
bytes_per_mask_uint8 = CROP_SIZE * CROP_SIZE * N_CHANNELS * 1   # uint8
bytes_per_crop_raw   = bytes_per_ct_int16 + bytes_per_roi_uint8 + bytes_per_mask_uint8

raw_total_gb     = N_CROPS * bytes_per_crop_raw / (1024**3)
raw_ct_only_gb   = N_CROPS * bytes_per_ct_int16 / (1024**3)

# P-C5 실제 압축 비율 추정
p_c5_size_mb    = p_c5.get("output_size_mb", 4.41)
p_c5_n_crops    = p_c5.get("crop_generation", {}).get("generated_crops", 110)
per_crop_mb     = p_c5_size_mb / p_c5_n_crops  # 압축 후 per crop MB
compressed_total_gb = N_CROPS * per_crop_mb / 1024

print(f"  raw (int16 CT + uint8 ROI/mask): {raw_total_gb:.2f} GB")
print(f"  estimated compressed (based on P-C5): {compressed_total_gb:.2f} GB")
print(f"  disk free: {disk_free_gb:.1f} GB → {'충분' if disk_free_gb > compressed_total_gb * 2 else '부족 가능'}")

disk_ok = disk_free_gb > compressed_total_gb * 1.5

# ===================== 6. crop format 확정 =====================
crop_format = {
    "crop_size":    96,
    "n_channels":   3,
    "mode":         "2.5D z-1/z/z+1",
    "z_center":     "local_z",
    "y_center":     "(y0+y1)//2",
    "x_center":     "(x0+x1)//2",
    "padding":      "reflect (edge fallback for size-1 axis)",
    "ct_dtype_save":   "int16",
    "roi_dtype_save":  "uint8",
    "mask_dtype_save": "uint8",
    "save_format":  "npz (numpy compressed)",
    "train_cast":   "float32 at dataloader",
    "float32_full_save": "비권장 (11.78GB 초과)",
}

# ===================== 7. label policy 확정 =====================
label_policy = {
    "policy":   "Option B",
    "description": "center patch 기준 candidate_label 유지 + mask_nonzero_warning flag 추가",
    "flags_added": [
        "mask_any_nonzero: bool — crop 3채널 mask.any()",
        "center_mask_nonzero: bool — mask[1].any() (center slice)",
        "adjacent_mask_nonzero: bool — mask[0].any() or mask[2].any() (z±1 slice)",
    ],
    "filter_excluded": False,
    "option_c_available": True,
    "option_c_note": "full crop 생성 후 mask_any_nonzero=True인 hard_negative를 학습에서 제외 가능. smoke 비율 5/72=6.9%이므로 full에서 비슷하면 약 5,500개 제외 예상.",
    "option_a_rejected": True,
    "option_a_reason": "crop 전체 mask 기준 재라벨링은 scoring 기준(center patch)과 불일치",
    "smoke_warn_rate": round(5/72, 4),
    "smoke_warn_note": "P-C6: 5/72=6.9% (center_nonzero 4, adjacent_only 1)",
    "full_expected_warn_count": round(n_hn * 5/72),
    "full_expected_warn_pct":   round(5/72 * 100, 1),
}

# ===================== 8. resume 구조 설계 =====================
resume_plan = {
    "strategy":   "npz 파일 존재 여부 기반 skip",
    "on_resume":  "FULL_CROP_DIR 내 기존 .npz 파일 목록을 candidate_id set으로 로드 → 이미 존재하면 skip",
    "labels_csv": "append 모드: 재개 시 기존 CSV 행 candidate_id set 확인 후 중복 방지",
    "incomplete":  "labels.csv 행 수 vs crop 파일 수 불일치 시 경고 출력, mismatched crop 재생성",
    "done_marker": "전체 완료 후 DONE.json 생성 (중간에 생성 금지)",
    "done_condition": "generated_crops == n_total AND labels_csv_rows == n_total",
    "batch_size":  "권장 500~1000 rows per batch (메모리 절약)",
    "error_csv":   f"{FULL_REPORT_DIR}/p_c8_errors.csv (실패 candidate 기록)",
    "runtime_summary": f"{FULL_REPORT_DIR}/p_c8_runtime_summary.json",
}

# ===================== 9. 예상 실행 시간 =====================
p_c5_elapsed = p_c5.get("elapsed_seconds", 3.4)
per_crop_sec = p_c5_elapsed / p_c5_n_crops
estimated_sec_1x = N_CROPS * per_crop_sec
estimated_sec_2x = estimated_sec_1x * 2
estimated_sec_3x = estimated_sec_1x * 3

runtime_estimate = {
    "p_c5_basis":          f"{p_c5_n_crops}개 / {p_c5_elapsed:.1f}초",
    "per_crop_sec":        round(per_crop_sec, 4),
    "full_n_crops":        N_CROPS,
    "estimated_min_1x":    round(estimated_sec_1x / 60, 1),
    "estimated_min_2x":    round(estimated_sec_2x / 60, 1),
    "estimated_min_3x":    round(estimated_sec_3x / 60, 1),
    "recommended_estimate": f"보수적 2x~3x: {round(estimated_sec_2x/60,0):.0f}~{round(estimated_sec_3x/60,0):.0f}분",
    "note": "CT/ROI/mask npy 반복 로드가 병목. 환자 단위 캐싱 구조로 개선 가능 (같은 환자 candidates를 묶어 처리).",
}

# ===================== 10. blocker 확인 =====================
blockers = []
if p_c6_verdict != "통과":
    blockers.append(f"P-C6 verdict={p_c6_verdict}")
if p_c5_verdict != "통과":
    blockers.append(f"P-C5 verdict={p_c5_verdict}")
if p_c3_verdict != "통과":
    blockers.append(f"P-C3 verdict={p_c3_verdict}")
if manifest_rows != n_total:
    blockers.append(f"manifest row mismatch: {manifest_rows} vs {n_total}")
if holdout_contamination != 0:
    blockers.append(f"holdout contamination={holdout_contamination}")
if collision:
    blockers.append(f"output collision: {existing_crop_count}개 기존 crop 존재")
if not disk_ok:
    blockers.append(f"disk space 부족 가능: free={disk_free_gb:.1f}GB, need≈{compressed_total_gb*1.5:.1f}GB")

# ===================== 판정 =====================
if len(blockers) == 0:
    verdict = "통과"
elif any("contamination" in b or "P-C6 verdict" in b or "P-C5 verdict" in b for b in blockers):
    verdict = "실패"
else:
    verdict = "부분통과"

print(f"\n[판정] {verdict}")
if blockers:
    print(f"  blockers: {blockers}")

# ===================== CSV 저장 =====================
now_str = datetime.datetime.now().isoformat(timespec="seconds")

def save_csv(path, rows, fieldnames=None):
    if not rows:
        with open(path, "w", newline="") as f: f.write("")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

# output path plan
save_csv(f"{OUT_DIR}/p_c7_output_path_plan.csv", [
    {"item": "full_crop_dir",     "path": FULL_CROP_DIR,     "exists": full_crop_exists,   "collision": collision},
    {"item": "full_report_dir",   "path": FULL_REPORT_DIR,   "exists": full_report_exists, "collision": False},
    {"item": "full_labels_csv",   "path": FULL_LABELS_CSV,   "exists": labels_exists,      "collision": False},
    {"item": "full_manifest_csv", "path": FULL_MANIFEST_CSV, "exists": False,              "collision": False},
    {"item": "done_marker",       "path": DONE_MARKER,       "exists": done_exists,        "collision": done_exists},
])

# storage estimate
save_csv(f"{OUT_DIR}/p_c7_storage_estimate.csv", [
    {"scenario": "raw_int16_CT+uint8_ROI+mask", "n_crops": N_CROPS, "crop_size": CROP_SIZE, "channels": N_CHANNELS, "dtype": "int16/uint8", "estimated_gb": round(raw_total_gb,2), "note": "압축 전 raw"},
    {"scenario": "estimated_npz_compressed",    "n_crops": N_CROPS, "crop_size": CROP_SIZE, "channels": N_CHANNELS, "dtype": "int16/uint8", "estimated_gb": round(compressed_total_gb,2), "note": f"P-C5 {per_crop_mb:.4f}MB/crop 기준"},
    {"scenario": "disk_free",                   "n_crops": "-",     "crop_size": "-",        "channels": "-",        "dtype": "-",           "estimated_gb": round(disk_free_gb,1),  "note": "현재 여유 공간"},
    {"scenario": "margin_needed_1.5x",          "n_crops": "-",     "crop_size": "-",        "channels": "-",        "dtype": "-",           "estimated_gb": round(compressed_total_gb*1.5,2), "note": "권장 최소 여유"},
    {"scenario": "disk_ok",                     "n_crops": "-",     "crop_size": "-",        "channels": "-",        "dtype": "-",           "estimated_gb": "-", "note": str(disk_ok)},
])

# label policy plan
save_csv(f"{OUT_DIR}/p_c7_label_policy_plan.csv", [
    {"option": "B (권장)", "action": "center patch 기준 label 유지", "flags": "mask_any_nonzero / center_mask_nonzero / adjacent_mask_nonzero", "filter": "없음", "note": "P-C6 권장"},
    {"option": "C (대안)", "action": "hn mask_any_nonzero=True 제외",  "flags": "mask_any_nonzero 기록 후 학습 제외", "filter": "있음", "note": f"예상 제외 {label_policy['full_expected_warn_count']}개 ({label_policy['full_expected_warn_pct']}%)"},
    {"option": "A (비권장)", "action": "crop mask 기준 hn→positive 재라벨링", "flags": "-", "filter": "없음", "note": "scoring 기준 불일치로 비권장"},
])

# resume plan
save_csv(f"{OUT_DIR}/p_c7_resume_plan.csv", [
    {"item": k, "value": str(v)} for k, v in resume_plan.items()
])

# disk space check
save_csv(f"{OUT_DIR}/p_c7_disk_space_check.csv", [
    {"item": "disk_total_gb",        "value": round(disk_total_gb,1)},
    {"item": "disk_used_gb",         "value": round(disk_used_gb,1)},
    {"item": "disk_free_gb",         "value": round(disk_free_gb,1)},
    {"item": "raw_estimate_gb",      "value": round(raw_total_gb,2)},
    {"item": "compressed_estimate_gb","value": round(compressed_total_gb,2)},
    {"item": "margin_needed_gb",     "value": round(compressed_total_gb*1.5,2)},
    {"item": "disk_ok",              "value": disk_ok},
])

# errors
save_csv(f"{OUT_DIR}/p_c7_errors.csv",
         [{"error": e} for e in errors] if errors else [{"error": "none"}])

# ===================== JSON =====================
result_json = {
    "step": "P-C7",
    "verdict": verdict,
    "created": now_str,
    "input_validation": {
        "p_c6_verdict": p_c6_verdict,
        "p_c5_verdict": p_c5_verdict,
        "p_c4_verdict": p_c4_verdict,
        "p_c3_verdict": p_c3_verdict,
    },
    "manifest": {
        "total_candidates": n_total,
        "manifest_file_rows": manifest_rows,
        "n_positive": n_positive,
        "n_hard_negative": n_hn,
        "pos_hn_ratio": round(n_hn/n_positive, 3),
        "stage2_holdout_contamination": holdout_contamination,
        "row_count_match": manifest_rows == n_total,
    },
    "output_path_plan": {
        "full_crop_dir":    FULL_CROP_DIR,
        "full_report_dir":  FULL_REPORT_DIR,
        "full_labels_csv":  FULL_LABELS_CSV,
        "full_manifest_csv":FULL_MANIFEST_CSV,
        "done_marker":      DONE_MARKER,
        "collision_detected": collision,
        "existing_crops":   existing_crop_count,
    },
    "crop_format": crop_format,
    "label_policy": label_policy,
    "storage_estimate": {
        "raw_int16_gb":        round(raw_total_gb,2),
        "compressed_est_gb":   round(compressed_total_gb,2),
        "disk_free_gb":        round(disk_free_gb,1),
        "margin_needed_gb":    round(compressed_total_gb*1.5,2),
        "disk_ok":             disk_ok,
    },
    "runtime_estimate": runtime_estimate,
    "resume_plan": resume_plan,
    "z_axis_policy": {
        "crop_z_basis": "local_z",
        "slice_index_usage": "crop 접근 금지 (global z 기준이므로 CT volume index로 사용 불가)",
        "local_z_definition": "lung-region-cropped volume 기준 z index",
        "confirmed": True,
    },
    "guardrails": {
        "full_crop_generated":       False,
        "training_executed":         False,
        "model_forward":             False,
        "scoring_rerun":             False,
        "feature_extraction":        False,
        "threshold_recalculated":    False,
        "metrics_recalculated":      False,
        "stage2_holdout_accessed":   False,
        "existing_results_modified": False,
    },
    "p_c8_readiness": {
        "ready": verdict in ("통과",),
        "next_step": "P-C8 full crop generation script 작성 + dry-check",
        "prerequisites": [
            "label policy Option B/C 사용자 확정",
            "P-C8 script dry-run (ALLOW_REAL=False)",
            "P-C8 실행 사용자 승인 (GPU/장시간 실행)",
        ],
    },
    "blockers": blockers,
    "errors": errors,
    "n_errors": len(errors),
}

with open(f"{OUT_DIR}/p_c7_full_crop_generation_preflight.json", "w") as f:
    json.dump(result_json, f, indent=2, ensure_ascii=False, default=str)

# ===================== MD =====================
hn_warn_expected = label_policy["full_expected_warn_count"]
hn_warn_pct      = label_policy["full_expected_warn_pct"]

md = f"""# P-C7 Full Crop Generation Preflight Report

**판정: {verdict}**
생성일시: {now_str}

---

## 1. 입력 검증

| 단계 | 판정 |
|------|------|
| P-C6 (crop smoke validation) | {p_c6_verdict} |
| P-C5 (crop smoke generation) | {p_c5_verdict} |
| P-C4 (crop generation preflight) | {p_c4_verdict} |
| P-C3 (candidate manifest) | {p_c3_verdict} |

## 2. full manifest 통계

| 항목 | 값 |
|------|----|
| total candidates | **{n_total:,}** |
| manifest file rows | {manifest_rows:,} |
| row count match | {manifest_rows == n_total} |
| positive | {n_positive:,} |
| hard_negative | {n_hn:,} |
| pos:hn ratio | 1:{round(n_hn/n_positive,2)} |
| stage2_holdout contamination | **{holdout_contamination}** |

## 3. crop format 확정

| 항목 | 값 |
|------|----|
| crop_size | 96px |
| channels | 3 (z-1 / z / z+1) |
| z_center | **local_z** (확정) |
| y_center | (y0+y1)//2 |
| x_center | (x0+x1)//2 |
| padding | reflect (edge fallback) |
| CT dtype 저장 | **int16** |
| ROI/mask dtype 저장 | **uint8** |
| 학습 시 변환 | float32 (DataLoader) |
| float32 full 저장 | 비권장 (11.78GB 초과) |
| 저장 형식 | npz (numpy compressed) |

## 4. z축 기준 확정

- **crop z 기준: `local_z`** (lung-region-cropped volume 기준 z index)
- **`slice_index` 사용 금지**: global z 기준이므로 CT volume 직접 접근 불가 (OOB 16/110 확인)
- local_z는 P-C5/P-C6에서 110/110 in-bound 확인 완료

## 5. label policy 확정

**채택: Option B** — center patch 기준 label 유지 + `mask_nonzero_warning` flag 추가

full crop 생성 시 npz에 다음 flag 기록:
- `mask_any_nonzero`: crop 3채널 전체 mask nonzero 여부
- `center_mask_nonzero`: ch1 (center slice) mask nonzero 여부
- `adjacent_mask_nonzero`: ch0 또는 ch2 (z±1) mask nonzero 여부

| Option | 설명 | 채택 |
|--------|------|------|
| **B** | center patch 기준 label 유지 + flag | **채택** |
| C | hn mask nonzero → 학습 제외 (filter) | 대안 (P-C8 실행 후 비율 보고 후 결정) |
| A | crop mask 기준 재라벨링 | 비권장 |

> **smoke 기준 경고율**: P-C6에서 5/72 = 6.9%
> full 예상 경고 건수: hard_negative {n_hn:,}개 × 6.9% ≈ **{hn_warn_expected:,}개**
> Option C 적용 시 이 건수가 학습에서 제외됨.

## 6. output path plan

| 항목 | 경로 | 존재 | 충돌 |
|------|------|------|------|
| full_crop_dir | `{FULL_CROP_DIR}` | {full_crop_exists} | {collision} |
| full_report_dir | `{FULL_REPORT_DIR}` | {full_report_exists} | False |
| full_labels_csv | `{FULL_LABELS_CSV}` | {labels_exists} | False |
| done_marker | `{DONE_MARKER}` | {done_exists} | {done_exists} |

## 7. disk space check

| 항목 | 값 |
|------|----|
| disk total | {disk_total_gb:.1f} GB |
| disk used | {disk_used_gb:.1f} GB |
| disk free | **{disk_free_gb:.1f} GB** |
| raw estimate (int16 CT + uint8) | {raw_total_gb:.2f} GB |
| compressed estimate (P-C5 기준) | **{compressed_total_gb:.2f} GB** |
| 권장 최소 여유 (1.5x) | {compressed_total_gb*1.5:.2f} GB |
| **disk OK** | **{disk_ok}** |

## 8. storage estimate

- npz raw (int16 CT + uint8 ROI + uint8 mask): **{raw_total_gb:.2f} GB**
- npz 압축 후 추정 (P-C5 {per_crop_mb:.4f} MB/crop 기준): **{compressed_total_gb:.2f} GB**
- 디스크 여유 {disk_free_gb:.1f} GB → {"충분" if disk_ok else "부족 가능성"}

## 9. 예상 실행 시간

- P-C5 기준: {p_c5_n_crops}개 / {p_c5_elapsed:.1f}초 = {per_crop_sec:.4f}초/crop
- full {N_CROPS:,}개 × {per_crop_sec:.4f}초 = {estimated_sec_1x/60:.1f}분 (1x)
- 보수 2x: **{estimated_sec_2x/60:.0f}분** / 보수 3x: **{estimated_sec_3x/60:.0f}분**
- 최적화: 환자 단위로 묶어 CT/ROI/mask npy 반복 로드 최소화 가능

## 10. resume 구조 설계

| 항목 | 내용 |
|------|------|
| skip 기준 | crop dir에 npz 파일 존재 시 skip |
| labels CSV | append 모드, 재개 시 기존 candidate_id set 확인 |
| incomplete 감지 | labels.csv 행 수 ≠ crop 파일 수 → 불일치 candidate 재생성 |
| DONE marker | 전체 완료 후에만 생성 (`generated == n_total AND labels_rows == n_total`) |
| error CSV | 실패 candidate 기록 → 완료 후 재시도 |
| batch size | 권장 500~1000 rows/batch |

## 11. guardrails 확인

| 항목 | 확인 |
|------|------|
| full crop 미생성 | True |
| 2차학습 없음 | True |
| model forward 없음 | True |
| scoring 재실행 없음 | True |
| stage2_holdout 미접근 | True |
| 기존 결과 무수정 | True |

## 12. blockers

{blockers if blockers else "없음 — P-C8 진행 가능"}

## 13. errors

{errors if errors else "없음"}

## 14. 다음 단계 추천

**P-C8 full crop generation script 작성 + dry-check** (권장)

사전 확정 필요:
1. **label policy: Option B 또는 C 사용자 확정**
2. P-C8 script 작성 (ALLOW_REAL=False dry-check 포함)
3. dry-run PASS 확인 후 → 실행 승인
"""

with open(f"{OUT_DIR}/p_c7_full_crop_generation_preflight.md", "w") as f:
    f.write(md)

print(f"\n[완료] 출력 경로: {OUT_DIR}")
print(f"  verdict: {verdict}")
print(f"  blockers: {len(blockers)}")
print(f"  errors: {len(errors)}")
