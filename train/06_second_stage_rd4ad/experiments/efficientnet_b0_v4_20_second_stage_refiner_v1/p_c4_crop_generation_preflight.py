"""
P-C4 Crop Generation Preflight
EfficientNet-B0 v4_20 ROI branch

금지: crop 생성 / 2차학습 / model forward / scoring / stage2_holdout 접근
허용: manifest read-only / 파일 존재 확인 / lightweight shape check / 용량 추정 / 계획 작성
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
MANIFEST_CSV = WORKSPACE / "outputs/candidates/p_c3_candidate_manifest/p_c3_candidate_manifest.csv"
MANIFEST_JSON = WORKSPACE / "outputs/candidates/p_c3_candidate_manifest/p_c3_candidate_manifest_summary.json"
SPLIT_CSV = BASE / "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1_balanced.csv"

CT_ROOT = Path("/mnt/c/Users/jinhy/Desktop/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1/volumes_npy")
ROI_ROOT = BASE / "outputs/mip-postprocess-research-v1/masks/refined_roi_v4_20_modeB_all_v1/lesion"

OUTPUT_DIR = WORKSPACE / "outputs/reports/p_c4_crop_generation_preflight"
if OUTPUT_DIR.exists():
    print(f"[ERROR] 출력 경로가 이미 존재합니다. 덮어쓰기 방지로 중단합니다: {OUTPUT_DIR}")
    sys.exit(1)
OUTPUT_DIR.mkdir(parents=True)

STAGE2_HOLDOUT_PATIENTS = set()  # split 로드 후 채움
CROP_SIZES = [64, 96, 128]
HALF_SIZES = {s: s // 2 for s in CROP_SIZES}

errors = []
t_start = datetime.datetime.now()

# ── 1. P-C3 summary 확인 ────────────────────────────────────────────────────
print("[1] P-C3 summary 확인")
with open(MANIFEST_JSON) as f:
    c3_summary = json.load(f)

c3_verdict = c3_summary["verdict"]
c3_n_total = c3_summary["candidate_counts"]["n_total"]
c3_n_pos = c3_summary["candidate_counts"]["n_positive"]
c3_n_hn = c3_summary["candidate_counts"]["n_hard_negative"]
c3_ratio = c3_summary["candidate_counts"]["pos_hn_ratio"]
c3_contamination = c3_summary["input_validation"]["stage2_holdout_contamination"]

print(f"  verdict={c3_verdict}  total={c3_n_total}  pos={c3_n_pos}  hn={c3_n_hn}  ratio={c3_ratio}  contamination={c3_contamination}")

if c3_verdict != "통과":
    print(f"[ERROR] P-C3 verdict={c3_verdict} ≠ 통과 → P-C4 진행 불가")
    sys.exit(1)
if c3_contamination != 0:
    print(f"[ERROR] P-C3 stage2_holdout_contamination={c3_contamination} ≠ 0")
    sys.exit(1)

# ── 2. split CSV 로드 ────────────────────────────────────────────────────────
print("[2] split CSV 로드")
split_df = pd.read_csv(SPLIT_CSV)
stage1_dev_set = set(split_df[split_df["stage_split"] == "stage1_dev"]["patient_id"])
STAGE2_HOLDOUT_PATIENTS = set(split_df[split_df["stage_split"] == "stage2_holdout"]["patient_id"])
safe_id_map = split_df.set_index("patient_id")["safe_id"].to_dict()
print(f"  stage1_dev={len(stage1_dev_set)}  stage2_holdout={len(STAGE2_HOLDOUT_PATIENTS)}")

# ── 3. manifest 로드 및 기본 검증 ────────────────────────────────────────────
print("[3] manifest 로드")
manifest = pd.read_csv(MANIFEST_CSV)

# row count 확인
assert len(manifest) == c3_n_total, f"row count mismatch: {len(manifest)} vs {c3_n_total}"
print(f"  row count={len(manifest):,}  OK")

# stage2_holdout contamination 재검사
cont = manifest["patient_id"].isin(STAGE2_HOLDOUT_PATIENTS).sum()
assert cont == 0, f"stage2_holdout contamination: {cont}"
print(f"  stage2_holdout contamination={cont}  OK")

# ── 4. 필수 crop 컬럼 검증 ──────────────────────────────────────────────────
print("[4] 필수 crop 컬럼 검증")
REQUIRED_CROP_COLS = [
    "patient_id", "safe_id", "slice_index", "local_z",
    "y0", "x0", "y1", "x1",
    "position_bin", "z_level",
    "candidate_label", "candidate_rule",
]
missing = [c for c in REQUIRED_CROP_COLS if c not in manifest.columns]
if missing:
    print(f"[ERROR] 필수 컬럼 누락: {missing}")
    sys.exit(1)
print(f"  필수 컬럼 {len(REQUIRED_CROP_COLS)}개 전부 존재  OK")

# 좌표 null count
COORD_COLS = ["slice_index", "local_z", "y0", "x0", "y1", "x1"]
coord_nulls = {c: int(manifest[c].isnull().sum()) for c in COORD_COLS}
total_null = sum(coord_nulls.values())
print(f"  좌표 null count={total_null}  {coord_nulls}")
assert total_null == 0, f"좌표 컬럼 null 존재: {coord_nulls}"

# schema summary
schema_rows = []
for col in REQUIRED_CROP_COLS + ["padim_score", "fallback_positive_below_p95", "no_hit_patient", "tiny_lesion_flag", "p_b3_risk6_flag"]:
    in_manifest = col in manifest.columns
    null_count = int(manifest[col].isnull().sum()) if in_manifest else None
    schema_rows.append({"column": col, "present": in_manifest, "null_count": null_count, "dtype": str(manifest[col].dtype) if in_manifest else None})
pd.DataFrame(schema_rows).to_csv(OUTPUT_DIR / "p_c4_manifest_schema_validation.csv", index=False)

# ── 5. CT/ROI/mask 파일 존재 확인 ────────────────────────────────────────────
print("[5] CT/ROI/mask 파일 존재 확인")

# stage1_dev 환자 unique safe_id 목록
patients_in_manifest = manifest[["patient_id", "safe_id"]].drop_duplicates()
print(f"  manifest 내 환자 수={len(patients_in_manifest)}")

file_avail_rows = []
n_ct_ok = n_ct_miss = 0
n_roi_ok = n_roi_miss = 0
n_mask_ok = n_mask_miss = 0
n_meta_ok = n_meta_miss = 0
shape_info = {}  # safe_id → shape_zyx

for _, row in patients_in_manifest.iterrows():
    pid = row["patient_id"]
    sid = row["safe_id"]

    # stage2_holdout 차단
    if pid in STAGE2_HOLDOUT_PATIENTS:
        errors.append({"patient_id": pid, "stage": "file_check", "error": "BLOCKED: stage2_holdout"})
        continue

    ct_path = CT_ROOT / sid / "ct_hu.npy"
    mask_path = CT_ROOT / sid / "lesion_mask_roi_0_0.npy"
    meta_path = CT_ROOT / sid / "meta.json"
    roi_path = ROI_ROOT / sid / "refined_roi.npy"

    ct_ok = ct_path.is_file()
    mask_ok = mask_path.is_file()
    meta_ok = meta_path.is_file()
    roi_ok = roi_path.is_file()

    if ct_ok: n_ct_ok += 1
    else: n_ct_miss += 1
    if mask_ok: n_mask_ok += 1
    else: n_mask_miss += 1
    if meta_ok: n_meta_ok += 1
    else: n_meta_miss += 1
    if roi_ok: n_roi_ok += 1
    else: n_roi_miss += 1

    # meta.json에서 shape 읽기 (CT 로드 없이)
    shape_zyx = None
    if meta_ok:
        try:
            with open(meta_path) as mf:
                meta = json.load(mf)
            shape_zyx = meta.get("shape_zyx")
            if shape_zyx:
                shape_info[sid] = shape_zyx
        except Exception as e:
            errors.append({"patient_id": pid, "stage": "meta_read", "error": str(e)})

    file_avail_rows.append({
        "patient_id": pid,
        "safe_id": sid,
        "ct_ok": ct_ok,
        "mask_ok": mask_ok,
        "meta_ok": meta_ok,
        "roi_ok": roi_ok,
        "shape_zyx": str(shape_zyx) if shape_zyx else None,
        "any_missing": not (ct_ok and mask_ok and roi_ok),
    })
    if not (ct_ok and mask_ok and roi_ok):
        errors.append({"patient_id": pid, "stage": "file_check",
                        "error": f"missing: ct={not ct_ok} mask={not mask_ok} roi={not roi_ok}"})

file_avail_df = pd.DataFrame(file_avail_rows)
file_avail_df.to_csv(OUTPUT_DIR / "p_c4_input_file_availability.csv", index=False)
print(f"  CT: ok={n_ct_ok}  miss={n_ct_miss}")
print(f"  mask: ok={n_mask_ok}  miss={n_mask_miss}")
print(f"  ROI: ok={n_roi_ok}  miss={n_roi_miss}")
print(f"  meta: ok={n_meta_ok}  miss={n_meta_miss}  shape 수집={len(shape_info)}")

# ── 6. 좌표 범위 검증 (meta.json shape 기준) ─────────────────────────────────
print("[6] 좌표 범위 검증")

coord_rows = []
total_oob_count = 0
total_checked = 0

for _, row in patients_in_manifest.iterrows():
    pid = row["patient_id"]
    sid = row["safe_id"]
    if pid in STAGE2_HOLDOUT_PATIENTS:
        continue
    if sid not in shape_info:
        continue

    shape_z, shape_y, shape_x = shape_info[sid]
    pat_rows = manifest[manifest["safe_id"] == sid]

    oob_z = int(((pat_rows["slice_index"] < 0) | (pat_rows["slice_index"] >= shape_z)).sum())
    oob_y0 = int((pat_rows["y0"] < 0).sum())
    oob_y1 = int((pat_rows["y1"] > shape_y).sum())
    oob_x0 = int((pat_rows["x0"] < 0).sum())
    oob_x1 = int((pat_rows["x1"] > shape_x).sum())
    oob_total = oob_z + oob_y0 + oob_y1 + oob_x0 + oob_x1

    total_oob_count += oob_total
    total_checked += len(pat_rows)

    coord_rows.append({
        "patient_id": pid,
        "safe_id": sid,
        "shape_z": shape_z, "shape_y": shape_y, "shape_x": shape_x,
        "n_rows": len(pat_rows),
        "oob_z": oob_z,
        "oob_y0": oob_y0, "oob_y1": oob_y1,
        "oob_x0": oob_x0, "oob_x1": oob_x1,
        "oob_total": oob_total,
    })
    if oob_total > 0:
        errors.append({"patient_id": pid, "stage": "coord_check",
                        "error": f"oob_total={oob_total} z={oob_z} y0={oob_y0} y1={oob_y1} x0={oob_x0} x1={oob_x1}"})

pd.DataFrame(coord_rows).to_csv(OUTPUT_DIR / "p_c4_crop_coordinate_validation.csv", index=False)
print(f"  검증 rows={total_checked:,}  oob_total={total_oob_count}")

# ── 7. crop 크기 후보별 boundary issue 계산 ─────────────────────────────────
print("[7] crop 크기 후보별 boundary issue 계산")

# patch 좌표는 (y0, x0, y1, x1) = 32px stride 그리드
# crop은 patch 중심에서 crop_size/2만큼 확장
manifest["cy"] = (manifest["y0"] + manifest["y1"]) // 2
manifest["cx"] = (manifest["x0"] + manifest["x1"]) // 2

crop_format_rows = []
for crop_sz in CROP_SIZES:
    half = crop_sz // 2
    # 환자별로 shape를 알아야 하므로 manifest에 shape 매핑
    manifest["_shape_y"] = manifest["safe_id"].map(
        {sid: v[1] for sid, v in shape_info.items()}
    )
    manifest["_shape_x"] = manifest["safe_id"].map(
        {sid: v[2] for sid, v in shape_info.items()}
    )
    manifest["_shape_z"] = manifest["safe_id"].map(
        {sid: v[0] for sid, v in shape_info.items()}
    )

    # 2D crop boundary issue
    oob_top = int((manifest["cy"] - half < 0).sum())
    oob_bot = int((manifest["cy"] + half > manifest["_shape_y"]).sum())
    oob_left = int((manifest["cx"] - half < 0).sum())
    oob_right = int((manifest["cx"] + half > manifest["_shape_x"]).sum())
    oob_2d = oob_top + oob_bot + oob_left + oob_right

    # 2.5D: z-1/z/z+1 boundary issue
    oob_z_prev = int((manifest["slice_index"] - 1 < 0).sum())
    oob_z_next = int((manifest["slice_index"] + 1 >= manifest["_shape_z"]).sum())
    oob_25d = oob_z_prev + oob_z_next

    crop_format_rows.append({
        "crop_size": crop_sz,
        "half_size": half,
        "n_total": len(manifest),
        "oob_2d_count": oob_2d,
        "oob_2d_pct": round(oob_2d / len(manifest) * 100, 2),
        "oob_25d_z_count": oob_25d,
        "oob_25d_z_pct": round(oob_25d / len(manifest) * 100, 2),
        "feasible_2d": oob_2d == 0,
        "feasible_25d": oob_25d == 0,
        "note": "boundary padding 적용 시 해결 가능" if oob_2d > 0 else "padding 불필요",
    })
    print(f"  crop_sz={crop_sz}: oob_2d={oob_2d}({oob_2d/len(manifest)*100:.1f}%)  oob_25d={oob_25d}({oob_25d/len(manifest)*100:.1f}%)")

pd.DataFrame(crop_format_rows).to_csv(OUTPUT_DIR / "p_c4_crop_format_plan.csv", index=False)

# ── 8. 저장 용량 추정 ────────────────────────────────────────────────────────
print("[8] 저장 용량 추정")
n_total = len(manifest)
DTYPE_BYTES = {"float32": 4, "int16": 2, "uint8": 1}

storage_rows = []
for crop_sz in CROP_SIZES:
    for n_ch in [1, 3]:  # 2D 1ch, 2.5D 3ch
        for dtype_name, dtype_bytes in [("float32", 4), ("int16", 2)]:
            bytes_per_crop = crop_sz * crop_sz * n_ch * dtype_bytes
            # full manifest: 114,381개
            total_bytes = n_total * bytes_per_crop
            total_mb = total_bytes / 1024 / 1024
            total_gb = total_mb / 1024
            # smoke 60개 기준
            smoke_bytes = 60 * bytes_per_crop
            smoke_mb = smoke_bytes / 1024 / 1024
            storage_rows.append({
                "crop_size": crop_sz,
                "n_channels": n_ch,
                "dtype": dtype_name,
                "bytes_per_crop": bytes_per_crop,
                "full_total_crops": n_total,
                "full_total_mb": round(total_mb, 1),
                "full_total_gb": round(total_gb, 2),
                "smoke_60_crops_mb": round(smoke_mb, 4),
            })
            if n_ch == 3:  # 2.5D만 출력
                print(f"  crop_sz={crop_sz} 3ch {dtype_name}: full={total_gb:.2f}GB  smoke60={smoke_mb:.2f}MB")

pd.DataFrame(storage_rows).to_csv(OUTPUT_DIR / "p_c4_storage_estimate.csv", index=False)

# 권장 용량 (2.5D 3ch float32, crop_size=96)
rec_gb = next(r["full_total_gb"] for r in storage_rows if r["crop_size"]==96 and r["n_channels"]==3 and r["dtype"]=="float32")
print(f"  권장 2.5D 3ch float32 96px 전체 용량: {rec_gb}GB")

# ── 9. smoke scope 계획 ──────────────────────────────────────────────────────
print("[9] smoke scope 계획")
NO_HIT = {"LUNG1-086", "LUNG1-386", "MSD_lung_096"}
TINY = {"LUNG1-156", "LUNG1-192", "LUNG1-311", "LUNG1-386"}
RISK6 = {"LUNG1-386", "LUNG1-156", "LUNG1-028", "LUNG1-306", "LUNG1-421", "LUNG1-295"}
SPECIAL_PIDS = NO_HIT | TINY | RISK6  # 필수 포함

pos_df = manifest[manifest["candidate_label"] == "positive"].copy()
hn_df = manifest[manifest["candidate_label"] == "hard_negative"].copy()

# smoke positive: 특수 환자 필수 + 일반 보완
smoke_pos_special = pos_df[pos_df["patient_id"].isin(SPECIAL_PIDS)].groupby("patient_id").head(2)
# 일반 환자에서 보완 (position_bin 다양성)
smoke_pos_general = (
    pos_df[~pos_df["patient_id"].isin(SPECIAL_PIDS)]
    .groupby("position_bin", group_keys=False)
    .apply(lambda g: g.nlargest(3, "padim_score"))
    .reset_index(drop=True)
)
smoke_pos = pd.concat([smoke_pos_special, smoke_pos_general]).drop_duplicates(subset=["candidate_id"])
# 50개 cap
if len(smoke_pos) > 50:
    smoke_pos = smoke_pos.head(50)

# smoke hard_negative: positive 수의 2배, position_bin 균형
smoke_hn_target = min(len(smoke_pos) * 2, 100)
smoke_hn = (
    hn_df
    .groupby("position_bin", group_keys=False)
    .apply(lambda g: g.nlargest(smoke_hn_target // 6, "padim_score"))
    .reset_index(drop=True)
)
if len(smoke_hn) > smoke_hn_target:
    smoke_hn = smoke_hn.head(smoke_hn_target)

smoke_scope = pd.concat([smoke_pos, smoke_hn]).reset_index(drop=True)
n_smoke_pos = (smoke_scope["candidate_label"] == "positive").sum()
n_smoke_hn = (smoke_scope["candidate_label"] == "hard_negative").sum()
smoke_ratio = n_smoke_hn / n_smoke_pos if n_smoke_pos > 0 else 0

smoke_rows = []
smoke_rows.append({"scope": "P-C5_smoke", "n_positive": int(n_smoke_pos), "n_hard_negative": int(n_smoke_hn),
    "n_total": len(smoke_scope), "ratio": round(float(smoke_ratio), 2),
    "no_hit_included": bool(smoke_scope["patient_id"].isin(NO_HIT).any()),
    "tiny_included": bool(smoke_scope["patient_id"].isin(TINY).any()),
    "risk6_included": bool(smoke_scope["patient_id"].isin(RISK6).any()),
    "fallback_included": bool(smoke_scope["fallback_positive_below_p95"].any()),
    "position_bins_covered": int(smoke_scope["position_bin"].nunique()),
    "patients_covered": int(smoke_scope["patient_id"].nunique()),
    "note": "P-C5 smoke 전용, crop 생성 승인 후 실행"})

smoke_rows.append({"scope": "full_manifest", "n_positive": int(n_pos := (manifest["candidate_label"]=="positive").sum()),
    "n_hard_negative": int((manifest["candidate_label"]=="hard_negative").sum()),
    "n_total": len(manifest), "ratio": round(c3_ratio, 2),
    "no_hit_included": True, "tiny_included": True, "risk6_included": True,
    "fallback_included": True, "position_bins_covered": int(manifest["position_bin"].nunique()),
    "patients_covered": int(manifest["patient_id"].nunique()),
    "note": "P-C5 smoke PASS 후 별도 승인 필요"})

pd.DataFrame(smoke_rows).to_csv(OUTPUT_DIR / "p_c4_smoke_scope_plan.csv", index=False)
print(f"  smoke scope: pos={n_smoke_pos}  hn={n_smoke_hn}  total={len(smoke_scope)}  ratio=1:{smoke_ratio:.2f}")
print(f"  no-hit={smoke_scope['patient_id'].isin(NO_HIT).any()}  tiny={smoke_scope['patient_id'].isin(TINY).any()}  risk6={smoke_scope['patient_id'].isin(RISK6).any()}")

# ── 10. 판정 ─────────────────────────────────────────────────────────────────
print("[10] 판정")
blocker_list = []
if c3_contamination != 0:
    blocker_list.append("stage2_holdout contamination > 0")
if n_ct_miss > 0:
    blocker_list.append(f"CT 누락 {n_ct_miss}명")
if n_roi_miss > 0:
    blocker_list.append(f"ROI 누락 {n_roi_miss}명")
if n_mask_miss > 0:
    blocker_list.append(f"mask 누락 {n_mask_miss}명")
if total_oob_count > 0:
    blocker_list.append(f"좌표 out-of-bound {total_oob_count}개")

major_blocker = (n_ct_miss + n_roi_miss + n_mask_miss) > len(patients_in_manifest) * 0.1
if major_blocker or c3_contamination != 0:
    verdict = "실패"
elif len(blocker_list) > 0:
    verdict = "부분통과"
else:
    verdict = "통과"

print(f"  판정: {verdict}  blockers={blocker_list}")

# ── 11. 출력 파일 생성 ────────────────────────────────────────────────────────
print("[11] 출력 파일 생성")
pd.DataFrame(errors).to_csv(OUTPUT_DIR / "p_c4_errors.csv", index=False)

elapsed = (datetime.datetime.now() - t_start).total_seconds()

summary = {
    "step": "P-C4",
    "verdict": verdict,
    "created": datetime.datetime.now().isoformat(),
    "elapsed_seconds": round(elapsed, 1),
    "p_c3_input": {
        "verdict": c3_verdict,
        "n_total": c3_n_total,
        "n_positive": c3_n_pos,
        "n_hard_negative": c3_n_hn,
        "ratio": float(c3_ratio),
        "stage2_holdout_contamination": int(c3_contamination),
    },
    "manifest_validation": {
        "row_count": int(len(manifest)),
        "row_count_match": bool(len(manifest) == c3_n_total),
        "stage2_holdout_contamination": int(cont),
        "required_crop_cols_ok": len(missing) == 0,
        "coord_null_total": int(total_null),
        "coord_null_detail": coord_nulls,
        "shape_mismatch_not_checked_npy_not_loaded": True,
        "shape_mismatch_deferred_to": "P-C5 crop smoke (CT/ROI/mask npy 실제 로드 후 확인)",
    },
    "file_availability": {
        "n_patients": int(len(patients_in_manifest)),
        "ct_ok": int(n_ct_ok), "ct_miss": int(n_ct_miss),
        "mask_ok": int(n_mask_ok), "mask_miss": int(n_mask_miss),
        "roi_ok": int(n_roi_ok), "roi_miss": int(n_roi_miss),
        "meta_ok": int(n_meta_ok), "shape_collected": int(len(shape_info)),
        "note": "파일 존재 여부만 확인. CT/ROI/mask npy value 분석 및 shape 일치 검증은 P-C5에서 수행",
    },
    "coordinate_validation": {
        "rows_checked": int(total_checked),
        "oob_total": int(total_oob_count),
    },
    "crop_format_plan": {
        "ct_source": "ct_hu.npy (int16 HU)",
        "roi_source": "refined_roi_v4_20_modeB_all_v1 refined_roi.npy",
        "mask_source": "lesion_mask_roi_0_0.npy",
        "crop_sizes_candidate": CROP_SIZES,
        "recommended_crop_size": 96,
        "recommended_channels": 3,
        "recommended_mode": "2.5D (z-1/z/z+1)",
        "save_format": ".npz",
        "label_csv": True,
        "boundary_handling": "reflect_padding",
    },
    "storage_estimate": {
        "full_114381_crops": {
            "2.5D_3ch_float32_96px_gb": float(rec_gb),
            "note": "전체 생성 시 사용자 별도 승인 필요",
        },
    },
    "smoke_scope": {
        "n_positive": int(n_smoke_pos),
        "n_hard_negative": int(n_smoke_hn),
        "n_total": len(smoke_scope),
        "ratio": round(float(smoke_ratio), 2),
        "no_hit_included": bool(smoke_scope["patient_id"].isin(NO_HIT).any()),
        "tiny_included": bool(smoke_scope["patient_id"].isin(TINY).any()),
        "risk6_included": bool(smoke_scope["patient_id"].isin(RISK6).any()),
        "fallback_included": bool(smoke_scope["fallback_positive_below_p95"].any()),
    },
    "blockers": blocker_list,
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
        "primary": "P-C5 crop smoke generation (사용자 승인 필요)",
        "smoke_size": len(smoke_scope),
        "full_after": "smoke PASS 후 full 114,381 crop 별도 승인",
        "p_c5_plan": {
            "step1_load_npy": "CT/ROI/mask npy 실제 로드 (smoke 대상만)",
            "step2_shape_check": "CT/ROI/mask shape 일치 확인 (shape_mismatch_not_checked_npy_not_loaded 해소)",
            "step3_crop_extract": "crop 1개 이상 실제 추출 및 값 확인",
            "step4_25d_padding": "2.5D z-1/z/z+1 boundary padding (reflect) 동작 확인",
            "step5_label_csv": "label CSV와 crop label 일치 확인",
        },
    },
}
with open(OUTPUT_DIR / "p_c4_crop_generation_preflight.json", "w") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

# ── 12. report MD ─────────────────────────────────────────────────────────────
report = f"""# P-C4 Crop Generation Preflight Report

**판정: {verdict}**
생성일: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  소요: {elapsed:.1f}초

## 1. P-C3 입력 검증

| 항목 | 결과 |
|------|------|
| P-C3 verdict | {c3_verdict} |
| manifest row 수 | {c3_n_total:,} (확인: {len(manifest):,}) |
| positive | {c3_n_pos:,} |
| hard_negative | {c3_n_hn:,} |
| positive:hard_negative 비율 | 1:{c3_ratio} |
| stage2_holdout contamination | {c3_contamination} |

## 2. manifest schema 검증

- 필수 crop 컬럼 {len(REQUIRED_CROP_COLS)}개: {"전부 OK" if not missing else f"누락: {missing}"}
- 좌표 null count: {total_null} ({"OK" if total_null == 0 else "FAIL"})
- 상세: {coord_nulls}

## 3. CT/ROI/mask 파일 존재 확인

| 파일 | 존재 | 누락 |
|------|------|------|
| CT (ct_hu.npy) | {n_ct_ok} | {n_ct_miss} |
| lesion mask | {n_mask_ok} | {n_mask_miss} |
| v4_20 ROI | {n_roi_ok} | {n_roi_miss} |
| meta.json | {n_meta_ok} | {n_meta_miss} |

shape 정보 수집 (meta.json 기반): {len(shape_info)}/{len(patients_in_manifest)}명

> ⚠️ **shape mismatch: not checked / deferred to P-C5**
> 이번 P-C4는 CT/ROI/mask npy를 실제 로드하지 않습니다.
> meta.json 기반 CT shape와 crop 좌표 범위만 검증합니다.
> ROI/mask shape mismatch 확인은 P-C5 crop smoke에서 수행합니다.
> (`shape_mismatch_not_checked_npy_not_loaded=True`)

## 4. 좌표 범위 검증

- 검증 rows: {total_checked:,}
- out-of-bound 총계: {total_oob_count}

## 5. crop 크기 후보별 boundary issue

| crop_size | 2D oob | 2D oob% | 2.5D z oob | 처리 방안 |
|-----------|--------|---------|-----------|---------|
"""
for row in crop_format_rows:
    report += f"| {row['crop_size']} | {row['oob_2d_count']} | {row['oob_2d_pct']}% | {row['oob_25d_z_count']} | {row['note']} |\n"

report += f"""
**권장**: crop_size=96, 2.5D 3ch, boundary padding=reflect

## 6. crop format 계획

| 항목 | 내용 |
|------|------|
| CT source | ct_hu.npy (int16 HU) |
| ROI source | refined_roi_v4_20_modeB_all_v1 refined_roi.npy |
| mask source | lesion_mask_roi_0_0.npy |
| 권장 crop_size | 96px |
| 채널 | 3ch (2.5D: z-1/z/z+1) |
| 저장 형식 | .npz |
| label CSV | 필요 |
| boundary | reflect padding |

## 7. 전체 crop 생성 저장 용량 추정

| crop_size | channels | dtype | full 용량 (GB) |
|-----------|----------|-------|--------------|
"""
for row in storage_rows:
    report += f"| {row['crop_size']} | {row['n_channels']}ch | {row['dtype']} | {row['full_total_gb']} |\n"

report += f"""
**권장 (2.5D 3ch float32 96px)**: 전체 {rec_gb}GB
⚠️ full crop 생성은 P-C5 smoke PASS 후 별도 승인 필요

## 8. P-C5 crop smoke scope 추천

| 항목 | 수 |
|------|----|
| smoke positive | {n_smoke_pos} |
| smoke hard_negative | {n_smoke_hn} |
| smoke 총계 | {len(smoke_scope)} |
| 비율 | 1:{smoke_ratio:.2f} |
| no-hit 3명 포함 | {smoke_scope["patient_id"].isin(NO_HIT).any()} |
| tiny lesion 4명 포함 | {smoke_scope["patient_id"].isin(TINY).any()} |
| risk6 6명 포함 | {smoke_scope["patient_id"].isin(RISK6).any()} |
| fallback 포함 | {smoke_scope["fallback_positive_below_p95"].any()} |
| position_bin 커버 | {smoke_scope["position_bin"].nunique()}개 |
| 환자 수 | {smoke_scope["patient_id"].nunique()}명 |

## 9. 가드레일 확인

- crop 생성: 없음
- 2차학습: 없음
- model forward: 없음
- scoring 재실행: 없음
- stage2_holdout 접근: 없음
- 기존 결과 수정: 없음

## 10. blockers

{"없음 (통과)" if not blocker_list else chr(10).join(f"- {b}" for b in blocker_list)}

## 11. 다음 단계

- **P-C5**: crop smoke generation (사용자 승인 필요)
  - smoke {len(smoke_scope)}개 (positive {n_smoke_pos} + hard_negative {n_smoke_hn})
  - P-C5 필수 수행 항목:
    1. CT/ROI/mask npy 실제 로드 (smoke 대상만)
    2. CT/ROI/mask shape 일치 확인 (`shape_mismatch_not_checked_npy_not_loaded` 해소)
    3. crop 1개 이상 실제 추출 및 값 확인
    4. 2.5D z-1/z/z+1 boundary padding (reflect) 동작 확인
    5. label CSV와 crop label 일치 확인
- full 114,381개 crop 생성은 smoke PASS 후 별도 승인
"""
with open(OUTPUT_DIR / "p_c4_crop_generation_preflight.md", "w") as f:
    f.write(report)

print(f"\n===== P-C4 완료 =====")
print(f"판정: {verdict}")
print(f"CT miss={n_ct_miss}  ROI miss={n_roi_miss}  mask miss={n_mask_miss}")
print(f"좌표 oob={total_oob_count}")
print(f"smoke scope: {len(smoke_scope)}개 (pos={n_smoke_pos} hn={n_smoke_hn})")
print(f"blockers={blocker_list}")
print(f"소요={elapsed:.1f}초")
print(f"출력: {OUTPUT_DIR}")
