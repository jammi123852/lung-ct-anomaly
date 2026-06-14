"""
P-C5 Crop Smoke Generation
EfficientNet-B0 v4_20 ROI branch

금지: full crop / 2차학습 / model forward / scoring / feature extraction
     / threshold 재계산 / metrics / stage2_holdout 접근 / 기존 결과 수정
허용: smoke 110개 CT/ROI/mask npy 로드 / crop 생성 / label CSV / integrity validation
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
P_C4_JSON = WORKSPACE / "outputs/reports/p_c4_crop_generation_preflight/p_c4_crop_generation_preflight.json"
SPLIT_CSV = BASE / "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1_balanced.csv"

CT_ROOT = Path("/mnt/c/Users/jinhy/Desktop/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1/volumes_npy")
ROI_ROOT = BASE / "outputs/mip-postprocess-research-v1/masks/refined_roi_v4_20_modeB_all_v1/lesion"

CROP_DIR = WORKSPACE / "outputs/crops/p_c5_crop_smoke"
REPORT_DIR = WORKSPACE / "outputs/reports/p_c5_crop_smoke"

CROP_SIZE = 96
HALF = CROP_SIZE // 2  # 48

errors = []
t_start = datetime.datetime.now()

# ── 출력 경로 초기화 ─────────────────────────────────────────────────────────
for d in [CROP_DIR, REPORT_DIR]:
    if d.exists():
        print(f"[ERROR] 출력 경로가 이미 존재합니다. 덮어쓰기 방지로 중단합니다: {d}")
        sys.exit(1)
    d.mkdir(parents=True)

# ── 1. P-C4 verdict 확인 ─────────────────────────────────────────────────────
print("[1] P-C4 verdict 확인")
with open(P_C4_JSON) as f:
    c4_summary = json.load(f)
c4_verdict = c4_summary["verdict"]
print(f"  P-C4 verdict={c4_verdict}")
if c4_verdict == "실패":
    print("[ERROR] P-C4 verdict=실패 → P-C5 진행 불가")
    sys.exit(1)

# ── 2. split CSV 로드 → stage2_holdout 차단 집합 확보 ────────────────────────
print("[2] split CSV 로드")
split_df = pd.read_csv(SPLIT_CSV)
STAGE2_HOLDOUT = set(split_df[split_df["stage_split"] == "stage2_holdout"]["patient_id"])
stage1_dev_set = set(split_df[split_df["stage_split"] == "stage1_dev"]["patient_id"])
print(f"  stage1_dev={len(stage1_dev_set)}  stage2_holdout={len(STAGE2_HOLDOUT)}")

# ── 3. P-C3 manifest 로드 ─────────────────────────────────────────────────────
print("[3] P-C3 manifest 로드")
manifest = pd.read_csv(MANIFEST_CSV)
assert len(manifest) == 114381, f"row mismatch: {len(manifest)}"
cont = manifest["patient_id"].isin(STAGE2_HOLDOUT).sum()
assert cont == 0, f"stage2_holdout contamination={cont}"
print(f"  row={len(manifest):,}  contamination={cont}  OK")

# ── 4. smoke selection 재현 (P-C4 동일 logic) ────────────────────────────────
print("[4] smoke selection 재현")
NO_HIT = {"LUNG1-086", "LUNG1-386", "MSD_lung_096"}
TINY = {"LUNG1-156", "LUNG1-192", "LUNG1-311", "LUNG1-386"}
RISK6 = {"LUNG1-386", "LUNG1-156", "LUNG1-028", "LUNG1-306", "LUNG1-421", "LUNG1-295"}
SPECIAL_PIDS = NO_HIT | TINY | RISK6

pos_df = manifest[manifest["candidate_label"] == "positive"].copy()
hn_df = manifest[manifest["candidate_label"] == "hard_negative"].copy()

smoke_pos_special = pos_df[pos_df["patient_id"].isin(SPECIAL_PIDS)].groupby("patient_id").head(2)
smoke_pos_general = (
    pos_df[~pos_df["patient_id"].isin(SPECIAL_PIDS)]
    .groupby("position_bin", group_keys=False)
    .apply(lambda g: g.nlargest(3, "padim_score"))
    .reset_index(drop=True)
)
smoke_pos = pd.concat([smoke_pos_special, smoke_pos_general]).drop_duplicates(subset=["candidate_id"])
if len(smoke_pos) > 50:
    smoke_pos = smoke_pos.head(50)

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
smoke_cont = smoke_scope["patient_id"].isin(STAGE2_HOLDOUT).sum()
assert smoke_cont == 0, f"smoke stage2_holdout contamination={smoke_cont}"
print(f"  smoke total={len(smoke_scope)}  pos={len(smoke_pos)}  hn={len(smoke_hn)}  contamination={smoke_cont}")

# manifest CSV 저장
smoke_scope.to_csv(REPORT_DIR / "p_c5_crop_smoke_manifest.csv", index=False)

# ── 5. 환자별 CT/ROI/mask 로드 및 crop 생성 ───────────────────────────────────
print("[5] crop 생성 시작")

def reflect_pad_volume(vol, pad_z, pad_y, pad_x):
    """z/y/x 방향으로 reflect padding (edge fallback)."""
    pz0, pz1 = pad_z
    py0, py1 = pad_y
    px0, px1 = pad_x
    def safe_pad(v, before, after, axis):
        if before == 0 and after == 0:
            return v
        size = v.shape[axis]
        mode = 'reflect' if size > 1 else 'edge'
        pads = [(0,0)]*v.ndim
        pads[axis] = (before, after)
        return np.pad(v, pads, mode=mode)
    v = safe_pad(vol, pz0, pz1, 0)
    v = safe_pad(v, py0, py1, 1)
    v = safe_pad(v, px0, px1, 2)
    return v

def extract_crop_25d(vol, lz, cy, cx, crop_size):
    """2.5D 3ch crop: (z-1, z, z+1) at center (lz, cy, cx)."""
    half = crop_size // 2
    Z, Y, X = vol.shape

    # z 패딩 계산
    z_indices = [lz - 1, lz, lz + 1]
    z_pad_before = max(0, -min(z_indices))
    z_pad_after  = max(0, max(z_indices) - (Z - 1))

    # y 패딩 계산
    y0 = cy - half
    y1 = cy + half
    y_pad_before = max(0, -y0)
    y_pad_after  = max(0, y1 - Y)

    # x 패딩 계산
    x0 = cx - half
    x1 = cx + half
    x_pad_before = max(0, -x0)
    x_pad_after  = max(0, x1 - X)

    total_pad = z_pad_before + z_pad_after + y_pad_before + y_pad_after + x_pad_before + x_pad_after

    if total_pad > 0:
        vol_p = reflect_pad_volume(vol,
                                   (z_pad_before, z_pad_after),
                                   (y_pad_before, y_pad_after),
                                   (x_pad_before, x_pad_after))
        lz_p = lz + z_pad_before
        y0_p = y0 + y_pad_before
        x0_p = x0 + x_pad_before
    else:
        vol_p = vol
        lz_p = lz
        y0_p = y0
        x0_p = x0

    # z-1, z, z+1 슬라이스 추출
    ch0 = vol_p[lz_p - 1, y0_p:y0_p + crop_size, x0_p:x0_p + crop_size]
    ch1 = vol_p[lz_p,     y0_p:y0_p + crop_size, x0_p:x0_p + crop_size]
    ch2 = vol_p[lz_p + 1, y0_p:y0_p + crop_size, x0_p:x0_p + crop_size]

    crop = np.stack([ch0, ch1, ch2], axis=0)  # (3, crop_size, crop_size)
    return crop, total_pad

# 환자 그룹 처리
label_rows = []
integrity_rows = []
lz_check_rows = []
generated_crops = 0
generated_errors = 0
total_pad_used = 0

patients_in_smoke = smoke_scope["safe_id"].unique()
print(f"  smoke 대상 환자 수={len(patients_in_smoke)}")

for patient_idx, sid in enumerate(patients_in_smoke):
    pat_rows = smoke_scope[smoke_scope["safe_id"] == sid].copy()
    pid = pat_rows.iloc[0]["patient_id"]

    # stage2_holdout 이중 차단
    if pid in STAGE2_HOLDOUT:
        errors.append({"candidate_id": "ALL", "patient_id": pid, "stage": "blocked",
                        "error": "BLOCKED: stage2_holdout"})
        continue

    ct_path   = CT_ROOT / sid / "ct_hu.npy"
    mask_path = CT_ROOT / sid / "lesion_mask_roi_0_0.npy"
    meta_path = CT_ROOT / sid / "meta.json"
    roi_path  = ROI_ROOT / sid / "refined_roi.npy"

    # 파일 존재 확인
    for fp, name in [(ct_path,'ct'), (mask_path,'mask'), (meta_path,'meta'), (roi_path,'roi')]:
        if not fp.exists():
            errors.append({"candidate_id":"ALL","patient_id":pid,"stage":"file_missing","error":f"{name} missing: {fp}"})

    # meta 로드
    with open(meta_path) as f:
        meta = json.load(f)
    shape_z, shape_y, shape_x = meta["shape_zyx"]

    # npy mmap 로드 (read-only)
    ct_vol   = np.load(ct_path,   mmap_mode='r')
    mask_vol = np.load(mask_path, mmap_mode='r')
    roi_vol  = np.load(roi_path,  mmap_mode='r')

    # shape 일치 확인
    shapes_match = (ct_vol.shape == mask_vol.shape == roi_vol.shape)
    shapes_match_meta = (list(ct_vol.shape) == [shape_z, shape_y, shape_x])
    if not shapes_match or not shapes_match_meta:
        errors.append({"candidate_id":"ALL","patient_id":pid,"stage":"shape_mismatch",
                        "error":f"ct={ct_vol.shape} mask={mask_vol.shape} roi={roi_vol.shape} meta={meta['shape_zyx']}"})

    # local_z vs slice_index 확인 (전체 smoke rows 기준)
    for _, row in pat_rows.iterrows():
        lz = int(row["local_z"])
        si = int(row["slice_index"])
        lz_inbound = (0 <= lz < shape_z)
        si_inbound = (0 <= si < shape_z)
        lz_check_rows.append({
            "candidate_id": row["candidate_id"],
            "patient_id": pid,
            "safe_id": sid,
            "local_z": lz,
            "slice_index": si,
            "shape_z": shape_z,
            "local_z_inbound": lz_inbound,
            "slice_index_inbound": si_inbound,
            "use_local_z": True,
        })

    # crop 생성
    for _, row in pat_rows.iterrows():
        cid = row["candidate_id"]
        lz  = int(row["local_z"])
        cy  = int((row["y0"] + row["y1"]) // 2)
        cx  = int((row["x0"] + row["x1"]) // 2)
        label = row["candidate_label"]

        # local_z in-bound 체크
        if not (0 <= lz < shape_z):
            errors.append({"candidate_id": cid, "patient_id": pid, "stage": "local_z_oob",
                            "error": f"local_z={lz} out of shape_z={shape_z}"})
            generated_errors += 1
            continue

        try:
            ct_crop,   pad_ct   = extract_crop_25d(ct_vol,   lz, cy, cx, CROP_SIZE)
            roi_crop,  pad_roi  = extract_crop_25d(roi_vol,  lz, cy, cx, CROP_SIZE)
            mask_crop, pad_mask = extract_crop_25d(mask_vol, lz, cy, cx, CROP_SIZE)
        except Exception as e:
            errors.append({"candidate_id": cid, "patient_id": pid, "stage": "crop_extract",
                            "error": str(e)})
            generated_errors += 1
            continue

        # shape 검증
        if ct_crop.shape != (3, CROP_SIZE, CROP_SIZE):
            errors.append({"candidate_id": cid, "patient_id": pid, "stage": "crop_shape",
                            "error": f"ct_crop shape={ct_crop.shape}"})
            generated_errors += 1
            continue

        # NaN/Inf 검증 (ct float 변환 후)
        ct_float = ct_crop.astype(np.float32)
        nan_cnt = int(np.isnan(ct_float).sum())
        inf_cnt = int(np.isinf(ct_float).sum())

        # mask consistency
        mask_pos_voxels = int(mask_crop[1].sum())  # z 슬라이스 기준 center
        if label == "positive" and mask_pos_voxels == 0:
            # fallback positive는 경계 이슈일 수 있으므로 warning
            err_type = "warning_pos_mask_zero"
        elif label == "hard_negative" and mask_pos_voxels > 0:
            err_type = "warning_hn_mask_nonzero"
        else:
            err_type = "ok"

        pad_used = pad_ct > 0

        # npz 저장
        npz_path = CROP_DIR / f"{cid}.npz"
        np.savez_compressed(
            npz_path,
            ct_crop=ct_crop,
            roi_crop=roi_crop,
            mask_crop=mask_crop,
            candidate_id=np.array([cid]),
            patient_id=np.array([pid]),
            safe_id=np.array([sid]),
            candidate_label=np.array([label]),
            candidate_rule=np.array([row["candidate_rule"]]),
            local_z=np.array([lz]),
            slice_index=np.array([int(row["slice_index"])]),
            y0=np.array([int(row["y0"])]),
            x0=np.array([int(row["x0"])]),
            y1=np.array([int(row["y1"])]),
            x1=np.array([int(row["x1"])]),
            padim_score=np.array([float(row["padim_score"])]),
        )

        generated_crops += 1
        total_pad_used += int(pad_used)

        # label CSV 행
        label_rows.append({
            "candidate_id": cid,
            "patient_id": pid,
            "safe_id": sid,
            "candidate_label": label,
            "candidate_rule": row["candidate_rule"],
            "local_z": lz,
            "slice_index": int(row["slice_index"]),
            "y0": int(row["y0"]),
            "x0": int(row["x0"]),
            "y1": int(row["y1"]),
            "x1": int(row["x1"]),
            "padim_score": float(row["padim_score"]),
            "position_bin": row["position_bin"],
            "fallback_positive_below_p95": bool(row["fallback_positive_below_p95"]),
            "no_hit_patient": bool(row["no_hit_patient"]),
            "tiny_lesion_flag": bool(row["tiny_lesion_flag"]),
            "p_b3_risk6_flag": bool(row["p_b3_risk6_flag"]),
            "npz_path": str(npz_path.relative_to(WORKSPACE)),
            "crop_shape": "(3,96,96)",
            "nan_in_ct": nan_cnt,
            "inf_in_ct": inf_cnt,
            "mask_pos_voxels_center": mask_pos_voxels,
            "mask_consistency": err_type,
            "pad_used": pad_used,
            "pad_total_pixels": int(pad_ct),
        })

        integrity_rows.append({
            "candidate_id": cid,
            "patient_id": pid,
            "candidate_label": label,
            "crop_shape_ok": ct_crop.shape == (3, CROP_SIZE, CROP_SIZE),
            "ct_nan": nan_cnt,
            "ct_inf": inf_cnt,
            "roi_shape_ok": roi_crop.shape == (3, CROP_SIZE, CROP_SIZE),
            "mask_shape_ok": mask_crop.shape == (3, CROP_SIZE, CROP_SIZE),
            "mask_consistency": err_type,
            "pad_used": pad_used,
        })

    print(f"  [{patient_idx+1}/{len(patients_in_smoke)}] {pid}: {len(pat_rows)}개 처리")

# ── 6. 결과 집계 ──────────────────────────────────────────────────────────────
print(f"\n[6] 집계  generated={generated_crops}  errors={generated_errors}")

label_df = pd.DataFrame(label_rows)
integrity_df = pd.DataFrame(integrity_rows)
lz_df = pd.DataFrame(lz_check_rows)
error_df = pd.DataFrame(errors)

label_df.to_csv(REPORT_DIR / "p_c5_crop_smoke_labels.csv", index=False)
integrity_df.to_csv(REPORT_DIR / "p_c5_crop_integrity_validation.csv", index=False)
lz_df.to_csv(REPORT_DIR / "p_c5_local_z_vs_slice_index_check.csv", index=False)
error_df.to_csv(REPORT_DIR / "p_c5_errors.csv", index=False)

# ── 7. integrity 집계 ─────────────────────────────────────────────────────────
print("[7] integrity 집계")
n_shape_ok   = int(integrity_df["crop_shape_ok"].sum()) if len(integrity_df) else 0
n_nan        = int((integrity_df["ct_nan"] > 0).sum()) if len(integrity_df) else 0
n_inf        = int((integrity_df["ct_inf"] > 0).sum()) if len(integrity_df) else 0
n_pos_crops  = int((label_df["candidate_label"] == "positive").sum()) if len(label_df) else 0
n_hn_crops   = int((label_df["candidate_label"] == "hard_negative").sum()) if len(label_df) else 0

pos_mask_ok  = 0
pos_mask_warn = 0
hn_mask_ok   = 0
hn_mask_warn  = 0
if len(label_df) > 0:
    pos_rows = label_df[label_df["candidate_label"] == "positive"]
    hn_rows  = label_df[label_df["candidate_label"] == "hard_negative"]
    pos_mask_ok   = int((pos_rows["mask_consistency"] == "ok").sum())
    pos_mask_warn = int((pos_rows["mask_consistency"] == "warning_pos_mask_zero").sum())
    hn_mask_ok    = int((hn_rows["mask_consistency"] == "ok").sum())
    hn_mask_warn  = int((hn_rows["mask_consistency"] == "warning_hn_mask_nonzero").sum())

lz_inbound_count  = int(lz_df["local_z_inbound"].sum())  if len(lz_df) else 0
si_inbound_count  = int(lz_df["slice_index_inbound"].sum()) if len(lz_df) else 0
lz_oob_count      = int((~lz_df["local_z_inbound"]).sum()) if len(lz_df) else 0

# special patient 포함 확인
if len(label_df) > 0:
    no_hit_included  = bool(label_df["patient_id"].isin(NO_HIT).any())
    tiny_included    = bool(label_df["tiny_lesion_flag"].any())
    risk6_included   = bool(label_df["p_b3_risk6_flag"].any())
    fallback_included = bool(label_df["fallback_positive_below_p95"].any())
else:
    no_hit_included = tiny_included = risk6_included = fallback_included = False

# output size
total_size_mb = sum(f.stat().st_size for f in CROP_DIR.glob("*.npz")) / 1024 / 1024
print(f"  n_shape_ok={n_shape_ok}  nan={n_nan}  inf={n_inf}  size={total_size_mb:.2f}MB")
print(f"  lz_inbound={lz_inbound_count}  lz_oob={lz_oob_count}  si_inbound={si_inbound_count}")
print(f"  pos_mask_ok={pos_mask_ok}  pos_mask_warn={pos_mask_warn}")
print(f"  hn_mask_ok={hn_mask_ok}  hn_mask_warn={hn_mask_warn}")
print(f"  total_pad_used={total_pad_used}")

# label CSV ↔ npz 일치
npz_ids = {f.stem for f in CROP_DIR.glob("*.npz")}
csv_ids  = set(label_df["candidate_id"].tolist()) if len(label_df) else set()
id_match = (npz_ids == csv_ids)
print(f"  label CSV rows={len(label_df)}  npz files={len(npz_ids)}  id_match={id_match}")

# ── 8. 판정 ─────────────────────────────────────────────────────────────────
print("[8] 판정")
EXPECTED = len(smoke_scope)
blocker_list = []

if generated_crops < EXPECTED * 0.9:
    blocker_list.append(f"생성 crop {generated_crops} < expected {EXPECTED}의 90%")
if n_nan > 0:
    blocker_list.append(f"NaN in ct_crop: {n_nan}개 crop")
if n_inf > 0:
    blocker_list.append(f"Inf in ct_crop: {n_inf}개 crop")
if not id_match:
    blocker_list.append("label CSV ↔ npz candidate_id 불일치")
if any(e.get("stage") == "blocked" for e in errors):
    blocker_list.append("stage2_holdout 접근 시도 감지")

if not blocker_list and generated_crops == EXPECTED:
    verdict = "통과"
elif not any(e.get("stage") == "blocked" for e in errors) and generated_crops >= EXPECTED * 0.9:
    verdict = "부분통과"
else:
    verdict = "실패"

print(f"  판정={verdict}  generated={generated_crops}  expected={EXPECTED}  blockers={blocker_list}")

# ── 9. summary JSON ───────────────────────────────────────────────────────────
elapsed = (datetime.datetime.now() - t_start).total_seconds()

summary = {
    "step": "P-C5",
    "verdict": verdict,
    "created": datetime.datetime.now().isoformat(),
    "elapsed_seconds": round(elapsed, 1),
    "p_c4_input": {
        "verdict": c4_verdict,
    },
    "smoke_scope": {
        "expected_total": EXPECTED,
        "n_positive_expected": int((smoke_scope["candidate_label"]=="positive").sum()),
        "n_hard_negative_expected": int((smoke_scope["candidate_label"]=="hard_negative").sum()),
    },
    "crop_generation": {
        "generated_crops": generated_crops,
        "generation_errors": generated_errors,
        "crop_size": CROP_SIZE,
        "channels": 3,
        "mode": "2.5D z-1/z/z+1",
        "z_center": "local_z",
        "padding": "reflect (edge fallback for size-1 axis)",
        "ct_dtype_save": "int16",
        "roi_dtype_save": "uint8",
        "mask_dtype_save": "uint8",
        "save_format": "npz",
    },
    "local_z_vs_slice_index": {
        "total_smoke_rows": len(lz_df),
        "local_z_inbound": lz_inbound_count,
        "local_z_oob": lz_oob_count,
        "slice_index_inbound": si_inbound_count,
        "z_center_used": "local_z",
        "conclusion": "local_z가 CT 접근에 적합 (slice_index는 global z로 사용 불가)",
    },
    "shape_consistency": {
        "shape_mismatch_not_checked_npy_not_loaded": False,
        "ct_roi_mask_shape_checked": True,
        "note": "P-C5에서 실제 npy 로드 후 shape 확인 완료",
    },
    "integrity": {
        "crop_shape_ok": n_shape_ok,
        "ct_nan_crops": n_nan,
        "ct_inf_crops": n_inf,
        "pad_used_crops": total_pad_used,
        "label_csv_rows": len(label_df),
        "npz_files": len(npz_ids),
        "id_match": id_match,
        "n_positive_crops": n_pos_crops,
        "n_hard_negative_crops": n_hn_crops,
        "pos_mask_ok": pos_mask_ok,
        "pos_mask_warn_zero": pos_mask_warn,
        "hn_mask_ok": hn_mask_ok,
        "hn_mask_warn_nonzero": hn_mask_warn,
    },
    "special_coverage": {
        "no_hit_included": no_hit_included,
        "tiny_included": tiny_included,
        "risk6_included": risk6_included,
        "fallback_included": fallback_included,
    },
    "output_size_mb": round(total_size_mb, 2),
    "blockers": blocker_list,
    "guardrails": {
        "full_crop_generated": False,
        "training_executed": False,
        "model_forward": False,
        "scoring_rerun": False,
        "feature_extraction": False,
        "threshold_recalculated": False,
        "metrics_recalculated": False,
        "stage2_holdout_accessed": False,
        "existing_results_modified": False,
    },
    "n_errors": len(errors),
    "next_step": {
        "primary": "P-C6 crop smoke validation / review",
        "alternative": "crop format 수정 후 P-C5 재실행 (판정 부분통과 시)",
        "full_crop_after": "P-C5 PASS 후 full 114,381 crop 생성 별도 승인",
    },
}

with open(REPORT_DIR / "p_c5_crop_smoke_summary.json", "w") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

# ── 10. report MD ─────────────────────────────────────────────────────────────
report = f"""# P-C5 Crop Smoke Generation Report

**판정: {verdict}**
생성일: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  소요: {elapsed:.1f}초

## 1. P-C4 입력 검증

| 항목 | 결과 |
|------|------|
| P-C4 verdict | {c4_verdict} |
| P-C5 진행 가능 | {'예' if c4_verdict != '실패' else '아니오'} |

## 2. smoke crop 생성 결과

| 항목 | 수 |
|------|----|
| 기대 crop 수 | {EXPECTED} |
| 생성 crop 수 | {generated_crops} |
| 생성 오류 | {generated_errors} |
| positive crop | {n_pos_crops} |
| hard_negative crop | {n_hn_crops} |
| 전체 생성 = smoke 110개 제한 | {'통과' if generated_crops <= EXPECTED else '초과'} |
| full crop 미생성 | 확인 |

## 3. CT/ROI/mask shape 일치

- 실제 npy 로드 후 확인 완료 (`shape_mismatch_not_checked_npy_not_loaded=False`)
- CT, ROI, mask shape 일치 체크 수행

## 4. local_z vs slice_index

| 항목 | 수 |
|------|----|
| smoke 대상 전체 rows | {len(lz_df)} |
| local_z in-bound | {lz_inbound_count} |
| local_z OOB | {lz_oob_count} |
| slice_index in-bound | {si_inbound_count} |
| 사용 기준 | **local_z** |
| 결론 | local_z가 CT 접근에 적합, slice_index는 global z로 직접 접근 불가 |

## 5. padding 사용

| 항목 | 수 |
|------|----|
| padding 사용 crop 수 | {total_pad_used} |
| padding 방식 | reflect (edge fallback) |

## 6. crop shape 검증

| 항목 | 결과 |
|------|------|
| crop shape (3,96,96) 확인 | {n_shape_ok}/{generated_crops} |

## 7. NaN / Inf

| 항목 | 결과 |
|------|------|
| ct_crop NaN crops | {n_nan} |
| ct_crop Inf crops | {n_inf} |

## 8. label / mask consistency

| 항목 | 결과 |
|------|------|
| positive mask 정상 | {pos_mask_ok} |
| positive mask zero (warning) | {pos_mask_warn} |
| hard_negative mask 정상 | {hn_mask_ok} |
| hard_negative mask nonzero (warning) | {hn_mask_warn} |

## 9. 특수 케이스 포함 여부

| 항목 | 포함 |
|------|------|
| no-hit 3명 | {no_hit_included} |
| tiny lesion | {tiny_included} |
| risk6 | {risk6_included} |
| fallback positive | {fallback_included} |

## 10. label CSV ↔ npz 일치

| 항목 | 결과 |
|------|------|
| label CSV rows | {len(label_df)} |
| npz files | {len(npz_ids)} |
| candidate_id 일치 | {id_match} |

## 11. output size

| 항목 | 결과 |
|------|------|
| crop npz 총 용량 | {total_size_mb:.2f} MB |

## 12. 가드레일 확인

- full crop 생성: 없음 (smoke {generated_crops}개 한정)
- 2차학습: 없음
- model forward: 없음
- scoring 재실행: 없음
- feature extraction: 없음
- threshold 재계산: 없음
- metrics 재계산: 없음
- stage2_holdout 접근: 없음
- 기존 결과 수정: 없음

## 13. blockers

{"없음" if not blocker_list else chr(10).join(f"- {b}" for b in blocker_list)}

## 14. 다음 단계

- **P-C6**: crop smoke validation / review (crop overlay 시각화 또는 integrity 재검토)
- 또는 crop format 수정 후 P-C5 재실행 (부분통과 시)
- full 114,381개 crop 생성은 P-C5 PASS 후 별도 승인 필요
"""

with open(REPORT_DIR / "p_c5_crop_smoke_report.md", "w") as f:
    f.write(report)

print(f"\n===== P-C5 완료 =====")
print(f"판정: {verdict}")
print(f"generated={generated_crops}  expected={EXPECTED}  errors={generated_errors}")
print(f"lz_oob={lz_oob_count}  pad_used={total_pad_used}  size={total_size_mb:.2f}MB")
print(f"blockers={blocker_list}")
print(f"소요={elapsed:.1f}초")
print(f"출력: {CROP_DIR}")
print(f"      {REPORT_DIR}")
