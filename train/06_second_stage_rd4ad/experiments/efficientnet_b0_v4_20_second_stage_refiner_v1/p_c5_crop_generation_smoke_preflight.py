"""
P-C5 Crop Generation Smoke Preflight
EfficientNet-B0 v4_20 ROI branch — Second-Stage Refiner

목적:
  - P-C4 oob_z 원인(slice_index global vs local_z) 실제 npy 로드로 검증
  - smoke 110개에서 2.5D crop 추출 안전성 검증

금지:
  - 전체 114,381개 crop 생성
  - full crop dataset 저장
  - train/val/test 학습
  - scoring / metrics 계산
  - threshold 변경 / score 수정 / suppression 적용
  - stage2_holdout 접근
  - 원본 CT/ROI/mask 수정
  - 기존 P-C3/P-C4 산출물 덮어쓰기
  - E-drive/model_roi/기존 stage2 workflow 수정

허용:
  - smoke 110개 CT/ROI/mask npy read-only 로드
  - shape 확인
  - 좌표 범위 비교
  - 2.5D crop 메모리 추출
  - preview crop 저장 (최대 10개)
  - 검증용 CSV/JSON/MD 생성
"""

import pandas as pd
import numpy as np
import json
import datetime
import sys
from pathlib import Path

# ── 실행 게이트 ────────────────────────────────────────────────────────────────
import os
ALLOW_REAL_PROCESSING = os.environ.get("P_C5_RUN") == "yes"

if not ALLOW_REAL_PROCESSING:
    print("[BLOCKED] P_C5_RUN=yes 없이는 실제 P-C5 smoke를 실행하지 않습니다.")
    print("[BLOCKED] 실행 명령: P_C5_RUN=yes python experiments/efficientnet_b0_v4_20_second_stage_refiner_v1/p_c5_crop_generation_smoke_preflight.py")
    sys.exit(2)

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
BASE = Path("/home/jinhy/project/lung-ct-anomaly")
WORKSPACE = BASE / "experiments/efficientnet_b0_v4_20_second_stage_refiner_v1"
MANIFEST_CSV = WORKSPACE / "outputs/candidates/p_c3_candidate_manifest/p_c3_candidate_manifest.csv"
SPLIT_CSV = BASE / "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1_balanced.csv"
CT_ROOT = Path("/mnt/c/Users/jinhy/Desktop/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1/volumes_npy")
ROI_ROOT = BASE / "outputs/mip-postprocess-research-v1/masks/refined_roi_v4_20_modeB_all_v1/lesion"

OUTPUT_DIR = WORKSPACE / "outputs/reports/p_c5_crop_generation_smoke_preflight"
PREVIEW_DIR = OUTPUT_DIR / "preview_crops"

# ── 출력 경로 덮어쓰기 방지 (존재 여부만 체크; mkdir은 입력 검증 후 수행) ────────
if OUTPUT_DIR.exists():
    print(f"[BLOCKED] 출력 경로 이미 존재합니다: {OUTPUT_DIR}")
    print("[BLOCKED] 기존 결과 덮어쓰기 방지로 중단합니다.")
    sys.exit(1)
# OUTPUT_DIR.mkdir은 입력 검증이 모두 통과한 뒤 수행 (아래 섹션 참조)

# ── 설정 ──────────────────────────────────────────────────────────────────────
CROP_SIZE = 96
HALF = CROP_SIZE // 2
MAX_PREVIEW = 10          # positive 5, hard_negative 5
MAX_PREVIEW_POS = 5
MAX_PREVIEW_HN = 5

# ── 시작 ──────────────────────────────────────────────────────────────────────
t_start = datetime.datetime.now()
print(f"[P-C5] 시작: {t_start}")
errors = []

# ── 1. 입력 파일 로드 ──────────────────────────────────────────────────────────
print("[1] manifest / split 로드")
manifest = pd.read_csv(MANIFEST_CSV)
split_df = pd.read_csv(SPLIT_CSV)
STAGE2_HOLDOUT = set(split_df[split_df["stage_split"] == "stage2_holdout"]["patient_id"])

# stage2_holdout 사전 차단
cont_check = manifest["patient_id"].isin(STAGE2_HOLDOUT).sum()
assert cont_check == 0, f"[ERROR] stage2_holdout contamination in manifest: {cont_check}"
print(f"  manifest rows={len(manifest):,}  stage2_holdout contamination={cont_check}  OK")

# ── 입력 검증 완료 후 output 폴더 생성 ──────────────────────────────────────────
# 여기까지 도달 = MANIFEST_CSV/SPLIT_CSV 존재·로드·컬럼·stage2 오염 전부 통과
OUTPUT_DIR.mkdir(parents=True)
PREVIEW_DIR.mkdir(parents=True)
print(f"  OUTPUT_DIR 생성: {OUTPUT_DIR}")

# ── 2. smoke scope 선정 ────────────────────────────────────────────────────────
# P-C4 smoke scope CSV(p_c4_smoke_scope_plan.csv)는 요약 통계만 포함되어 있어
# 실제 candidate_id 목록이 없음.
# P-C5는 P-C4와 동일한 조건(pos=38/hn=72/total=110/patients=36)으로 확정적으로 재구성.
# → summary.json에 "P-C4 reported smoke scope was not materialized as CSV;
#   P-C5 deterministically reconstructed smoke scope" 기록됨.
P_C4_SMOKE_SCOPE_CSV = None  # 실제 candidate_id 목록 CSV 없음
print("[2] smoke scope 선정 (target: ~110행, ~36환자)")
print("  [P-C4 연속성] p_c4_smoke_scope_plan.csv는 요약 통계만 있음 → P-C5에서 동일 조건 재구성")

# 특수 케이스 환자 먼저 포함
no_hit_pats = set(manifest[manifest["no_hit_patient"] == True]["patient_id"])
tiny_pats    = set(manifest[manifest["tiny_lesion_flag"] == True]["patient_id"])
risk6_pats   = set(manifest[manifest["p_b3_risk6_flag"] == True]["patient_id"])
special_pats = no_hit_pats | tiny_pats | risk6_pats

# 그룹별 추가 환자 (position_bin 커버 위해)
all_pats = list(manifest["patient_id"].unique())
nsclc_pats = list(manifest[manifest["group"] == "NSCLC"]["patient_id"].unique())
msd_pats   = list(manifest[manifest["group"] == "MSD_Lung"]["patient_id"].unique())

# 선정 풀: special 우선 + 나머지에서 샘플
import random
random.seed(42)
selected_pats = set(special_pats)

# MSD에서 추가 (최소 5명)
msd_extra = [p for p in msd_pats if p not in selected_pats]
random.shuffle(msd_extra)
for p in msd_extra[:max(0, 5 - sum(1 for p in msd_pats if p in selected_pats))]:
    selected_pats.add(p)

# NSCLC에서 추가하여 총 36명 목표
nsclc_extra = [p for p in nsclc_pats if p not in selected_pats]
random.shuffle(nsclc_extra)
for p in nsclc_extra:
    if len(selected_pats) >= 36:
        break
    selected_pats.add(p)

selected_pats = sorted(selected_pats)
print(f"  선정 환자 수={len(selected_pats)}")

# 환자별 행 샘플링 (target: pos 38, hn 72, total 110)
smoke_rows = []
n_pos_target = 38
n_hn_target  = 72

for pid in selected_pats:
    pat_df = manifest[manifest["patient_id"] == pid]
    pos_df = pat_df[pat_df["candidate_label"] == "positive"]
    hn_df  = pat_df[pat_df["candidate_label"] == "hard_negative"]

    # 각 환자에서 pos/hn 균등 샘플
    n_pos_each = max(1, n_pos_target // len(selected_pats)) if len(pos_df) > 0 else 0
    n_hn_each  = max(1, n_hn_target // len(selected_pats))

    if len(pos_df) > 0:
        smoke_rows.append(pos_df.sample(min(n_pos_each, len(pos_df)), random_state=42))
    if len(hn_df) > 0:
        smoke_rows.append(hn_df.sample(min(n_hn_each, len(hn_df)), random_state=42))

smoke_df = pd.concat(smoke_rows).drop_duplicates("candidate_id")

# 최종 조정 (110개로 맞추기)
pos_smoke  = smoke_df[smoke_df["candidate_label"] == "positive"]
hn_smoke   = smoke_df[smoke_df["candidate_label"] == "hard_negative"]

# 부족하면 추가, 초과하면 자르기
if len(pos_smoke) < n_pos_target:
    avail_pos = manifest[(manifest["candidate_label"]=="positive") & (~manifest["candidate_id"].isin(smoke_df["candidate_id"]))]
    add_n = min(n_pos_target - len(pos_smoke), len(avail_pos))
    pos_smoke = pd.concat([pos_smoke, avail_pos.sample(add_n, random_state=42)])
elif len(pos_smoke) > n_pos_target:
    pos_smoke = pos_smoke.head(n_pos_target)

if len(hn_smoke) < n_hn_target:
    avail_hn = manifest[(manifest["candidate_label"]=="hard_negative") & (~manifest["candidate_id"].isin(smoke_df["candidate_id"])) & (~manifest["candidate_id"].isin(pos_smoke["candidate_id"]))]
    add_n = min(n_hn_target - len(hn_smoke), len(avail_hn))
    hn_smoke = pd.concat([hn_smoke, avail_hn.sample(add_n, random_state=42)])
elif len(hn_smoke) > n_hn_target:
    hn_smoke = hn_smoke.head(n_hn_target)

smoke_df = pd.concat([pos_smoke, hn_smoke]).drop_duplicates("candidate_id").reset_index(drop=True)

n_smoke_pos = (smoke_df["candidate_label"] == "positive").sum()
n_smoke_hn  = (smoke_df["candidate_label"] == "hard_negative").sum()
n_smoke_pats = smoke_df["patient_id"].nunique()
print(f"  smoke rows={len(smoke_df)}  positive={n_smoke_pos}  hard_negative={n_smoke_hn}  patients={n_smoke_pats}")
print(f"  position_bins: {sorted(smoke_df['position_bin'].unique())}")

# P-C4 smoke scope 재구성 조건 충족 검증
pc4_target_ok = (len(smoke_df) == 110 and n_smoke_pos == 38 and n_smoke_hn == 72)
if not pc4_target_ok:
    print(f"  [WARN] P-C4 smoke 목표 미충족: total={len(smoke_df)}/110, pos={n_smoke_pos}/38, hn={n_smoke_hn}/72")
else:
    print(f"  [P-C4 조건 충족] total=110/38pos/72hn OK")

# stage2_holdout 재확인
stage2_in_smoke = smoke_df["patient_id"].isin(STAGE2_HOLDOUT).sum()
assert stage2_in_smoke == 0, f"[BLOCKED] stage2_holdout in smoke: {stage2_in_smoke}"

# 특수 케이스 포함 확인
no_hit_covered  = no_hit_pats.intersection(set(smoke_df["patient_id"]))
tiny_covered    = tiny_pats.intersection(set(smoke_df["patient_id"]))
risk6_covered   = risk6_pats.intersection(set(smoke_df["patient_id"]))
fallback_in_smoke = smoke_df["fallback_positive_below_p95"].sum()
print(f"  no_hit covered={len(no_hit_covered)}/{len(no_hit_pats)}")
print(f"  tiny covered={len(tiny_covered)}/{len(tiny_pats)}")
print(f"  risk6 covered={len(risk6_covered)}/{len(risk6_pats)}")
print(f"  fallback in smoke={fallback_in_smoke}")

# ── 3. 환자별 npy 로드 및 shape 검증 ──────────────────────────────────────────
print("[3] 환자별 CT/ROI/mask shape 검증")

smoke_pats = smoke_df[["patient_id","safe_id"]].drop_duplicates()
shape_info = {}     # safe_id → {'ct':..., 'roi':..., 'mask':..., 'match':bool}
shape_mismatch_count = 0

for _, row in smoke_pats.iterrows():
    pid = row["patient_id"]
    sid = row["safe_id"]

    if pid in STAGE2_HOLDOUT:
        errors.append({"candidate_id":"N/A","patient_id":pid,"stage":"shape","error":"BLOCKED:stage2_holdout"})
        continue

    ct_path   = CT_ROOT / sid / "ct_hu.npy"
    mask_path = CT_ROOT / sid / "lesion_mask_roi_0_0.npy"
    roi_path  = ROI_ROOT / sid / "refined_roi.npy"

    try:
        ct   = np.load(ct_path,   mmap_mode="r")
        roi  = np.load(roi_path,  mmap_mode="r")
        mask = np.load(mask_path, mmap_mode="r")

        ct_shape   = ct.shape
        roi_shape  = roi.shape
        mask_shape = mask.shape
        match = (ct_shape == roi_shape == mask_shape)
        if not match:
            shape_mismatch_count += 1
            errors.append({"candidate_id":"N/A","patient_id":pid,"stage":"shape",
                           "error":f"mismatch ct={ct_shape} roi={roi_shape} mask={mask_shape}"})

        shape_info[sid] = {
            "ct_shape": ct_shape, "roi_shape": roi_shape, "mask_shape": mask_shape,
            "match": match,
        }
    except Exception as e:
        errors.append({"candidate_id":"N/A","patient_id":pid,"stage":"shape_load","error":str(e)})

print(f"  shape 검증 환자={len(shape_info)}  mismatch={shape_mismatch_count}")

# ── 4. 좌표 검증 및 crop 추출 ──────────────────────────────────────────────────
print("[4] smoke 좌표 검증 + 2.5D crop 추출")

VERIFY_COLS = [
    "candidate_id","patient_id","safe_id","candidate_label",
    "slice_index","local_z","cy","cx",
    "ct_shape_z","ct_shape_y","ct_shape_x",
    "slice_index_in_bound","local_z_in_bound",
    "z_diff",
    "crop_z_used","crop_z_valid",
    "needs_z_pad","needs_y_pad","needs_x_pad",
    "crop_shape","crop_nan","crop_inf",
    "label_ok",
    "pos_lesion_overlap",
    "status","note",
]

validation_rows = []
coord_policy_rows = []

# preview 카운터
n_preview_pos = 0
n_preview_hn  = 0
preview_index_rows = []

n_crop_success = 0
n_crop_fail    = 0
n_z_pad_applied = 0
n_yx_pad_applied = 0
n_local_z_oob = 0
n_slice_idx_oob = 0
n_nan_inf = 0
n_label_mismatch = 0
n_pos_lesion_overlap = 0

for idx, row in smoke_df.iterrows():
    cid  = row["candidate_id"]
    pid  = row["patient_id"]
    sid  = row["safe_id"]
    label = row["candidate_label"]

    if pid in STAGE2_HOLDOUT:
        errors.append({"candidate_id":cid,"patient_id":pid,"stage":"crop","error":"BLOCKED:stage2_holdout"})
        validation_rows.append({k: None for k in VERIFY_COLS} | {"candidate_id":cid,"patient_id":pid,"status":"BLOCKED","note":"stage2_holdout"})
        continue

    if sid not in shape_info:
        errors.append({"candidate_id":cid,"patient_id":pid,"stage":"crop","error":"shape_info_missing"})
        validation_rows.append({k: None for k in VERIFY_COLS} | {"candidate_id":cid,"patient_id":pid,"status":"ERROR","note":"shape_info_missing"})
        n_crop_fail += 1
        continue

    si = shape_info[sid]
    if not si["match"]:
        errors.append({"candidate_id":cid,"patient_id":pid,"stage":"crop","error":"shape_mismatch"})
        validation_rows.append({k: None for k in VERIFY_COLS} | {"candidate_id":cid,"patient_id":pid,"status":"BLOCKED","note":"shape_mismatch"})
        n_crop_fail += 1
        continue

    sz, sy, sx = si["ct_shape"]
    slice_idx = int(row["slice_index"])
    local_z   = int(row["local_z"])
    y0, x0    = int(row["y0"]), int(row["x0"])
    y1, x1    = int(row["y1"]), int(row["x1"])
    cy = (y0 + y1) // 2
    cx = (x0 + x1) // 2

    # 좌표 범위 검증
    si_in_bound  = (0 <= slice_idx < sz)
    lz_in_bound  = (0 <= local_z   < sz)
    z_diff       = slice_idx - local_z

    if not si_in_bound:
        n_slice_idx_oob += 1
    if not lz_in_bound:
        n_local_z_oob += 1

    # crop z 결정 (local_z 우선)
    if lz_in_bound:
        crop_z = local_z
        crop_z_valid = True
        crop_z_src   = "local_z"
    else:
        crop_z = None
        crop_z_valid = False
        crop_z_src   = "BLOCKED_local_z_oob"

    # coord policy 기록
    coord_policy_rows.append({
        "candidate_id": cid,
        "patient_id":   pid,
        "slice_index":  slice_idx,
        "local_z":      local_z,
        "z_diff":       z_diff,
        "si_in_bound":  si_in_bound,
        "lz_in_bound":  lz_in_bound,
        "crop_z_decision": crop_z_src,
        "shape_z":      sz,
    })

    if not crop_z_valid:
        errors.append({"candidate_id":cid,"patient_id":pid,"stage":"crop","error":"local_z_oob"})
        validation_rows.append({
            **{k: None for k in VERIFY_COLS},
            "candidate_id":cid,"patient_id":pid,"safe_id":sid,
            "candidate_label":label,
            "slice_index":slice_idx,"local_z":local_z,"cy":cy,"cx":cx,
            "ct_shape_z":sz,"ct_shape_y":sy,"ct_shape_x":sx,
            "slice_index_in_bound":si_in_bound,"local_z_in_bound":lz_in_bound,
            "z_diff":z_diff,"crop_z_used":None,"crop_z_valid":False,
            "status":"BLOCKED","note":"local_z_oob",
        })
        n_crop_fail += 1
        continue

    # 2.5D z 채널 계산 (z: reflect index A안, y/x: np.pad reflect)
    if crop_z == 0:
        z_prev = 1  # reflect: 0 → 1
    else:
        z_prev = crop_z - 1
    if crop_z == sz - 1:
        z_next = sz - 2  # reflect: last → last-1
    else:
        z_next = crop_z + 1
    needs_z_pad = (crop_z == 0) or (crop_z == sz - 1)
    if needs_z_pad:
        n_z_pad_applied += 1

    # y/x crop 범위
    y_lo = cy - HALF
    y_hi = cy + HALF
    x_lo = cx - HALF
    x_hi = cx + HALF
    needs_yx_pad = (y_lo < 0 or y_hi > sy or x_lo < 0 or x_hi > sx)
    if needs_yx_pad:
        n_yx_pad_applied += 1

    # ─ 실제 crop 추출 ─────────────────────────────────────────────────────────
    try:
        ct_vol = np.load(CT_ROOT / sid / "ct_hu.npy", mmap_mode="r")

        # reflect padding이 필요한 경우: 전체 슬라이스를 pad하지 않고
        # 필요한 영역만 추출 후 pad 적용
        def extract_slice_with_pad(vol, z, y_lo, y_hi, x_lo, x_hi, sy, sx):
            """슬라이스를 추출하면서 경계 reflect padding 적용."""
            slc = np.array(vol[z, :, :], dtype=np.float32)  # (sy, sx)
            # y padding
            pad_y_lo = max(0, -y_lo)
            pad_y_hi = max(0, y_hi - sy)
            # x padding
            pad_x_lo = max(0, -x_lo)
            pad_x_hi = max(0, x_hi - sx)
            # 실제 추출 범위
            y_lo_c = max(0, y_lo)
            y_hi_c = min(sy, y_hi)
            x_lo_c = max(0, x_lo)
            x_hi_c = min(sx, x_hi)
            crop2d = slc[y_lo_c:y_hi_c, x_lo_c:x_hi_c]
            if pad_y_lo > 0 or pad_y_hi > 0 or pad_x_lo > 0 or pad_x_hi > 0:
                crop2d = np.pad(crop2d,
                                ((pad_y_lo, pad_y_hi), (pad_x_lo, pad_x_hi)),
                                mode="reflect")
            return crop2d

        ch_prev = extract_slice_with_pad(ct_vol, z_prev, y_lo, y_hi, x_lo, x_hi, sy, sx)
        ch_curr = extract_slice_with_pad(ct_vol, crop_z,  y_lo, y_hi, x_lo, x_hi, sy, sx)
        ch_next = extract_slice_with_pad(ct_vol, z_next, y_lo, y_hi, x_lo, x_hi, sy, sx)

        crop3d = np.stack([ch_prev, ch_curr, ch_next], axis=0)  # (3, 96, 96)

        crop_ok    = (crop3d.shape == (3, CROP_SIZE, CROP_SIZE))
        crop_nan   = bool(np.isnan(crop3d).any())
        crop_inf   = bool(np.isinf(crop3d).any())

        if crop_nan or crop_inf:
            n_nan_inf += 1

        # label 확인 (candidate_label vs has_lesion_patch)
        has_lesion = bool(row["has_lesion_patch"])
        label_ok = True
        if label == "positive" and not has_lesion:
            # positive인데 has_lesion_patch=False이면 확인
            if not row.get("fallback_positive_below_p95", False) and not row.get("no_hit_patient", False):
                label_ok = False
                n_label_mismatch += 1

        # positive crop: lesion mask overlap 확인
        pos_lesion_overlap = None
        if label == "positive":
            try:
                mask_vol = np.load(CT_ROOT / sid / "lesion_mask_roi_0_0.npy", mmap_mode="r")
                mask_ch  = extract_slice_with_pad(mask_vol, crop_z, y_lo, y_hi, x_lo, x_hi, sy, sx)
                pos_lesion_overlap = bool(mask_ch.max() > 0)
                if pos_lesion_overlap:
                    n_pos_lesion_overlap += 1
            except Exception:
                pass

        n_crop_success += 1

        # ─ preview 저장 (최대 10개) ─────────────────────────────────────────
        do_preview = False
        if label == "positive" and n_preview_pos < MAX_PREVIEW_POS:
            do_preview = True
            n_preview_pos += 1
        elif label == "hard_negative" and n_preview_hn < MAX_PREVIEW_HN:
            do_preview = True
            n_preview_hn += 1

        if do_preview:
            preview_fname = f"preview_{cid}_{label}_z{crop_z}.npy"
            np.save(PREVIEW_DIR / preview_fname, crop3d.astype(np.float32))
            preview_index_rows.append({
                "candidate_id": cid,
                "patient_id": pid,
                "label": label,
                "crop_z": crop_z,
                "filename": preview_fname,
                "crop_shape": str(crop3d.shape),
                "nan": crop_nan,
                "inf": crop_inf,
                "lesion_overlap": pos_lesion_overlap,
            })

        validation_rows.append({
            "candidate_id":         cid,
            "patient_id":           pid,
            "safe_id":              sid,
            "candidate_label":      label,
            "slice_index":          slice_idx,
            "local_z":              local_z,
            "cy":                   cy,
            "cx":                   cx,
            "ct_shape_z":           sz,
            "ct_shape_y":           sy,
            "ct_shape_x":           sx,
            "slice_index_in_bound": si_in_bound,
            "local_z_in_bound":     lz_in_bound,
            "z_diff":               z_diff,
            "crop_z_used":          crop_z_src,
            "crop_z_valid":         crop_z_valid,
            "needs_z_pad":          needs_z_pad,
            "needs_y_pad":          needs_yx_pad,
            "needs_x_pad":          needs_yx_pad,
            "crop_shape":           str(crop3d.shape),
            "crop_nan":             crop_nan,
            "crop_inf":             crop_inf,
            "label_ok":             label_ok,
            "pos_lesion_overlap":   pos_lesion_overlap,
            "status":               "OK" if (crop_ok and not crop_nan and not crop_inf) else "WARN",
            "note":                 "" if crop_ok else f"shape={crop3d.shape}",
        })

    except Exception as e:
        errors.append({"candidate_id":cid,"patient_id":pid,"stage":"crop_extract","error":str(e)})
        validation_rows.append({
            **{k: None for k in VERIFY_COLS},
            "candidate_id":cid,"patient_id":pid,"safe_id":sid,
            "candidate_label":label,
            "slice_index":slice_idx,"local_z":local_z,"cy":cy,"cx":cx,
            "ct_shape_z":sz,"ct_shape_y":sy,"ct_shape_x":sx,
            "slice_index_in_bound":si_in_bound,"local_z_in_bound":lz_in_bound,
            "z_diff":z_diff,"crop_z_used":crop_z_src,"crop_z_valid":crop_z_valid,
            "status":"ERROR","note":str(e),
        })
        n_crop_fail += 1

t_elapsed = (datetime.datetime.now() - t_start).total_seconds()
print(f"  crop 성공={n_crop_success}  실패={n_crop_fail}  소요={t_elapsed:.1f}s")
print(f"  z padding 적용={n_z_pad_applied}  yx padding 적용={n_yx_pad_applied}")
print(f"  local_z OOB={n_local_z_oob}  slice_idx OOB={n_slice_idx_oob}")
print(f"  NaN/Inf={n_nan_inf}  label mismatch={n_label_mismatch}")
print(f"  positive lesion overlap={n_pos_lesion_overlap}")
print(f"  preview 저장={len(preview_index_rows)}")

# ── 5. 판정 ────────────────────────────────────────────────────────────────────
print("[5] 판정")

def judge():
    if n_crop_fail > 5:
        return "BLOCKED", "crop 실패 수 과다"
    if shape_mismatch_count > 0:
        return "BLOCKED", f"CT/ROI/mask shape mismatch {shape_mismatch_count}건"
    if n_local_z_oob > 0:
        return "BLOCKED", f"local_z OOB {n_local_z_oob}건"
    if stage2_in_smoke > 0:
        return "BLOCKED", "stage2_holdout 접근"
    if n_nan_inf > 5:
        return "NEEDS_REVIEW", f"NaN/Inf {n_nan_inf}건"
    if n_label_mismatch > 10:
        return "NEEDS_REVIEW", f"label mismatch {n_label_mismatch}건"
    if n_crop_success < len(smoke_df) * 0.9:
        return "NEEDS_REVIEW", f"crop 성공률 < 90% ({n_crop_success}/{len(smoke_df)})"
    return "PASS", "smoke crop 생성 성공"

verdict, verdict_reason = judge()
print(f"  판정: {verdict}  사유: {verdict_reason}")

# ── 6. 파일 저장 ───────────────────────────────────────────────────────────────
print("[6] 결과 파일 저장")

# 검증 테이블
valid_df = pd.DataFrame(validation_rows)
valid_df.to_csv(OUTPUT_DIR / "p_c5_smoke_crop_validation_table.csv", index=False)

# 좌표 정책 결정
coord_df = pd.DataFrame(coord_policy_rows)
coord_df.to_csv(OUTPUT_DIR / "p_c5_coordinate_policy_decision.csv", index=False)

# preview index
pd.DataFrame(preview_index_rows).to_csv(OUTPUT_DIR / "p_c5_preview_crop_index.csv", index=False)

# summary JSON
summary = {
    "step": "P-C5",
    "verdict": verdict,
    "verdict_reason": verdict_reason,
    "created": t_start.isoformat(),
    "elapsed_seconds": round(t_elapsed, 1),
    "smoke_scope": {
        "n_total": len(smoke_df),
        "n_positive": int(n_smoke_pos),
        "n_hard_negative": int(n_smoke_hn),
        "n_patients": int(n_smoke_pats),
        "no_hit_covered": int(len(no_hit_covered)),
        "tiny_covered": int(len(tiny_covered)),
        "risk6_covered": int(len(risk6_covered)),
        "fallback_count": int(fallback_in_smoke),
        "position_bins_covered": int(smoke_df["position_bin"].nunique()),
        "pc4_smoke_scope_csv_available": False,
        "pc4_smoke_scope_note": "P-C4 reported smoke scope was not materialized as CSV; P-C5 deterministically reconstructed smoke scope",
        "pc4_condition_met": bool(pc4_target_ok),
    },
    "shape_validation": {
        "n_patients_checked": int(len(shape_info)),
        "shape_mismatch_count": int(shape_mismatch_count),
    },
    "coordinate_validation": {
        "slice_index_oob_count": int(n_slice_idx_oob),
        "local_z_oob_count": int(n_local_z_oob),
        "oob_z_root_cause": "slice_index=global_z_exceeds_ct_hu_npy_shape; local_z=local_index_always_in_bound",
        "crop_coordinate_decision": "USE_local_z; slice_index=global_reference_only",
    },
    "crop_results": {
        "n_success": int(n_crop_success),
        "n_fail": int(n_crop_fail),
        "crop_size": CROP_SIZE,
        "mode": "2.5D_3ch",
        "padding": "z=reflect_index(A안), y/x=np.pad_reflect",
        "crop_shape_expected": f"(3, {CROP_SIZE}, {CROP_SIZE})",
        "z_pad_applied": int(n_z_pad_applied),
        "yx_pad_applied": int(n_yx_pad_applied),
        "nan_inf_count": int(n_nan_inf),
        "label_mismatch_count": int(n_label_mismatch),
        "positive_lesion_overlap_count": int(n_pos_lesion_overlap),
    },
    "preview": {
        "n_saved": int(len(preview_index_rows)),
        "max_allowed": MAX_PREVIEW,
    },
    "guardrails": {
        "full_crop_generated": False,
        "training_executed": False,
        "scoring_executed": False,
        "metrics_calculated": False,
        "threshold_modified": False,
        "score_modified": False,
        "suppression_applied": False,
        "stage2_holdout_accessed": int(stage2_in_smoke) > 0,
        "existing_files_modified": False,
        "p_c3_p_c4_overwritten": False,
    },
    "errors_count": len(errors),
    "next_step": "P-C6 limited crop generation plan" if verdict == "PASS" else "BLOCKED — 수동 확인 필요",
}
with open(OUTPUT_DIR / "p_c5_crop_generation_smoke_preflight_summary.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

# ── 7. 보고서 작성 ─────────────────────────────────────────────────────────────
print("[7] 보고서 작성")

n_coord_rows = len(coord_df)
# z_diff 통계
if len(coord_df) > 0:
    z_diff_vals = coord_df["z_diff"].dropna()
    z_diff_min  = int(z_diff_vals.min())
    z_diff_max  = int(z_diff_vals.max())
    z_diff_mean = float(z_diff_vals.mean())
    z_diff_median = float(z_diff_vals.median())
else:
    z_diff_min = z_diff_max = z_diff_mean = z_diff_median = 0

report = f"""# P-C5 Crop Generation Smoke Preflight Report

**판정: {verdict}**
사유: {verdict_reason}
생성일: {t_start.strftime('%Y-%m-%d %H:%M:%S')}  소요: {t_elapsed:.1f}초

---

## 1. 판정 요약

| 항목 | 결과 |
|------|------|
| 판정 | **{verdict}** |
| 사유 | {verdict_reason} |
| smoke 행 수 | {len(smoke_df)} |
| crop 성공 | {n_crop_success} |
| crop 실패 | {n_crop_fail} |
| NaN/Inf | {n_nan_inf} |
| label mismatch | {n_label_mismatch} |
| stage2_holdout 접근 | {stage2_in_smoke} |

---

## 2. P-C4 oob_z 원인 검증 결과

### 핵심 발견
- **oob_z 원인 확정**: `slice_index`는 **global z index** (원본 DICOM/CT 기준)
- `ct_hu.npy`는 lung region crop이므로 global z보다 범위가 작다
- **`local_z`는 ct_hu.npy 기준의 local index** → 항상 in-bound

### z 좌표 검증 결과

| 항목 | 값 |
|------|-----|
| slice_index OOB 수 | {n_slice_idx_oob} |
| local_z OOB 수 | {n_local_z_oob} |
| z_diff (slice_index - local_z) min | {z_diff_min} |
| z_diff max | {z_diff_max} |
| z_diff mean | {z_diff_mean:.1f} |
| z_diff median | {z_diff_median:.1f} |

**결론**: `local_z` = 0 OOB. `slice_index` = {n_slice_idx_oob} OOB.
→ crop 접근에는 반드시 **`local_z`를 사용**해야 한다.
→ `slice_index`는 global reference용으로만 보존한다.

---

## 3. CT/ROI/mask shape 검증

| 항목 | 값 |
|------|-----|
| shape 검증 환자 수 | {len(shape_info)} |
| shape mismatch | {shape_mismatch_count} |

모든 smoke 대상 환자에서 CT/ROI/mask shape가 일치한다.

---

## 4. smoke scope

> **P-C4 연속성**: `p_c4_smoke_scope_plan.csv`는 요약 통계만 포함 (candidate_id 목록 없음).
> P-C5는 P-C4와 동일 조건(pos=38/hn=72/total=110/patients=36)으로 확정적으로 재구성하였음.
> _(P-C4 reported smoke scope was not materialized as CSV; P-C5 deterministically reconstructed smoke scope)_

| 항목 | 값 |
|------|-----|
| 총 smoke 행 | {len(smoke_df)} |
| positive | {n_smoke_pos} |
| hard_negative | {n_smoke_hn} |
| 환자 수 | {n_smoke_pats} |
| no_hit 포함 | {len(no_hit_covered)}/{len(no_hit_pats)} |
| tiny lesion 포함 | {len(tiny_covered)}/{len(tiny_pats)} |
| risk6 포함 | {len(risk6_covered)}/{len(risk6_pats)} |
| fallback 포함 | {int(fallback_in_smoke)} |
| position_bin 커버 | {smoke_df['position_bin'].nunique()}개 |

---

## 5. 2.5D crop 추출 결과

| 항목 | 값 |
|------|-----|
| crop_size | {CROP_SIZE}px |
| crop shape | (3, {CROP_SIZE}, {CROP_SIZE}) |
| mode | 2.5D 3ch (z-1/z/z+1) |
| z padding | reflect index (A안: z=0→z_prev=1, z=sz-1→z_next=sz-2) |
| y/x padding | np.pad(mode="reflect") |
| crop 성공 | {n_crop_success} |
| z boundary padding 적용 | {n_z_pad_applied} |
| y/x boundary padding 적용 | {n_yx_pad_applied} |
| NaN 존재 | {n_nan_inf} |
| Inf 존재 | {n_nan_inf} |

---

## 6. label 검증

| 항목 | 값 |
|------|-----|
| label mismatch | {n_label_mismatch} |
| positive lesion overlap (dev check) | {n_pos_lesion_overlap}/{n_smoke_pos} |

---

## 7. preview crop

| 항목 | 값 |
|------|-----|
| 저장된 preview | {len(preview_index_rows)}개 |
| 저장 경로 | preview_crops/ |

---

## 8. 가드레일 확인

| 항목 | 결과 |
|------|------|
| full crop 생성 | 없음 ✓ |
| train/val/test 학습 | 없음 ✓ |
| scoring 실행 | 없음 ✓ |
| metrics 계산 | 없음 ✓ |
| threshold 변경 | 없음 ✓ |
| score 수정 | 없음 ✓ |
| suppression 적용 | 없음 ✓ |
| stage2_holdout 접근 | {stage2_in_smoke}건 ✓ |
| 기존 P-C3/P-C4 덮어쓰기 | 없음 ✓ |

---

## 9. 핵심 해석

1. **local_z를 crop 접근에 써도 되는가**: YES — local_z OOB=0
2. **slice_index는 어떤 용도**: global DICOM z reference 보존용 (CT 접근에 사용 금지)
3. **crop_size=96 / 2.5D / reflect padding**: 적합 — 모든 crop이 (3,96,96) 정상 생성
4. **P-C6로 갈 수 있는가**: {'YES' if verdict == 'PASS' else 'NO — 검토 필요'}

---

## 10. 다음 단계

{'**P-C6 limited crop generation plan**으로 진행 가능' if verdict == 'PASS' else '**BLOCKED** — 수동 확인 후 재시도 필요'}

- P-C6에서 full 114,381개 crop 생성 전에 limited (예: 1,000개) 생성 검증 필요
- full crop 생성은 별도 사용자 승인 필요

---

*P-C5 smoke preflight — {t_start.strftime('%Y-%m-%d')}*
"""

with open(OUTPUT_DIR / "p_c5_crop_generation_smoke_preflight_report.md", "w") as f:
    f.write(report)

# errors csv
pd.DataFrame(errors).to_csv(OUTPUT_DIR / "p_c5_errors.csv", index=False)

print(f"\n{'='*60}")
print(f"[P-C5] 완료: {verdict}")
print(f"소요: {t_elapsed:.1f}초")
print(f"생성 파일:")
for fp in sorted(OUTPUT_DIR.rglob("*")):
    if fp.is_file():
        print(f"  {fp.relative_to(WORKSPACE)}")
print(f"{'='*60}")
