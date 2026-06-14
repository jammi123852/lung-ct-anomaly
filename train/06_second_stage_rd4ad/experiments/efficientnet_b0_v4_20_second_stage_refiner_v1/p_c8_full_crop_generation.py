"""
P-C8: EfficientNet-B0 v4_20 second-stage full crop generation
- `--full-run` 없으면 dry-check 모드만 실행 (crop 생성 금지)
- `--full-run` 있어야만 실제 114,381개 crop 생성
- label policy: Option B (center patch 기준 유지 + warning flag 3종)
- z 기준: local_z 확정, slice_index crop 접근 금지
"""

import argparse, os, sys, json, csv, datetime, shutil
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# 경로 상수
# ============================================================
BASE      = Path("/home/jinhy/project/lung-ct-anomaly")
WORKSPACE = BASE / "experiments/efficientnet_b0_v4_20_second_stage_refiner_v1"

CT_ROOT  = Path("/mnt/c/Users/jinhy/Desktop/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1/volumes_npy")
ROI_ROOT = BASE / "outputs/mip-postprocess-research-v1/masks/refined_roi_v4_20_modeB_all_v1/lesion"

MANIFEST_CSV  = WORKSPACE / "outputs/candidates/p_c3_candidate_manifest/p_c3_candidate_manifest.csv"
P_C3_JSON     = WORKSPACE / "outputs/candidates/p_c3_candidate_manifest/p_c3_candidate_manifest_summary.json"
P_C6_JSON     = WORKSPACE / "outputs/reports/p_c6_crop_smoke_validation/p_c6_crop_smoke_validation.json"
P_C7_JSON     = WORKSPACE / "outputs/reports/p_c7_full_crop_generation_preflight/p_c7_full_crop_generation_preflight.json"
SPLIT_CSV     = BASE / "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1_balanced.csv"

CROP_DIR      = WORKSPACE / "outputs/crops/p_c8_full_crops"
REPORT_DIR    = WORKSPACE / "outputs/reports/p_c8_full_crop_generation"
LABELS_CSV    = REPORT_DIR / "p_c8_full_crop_labels.csv"
MANIFEST_COPY = REPORT_DIR / "p_c8_full_crop_manifest.csv"
SUMMARY_JSON  = REPORT_DIR / "p_c8_full_crop_generation_summary.json"
REPORT_MD     = REPORT_DIR / "p_c8_full_crop_generation_report.md"
INTEGRITY_CSV = REPORT_DIR / "p_c8_full_crop_integrity.csv"
WARN_CSV      = REPORT_DIR / "p_c8_mask_warning_summary.csv"
ERROR_CSV     = REPORT_DIR / "p_c8_errors.csv"
DONE_MARKER   = REPORT_DIR / "DONE.json"

DRYCHECK_DIR  = WORKSPACE / "outputs/reports/p_c8_full_crop_generation_drycheck"

# crop params
CROP_SIZE = 96
HALF      = CROP_SIZE // 2
N_CHANNELS = 3
EXPECTED_SHAPE = (N_CHANNELS, CROP_SIZE, CROP_SIZE)

REQUIRED_COLS = [
    "candidate_id", "patient_id", "safe_id", "split",
    "local_z", "slice_index",
    "y0", "x0", "y1", "x1",
    "padim_score", "candidate_label", "candidate_rule",
    "no_hit_patient", "tiny_lesion_flag", "p_b3_risk6_flag",
    "fallback_positive_below_p95", "source_branch",
]

LABEL_CSV_COLS = [
    "candidate_id", "crop_path",
    "patient_id", "safe_id",
    "candidate_label", "candidate_rule",
    "local_z", "slice_index",
    "y0", "x0", "y1", "x1",
    "padim_score",
    "mask_nonzero_warning", "center_mask_nonzero", "adjacent_mask_nonzero",
    "no_hit_patient", "tiny_lesion_flag", "p_b3_risk6_flag",
    "fallback_positive_below_p95", "source_branch",
    "crop_shape", "ct_nan", "ct_inf", "mask_consistency",
    "pad_used",
]

INTEGRITY_CSV_COLS = [
    "candidate_id", "crop_exists", "crop_shape",
    "ct_nan", "ct_inf",
    "roi_binary_valid", "mask_binary_valid",
    "mask_nonzero_warning", "center_mask_nonzero", "adjacent_mask_nonzero",
    "mask_consistency", "pad_used", "error",
]

WARN_CSV_COLS = [
    "candidate_id", "patient_id", "candidate_rule",
    "mask_nonzero_warning", "center_mask_nonzero", "adjacent_mask_nonzero",
]

# ============================================================
# 공통 유틸
# ============================================================
def load_verdicts():
    """P-C7/C6/C3 verdict 로드 및 검증. 실패 시 sys.exit."""
    errs = []
    with open(P_C7_JSON) as f: p_c7 = json.load(f)
    with open(P_C6_JSON) as f: p_c6 = json.load(f)
    with open(P_C3_JSON) as f: p_c3 = json.load(f)

    if p_c7["verdict"] != "통과":
        errs.append(f"P-C7 verdict={p_c7['verdict']} (통과 필요)")
    if p_c6["verdict"] != "통과":
        errs.append(f"P-C6 verdict={p_c6['verdict']} (통과 필요)")
    if p_c3["verdict"] != "통과":
        errs.append(f"P-C3 verdict={p_c3['verdict']} (통과 필요)")

    n_total  = p_c3["candidate_counts"]["n_total"]
    holdout  = p_c3["input_validation"]["stage2_holdout_contamination"]
    n_pos    = p_c3["candidate_counts"]["n_positive"]
    n_hn     = p_c3["candidate_counts"]["n_hard_negative"]

    if holdout != 0:
        errs.append(f"stage2_holdout_contamination={holdout} (0 필요)")
    if n_total != 114381:
        errs.append(f"P-C3 n_total={n_total} (114381 필요)")

    if errs:
        print("[ERROR] 사전 검증 실패:")
        for e in errs: print(f"  {e}")
        sys.exit(2)

    return p_c7, p_c6, p_c3, n_total, n_pos, n_hn


def load_stage2_holdout():
    """stage2_holdout patient_id set 로드."""
    split_df = pd.read_csv(SPLIT_CSV)
    holdout  = set(split_df[split_df["stage_split"] == "stage2_holdout"]["patient_id"])
    stage1   = set(split_df[split_df["stage_split"] == "stage1_dev"]["patient_id"])
    return holdout, stage1


def load_manifest(holdout_set):
    """manifest 로드 + schema/null/holdout 검증."""
    df = pd.read_csv(MANIFEST_CSV)

    # 행 수
    if len(df) != 114381:
        print(f"[ERROR] manifest rows={len(df)} (114381 필요)")
        sys.exit(2)

    # 필수 컬럼
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"[ERROR] manifest missing columns: {missing}")
        sys.exit(2)

    # coordinate null
    coord_cols = ["local_z", "y0", "x0", "y1", "x1"]
    null_cnt = df[coord_cols].isnull().sum().sum()
    if null_cnt > 0:
        print(f"[ERROR] coordinate null count={null_cnt}")
        sys.exit(2)

    # stage2_holdout 오염
    cont = df[df["patient_id"].isin(holdout_set)]
    if len(cont) > 0:
        print(f"[ERROR] stage2_holdout contamination={len(cont)}")
        sys.exit(2)

    # split 확인
    if "split" in df.columns:
        bad_split = df[df["split"] != "stage1_dev"]
        if len(bad_split) > 0:
            print(f"[WARN] non-stage1_dev rows: {len(bad_split)}")

    return df


def check_output_collision(is_resume: bool, is_fullrun: bool):
    """output path collision 확인. DONE marker 있으면 항상 중단."""
    if DONE_MARKER.exists():
        print("[ERROR] DONE.json 이미 존재 → 이미 완료된 실행. 재실행하려면 DONE.json 삭제 후 진행.")
        sys.exit(2)

    if CROP_DIR.exists() and not is_resume:
        existing = list(CROP_DIR.glob("*.npz"))
        if len(existing) > 0 and is_fullrun:
            print(f"[ERROR] crop dir 이미 존재하고 {len(existing)}개 npz 있음. --resume 없이 overwrite 불가.")
            sys.exit(2)


def get_resume_set():
    """이미 생성된 candidate_id set 반환. npz + label 행 둘 다 있어야만 완료로 처리."""
    npz_ids = set()
    if CROP_DIR.exists():
        for f in CROP_DIR.glob("*.npz"):
            npz_ids.add(f.stem)
    label_ids = set()
    if LABELS_CSV.exists():
        with open(LABELS_CSV) as f:
            reader = csv.DictReader(f)
            for row in reader:
                label_ids.add(row["candidate_id"])
    # npz 와 label 행 둘 다 있는 것만 완료 처리
    return npz_ids & label_ids


def extract_crop_3ch(vol, local_z, cy, cx):
    """2.5D 3ch crop 추출. z=local_z ±1, reflect padding."""
    Z, H, W = vol.shape

    if local_z < 0 or local_z >= Z:
        raise ValueError(f"local_z={local_z} out of bounds [0, {Z})")

    z_indices = [local_z - 1, local_z, local_z + 1]
    slices = []
    for zi in z_indices:
        zi_c = max(0, min(Z - 1, zi))
        slices.append(vol[zi_c])
    arr = np.stack(slices, axis=0)  # (3, H, W)

    y0_c = cy - HALF
    y1_c = cy + HALF
    x0_c = cx - HALF
    x1_c = cx + HALF

    pt  = max(0, -y0_c)
    pb  = max(0, y1_c - H)
    pl  = max(0, -x0_c)
    pr  = max(0, x1_c - W)

    y0s = max(0, y0_c);  y1s = min(H, y1_c)
    x0s = max(0, x0_c);  x1s = min(W, x1_c)

    patch     = arr[:, y0s:y1s, x0s:x1s]
    pad_used  = any([pt, pb, pl, pr])

    if pad_used:
        mode = "reflect" if (patch.shape[1] > 1 and patch.shape[2] > 1) else "edge"
        patch = np.pad(patch, ((0, 0), (pt, pb), (pl, pr)), mode=mode)

    return patch, pad_used


def compute_mask_flags(mask_crop):
    """mask warning flag 3종 계산."""
    mask_any      = bool(mask_crop.any())
    center_nz     = bool(mask_crop[1].any())   # ch1 = center slice
    adjacent_nz   = bool(mask_crop[0].any() or mask_crop[2].any())
    return mask_any, center_nz, adjacent_nz


# ============================================================
# DRY-CHECK 모드
# ============================================================
def run_drycheck():
    now_str   = datetime.datetime.now().isoformat(timespec="seconds")
    dc_errors = []

    print("=" * 60)
    print("P-C8 dry-check 모드 (--full-run 없음, crop 생성 없음)")
    print("=" * 60)

    DRYCHECK_DIR.mkdir(parents=True, exist_ok=True)

    # 1. verdict 확인
    print("[1] verdict 확인...")
    p_c7, p_c6, p_c3, n_total, n_pos, n_hn = load_verdicts()
    print(f"  P-C7={p_c7['verdict']}, P-C6={p_c6['verdict']}, P-C3={p_c3['verdict']}")

    # 2. holdout set
    print("[2] stage2_holdout set 로드...")
    holdout_set, stage1_set = load_stage2_holdout()
    print(f"  stage1_dev={len(stage1_set)}, stage2_holdout={len(holdout_set)}")

    # 3. manifest 검증
    print("[3] manifest 검증...")
    df = load_manifest(holdout_set)
    manifest_rows     = len(df)
    missing_cols      = [c for c in REQUIRED_COLS if c not in df.columns]
    coord_null        = df[["local_z","y0","x0","y1","x1"]].isnull().sum().sum()
    n_pos_manifest    = (df["candidate_label"] == "positive").sum()
    n_hn_manifest     = (df["candidate_label"] == "hard_negative").sum()
    holdout_cont      = df["patient_id"].isin(holdout_set).sum()
    n_patients        = df["patient_id"].nunique()
    print(f"  rows={manifest_rows}, pos={n_pos_manifest}, hn={n_hn_manifest}, patients={n_patients}")
    print(f"  missing_cols={missing_cols}, coord_null={coord_null}, holdout_cont={holdout_cont}")

    # 4. output path 확인
    print("[4] output path collision 확인...")
    crop_dir_exists   = CROP_DIR.exists()
    report_dir_exists = REPORT_DIR.exists()
    done_exists       = DONE_MARKER.exists()
    labels_exists     = LABELS_CSV.exists()
    existing_crops    = len(list(CROP_DIR.glob("*.npz"))) if crop_dir_exists else 0
    collision         = done_exists or (crop_dir_exists and existing_crops > 0)
    if done_exists:
        dc_errors.append("DONE.json 이미 존재 → 완료된 실행")
    print(f"  crop_dir={crop_dir_exists}({existing_crops}개), done={done_exists}, labels={labels_exists}")

    # 5. disk space
    print("[5] disk space 확인...")
    disk = shutil.disk_usage(str(BASE))
    disk_free_gb = disk.free / (1024**3)
    # P-C5 기준 압축 후 4.48 GB 예상 (P-C7 추정)
    need_gb = 4.48 * 1.5
    disk_ok = disk_free_gb > need_gb
    print(f"  free={disk_free_gb:.1f}GB, need≈{need_gb:.1f}GB → {'OK' if disk_ok else 'WARNING'}")

    # 6. CT/ROI/mask 파일 존재 샘플 확인 (상위 10 환자만, 전체 로드 금지)
    print("[6] CT/ROI/mask 파일 존재 샘플 확인 (10명)...")
    sample_patients = df["safe_id"].unique()[:10]
    file_check_rows = []
    missing_ct = 0; missing_roi = 0; missing_mask = 0
    for sid in sample_patients:
        ct_p   = CT_ROOT  / sid / "ct_hu.npy"
        mask_p = CT_ROOT  / sid / "lesion_mask_roi_0_0.npy"
        meta_p = CT_ROOT  / sid / "meta.json"
        roi_p  = ROI_ROOT / sid / "refined_roi.npy"
        ct_ok   = ct_p.exists()
        mask_ok = mask_p.exists()
        meta_ok = meta_p.exists()
        roi_ok  = roi_p.exists()
        if not ct_ok:   missing_ct   += 1
        if not roi_ok:  missing_roi  += 1
        if not mask_ok: missing_mask += 1
        file_check_rows.append({
            "safe_id": sid, "ct_ok": ct_ok, "mask_ok": mask_ok,
            "meta_ok": meta_ok, "roi_ok": roi_ok,
        })
    print(f"  sample 10명: ct_missing={missing_ct}, roi_missing={missing_roi}, mask_missing={missing_mask}")
    if missing_ct > 0 or missing_roi > 0:
        dc_errors.append(f"sample 10명 중 CT missing={missing_ct}, ROI missing={missing_roi}")

    # 7. resume 상태 확인
    print("[7] resume 상태 확인...")
    resume_set      = get_resume_set()
    resume_possible = len(resume_set) > 0
    remaining       = n_total - len(resume_set)
    print(f"  already done={len(resume_set)}, remaining={remaining}")

    # 8. z축 policy 확인
    print("[8] z축 policy 확인...")
    lz_range  = (int(df["local_z"].min()), int(df["local_z"].max()))
    si_range  = (int(df["slice_index"].min()), int(df["slice_index"].max()))
    print(f"  local_z range={lz_range}, slice_index range={si_range}")
    print("  crop z 기준: local_z 확정, slice_index: crop 접근 금지")

    # 9. label 분포
    label_dist = df["candidate_label"].value_counts().to_dict()
    print(f"[9] label 분포: {label_dist}")

    # 10. 판정
    blockers = []
    if holdout_cont > 0:      blockers.append(f"holdout_contamination={holdout_cont}")
    if done_exists:           blockers.append("DONE.json 이미 존재")
    if not disk_ok:           blockers.append(f"disk free={disk_free_gb:.1f}GB < need={need_gb:.1f}GB")
    if missing_cols:          blockers.append(f"missing manifest cols: {missing_cols}")
    if coord_null > 0:        blockers.append(f"coord_null={coord_null}")
    blockers.extend(dc_errors)

    verdict = "통과" if len(blockers) == 0 else ("실패" if holdout_cont > 0 or done_exists else "부분통과")
    print(f"\n[판정] {verdict}")
    if blockers:
        print(f"  blockers: {blockers}")

    # 출력 파일 저장
    dc_json = {
        "step": "P-C8 dry-check",
        "verdict": verdict,
        "created": now_str,
        "input_validation": {
            "p_c7_verdict": p_c7["verdict"],
            "p_c6_verdict": p_c6["verdict"],
            "p_c3_verdict": p_c3["verdict"],
        },
        "manifest": {
            "rows": manifest_rows, "expected": 114381,
            "n_positive": int(n_pos_manifest), "n_hard_negative": int(n_hn_manifest),
            "n_patients": n_patients,
            "missing_cols": missing_cols,
            "coord_null": int(coord_null),
            "holdout_contamination": int(holdout_cont),
        },
        "output_paths": {
            "crop_dir":           str(CROP_DIR),
            "report_dir":         str(REPORT_DIR),
            "labels_csv":         str(LABELS_CSV),
            "manifest_copy":      str(MANIFEST_COPY),
            "summary_json":       str(SUMMARY_JSON),
            "done_marker":        str(DONE_MARKER),
            "crop_dir_exists":    crop_dir_exists,
            "done_exists":        done_exists,
            "existing_crops":     existing_crops,
            "collision_detected": collision,
        },
        "disk_space": {
            "free_gb": round(disk_free_gb, 1),
            "need_gb": round(need_gb, 2),
            "ok":      disk_ok,
        },
        "file_availability_sample": {
            "n_sampled": len(sample_patients),
            "missing_ct": missing_ct,
            "missing_roi": missing_roi,
            "missing_mask": missing_mask,
        },
        "resume": {
            "already_done": len(resume_set),
            "remaining": remaining,
            "resume_possible": resume_possible,
        },
        "z_policy": {
            "crop_z_basis": "local_z",
            "slice_index_crop_access": "금지",
            "local_z_range": lz_range,
            "slice_index_range": si_range,
        },
        "crop_format": {
            "crop_size": CROP_SIZE, "n_channels": N_CHANNELS,
            "ct_dtype": "int16", "roi_dtype": "uint8", "mask_dtype": "uint8",
            "padding": "reflect (edge fallback)",
        },
        "label_policy": {
            "policy": "Option B",
            "center_patch_label_kept": True,
            "warning_flags": ["mask_nonzero_warning", "center_mask_nonzero", "adjacent_mask_nonzero"],
        },
        "guardrails": {
            "full_crop_generated":       False,
            "training_executed":         False,
            "model_forward":             False,
            "scoring_rerun":             False,
            "stage2_holdout_accessed":   False,
            "existing_results_modified": False,
        },
        "p_c9_readiness": {
            "ready": verdict == "통과",
            "run_command": "source ~/ai_env/bin/activate && python3 experiments/efficientnet_b0_v4_20_second_stage_refiner_v1/p_c8_full_crop_generation.py --full-run",
            "resume_command": "source ~/ai_env/bin/activate && python3 experiments/efficientnet_b0_v4_20_second_stage_refiner_v1/p_c8_full_crop_generation.py --full-run --resume",
        },
        "blockers": blockers,
        "n_errors": len(blockers),
    }

    with open(DRYCHECK_DIR / "p_c8_full_crop_generation_drycheck.json", "w") as f:
        json.dump(dc_json, f, indent=2, ensure_ascii=False, default=str)

    md_lines = [
        "# P-C8 Full Crop Generation Dry-Check Report",
        "",
        f"**판정: {verdict}**",
        f"생성일시: {now_str}",
        "",
        "---",
        "",
        "## 1. 입력 검증",
        "",
        "| 단계 | 판정 |",
        "|------|------|",
        f"| P-C7 | {p_c7['verdict']} |",
        f"| P-C6 | {p_c6['verdict']} |",
        f"| P-C3 | {p_c3['verdict']} |",
        "",
        "## 2. manifest 검증",
        "",
        "| 항목 | 값 |",
        "|------|----|",
        f"| rows | {manifest_rows} / 114,381 |",
        f"| positive | {n_pos_manifest:,} |",
        f"| hard_negative | {n_hn_manifest:,} |",
        f"| patients | {n_patients} |",
        f"| missing cols | {missing_cols or '없음'} |",
        f"| coord null | {coord_null} |",
        f"| holdout contamination | {holdout_cont} |",
        "",
        "## 3. output path",
        "",
        f"- crop dir: `{CROP_DIR}` → exists={crop_dir_exists}, existing={existing_crops}개",
        f"- DONE marker: {done_exists}",
        f"- collision: {collision}",
        "",
        "## 4. disk space",
        "",
        f"- free: **{disk_free_gb:.1f} GB**, need≈{need_gb:.1f} GB → {'OK' if disk_ok else 'WARNING'}",
        "",
        "## 5. CT/ROI/mask 파일 존재 (샘플 10명)",
        "",
        f"- CT missing: {missing_ct}/10",
        f"- ROI missing: {missing_roi}/10",
        f"- mask missing: {missing_mask}/10",
        "",
        "## 6. z축 policy 확정",
        "",
        "- **crop z 기준: `local_z`** (확정)",
        "- **`slice_index`: crop 접근 금지** (global z, OOB 발생 확인)",
        f"- local_z range: {lz_range}",
        "",
        "## 7. crop format 확정",
        "",
        "| 항목 | 값 |",
        "|------|----|",
        "| crop_size | 96px |",
        "| channels | 3 (z-1/z/z+1) |",
        "| CT dtype | int16 |",
        "| ROI/mask dtype | uint8 |",
        "| padding | reflect (edge fallback) |",
        "",
        "## 8. label policy 확정",
        "",
        "**Option B** — center patch 기준 `candidate_label` 유지 + warning flag 3종",
        "",
        "| flag | 정의 |",
        "|------|------|",
        "| `mask_nonzero_warning` | crop 3채널 전체 mask.any() |",
        "| `center_mask_nonzero` | mask[1].any() (center slice) |",
        "| `adjacent_mask_nonzero` | mask[0].any() or mask[2].any() |",
        "",
        "## 9. resume 구조",
        "",
        f"- 이미 완료: {len(resume_set)}개",
        f"- 남은 작업: {remaining:,}개",
        "- skip 기준: crop dir에 `{candidate_id}.npz` 존재 시 skip",
        "- labels CSV: append 방식, 재개 시 기존 candidate_id set 확인",
        "- DONE marker: 전체 완료 후에만 생성",
        "",
        "## 10. guardrails 확인",
        "",
        "| 항목 | 확인 |",
        "|------|------|",
        "| full crop 미생성 | True |",
        "| 2차학습 없음 | True |",
        "| model forward 없음 | True |",
        "| scoring 재실행 없음 | True |",
        "| stage2_holdout 미접근 | True |",
        "| 기존 결과 무수정 | True |",
        "",
        "## 11. blockers",
        "",
        str(blockers) if blockers else "없음 — P-C9 (full-run) 진행 가능",
        "",
        "## 12. P-C9 실행 명령",
        "",
        "```bash",
        "source ~/ai_env/bin/activate && python3 experiments/efficientnet_b0_v4_20_second_stage_refiner_v1/p_c8_full_crop_generation.py --full-run",
        "```",
        "",
        "재개 시:",
        "```bash",
        "source ~/ai_env/bin/activate && python3 experiments/efficientnet_b0_v4_20_second_stage_refiner_v1/p_c8_full_crop_generation.py --full-run --resume",
        "```",
    ]

    with open(DRYCHECK_DIR / "p_c8_full_crop_generation_drycheck.md", "w") as f:
        f.write("\n".join(md_lines))

    print(f"\n[dry-check 완료] 결과: {DRYCHECK_DIR}")
    print(f"  JSON: p_c8_full_crop_generation_drycheck.json")
    print(f"  MD:   p_c8_full_crop_generation_drycheck.md")


# ============================================================
# FULL-RUN 모드
# ============================================================
def run_fullrun(resume: bool):
    now_str = datetime.datetime.now().isoformat(timespec="seconds")
    t_start = datetime.datetime.now()

    print("=" * 60)
    print("P-C8 FULL-RUN 모드")
    print("=" * 60)

    # 가드
    p_c7, p_c6, p_c3, n_total, n_pos, n_hn = load_verdicts()
    holdout_set, stage1_set = load_stage2_holdout()
    df = load_manifest(holdout_set)
    check_output_collision(is_resume=resume, is_fullrun=True)

    CROP_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # resume set
    resume_set = get_resume_set() if resume else set()
    print(f"  resume: {len(resume_set)}개 skip")

    # manifest copy
    import shutil as _sh
    _sh.copy2(MANIFEST_CSV, MANIFEST_COPY)

    # labels CSV 준비
    label_file_exists = LABELS_CSV.exists()
    label_f  = open(LABELS_CSV, "a", newline="")
    label_w  = csv.DictWriter(label_f, fieldnames=LABEL_CSV_COLS)
    if not label_file_exists:
        label_w.writeheader()

    # error CSV
    error_f  = open(ERROR_CSV, "a", newline="")
    error_w  = csv.DictWriter(error_f, fieldnames=["candidate_id","patient_id","error"])
    if not (ERROR_CSV.exists() and os.path.getsize(ERROR_CSV) > 0):
        error_w.writeheader()

    # integrity CSV
    integ_file_exists = INTEGRITY_CSV.exists()
    integ_f = open(INTEGRITY_CSV, "a", newline="")
    integ_w = csv.DictWriter(integ_f, fieldnames=INTEGRITY_CSV_COLS)
    if not integ_file_exists:
        integ_w.writeheader()

    # 환자별 묶기
    groups = df.groupby("safe_id")

    generated   = 0
    skipped     = 0
    n_errors    = 0
    warn_rows   = []

    for sid, pat_rows in groups:
        # stage2_holdout 이중 차단
        if pat_rows["patient_id"].iloc[0] in holdout_set:
            for _, row in pat_rows.iterrows():
                error_w.writerow({"candidate_id": row["candidate_id"],
                                   "patient_id": row["patient_id"],
                                   "error": "BLOCKED: stage2_holdout"})
            continue

        # CT/ROI/mask 로드
        ct_path   = CT_ROOT  / sid / "ct_hu.npy"
        mask_path = CT_ROOT  / sid / "lesion_mask_roi_0_0.npy"
        meta_path = CT_ROOT  / sid / "meta.json"
        roi_path  = ROI_ROOT / sid / "refined_roi.npy"

        for fp, name in [(ct_path,"ct"),(mask_path,"mask"),(meta_path,"meta"),(roi_path,"roi")]:
            if not fp.exists():
                err_msg = f"missing {name}: {fp}"
                for _, row in pat_rows.iterrows():
                    error_w.writerow({"candidate_id": row["candidate_id"],
                                       "patient_id": row["patient_id"],
                                       "error": err_msg})
                    integ_w.writerow({"candidate_id": row["candidate_id"],
                                      "crop_exists": False, "crop_shape": "",
                                      "ct_nan": "", "ct_inf": "",
                                      "roi_binary_valid": "", "mask_binary_valid": "",
                                      "mask_nonzero_warning": "", "center_mask_nonzero": "",
                                      "adjacent_mask_nonzero": "", "mask_consistency": "",
                                      "pad_used": "", "error": err_msg})
                n_errors += len(pat_rows)
                break
        else:
            ct_vol   = np.load(ct_path,   mmap_mode="r")
            mask_vol = np.load(mask_path, mmap_mode="r")
            roi_vol  = np.load(roi_path,  mmap_mode="r")

            if not (ct_vol.shape == mask_vol.shape == roi_vol.shape):
                err_msg = f"shape mismatch ct={ct_vol.shape} mask={mask_vol.shape} roi={roi_vol.shape}"
                for _, row in pat_rows.iterrows():
                    error_w.writerow({"candidate_id": row["candidate_id"],
                                       "patient_id": row["patient_id"],
                                       "error": err_msg})
                    integ_w.writerow({"candidate_id": row["candidate_id"],
                                      "crop_exists": False, "crop_shape": "",
                                      "ct_nan": "", "ct_inf": "",
                                      "roi_binary_valid": "", "mask_binary_valid": "",
                                      "mask_nonzero_warning": "", "center_mask_nonzero": "",
                                      "adjacent_mask_nonzero": "", "mask_consistency": "",
                                      "pad_used": "", "error": err_msg})
                n_errors += len(pat_rows)
                continue

            for _, row in pat_rows.iterrows():
                cid = row["candidate_id"]

                # resume skip
                if cid in resume_set:
                    skipped += 1
                    continue

                local_z = int(row["local_z"])
                y0i = int(row["y0"]); y1i = int(row["y1"])
                x0i = int(row["x0"]); x1i = int(row["x1"])
                cy  = (y0i + y1i) // 2
                cx  = (x0i + x1i) // 2

                try:
                    # volume 전체 astype 금지: crop 추출 후 crop에만 변환
                    ct_crop,   pad_ct  = extract_crop_3ch(ct_vol,   local_z, cy, cx)
                    mask_crop, _       = extract_crop_3ch(mask_vol, local_z, cy, cx)
                    roi_crop,  _       = extract_crop_3ch(roi_vol,  local_z, cy, cx)
                    ct_crop   = ct_crop.astype(np.int16,  copy=False)
                    mask_crop = mask_crop.astype(np.uint8, copy=False)
                    roi_crop  = roi_crop.astype(np.uint8,  copy=False)
                except ValueError as e:
                    err_msg = str(e)
                    error_w.writerow({"candidate_id": cid, "patient_id": row["patient_id"], "error": err_msg})
                    integ_w.writerow({"candidate_id": cid,
                                      "crop_exists": False, "crop_shape": "",
                                      "ct_nan": "", "ct_inf": "",
                                      "roi_binary_valid": "", "mask_binary_valid": "",
                                      "mask_nonzero_warning": "", "center_mask_nonzero": "",
                                      "adjacent_mask_nonzero": "", "mask_consistency": "",
                                      "pad_used": "", "error": err_msg})
                    n_errors += 1
                    continue

                # 검증
                if ct_crop.shape != EXPECTED_SHAPE:
                    err_msg = f"shape {ct_crop.shape} != {EXPECTED_SHAPE}"
                    error_w.writerow({"candidate_id": cid, "patient_id": row["patient_id"],
                                       "error": err_msg})
                    integ_w.writerow({"candidate_id": cid,
                                      "crop_exists": False, "crop_shape": str(ct_crop.shape),
                                      "ct_nan": "", "ct_inf": "",
                                      "roi_binary_valid": "", "mask_binary_valid": "",
                                      "mask_nonzero_warning": "", "center_mask_nonzero": "",
                                      "adjacent_mask_nonzero": "", "mask_consistency": "",
                                      "pad_used": pad_ct, "error": err_msg})
                    n_errors += 1
                    continue

                ct_f   = ct_crop.astype(np.float32)
                ct_nan = int(np.isnan(ct_f).sum())
                ct_inf = int(np.isinf(ct_f).sum())

                # warning flags
                mask_any, center_nz, adjacent_nz = compute_mask_flags(mask_crop)
                label = row["candidate_label"]
                if label == "hard_negative" and mask_any:
                    warn_rows.append({
                        "candidate_id": cid, "patient_id": row["patient_id"],
                        "candidate_rule": row["candidate_rule"],
                        "mask_nonzero_warning": mask_any,
                        "center_mask_nonzero": center_nz,
                        "adjacent_mask_nonzero": adjacent_nz,
                    })

                mask_consistency = "ok"
                if label == "positive" and not mask_any:
                    mask_consistency = "warn_pos_mask_zero"
                elif label == "hard_negative" and mask_any:
                    mask_consistency = "warning_hn_mask_nonzero"

                # npz 저장
                npz_path = CROP_DIR / f"{cid}.npz"
                np.savez_compressed(
                    npz_path,
                    ct_crop=ct_crop.astype(np.int16),
                    roi_crop=roi_crop.astype(np.uint8),
                    mask_crop=mask_crop.astype(np.uint8),
                    candidate_id=np.array([cid]),
                    patient_id=np.array([row["patient_id"]]),
                    safe_id=np.array([sid]),
                    candidate_label=np.array([label]),
                    candidate_rule=np.array([row["candidate_rule"]]),
                    local_z=np.array([local_z]),
                    slice_index=np.array([int(row["slice_index"])]),
                    y0=np.array([y0i]), x0=np.array([x0i]),
                    y1=np.array([y1i]), x1=np.array([x1i]),
                    padim_score=np.array([float(row["padim_score"])]),
                    mask_nonzero_warning=np.array([mask_any]),
                    center_mask_nonzero=np.array([center_nz]),
                    adjacent_mask_nonzero=np.array([adjacent_nz]),
                )

                # label CSV append
                label_w.writerow({
                    "candidate_id":   cid,
                    "crop_path":      str(npz_path.relative_to(WORKSPACE)),
                    "patient_id":     row["patient_id"],
                    "safe_id":        sid,
                    "candidate_label": label,
                    "candidate_rule": row["candidate_rule"],
                    "local_z":        local_z,
                    "slice_index":    int(row["slice_index"]),
                    "y0": y0i, "x0": x0i, "y1": y1i, "x1": x1i,
                    "padim_score":    float(row["padim_score"]),
                    "mask_nonzero_warning":  mask_any,
                    "center_mask_nonzero":   center_nz,
                    "adjacent_mask_nonzero": adjacent_nz,
                    "no_hit_patient": row.get("no_hit_patient", False),
                    "tiny_lesion_flag": row.get("tiny_lesion_flag", False),
                    "p_b3_risk6_flag":  row.get("p_b3_risk6_flag", False),
                    "fallback_positive_below_p95": row.get("fallback_positive_below_p95", False),
                    "source_branch":  row.get("source_branch", ""),
                    "crop_shape":     str(ct_crop.shape),
                    "ct_nan":         ct_nan,
                    "ct_inf":         ct_inf,
                    "mask_consistency": mask_consistency,
                    "pad_used":       pad_ct,
                })

                # integrity CSV append
                integ_w.writerow({
                    "candidate_id":         cid,
                    "crop_exists":          True,
                    "crop_shape":           str(ct_crop.shape),
                    "ct_nan":               ct_nan,
                    "ct_inf":               ct_inf,
                    "roi_binary_valid":     bool(np.isin(roi_crop, [0, 1]).all()),
                    "mask_binary_valid":    bool(np.isin(mask_crop, [0, 1]).all()),
                    "mask_nonzero_warning": mask_any,
                    "center_mask_nonzero":  center_nz,
                    "adjacent_mask_nonzero": adjacent_nz,
                    "mask_consistency":     mask_consistency,
                    "pad_used":             pad_ct,
                    "error":                "",
                })

                generated += 1

                if generated % 5000 == 0:
                    elapsed = (datetime.datetime.now() - t_start).total_seconds()
                    print(f"  [{generated}/{n_total}] elapsed={elapsed:.0f}s, errors={n_errors}")

    label_f.close()
    error_f.close()
    integ_f.close()

    # warning summary: integrity CSV 전체 기준으로 재생성 (resume 포함 전체 crop 반영)
    # warn_rows는 현재 실행분만 담으므로, integrity CSV를 읽어 mask_nonzero_warning==True 행으로 대체
    warn_from_integ = []
    if INTEGRITY_CSV.exists():
        with open(INTEGRITY_CSV, newline="") as f_integ:
            for irow in csv.DictReader(f_integ):
                if str(irow.get("mask_nonzero_warning", "")).lower() in ("true", "1"):
                    warn_from_integ.append({
                        "candidate_id":          irow.get("candidate_id", ""),
                        "patient_id":            "",   # integrity CSV에 patient_id 없음 → labels CSV join 생략
                        "candidate_rule":        "",
                        "mask_nonzero_warning":  irow.get("mask_nonzero_warning", ""),
                        "center_mask_nonzero":   irow.get("center_mask_nonzero", ""),
                        "adjacent_mask_nonzero": irow.get("adjacent_mask_nonzero", ""),
                    })
    # patient_id / candidate_rule 보강: labels CSV join
    if warn_from_integ and LABELS_CSV.exists():
        label_lookup = {}
        with open(LABELS_CSV, newline="") as f_lbl:
            for lrow in csv.DictReader(f_lbl):
                label_lookup[lrow["candidate_id"]] = {
                    "patient_id":     lrow.get("patient_id", ""),
                    "candidate_rule": lrow.get("candidate_rule", ""),
                }
        for wr in warn_from_integ:
            info = label_lookup.get(wr["candidate_id"], {})
            wr["patient_id"]     = info.get("patient_id", "")
            wr["candidate_rule"] = info.get("candidate_rule", "")
    with open(WARN_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=WARN_CSV_COLS)
        w.writeheader()
        if warn_from_integ:
            w.writerows(warn_from_integ)

    elapsed_total = (datetime.datetime.now() - t_start).total_seconds()
    total_done    = generated + skipped

    print(f"\n[완료] generated={generated}, skipped={skipped}, errors={n_errors}, total={total_done}")

    # 최종 카운트 검증
    crop_count_final  = len(list(CROP_DIR.glob("*.npz")))
    label_count_final = (sum(1 for _ in open(LABELS_CSV)) - 1) if LABELS_CSV.exists() else 0
    integ_count_final = (sum(1 for _ in open(INTEGRITY_CSV)) - 1) if INTEGRITY_CSV.exists() else 0
    print(f"  npz={crop_count_final}, labels_csv={label_count_final}, integrity_csv={integ_count_final}")

    # DONE 조건 5가지 강화
    done_conditions = {
        "generated_plus_skipped_eq_total": (generated + skipped) == n_total,
        "errors_zero":            n_errors == 0,
        "crop_npz_count_ok":      crop_count_final == n_total,
        "labels_csv_count_ok":    label_count_final == n_total,
        "integrity_csv_count_ok": integ_count_final == n_total,
    }
    all_done = all(done_conditions.values())
    print(f"  DONE 조건: {done_conditions}")

    verdict_full = "통과" if all_done else "부분통과"

    # summary JSON
    summary = {
        "step": "P-C8",
        "verdict": verdict_full,
        "created": now_str,
        "elapsed_seconds": round(elapsed_total, 1),
        "generated":  generated,
        "skipped":    skipped,
        "n_errors":   n_errors,
        "total_done": total_done,
        "expected":   n_total,
        "crop_count_final":  crop_count_final,
        "label_count_final": label_count_final,
        "integ_count_final": integ_count_final,
        "hn_warn_count": len(warn_from_integ),
        "done_conditions": done_conditions,
        "guardrails": {
            "full_crop_generated":       True,
            "training_executed":         False,
            "stage2_holdout_accessed":   False,
            "existing_results_modified": False,
        },
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # DONE marker — 5가지 조건 모두 통과 시에만
    if all_done:
        with open(DONE_MARKER, "w") as f:
            json.dump({
                "done": True,
                "generated":     generated,
                "skipped":       skipped,
                "n_errors":      n_errors,
                "crop_count":    crop_count_final,
                "label_count":   label_count_final,
                "integ_count":   integ_count_final,
                "elapsed_seconds": round(elapsed_total, 1),
                "created": now_str,
                "done_conditions": done_conditions,
            }, f, indent=2)
        print("  DONE.json 생성 완료")
    else:
        print(f"  [WARN] DONE 미생성: {done_conditions}")

    # report.md 생성
    md_lines = [
        "# P-C8 Full Crop Generation Report",
        "",
        f"**판정: {verdict_full}**",
        f"생성일시: {now_str}",
        "",
        "## 결과 요약",
        "",
        "| 항목 | 값 |",
        "|------|----|",
        f"| generated | {generated:,} |",
        f"| skipped | {skipped:,} |",
        f"| errors | {n_errors} |",
        f"| crop npz count | {crop_count_final:,} |",
        f"| labels CSV count | {label_count_final:,} |",
        f"| integrity CSV count | {integ_count_final:,} |",
        f"| expected | {n_total:,} |",
        f"| elapsed | {round(elapsed_total, 1)}s |",
        f"| DONE | {all_done} |",
        "",
        "## DONE 조건",
        "",
    ] + [f"- {k}: {v}" for k, v in done_conditions.items()] + [
        "",
        "## guardrails",
        "",
        "| 항목 | 확인 |",
        "|------|------|",
        "| training_executed | False |",
        "| stage2_holdout_accessed | False |",
        "| existing_results_modified | False |",
    ]
    with open(REPORT_MD, "w") as f:
        f.write("\n".join(md_lines))
    print(f"  report.md 생성: {REPORT_MD}")


# ============================================================
# ENTRY POINT
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="P-C8 full crop generation")
    parser.add_argument("--full-run", action="store_true",
                        help="실제 crop 생성 활성화 (기본: dry-check 모드)")
    parser.add_argument("--resume", action="store_true",
                        help="기존 crop skip하고 이어서 생성")
    args = parser.parse_args()

    if not args.full_run:
        run_drycheck()
    else:
        run_fullrun(resume=args.resume)


if __name__ == "__main__":
    main()
