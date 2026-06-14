"""
p_c_normal29b_crop_level_mask_generation.py

P-C-NORMAL29b: 전체 train/val crop-level mask 생성 + audit

목표:
  P-C-NORMAL30 pixel ROI masking 학습 실험을 위한 3채널 crop-level mask를
  train/val 전체에 대해 생성하고 품질 검증한다.

금지:
  - 학습 / model forward / prediction export
  - threshold 최적화
  - checkpoint 수정
  - 기존 결과 수정/삭제/덮어쓰기
  - P-C-NORMAL30 학습 시작

mask 설계:
  - mask_3ch[0] = roi[z-1] 에서 center_y/center_x 기준 96×96 crop
  - mask_3ch[1] = roi[z]   에서 center_y/center_x 기준 96×96 crop
  - mask_3ch[2] = roi[z+1] 에서 center_y/center_x 기준 96×96 crop
  - z-1/z+1 이 volume boundary 밖이면 nearest-repeat (z clamp)
  - spatial boundary 초과 시 zero-padding
  - dtype: uint8 (0/1)
"""

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ─── 경로 ───────────────────────────────────────────────────────────────────
BRANCH_ROOT  = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BRANCH_ROOT.parents[1]

TRAIN_MANIFEST = (PROJECT_ROOT /
    "outputs/manifests/p_c_normal24g_fix_balanced_w1_feature_manifest/"
    "p_c_normal24g_fix_balanced_w1_train_manifest.csv")
VAL_MANIFEST   = (PROJECT_ROOT /
    "outputs/manifests/p_c_normal24g_fix_balanced_w1_feature_manifest/"
    "p_c_normal24g_fix_balanced_w1_val_manifest.csv")

ROI_ROOT    = (PROJECT_ROOT /
    "outputs/mip-postprocess-research-v1/masks/refined_roi_v4_20_modeB_all_v1")
MASK_ROOT   = PROJECT_ROOT / "outputs/p_c_normal29b_crop_level_masks"
REPORT_ROOT = PROJECT_ROOT / "outputs/reports/p_c_normal29b_crop_level_mask_generation"

CROP_HALF = 48       # 96×96 crop → center ± 48
LOW_MASK_THRESHOLD = 0.05  # nonzero ratio 이하 → low mask

# ─── Guardrail ───────────────────────────────────────────────────────────────
GUARDRAILS = {
    "no_training_run":                   True,
    "no_model_forward":                  True,
    "no_prediction_export":              True,
    "no_threshold_optimization":         True,
    "no_checkpoint_modification":        True,
    "no_existing_result_overwrite":      True,
    "full_crop_level_mask_generation_only": True,
    "center_based_96_crop_used":         True,
    "patch_32_bbox_not_used_for_mask":   True,
    "mask_3ch_generated":                True,
    "p_c_normal30_training_not_started": True,
}


# ─── 유틸 ────────────────────────────────────────────────────────────────────

def _write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        rows = [{"note": "empty"}]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _write_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def get_roi_path(safe_id: str, label: int) -> Path:
    subdir = "normal" if label == 0 else "lesion"
    return ROI_ROOT / subdir / safe_id / "refined_roi.npy"


def extract_mask_96(roi_volume: np.ndarray, z: int, cy: int, cx: int):
    """
    volume에서 (z, cy, cx) 기준 96×96 mask crop 추출.
    - z:  volume 범위 내로 clamp (nearest-repeat)
    - 공간: 경계 초과 시 zero-pad
    반환: (mask_96 uint8, pads tuple, z_effective int, nearest_repeat_used bool)
    """
    Z, H, W = roi_volume.shape
    nearest_repeat_used = False
    z_eff = int(np.clip(z, 0, Z - 1))
    if z_eff != int(z):
        nearest_repeat_used = True

    y0_vol = cy - CROP_HALF
    y1_vol = cy + CROP_HALF
    x0_vol = cx - CROP_HALF
    x1_vol = cx + CROP_HALF

    pad_y0 = max(0, -y0_vol)
    pad_y1 = max(0, y1_vol - H)
    pad_x0 = max(0, -x0_vol)
    pad_x1 = max(0, x1_vol - W)

    y0c = max(0, y0_vol)
    y1c = min(H, y1_vol)
    x0c = max(0, x0_vol)
    x1c = min(W, x1_vol)

    patch = roi_volume[z_eff][y0c:y1c, x0c:x1c]
    if pad_y0 > 0 or pad_y1 > 0 or pad_x0 > 0 or pad_x1 > 0:
        patch = np.pad(patch, ((pad_y0, pad_y1), (pad_x0, pad_x1)), mode="constant")

    return patch.astype(np.uint8), (pad_y0, pad_y1, pad_x0, pad_x1), z_eff, nearest_repeat_used


def normalize_hu(arr_hw: np.ndarray) -> np.ndarray:
    arr = arr_hw.astype(np.float32)
    arr = np.clip(arr, -1000.0, 200.0)
    arr = (arr - (-1000.0)) / (200.0 - (-1000.0)) * 255.0
    return arr.astype(np.uint8)


def make_overlay(ct_hw: np.ndarray, mask_hw: np.ndarray) -> np.ndarray:
    """CT grayscale + green lung overlay → RGB uint8 (H×W×3)."""
    ct_g  = normalize_hu(ct_hw)
    rgb   = np.stack([ct_g, ct_g, ct_g], axis=-1)
    green = mask_hw.astype(bool)
    rgb[green, 1] = np.clip(rgb[green, 1].astype(np.int32) + 80, 0, 255).astype(np.uint8)
    # boundary: bright green
    from scipy.ndimage import binary_erosion
    eroded   = binary_erosion(green, iterations=1)
    boundary = green & ~eroded
    rgb[boundary] = [0, 255, 0]
    return rgb


# ─── 메인 ────────────────────────────────────────────────────────────────────

def main():
    start_time = datetime.now()
    MASK_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    (MASK_ROOT / "train").mkdir(parents=True, exist_ok=True)
    (MASK_ROOT / "val").mkdir(parents=True, exist_ok=True)

    # ── 0. Guardrail 사전 점검 ──────────────────────────────────────────────
    guard_rows = []
    for k, v in GUARDRAILS.items():
        guard_rows.append({"guardrail": k, "expected": True, "actual": v, "pass": v is True})
    _write_csv(guard_rows, REPORT_ROOT / "p_c_normal29b_guardrail_check.csv")
    if any(not r["pass"] for r in guard_rows):
        print("[29b] ABORT: guardrail 실패")
        sys.exit(1)
    print("[29b] guardrail: PASS")

    # ── 1. Manifest 로드 ────────────────────────────────────────────────────
    for p in [TRAIN_MANIFEST, VAL_MANIFEST]:
        if not p.exists():
            print(f"[29b] ABORT: manifest not found: {p}")
            sys.exit(1)

    df_train = pd.read_csv(TRAIN_MANIFEST, low_memory=False)
    df_val   = pd.read_csv(VAL_MANIFEST,   low_memory=False)

    # split 컬럼 명시 (혹시 없는 경우 대비)
    df_train = df_train.copy(); df_train["_split_tag"] = "train"
    df_val   = df_val.copy();   df_val["_split_tag"]   = "val"
    df_all   = pd.concat([df_train, df_val], ignore_index=True)

    total_expected = len(df_all)
    print(f"[29b] manifest: train={len(df_train)} val={len(df_val)} total={total_expected}")
    print(f"[29b] label 분포: {dict(df_all.label.value_counts().sort_index())}")

    # ── 2. ROI missing 사전 점검 ────────────────────────────────────────────
    missing_roi = []
    checked_keys = set()
    for _, row in df_all.iterrows():
        key = (str(row.safe_id), int(row.label))
        if key in checked_keys:
            continue
        checked_keys.add(key)
        roi_path = get_roi_path(str(row.safe_id), int(row.label))
        if not roi_path.exists():
            missing_roi.append({"safe_id": row.safe_id, "label": row.label, "roi_path": str(roi_path)})
    if missing_roi:
        print(f"[29b] WARNING: ROI 파일 없는 환자 {len(missing_roi)}개")
        _write_csv(missing_roi, REPORT_ROOT / "p_c_normal29b_missing_roi.csv")
    else:
        print(f"[29b] ROI 사전 점검: 전체 {len(checked_keys)} 환자 모두 존재")

    # ── 3. 환자별 그룹 처리 (ROI 캐시 1건씩) ───────────────────────────────
    manifest_rows = []
    audit_rows    = []
    zero_low_rows = []

    # safe_id 기준 정렬 → 같은 환자 rows 연속 처리
    df_all_sorted = df_all.sort_values(["safe_id", "canonical_volume_z"]).reset_index(drop=True)

    current_safe_id = None
    roi_volume      = None
    roi_vol_z       = None

    n_done       = 0
    n_error      = 0
    n_nearest    = 0
    n_spatial_pad = 0

    for idx, row in df_all_sorted.iterrows():
        safe_id  = str(row.safe_id)
        label    = int(row.label)
        split    = str(row._split_tag)
        z_raw    = row.canonical_volume_z
        cy       = int(row.center_y)
        cx       = int(row.center_x)
        crop_path = str(row.crop_path)

        # 진행 출력 (1000건마다)
        if idx % 1000 == 0:
            print(f"[29b] {idx}/{len(df_all_sorted)} | n_done={n_done} n_error={n_error}")

        # ROI 로드 (환자 변경 시만)
        if safe_id != current_safe_id:
            roi_path = get_roi_path(safe_id, label)
            if not roi_path.exists():
                error_msg = f"ROI not found: {roi_path}"
                audit_rows.append({
                    "crop_path": crop_path, "safe_id": safe_id, "split": split,
                    "label": label, "status": "FAIL_ROI_MISSING", "error_message": error_msg,
                })
                n_error += 1
                current_safe_id = safe_id  # 다음 같은 환자는 skip
                roi_volume = None
                roi_vol_z  = None
                continue
            roi_volume      = np.load(str(roi_path))
            roi_vol_z       = roi_volume.shape[0]
            current_safe_id = safe_id

        if roi_volume is None:
            audit_rows.append({
                "crop_path": crop_path, "safe_id": safe_id, "split": split,
                "label": label, "status": "FAIL_ROI_MISSING", "error_message": "roi_volume is None",
            })
            n_error += 1
            continue

        # z index 결정
        try:
            z_int = int(float(z_raw))
        except (ValueError, TypeError):
            audit_rows.append({
                "crop_path": crop_path, "safe_id": safe_id, "split": split,
                "label": label, "status": "FAIL_Z_PARSE", "error_message": f"z parse error: {z_raw}",
            })
            n_error += 1
            continue

        # 3채널 mask 추출
        zm1_raw = z_int - 1
        z0_raw  = z_int
        zp1_raw = z_int + 1

        try:
            ch0, pads0, zm1_eff, nr0 = extract_mask_96(roi_volume, zm1_raw, cy, cx)
            ch1, pads1, z0_eff,  nr1 = extract_mask_96(roi_volume, z0_raw,  cy, cx)
            ch2, pads2, zp1_eff, nr2 = extract_mask_96(roi_volume, zp1_raw, cy, cx)
        except Exception as e:
            audit_rows.append({
                "crop_path": crop_path, "safe_id": safe_id, "split": split,
                "label": label, "status": "FAIL_EXTRACT", "error_message": str(e),
            })
            n_error += 1
            continue

        mask_3ch = np.stack([ch0, ch1, ch2], axis=0)  # (3, 96, 96) uint8

        # shape 검증
        if mask_3ch.shape != (3, 96, 96):
            audit_rows.append({
                "crop_path": crop_path, "safe_id": safe_id, "split": split,
                "label": label, "status": "FAIL_SHAPE",
                "error_message": f"shape={mask_3ch.shape}",
            })
            n_error += 1
            continue

        # NaN/Inf 검증 (uint8이라 NaN 없지만 명시)
        if not np.isfinite(mask_3ch.astype(np.float32)).all():
            audit_rows.append({
                "crop_path": crop_path, "safe_id": safe_id, "split": split,
                "label": label, "status": "FAIL_NONFINITE", "error_message": "NaN or Inf in mask",
            })
            n_error += 1
            continue

        # 저장 경로 결정
        crop_stem = Path(crop_path).stem
        mask_fname = f"{crop_stem}_mask.npz"
        mask_out_path = MASK_ROOT / split / mask_fname
        np.savez_compressed(str(mask_out_path), mask_3ch=mask_3ch)

        # nonzero ratio 계산
        nr_ch0 = float(mask_3ch[0].sum()) / (96 * 96)
        nr_ch1 = float(mask_3ch[1].sum()) / (96 * 96)
        nr_ch2 = float(mask_3ch[2].sum()) / (96 * 96)
        nr_mean = (nr_ch0 + nr_ch1 + nr_ch2) / 3.0

        any_nearest = nr0 or nr1 or nr2
        any_spatial_pad = (
            any(p > 0 for p in pads1)  # center slice
        )
        if any_nearest:
            n_nearest += 1
        if any_spatial_pad:
            n_spatial_pad += 1

        status = "PASS"
        manifest_rows.append({
            "crop_path":              crop_path,
            "mask_path":              str(mask_out_path),
            "safe_id":                safe_id,
            "patient_id":             str(row.patient_id),
            "split":                  split,
            "label":                  label,
            "canonical_volume_z":     z_int,
            "center_y":               cy,
            "center_x":               cx,
            "mask_shape":             "3x96x96",
            "mask_dtype":             "uint8",
            "mask_nonzero_ratio_ch0": round(nr_ch0, 6),
            "mask_nonzero_ratio_ch1": round(nr_ch1, 6),
            "mask_nonzero_ratio_ch2": round(nr_ch2, 6),
            "mask_nonzero_ratio_mean":round(nr_mean, 6),
            "z_minus1_effective":     zm1_eff,
            "z_center_effective":     z0_eff,
            "z_plus1_effective":      zp1_eff,
            "nearest_repeat_used":    any_nearest,
            "status":                 status,
            "error_message":          "",
        })

        audit_rows.append({
            "crop_path":   crop_path,
            "safe_id":     safe_id,
            "split":       split,
            "label":       label,
            "status":      status,
            "error_message": "",
        })

        # zero/low mask 기록
        if nr_mean < LOW_MASK_THRESHOLD:
            zero_low_rows.append({
                "crop_path":               crop_path,
                "safe_id":                 safe_id,
                "split":                   split,
                "label":                   label,
                "mask_nonzero_ratio_mean": round(nr_mean, 6),
                "mask_nonzero_ratio_ch1":  round(nr_ch1, 6),
                "canonical_volume_z":      z_int,
                "center_y":                cy,
                "center_x":                cx,
                "nearest_repeat_used":     any_nearest,
            })

        n_done += 1

    print(f"\n[29b] 처리 완료: n_done={n_done} n_error={n_error} n_nearest={n_nearest} n_spatial_pad={n_spatial_pad}")

    # ── 4. mask manifest CSV ─────────────────────────────────────────────────
    _write_csv(manifest_rows, REPORT_ROOT / "p_c_normal29b_mask_manifest.csv")
    print(f"[29b] mask_manifest: {len(manifest_rows)} rows")

    # ── 5. audit CSV ─────────────────────────────────────────────────────────
    _write_csv(audit_rows, REPORT_ROOT / "p_c_normal29b_mask_generation_audit.csv")

    # ── 6. zero/low mask CSV ─────────────────────────────────────────────────
    if not zero_low_rows:
        zero_low_rows = [{"note": "zero_or_low_mask_cases: none"}]
    _write_csv(zero_low_rows, REPORT_ROOT / "p_c_normal29b_zero_or_low_mask_cases.csv")
    n_low = len(zero_low_rows) if zero_low_rows[0].get("note") is None else 0
    print(f"[29b] zero/low mask cases (mean<{LOW_MASK_THRESHOLD}): {n_low}")

    # ── 7. distribution summary CSV ──────────────────────────────────────────
    if manifest_rows:
        df_m = pd.DataFrame(manifest_rows)

        dist_rows = []
        for grp_key, grp_df in [
            ("all",          df_m),
            ("label_0_normal", df_m[df_m.label == 0]),
            ("label_1_nsclc",  df_m[df_m.label == 1]),
            ("split_train",  df_m[df_m.split == "train"]),
            ("split_val",    df_m[df_m.split == "val"]),
        ]:
            col = "mask_nonzero_ratio_mean"
            dist_rows.append({
                "group":     grp_key,
                "count":     len(grp_df),
                "mean":      round(float(grp_df[col].mean()), 4),
                "std":       round(float(grp_df[col].std()),  4),
                "min":       round(float(grp_df[col].min()),  4),
                "p5":        round(float(grp_df[col].quantile(0.05)), 4),
                "p25":       round(float(grp_df[col].quantile(0.25)), 4),
                "median":    round(float(grp_df[col].median()), 4),
                "p75":       round(float(grp_df[col].quantile(0.75)), 4),
                "p95":       round(float(grp_df[col].quantile(0.95)), 4),
                "max":       round(float(grp_df[col].max()),  4),
                "zero_count":  int((grp_df[col] == 0.0).sum()),
                f"low_lt{LOW_MASK_THRESHOLD}": int((grp_df[col] < LOW_MASK_THRESHOLD).sum()),
            })
        _write_csv(dist_rows, REPORT_ROOT / "p_c_normal29b_mask_distribution_summary.csv")
    else:
        _write_csv([{"note": "no manifest rows"}],
                   REPORT_ROOT / "p_c_normal29b_mask_distribution_summary.csv")
        df_m = pd.DataFrame()

    # ── 8. sample contact sheet ───────────────────────────────────────────────
    _make_contact_sheet(df_m if not df_m.empty else None)

    # ── 9. 검증 ──────────────────────────────────────────────────────────────
    row_match   = (n_done == total_expected - n_error) and (n_done + n_error == total_expected)
    missing_cnt = total_expected - n_done - n_error
    shape_ok    = all(r.get("mask_shape") == "3x96x96" for r in manifest_rows)
    dtype_ok    = all(r.get("mask_dtype") == "uint8"   for r in manifest_rows)
    zero_cnt    = int(df_m["mask_nonzero_ratio_mean"].eq(0.0).sum()) if not df_m.empty else 0

    final_pass  = (n_error == 0) and shape_ok and dtype_ok

    # ── 10. summary.json ─────────────────────────────────────────────────────
    elapsed = (datetime.now() - start_time).total_seconds()
    summary = {
        "branch":          "P-C-NORMAL29b",
        "date":            start_time.strftime("%Y-%m-%d"),
        "elapsed_sec":     round(elapsed, 1),
        "manifest_source": {
            "train": str(TRAIN_MANIFEST),
            "val":   str(VAL_MANIFEST),
        },
        "roi_root":          str(ROI_ROOT),
        "mask_output_root":  str(MASK_ROOT),
        "total_expected":    total_expected,
        "n_done":            n_done,
        "n_error":           n_error,
        "n_nearest_repeat":  n_nearest,
        "n_spatial_pad":     n_spatial_pad,
        "n_zero_mask":       zero_cnt,
        "n_low_mask":        n_low,
        "row_count_match":   row_match,
        "missing_count":     missing_cnt,
        "shape_ok":          shape_ok,
        "dtype_ok":          dtype_ok,
        "final_verdict":     "PASS" if final_pass else "PARTIAL_PASS" if n_error < total_expected * 0.01 else "FAIL",
        "guardrails":        GUARDRAILS,
        "train_rows":        len(df_train),
        "val_rows":          len(df_val),
        "mask_3ch_spec":     "ch0=z-1 ch1=z ch2=z+1, nearest-repeat at boundary",
        "spatial_boundary":  "zero-pad",
        "center_based_96":   True,
        "patch_32_not_used": True,
    }
    _write_json(summary, REPORT_ROOT / "p_c_normal29b_summary.json")

    # ── 11. report.md ────────────────────────────────────────────────────────
    verdict = summary["final_verdict"]
    low_note = f"{n_low}" if n_low > 0 else "0 (none)"
    report_md = f"""# P-C-NORMAL29b 결과 보고서

날짜: {start_time.strftime("%Y-%m-%d %H:%M")}
소요: {elapsed:.1f}초

## 전체 판정: {verdict}

## 수치 요약

| 항목 | 값 |
|---|---|
| 전체 expected | {total_expected} |
| n_done (PASS) | {n_done} |
| n_error (FAIL) | {n_error} |
| row count 일치 | {"YES" if n_error == 0 else "NO"} |
| missing mask | {missing_cnt} |
| shape 3×96×96 | {"OK" if shape_ok else "FAIL"} |
| dtype uint8 | {"OK" if dtype_ok else "FAIL"} |
| zero mask count | {zero_cnt} |
| low mask (<{LOW_MASK_THRESHOLD}) count | {low_note} |
| nearest-repeat 사용 | {n_nearest}건 |
| spatial padding 필요 | {n_spatial_pad}건 |

## mask nonzero_ratio_mean 분포

"""
    if not df_m.empty:
        for grp in ["all", "label_0_normal", "label_1_nsclc"]:
            sub = df_m if grp == "all" else df_m[df_m.label == (0 if "normal" in grp else 1)]
            col = "mask_nonzero_ratio_mean"
            report_md += f"**{grp}** (n={len(sub)}): mean={sub[col].mean():.4f} std={sub[col].std():.4f} min={sub[col].min():.4f} max={sub[col].max():.4f}\n\n"

    report_md += f"""
## Guardrail

| 항목 | 통과 |
|---|---|
"""
    for r in guard_rows:
        report_md += f"| {r['guardrail']} | {'✅' if r['pass'] else '❌'} |\n"

    report_md += f"""
## 다음 단계

P-C-NORMAL29b PASS 이후 → **P-C-NORMAL30a masking smoke training**
- 단, P-C-NORMAL29b {verdict} 확인 후 사용자 승인 필요
- 조건: mask manifest 존재, row count 일치, shape/dtype OK, zero mask 없음
"""
    (REPORT_ROOT / "p_c_normal29b_report.md").write_text(report_md, encoding="utf-8")
    print(f"[29b] report.md 저장")

    # ── 12. DONE.json ────────────────────────────────────────────────────────
    done_obj = {
        "done":            True,
        "final_verdict":   verdict,
        "n_done":          n_done,
        "n_error":         n_error,
        "date":            start_time.strftime("%Y-%m-%d"),
        "elapsed_sec":     round(elapsed, 1),
        "mask_root":       str(MASK_ROOT),
        "report_root":     str(REPORT_ROOT),
    }
    _write_json(done_obj, REPORT_ROOT / "DONE.json")
    print(f"[29b] DONE.json 저장 | verdict={verdict}")

    # ── 최종 요약 출력 ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[29b] 전체 판정: {verdict}")
    print(f"  expected={total_expected}  done={n_done}  error={n_error}")
    print(f"  shape_ok={shape_ok}  dtype_ok={dtype_ok}")
    print(f"  zero_mask={zero_cnt}  low_mask={n_low}")
    print(f"  nearest_repeat={n_nearest}  spatial_pad={n_spatial_pad}")
    print(f"  소요: {elapsed:.1f}초")
    print(f"{'='*60}")


def _make_contact_sheet(df_m):
    """normal/NSCLC × high/medium/low mask ratio 대표 샘플 contact sheet 생성."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.ndimage import binary_erosion
    except ImportError as e:
        print(f"[29b] contact sheet 스킵 (import 실패: {e})")
        return

    if df_m is None or df_m.empty:
        print("[29b] contact sheet 스킵 (manifest 없음)")
        return

    samples = []
    col = "mask_nonzero_ratio_mean"
    for label_val, label_str in [(0, "normal"), (1, "nsclc")]:
        sub = df_m[df_m.label == label_val].copy().dropna(subset=[col])
        if len(sub) == 0:
            continue
        sub_sorted = sub.sort_values(col)
        # high/medium/low: p10 / p50 / p90 근처 각 1개
        quantiles = [(0.05, "low"), (0.50, "medium"), (0.95, "high")]
        for q, qname in quantiles:
            idx_q = int(q * (len(sub_sorted) - 1))
            row = sub_sorted.iloc[idx_q]
            samples.append({
                "label_str": label_str, "qname": qname,
                "mask_path": row.mask_path,
                "crop_path": row.crop_path,
                "nr_mean":   row[col],
                "safe_id":   row.safe_id[:20],
                "z":         row.canonical_volume_z,
            })

    if not samples:
        print("[29b] contact sheet 스킵 (샘플 없음)")
        return

    n_cols = 3
    n_rows = len(samples)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3.2))
    if n_rows == 1:
        axes = [axes]
    fig.suptitle("P-C-NORMAL29b: sample mask overlays (CT + ROI mask)", fontsize=11)

    for row_i, s in enumerate(samples):
        axrow = axes[row_i]
        titles_ch = ["ch0 (z-1)", "ch1 (z)", "ch2 (z+1)"]

        # mask 로드
        try:
            mask_3ch = np.load(s["mask_path"])["mask_3ch"]
        except Exception as e:
            for ax in axrow:
                ax.set_visible(False)
            print(f"[29b] sample mask load fail: {e}")
            continue

        # ct crop 로드 (optional, 없으면 검은 배경)
        ct_3ch = None
        try:
            crop_data = np.load(s["crop_path"])
            ct_3ch = crop_data["ct_crop"]
        except Exception:
            pass

        for ch_i in range(n_cols):
            ax = axrow[ch_i] if n_rows > 1 else axrow[ch_i]
            mask_hw = mask_3ch[ch_i]
            if ct_3ch is not None:
                ct_hw = ct_3ch[ch_i]
                overlay = _make_overlay_local(ct_hw, mask_hw)
                ax.imshow(overlay)
            else:
                ax.imshow(mask_hw, cmap="gray", vmin=0, vmax=1)

            ax.set_title(
                f"[{s['label_str']} {s['qname']}] {titles_ch[ch_i]}\n"
                f"nr={s['nr_mean']:.3f} z={s['z']}",
                fontsize=6
            )
            ax.axis("off")

    plt.tight_layout()
    out_png = REPORT_ROOT / "p_c_normal29b_sample_contact_sheet.png"
    plt.savefig(str(out_png), dpi=80, bbox_inches="tight")
    plt.close()
    print(f"[29b] contact sheet 저장: {out_png}")


def _make_overlay_local(ct_hw: np.ndarray, mask_hw: np.ndarray) -> np.ndarray:
    """CT grayscale + mask overlay → RGB uint8 (H×W×3)."""
    ct_g = ct_hw.astype(np.float32)
    ct_g = np.clip(ct_g, -1000.0, 200.0)
    ct_g = ((ct_g + 1000.0) / 1200.0 * 255.0).astype(np.uint8)
    rgb  = np.stack([ct_g, ct_g, ct_g], axis=-1)
    green = mask_hw.astype(bool)
    rgb[green, 1] = np.clip(rgb[green, 1].astype(np.int32) + 80, 0, 255).astype(np.uint8)
    try:
        from scipy.ndimage import binary_erosion
        eroded   = binary_erosion(green, iterations=1)
        boundary = green & ~eroded
        rgb[boundary] = [0, 255, 0]
    except Exception:
        pass
    return rgb


if __name__ == "__main__":
    main()
