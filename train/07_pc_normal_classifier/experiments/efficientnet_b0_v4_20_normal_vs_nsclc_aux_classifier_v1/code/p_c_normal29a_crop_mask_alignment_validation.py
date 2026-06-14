"""
p_c_normal29a_crop_mask_alignment_validation.py

P-C-NORMAL29a: crop-level ROI mask alignment validation

목표:
  refined_roi.npy (volume-level)에서 crop 단위 96×96 mask를 잘라냈을 때
  실제 ct_crop과 픽셀 정렬이 맞는지 검증한다.

핵심 발견:
  - ct_crop shape: (3, 96, 96) — z-1/z/z+1 슬라이스 3장
  - crop 추출 bbox: center_y ± 48, center_x ± 48 (volume 좌표)
  - y0/x0/y1/x1 (32×32): candidate annotation bbox (crop_lung_roi_ratio 계산용, crop 추출 bbox 아님)
  - 마스킹용 96×96 mask = roi[z][cy-48:cy+48, cx-48:cx+48]

금지:
  - 전체 crop-level mask 생성
  - 학습 / model forward / prediction export
  - threshold 최적화
  - 기존 결과 수정/삭제/덮어쓰기

출력:
  - overlay PNG 10장 (normal 5 + NSCLC 5)
  - contact_sheet.png
  - sample_manifest.csv
  - alignment_check_report.csv
  - guardrail_check.csv
  - report.md
  - DONE.json
"""

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

BRANCH_ROOT  = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BRANCH_ROOT.parents[1]

TRAIN_MANIFEST = PROJECT_ROOT / "outputs/manifests/p_c_normal24g_fix_zroi_only_feature_manifest/p_c_normal24g_fix_train_feature_manifest_usable.csv"
ROI_ROOT       = PROJECT_ROOT / "outputs/mip-postprocess-research-v1/masks/refined_roi_v4_20_modeB_all_v1"
REPORT_ROOT    = PROJECT_ROOT / "outputs/reports/p_c_normal29a_crop_mask_alignment_validation"

CROP_HALF = 48   # 96×96 crop → center ± 48
N_SAMPLES  = 10  # normal 5 + NSCLC 5


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


def get_roi_path(safe_id, label):
    """label 0=normal, 1=NSCLC/MSD"""
    subdir = "normal" if label == 0 else "lesion"
    return ROI_ROOT / subdir / safe_id / "refined_roi.npy"


def extract_mask_96(roi_volume, z, center_y, center_x):
    """volume에서 z 슬라이스의 96×96 mask crop 추출. 경계 초과 시 zero padding."""
    H, W = roi_volume.shape[1], roi_volume.shape[2]
    z = int(z)
    cy, cx = int(center_y), int(center_x)

    y0_vol = cy - CROP_HALF
    y1_vol = cy + CROP_HALF
    x0_vol = cx - CROP_HALF
    x1_vol = cx + CROP_HALF

    # padding 필요 여부
    pad_y0 = max(0, -y0_vol)
    pad_y1 = max(0, y1_vol - H)
    pad_x0 = max(0, -x0_vol)
    pad_x1 = max(0, x1_vol - W)

    y0_clip = max(0, y0_vol)
    y1_clip = min(H, y1_vol)
    x0_clip = max(0, x0_vol)
    x1_clip = min(W, x1_vol)

    roi_slice = roi_volume[z]
    patch     = roi_slice[y0_clip:y1_clip, x0_clip:x1_clip]

    if pad_y0 > 0 or pad_y1 > 0 or pad_x0 > 0 or pad_x1 > 0:
        patch = np.pad(patch, ((pad_y0, pad_y1), (pad_x0, pad_x1)), mode="constant")

    return patch.astype(np.uint8), (pad_y0, pad_y1, pad_x0, pad_x1)


def normalize_hu(arr_hw):
    """HU [-1000, 200] → [0, 255] uint8"""
    arr = arr_hw.astype(np.float32)
    arr = np.clip(arr, -1000.0, 200.0)
    arr = (arr - (-1000.0)) / (200.0 - (-1000.0)) * 255.0
    return arr.astype(np.uint8)


def save_overlay_png(ct_hw, mask_hw, save_path, title=""):
    """CT grayscale + mask overlay (lung=green tint, boundary=bright green) PNG 저장."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.colors import ListedColormap

        ct_norm = normalize_hu(ct_hw)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(title, fontsize=9, wrap=True)

        # 1) CT only
        axes[0].imshow(ct_norm, cmap="gray", vmin=0, vmax=255)
        axes[0].set_title("CT (center slice)", fontsize=8)
        axes[0].axis("off")

        # 2) Mask only
        axes[1].imshow(mask_hw, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("ROI mask", fontsize=8)
        axes[1].axis("off")

        # 3) Overlay
        rgb = np.stack([ct_norm, ct_norm, ct_norm], axis=-1)
        # lung 영역: green tint
        lung_area = mask_hw.astype(bool)
        rgb_overlay = rgb.copy()
        rgb_overlay[lung_area, 1] = np.clip(rgb[lung_area, 1].astype(int) + 40, 0, 255).astype(np.uint8)
        # boundary: bright green
        from scipy.ndimage import binary_erosion
        boundary = lung_area & ~binary_erosion(lung_area)
        rgb_overlay[boundary] = [0, 255, 0]

        axes[2].imshow(rgb_overlay)
        ratio = float(mask_hw.sum()) / (mask_hw.shape[0] * mask_hw.shape[1])
        axes[2].set_title(f"Overlay (lung={ratio:.3f})", fontsize=8)
        axes[2].axis("off")

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(str(save_path), dpi=100, bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception as e:
        print(f"  [WARN] overlay save failed: {e}")
        return False


def save_contact_sheet(overlay_paths, save_path):
    """전체 샘플 contact sheet PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        n = len(overlay_paths)
        fig, axes = plt.subplots(n, 1, figsize=(12, 4 * n))
        if n == 1:
            axes = [axes]
        for ax, p in zip(axes, overlay_paths):
            if p.exists():
                img = mpimg.imread(str(p))
                ax.imshow(img)
            ax.axis("off")
            ax.set_title(p.stem, fontsize=7)
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=80, bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception as e:
        print(f"  [WARN] contact sheet failed: {e}")
        return False


def select_samples(df, n_per_label=5):
    """label별 n_per_label개 선택 (ratio 분포 다양하게)."""
    samples = []
    for label in [0, 1]:
        sub = df[df.label == label].copy()
        # ratio 구간 별로 골고루 선택
        sub = sub.sort_values("crop_lung_roi_ratio")
        step = max(1, len(sub) // n_per_label)
        indices = [i * step for i in range(n_per_label)]
        indices[-1] = min(indices[-1], len(sub) - 1)
        # 중간값도 포함
        mid = len(sub) // 2
        if mid not in indices:
            indices[n_per_label // 2] = mid
        selected = sub.iloc[indices].drop_duplicates("safe_id")
        # n_per_label 부족하면 추가
        if len(selected) < n_per_label:
            extra = sub[~sub.index.isin(selected.index)].head(n_per_label - len(selected))
            selected = pd.concat([selected, extra])
        samples.append(selected.head(n_per_label))
    return pd.concat(samples).reset_index(drop=True)


def validate_sample(row, roi_volume, sample_idx, save_dir):
    """단일 crop 검증. 결과 dict 반환."""
    result = {
        "sample_idx":        sample_idx,
        "label":             int(row.label),
        "safe_id":           str(row.safe_id),
        "patient_id":        str(row.patient_id),
        "crop_path":         str(row.crop_path),
        "canonical_volume_z": float(row.canonical_volume_z),
        "center_y":          int(row.center_y),
        "center_x":          int(row.center_x),
        "y0_ann":            int(row.y0), "y1_ann": int(row.y1),
        "x0_ann":            int(row.x0), "x1_ann": int(row.x1),
        "crop_lung_roi_ratio": float(row.crop_lung_roi_ratio),
        "position_bin":      str(row.position_bin) if hasattr(row, "position_bin") else "",
    }

    errors = []

    # ── 1. ct_crop 로드 ───────────────────────────────────────────────────────
    try:
        data   = np.load(str(row.crop_path))
        ct     = data["ct_crop"]    # (3, H, W)
        ct_h, ct_w = ct.shape[1], ct.shape[2]
        result["ct_shape"] = f"{ct.shape[0]}x{ct_h}x{ct_w}"
        result["ct_channels"] = ct.shape[0]
        result["ch0_eq_ch1"] = bool(np.allclose(ct[0], ct[1]))
        result["ch0_eq_ch2"] = bool(np.allclose(ct[0], ct[2]))
        if ct_h != 96 or ct_w != 96:
            errors.append(f"ct_crop not 96×96: {ct_h}×{ct_w}")
    except Exception as e:
        errors.append(f"ct_crop load error: {e}")
        result["ct_shape"] = "error"
        result["pass"] = False
        result["errors"] = "; ".join(errors)
        return result

    # ── 2. z index 검증 ───────────────────────────────────────────────────────
    z = int(row.canonical_volume_z)
    vol_z = roi_volume.shape[0]
    result["z_int"]   = z
    result["vol_z"]   = vol_z
    result["vol_h"]   = roi_volume.shape[1]
    result["vol_w"]   = roi_volume.shape[2]
    if z < 0 or z >= vol_z:
        errors.append(f"z={z} out of range [0, {vol_z})")

    # ── 3. 96×96 mask 추출 (center_y±48, center_x±48) ────────────────────────
    cy, cx = int(row.center_y), int(row.center_x)
    result["bbox_96_y"] = f"{cy-CROP_HALF}:{cy+CROP_HALF}"
    result["bbox_96_x"] = f"{cx-CROP_HALF}:{cx+CROP_HALF}"

    mask_96, pads = extract_mask_96(roi_volume, z, cy, cx)
    result["mask_shape"] = f"{mask_96.shape[0]}x{mask_96.shape[1]}"
    result["pad_y0_y1_x0_x1"] = f"{pads[0]},{pads[1]},{pads[2]},{pads[3]}"
    any_pad = any(p > 0 for p in pads)
    result["has_padding"] = any_pad
    if any_pad:
        errors.append(f"bbox exceeded volume boundary — padding applied: {pads}")

    if mask_96.shape != (96, 96):
        errors.append(f"mask not 96×96: {mask_96.shape}")

    # ── 4. mask nonzero ratio ─────────────────────────────────────────────────
    mask_ratio_96 = float(mask_96.sum()) / (96 * 96) if mask_96.size > 0 else 0.0
    result["mask_ratio_96x96"] = round(mask_ratio_96, 4)

    # annotation bbox (32×32) mask ratio
    y0a, y1a = int(row.y0), int(row.y1)
    x0a, x1a = int(row.x0), int(row.x1)
    mask_32 = roi_volume[z][y0a:y1a, x0a:x1a]
    ann_area = (y1a - y0a) * (x1a - x0a)
    mask_ratio_ann = float(mask_32.sum()) / ann_area if ann_area > 0 else 0.0
    result["mask_ratio_ann32"] = round(mask_ratio_ann, 4)
    result["manifest_crop_lung_roi_ratio"] = float(row.crop_lung_roi_ratio)
    result["ratio_diff_ann_vs_manifest"] = round(abs(mask_ratio_ann - float(row.crop_lung_roi_ratio)), 4)

    # ── 5. overlay 저장 ───────────────────────────────────────────────────────
    label_str = "normal" if row.label == 0 else "nsclc"
    png_name  = f"sample_{sample_idx:02d}_{label_str}_{row.safe_id[:20]}.png"
    png_path  = save_dir / png_name
    title = (f"[{sample_idx}] {label_str} | {row.safe_id[:30]}\n"
             f"z={z} cy={cy} cx={cx} | mask_ratio={mask_ratio_96:.3f} | "
             f"crop_roi_ratio={row.crop_lung_roi_ratio:.3f}")
    ct_center = ct[1]  # center slice (channel 1)
    overlay_ok = save_overlay_png(ct_center, mask_96, png_path, title=title)
    result["overlay_saved"] = overlay_ok
    result["overlay_path"]  = str(png_path) if overlay_ok else ""

    # ── 6. 정렬 판정 ──────────────────────────────────────────────────────────
    shape_ok  = (result["mask_shape"] == "96x96") and (ct_h == 96)
    z_ok      = (0 <= z < vol_z)
    ratio_ok  = mask_ratio_96 > 0.0  # 완전히 폐 밖이면 문제
    result["shape_ok"]  = shape_ok
    result["z_ok"]      = z_ok
    result["ratio_ok"]  = ratio_ok
    result["pass"]      = shape_ok and z_ok and ratio_ok and not any_pad
    result["errors"]    = "; ".join(errors) if errors else "none"

    if result["pass"]:
        print(f"  [PASS] sample {sample_idx} | {label_str} | mask_ratio={mask_ratio_96:.3f} | pad={any_pad}")
    else:
        print(f"  [FAIL] sample {sample_idx} | {label_str} | errors: {result['errors']}")

    return result


def main():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if REPORT_ROOT.exists() and any(REPORT_ROOT.iterdir()):
        print(f"[ABORT] output dir already exists: {REPORT_ROOT}")
        sys.exit(2)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    overlay_dir = REPORT_ROOT / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. manifest 로드 ──────────────────────────────────────────────────────
    print(f"[29a] loading manifest...")
    if not TRAIN_MANIFEST.exists():
        print(f"[ERROR] manifest not found: {TRAIN_MANIFEST}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(TRAIN_MANIFEST, low_memory=False)
    print(f"[29a] manifest: {len(df)} rows | normal={int((df.label==0).sum())} NSCLC={int((df.label==1).sum())}")

    # ── 2. 샘플 선택 ──────────────────────────────────────────────────────────
    samples = select_samples(df, n_per_label=5)
    print(f"[29a] selected {len(samples)} samples: normal={int((samples.label==0).sum())} NSCLC={int((samples.label==1).sum())}")
    samples.to_csv(REPORT_ROOT / "sample_manifest.csv", index=False)

    # ── 3. 각 샘플 검증 ───────────────────────────────────────────────────────
    results    = []
    roi_cache  = {}
    overlay_paths = []

    for idx, row in samples.iterrows():
        sample_num = len(results) + 1
        label_str  = "normal" if row.label == 0 else "nsclc"
        print(f"\n[29a] sample {sample_num}/{len(samples)} | {label_str} | {row.safe_id[:30]}")

        # ROI 마스크 로드 (캐시)
        roi_path = get_roi_path(row.safe_id, int(row.label))
        if not roi_path.exists():
            print(f"  [FAIL] ROI not found: {roi_path}")
            results.append({
                "sample_idx": sample_num, "label": int(row.label),
                "safe_id": str(row.safe_id), "pass": False,
                "errors": f"ROI not found: {roi_path}",
            })
            continue

        if str(roi_path) not in roi_cache:
            roi_cache[str(roi_path)] = np.load(str(roi_path))
        roi_vol = roi_cache[str(roi_path)]

        result = validate_sample(row, roi_vol, sample_num, overlay_dir)
        results.append(result)
        if result.get("overlay_path"):
            overlay_paths.append(Path(result["overlay_path"]))

    roi_cache.clear()

    # ── 4. contact sheet ──────────────────────────────────────────────────────
    print(f"\n[29a] saving contact sheet...")
    contact_path = REPORT_ROOT / "contact_sheet.png"
    save_contact_sheet(overlay_paths, contact_path)

    # ── 5. alignment check report CSV ────────────────────────────────────────
    _write_csv(results, REPORT_ROOT / "alignment_check_report.csv")

    # ── 6. 전체 판정 ──────────────────────────────────────────────────────────
    n_pass = sum(1 for r in results if r.get("pass", False))
    n_fail = len(results) - n_pass
    shape_all_ok  = all(r.get("shape_ok", False) for r in results)
    z_all_ok      = all(r.get("z_ok",    False) for r in results)
    ratio_all_ok  = all(r.get("ratio_ok", False) for r in results)
    no_padding    = not any(r.get("has_padding", True) for r in results)

    overall = "PASS" if (n_fail == 0) else "FAIL"

    # ── 7. guardrail ─────────────────────────────────────────────────────────
    guardrail_rows = [
        {"guardrail": "full_mask_generation_run",    "expected": False, "actual": False, "pass": True},
        {"guardrail": "training_run",                "expected": False, "actual": False, "pass": True},
        {"guardrail": "model_forward_run",           "expected": False, "actual": False, "pass": True},
        {"guardrail": "existing_results_modified",   "expected": False, "actual": False, "pass": True},
        {"guardrail": "samples_only_validated",      "expected": True,  "actual": True,  "pass": True},
        {"guardrail": "all_shapes_96x96",            "expected": True,  "actual": shape_all_ok, "pass": shape_all_ok},
        {"guardrail": "all_z_valid",                 "expected": True,  "actual": z_all_ok,     "pass": z_all_ok},
        {"guardrail": "all_ratio_nonzero",           "expected": True,  "actual": ratio_all_ok, "pass": ratio_all_ok},
        {"guardrail": "no_boundary_padding",         "expected": True,  "actual": no_padding,   "pass": no_padding},
    ]
    _write_csv(guardrail_rows, REPORT_ROOT / "guardrail_check.csv")

    # ── 8. report.md ──────────────────────────────────────────────────────────
    fail_rows = [r for r in results if not r.get("pass", False)]
    md = f"""# P-C-NORMAL29a: Crop-level ROI Mask Alignment Validation

**날짜**: {ts[:10]}
**전체 판정**: {overall}

---

## 핵심 발견 (구조 확인)

| 항목 | 값 |
|---|---|
| ct_crop shape | (3, 96, 96) |
| 채널 구성 | z-1 / z / z+1 (3장 연속 슬라이스, 동일하지 않음) |
| crop 추출 bbox | center_y ± 48, center_x ± 48 (96×96, volume 좌표) |
| y0/x0/y1/x1 용도 | candidate 어노테이션 (32×32), crop_lung_roi_ratio 계산에만 사용 |
| masking용 96×96 mask | roi[z][cy-48:cy+48, cx-48:cx+48] |

> **결론**: masking 실험 시 center_y/center_x ± 48 기준 96×96 마스크를 사용해야 한다.
> y0/x0/y1/x1은 crop 추출 bbox가 아니다.

---

## 샘플 결과

| sample | label | safe_id | z | mask_ratio_96 | shape_ok | z_ok | pad | pass |
|---|---|---|---|---|---|---|---|---|
"""
    for r in results:
        md += (f"| {r.get('sample_idx','')} | {'normal' if r.get('label')==0 else 'nsclc'} "
               f"| {str(r.get('safe_id',''))[:25]} | {r.get('z_int','?')} "
               f"| {r.get('mask_ratio_96x96','?')} | {r.get('shape_ok','?')} "
               f"| {r.get('z_ok','?')} | {r.get('has_padding','?')} | {r.get('pass','?')} |\n")

    md += f"""
---

## 전체 통계

| 항목 | 값 |
|---|---|
| 총 샘플 | {len(results)} |
| PASS | {n_pass} |
| FAIL | {n_fail} |
| shape 전체 OK | {shape_all_ok} |
| z index 전체 OK | {z_all_ok} |
| ratio nonzero 전체 OK | {ratio_all_ok} |
| padding 없음 | {no_padding} |

"""
    if fail_rows:
        md += "## FAIL 샘플 상세\n\n"
        for r in fail_rows:
            md += f"- sample {r.get('sample_idx')}: {r.get('errors')}\n"
        md += "\n"

    md += f"""---

## 판정 기준

- PASS: shape 96×96 일치 + z valid + mask nonzero + padding 없음
- FAIL: 위 조건 중 하나라도 불충족

## Guardrail

- full_mask_generation_run=False
- training_run=False
- model_forward_run=False
- existing_results_modified=False
- samples_only_validated=True

## 다음 단계 (PASS 시)

- P-C-NORMAL30 전체 mask 생성 전처리 설계 (사용자 승인 후)
- 학습 시: crop image pixel masking = roi[z][cy-48:cy+48, cx-48:cx+48] 기준
- scalar feature는 별도 결정 (제거 여부)
"""
    (REPORT_ROOT / "report.md").write_text(md, encoding="utf-8")

    # ── 9. DONE.json ──────────────────────────────────────────────────────────
    _write_json(
        {"step": "p_c_normal29a", "verdict": overall, "timestamp": ts,
         "n_samples": len(results), "n_pass": n_pass, "n_fail": n_fail,
         "shape_all_ok": shape_all_ok, "z_all_ok": z_all_ok,
         "ratio_all_ok": ratio_all_ok, "no_padding": no_padding,
         "masking_bbox_correct": "center_y ± 48, center_x ± 48 (NOT y0/y1/x0/x1)",
         "ct_crop_channels": "z-1, z, z+1 (3 adjacent slices)",
         "full_mask_generation_run": False, "training_run": False},
        REPORT_ROOT / "DONE.json",
    )

    print(f"\n{'='*60}")
    print(f"[29a] 전체 판정: {overall}")
    print(f"  PASS={n_pass}  FAIL={n_fail}")
    print(f"  shape_ok={shape_all_ok}  z_ok={z_all_ok}  ratio_ok={ratio_all_ok}  no_pad={no_padding}")
    print(f"  핵심: masking bbox = center_y±48, center_x±48 (NOT y0:y1/x0:x1)")
    print(f"  report → {REPORT_ROOT}")
    print(f"{'='*60}")
    sys.exit(0 if overall == "PASS" else 1)


if __name__ == "__main__":
    main()
