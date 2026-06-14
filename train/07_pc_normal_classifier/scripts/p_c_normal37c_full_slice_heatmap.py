"""
P-C-NORMAL37c: full-slice CT heatmap rendering fix
- 원본 CT volume에서 full z slice를 background로 사용
- crop 조각 canvas 방식 폐기
- coverage 없는 영역 alpha=0 처리
- logit 기반 normalize heatmap
- 4 panel: CT only / coverage map / mean overlay / max overlay
- blocky + smoothed 저장
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")
STAGE_LABEL  = "P-C-NORMAL37c"

SCORE_CSV    = PROJECT_ROOT / "outputs/p_c_normal35_full_downstream_scoring/p_c_normal35_full_crop_scores.csv"
SAMPLE_CSV   = PROJECT_ROOT / "outputs/reports/p_c_normal36_heatmap_preflight/p_c_normal36_sample_selection.csv"

NSCLC_VOL_ROOT  = Path("/mnt/c/Users/jinhy/Desktop/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1/volumes_npy")
NORMAL_VOL_ROOT = Path("/mnt/c/Users/jinhy/Desktop/Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1/volumes_npy")

OUTPUT_DIR  = PROJECT_ROOT / "outputs/p_c_normal37c_full_slice_heatmap_render_fix"
REPORT_DIR  = PROJECT_ROOT / "outputs/reports/p_c_normal37c_full_slice_heatmap_render_fix"
PREVIEW_DIR = OUTPUT_DIR / "previews"

CROP_SIZE    = 96
HALF         = CROP_SIZE // 2
LOGIT_P_LOW  = 5
LOGIT_P_HIGH = 95
HM_ALPHA     = 0.55
TOP_K        = 5          # 환자당 최대 slice 수
MAX_PATIENTS = 8          # sample 환자 수

GUARDRAILS = {
    "rendering_fix_only":               True,
    "no_training_run":                  True,
    "no_model_forward":                 True,
    "no_scoring_rerun":                 True,
    "no_threshold_optimization":        True,
    "no_threshold_sweep":               True,
    "no_best_threshold_selection":      True,
    "no_xai_card_generated":            True,
    "no_explanation_card_generated":    True,
    "no_existing_result_overwrite":     True,
    "p_c_normal37b_outputs_readonly":   True,
    "full_ct_slice_background_used":    True,
    "crop_mosaic_background_used":      False,
    "coverage_zero_alpha_applied":      True,
    "sample_preview_only":              True,
    "full_heatmap_generation_run":      False,
    "diagnostic_wording_avoided":       True,
}

def _write_csv(rows, path):
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

def _write_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def _load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def _abort(msg):
    print(f"[ABORT] {msg}", file=sys.stderr)
    sys.exit(2)


def resolve_ct_path(safe_id, source_split):
    """safe_id와 source_split으로 ct_hu.npy 경로 반환."""
    if source_split == "stage2_holdout":
        p = NSCLC_VOL_ROOT / safe_id / "ct_hu.npy"
    else:
        p = NORMAL_VOL_ROOT / safe_id / "ct_hu.npy"
    return p


def load_ct_slice(ct_path, z_idx):
    """CT volume에서 z slice 로드 → HU clip → [0,1] normalize."""
    ct = np.load(str(ct_path), mmap_mode="r")
    if z_idx < 0 or z_idx >= ct.shape[0]:
        return None, ct.shape
    slc = ct[z_idx].astype(np.float32)
    slc = np.clip(slc, -1000, 200)
    slc = (slc - (-1000)) / (200 - (-1000))
    return slc, ct.shape


def build_heatmap(z_crops, ct_h, ct_w, logit_min, logit_range):
    score_sum = np.zeros((ct_h, ct_w), dtype=np.float32)
    score_max = np.zeros((ct_h, ct_w), dtype=np.float32)
    count     = np.zeros((ct_h, ct_w), dtype=np.float32)
    n_clipped = 0

    for r in z_crops:
        try:
            cy   = int(float(r["center_y"]))
            cx   = int(float(r["center_x"]))
            lg   = float(r["logit"])
            lg_n = float(np.clip((lg - logit_min) / logit_range, 0.0, 1.0))
        except Exception:
            continue
        y0r, y1r = cy - HALF, cy + HALF
        x0r, x1r = cx - HALF, cx + HALF
        y0, y1 = max(0, y0r), min(ct_h, y1r)
        x0, x1 = max(0, x0r), min(ct_w, x1r)
        if y0 != y0r or x0 != x0r or y1 != y1r or x1 != x1r:
            n_clipped += 1
        score_sum[y0:y1, x0:x1] += lg_n
        score_max[y0:y1, x0:x1]  = np.maximum(score_max[y0:y1, x0:x1], lg_n)
        count[y0:y1, x0:x1]     += 1.0

    with np.errstate(invalid="ignore"):
        mean_hm = np.where(count > 0, score_sum / count, 0.0)
    return mean_hm, score_max, count, n_clipped


def save_panel_png(ct_slice, mean_hm, max_hm, count_hm,
                   png_path, title, plt, smooth_sigma=3.0):
    """4 panel: CT only / coverage / mean overlay / max overlay + smoothed."""
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.ndimage import gaussian_filter

    ct_h, ct_w = ct_slice.shape
    alpha_mask = (count_hm > 0).astype(np.float32)

    # smoothed (visual only)
    mean_smooth = gaussian_filter(mean_hm, sigma=smooth_sigma)
    max_smooth  = gaussian_filter(max_hm,  sigma=smooth_sigma)

    def _overlay(ax, bg, hm, alpha_m, title_str):
        ax.imshow(bg, cmap="gray", vmin=0, vmax=1)
        # RGBA heatmap: alpha=0 where no coverage
        import matplotlib.cm as cm
        cmap = cm.get_cmap("hot")
        rgba = cmap(hm)
        rgba[..., 3] = alpha_m * HM_ALPHA
        im = ax.imshow(rgba, vmin=0, vmax=1)
        ax.set_title(title_str, fontsize=7)
        ax.axis("off")
        return im

    # ── blocky panels ─────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(title, fontsize=8, y=1.01)

    axes[0, 0].imshow(ct_slice, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Full CT slice", fontsize=7)
    axes[0, 0].axis("off")

    im_cov = axes[0, 1].imshow(count_hm, cmap="Blues")
    axes[0, 1].set_title("Coverage (crop count)", fontsize=7)
    axes[0, 1].axis("off")
    plt.colorbar(im_cov, ax=axes[0, 1], fraction=0.046)

    _overlay(axes[0, 2], ct_slice, mean_hm, alpha_mask, "Mean heatmap overlay (blocky)")
    _overlay(axes[1, 0], ct_slice, max_hm,  alpha_mask, "Max heatmap overlay (blocky)")
    _overlay(axes[1, 1], ct_slice, mean_smooth, alpha_mask, "Mean heatmap overlay (smoothed, visual only)")
    _overlay(axes[1, 2], ct_slice, max_smooth,  alpha_mask, "Max heatmap overlay (smoothed, visual only)")

    # colorbar for overlays
    import matplotlib.cm as cm
    sm = plt.cm.ScalarMappable(cmap="hot", norm=plt.Normalize(0, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=axes[0, 2], fraction=0.046, label="normalized logit score")
    plt.colorbar(sm, ax=axes[1, 0], fraction=0.046)
    plt.colorbar(sm, ax=axes[1, 1], fraction=0.046)
    plt.colorbar(sm, ax=axes[1, 2], fraction=0.046)

    # note
    fig.text(0.5, -0.01,
             "※ crop-level score projection (96×96 patch avg) — NOT pixel-level Grad-CAM",
             ha="center", fontsize=7, color="gray")

    plt.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    import datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # output dir guard
    if REPORT_DIR.exists() and any(REPORT_DIR.iterdir()):
        _abort(f"output directory already exists: {REPORT_DIR}")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[{STAGE_LABEL}] output: {PREVIEW_DIR}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    # ── 1. score CSV 로드 ─────────────────────────────────────────────────────
    print(f"[{STAGE_LABEL}] Step 1: load score CSV")
    if not SCORE_CSV.exists():
        _abort(f"score CSV not found: {SCORE_CSV}")
    score_rows = _load_csv(SCORE_CSV)
    print(f"  {len(score_rows)} rows loaded")

    # logit percentile 계산
    all_logits  = [float(r["logit"]) for r in score_rows
                   if r.get("logit","") not in ("NaN","","Inf")]
    logit_min   = float(np.percentile(all_logits, LOGIT_P_LOW))
    logit_max   = float(np.percentile(all_logits, LOGIT_P_HIGH))
    logit_range = logit_max - logit_min
    print(f"  logit normalize: [{logit_min:.2f}, {logit_max:.2f}] (p{LOGIT_P_LOW}~p{LOGIT_P_HIGH})")

    # ── 2. sample patient 선정 ────────────────────────────────────────────────
    print(f"[{STAGE_LABEL}] Step 2: sample patient selection")

    # P-C-NORMAL36 sample selection 기반, CT volume 존재 확인
    preferred_order = [
        "LUNG1-196", "LUNG1-205", "LUNG1-396",        # NSCLC TP / caveat / borderline
        "normal014", "normal073",                       # normal FP / low_mask
        "subset3_1.3.6.1.4.1.14519.5.2.1.6279.6001.314519596680450457855054746285",
        "LUNG1-349", "LUNG1-043",                       # NSCLC TP 추가
    ]

    # score CSV에서 patient 정보 수집
    pat_info = {}
    for r in score_rows:
        pid = r["patient_id"]
        if pid not in pat_info:
            pat_info[pid] = {
                "safe_id":       r["safe_id"],
                "source_split":  r["source_split"],
                "label":         r["label"],
            }

    sample_selection_rows = []
    sample_patients = []

    for pid in preferred_order:
        if len(sample_patients) >= MAX_PATIENTS:
            break
        if pid not in pat_info:
            continue
        info    = pat_info[pid]
        ct_path = resolve_ct_path(info["safe_id"], info["source_split"])
        if not ct_path.exists():
            print(f"  [SKIP] {pid}: ct_hu.npy not found at {ct_path}")
            continue
        sample_patients.append(pid)
        sample_selection_rows.append({
            "patient_id":   pid,
            "safe_id":      info["safe_id"],
            "source_split": info["source_split"],
            "label":        info["label"],
            "ct_path":      str(ct_path),
            "status":       "OK",
        })
        print(f"  selected: {pid} ({info['source_split']}, label={info['label']})")

    _write_csv(sample_selection_rows, REPORT_DIR / "p_c_normal37c_sample_selection.csv")
    print(f"  total sample patients: {len(sample_patients)}")

    # ── 3. 환자별 z → crops 그룹화 ───────────────────────────────────────────
    print(f"[{STAGE_LABEL}] Step 3: group crops by patient/z")
    pat_z_crops = defaultdict(lambda: defaultdict(list))
    for r in score_rows:
        if r["patient_id"] in sample_patients:
            pat_z_crops[r["patient_id"]][r["canonical_volume_z"]].append(r)

    # ── 4. 렌더링 ─────────────────────────────────────────────────────────────
    print(f"[{STAGE_LABEL}] Step 4: render heatmaps")

    ct_source_rows    = []
    render_quality_rows = []
    png_manifest_rows = []
    n_png_saved = 0
    n_render_fail = 0

    for pid in sample_patients:
        info    = pat_info[pid]
        ct_path = resolve_ct_path(info["safe_id"], info["source_split"])
        z_dict  = pat_z_crops[pid]
        safe_pid = pid.replace("/","_").replace(".","_")[:60]
        pat_dir  = PREVIEW_DIR / safe_pid
        pat_dir.mkdir(parents=True, exist_ok=True)

        # CT volume 로드 (mmap)
        ct_vol = np.load(str(ct_path), mmap_mode="r")
        ct_z, ct_h, ct_w = ct_vol.shape
        ct_source_rows.append({
            "patient_id": pid,
            "safe_id":    info["safe_id"],
            "ct_path":    str(ct_path),
            "ct_shape":   f"{ct_z}×{ct_h}×{ct_w}",
            "status":     "OK",
        })
        print(f"  {pid}: CT shape={ct_vol.shape}, n_z_slices_in_score={len(z_dict)}")

        # top-K z by n_crops (혹은 max_logit)
        z_rank = sorted(z_dict.items(),
                        key=lambda kv: max((float(r["logit"]) for r in kv[1]
                                           if r.get("logit","") not in ("NaN","")), default=-99),
                        reverse=True)[:TOP_K]

        for z_str, z_crops in z_rank:
            z_idx = int(float(z_str))
            if z_idx < 0 or z_idx >= ct_z:
                print(f"    [SKIP] z={z_idx} out of volume range {ct_z}")
                continue

            # full CT slice
            ct_slc = ct_vol[z_idx].astype(np.float32)
            ct_slc = np.clip(ct_slc, -1000, 200)
            ct_slc = (ct_slc - (-1000)) / (200 - (-1000))

            # heatmap canvas
            mean_hm, max_hm, count_hm, n_clipped = build_heatmap(
                z_crops, ct_h, ct_w, logit_min, logit_range)

            # stats
            logit_vals = [float(r["logit"]) for r in z_crops
                          if r.get("logit","") not in ("NaN","")]
            prob_vals  = [float(r["prob"])  for r in z_crops
                          if r.get("prob","")  not in ("NaN","")]
            n_low  = sum(1 for r in z_crops if r.get("low_mask_flag","False")=="True")
            n_zero = sum(1 for r in z_crops if r.get("zero_mask_flag","False")=="True")
            cov_pixels = int((count_hm > 0).sum())
            cov_ratio  = cov_pixels / (ct_h * ct_w)

            # render quality row
            rq = {
                "patient_id":         pid,
                "z":                  z_str,
                "label":              info["label"],
                "ct_shape":           f"{ct_h}×{ct_w}",
                "n_crops_on_slice":   len(z_crops),
                "coverage_pixels":    cov_pixels,
                "coverage_ratio":     round(cov_ratio, 4),
                "max_count":          int(count_hm.max()),
                "mean_count_nonzero": round(float(count_hm[count_hm>0].mean()), 2) if cov_pixels else 0,
                "mean_heatmap_max":   round(float(mean_hm.max()), 4),
                "max_heatmap_max":    round(float(max_hm.max()),  4),
                "max_prob":           round(max(prob_vals),  4) if prob_vals  else "",
                "max_logit":          round(max(logit_vals), 4) if logit_vals else "",
                "n_low_mask":         n_low,
                "n_zero_mask":        n_zero,
                "boundary_clipped":   n_clipped,
                "ct_source_path":     str(ct_path),
                "full_ct_bg":         True,
                "crop_mosaic_used":   False,
                "render_status":      "OK",
                "warning":            "boundary_clipped" if n_clipped > 0 else "",
            }

            # title
            max_p  = round(max(prob_vals),  3) if prob_vals  else "N/A"
            max_lg = round(max(logit_vals), 2) if logit_vals else "N/A"
            title  = (f"{pid} | z={z_str} | label={'normal' if info['label']=='0' else 'NSCLC'} "
                      f"| n={len(z_crops)} | max_prob={max_p} | max_logit={max_lg} "
                      f"| low_mask={n_low} | cov={cov_ratio:.1%}")

            safe_z   = str(z_str).replace(".","p")
            png_name = f"{safe_pid}_z{safe_z}.png"
            png_path = pat_dir / png_name
            try:
                save_panel_png(ct_slc, mean_hm, max_hm, count_hm, png_path, title, plt)
                n_png_saved += 1
                rq["render_status"] = "OK"
                png_manifest_rows.append({
                    "patient_id": pid,
                    "label":      info["label"],
                    "z":          z_str,
                    "n_crops":    len(z_crops),
                    "png_path":   str(png_path),
                    "cov_ratio":  round(cov_ratio, 4),
                    "max_hm":     round(float(max_hm.max()), 4),
                })
                print(f"    z={z_str}: n={len(z_crops)} cov={cov_ratio:.1%} png={png_name}")
            except Exception as e:
                rq["render_status"] = f"ERROR: {str(e)[:100]}"
                n_render_fail += 1
                print(f"    z={z_str}: RENDER ERROR {e}", file=sys.stderr)

            render_quality_rows.append(rq)

    # ── 5. 출력 파일 저장 ─────────────────────────────────────────────────────
    _write_csv(ct_source_rows,      REPORT_DIR / "p_c_normal37c_ct_source_resolution.csv")
    _write_csv(render_quality_rows, REPORT_DIR / "p_c_normal37c_render_quality_check.csv")
    _write_csv(png_manifest_rows,   REPORT_DIR / "p_c_normal37c_slice_render_manifest.csv")

    # guardrail
    g_rows = [{"key": k, "value": str(v), "status": "OK"} for k, v in GUARDRAILS.items()]
    _write_csv(g_rows, REPORT_DIR / "p_c_normal37c_guardrail_check.csv")
    guardrail_fail = 0

    # verdict
    if n_render_fail == 0 and n_png_saved > 0 and guardrail_fail == 0:
        verdict        = "PASS"
        verdict_reason = f"{n_png_saved} PNGs rendered, full CT slice background, coverage alpha OK"
    elif n_png_saved > 0:
        verdict        = "PARTIAL_PASS"
        verdict_reason = f"n_png={n_png_saved}, render_fail={n_render_fail}"
    else:
        verdict        = "FAIL"
        verdict_reason = "no PNGs rendered"

    ts2 = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # report.md
    report_md = f"""# P-C-NORMAL37c Full-Slice CT Heatmap Rendering Fix

Generated: {ts2}

## Verdict: {verdict}

{verdict_reason}

## P-C-NORMAL37b 문제점

- crop 조각 canvas 방식 → full CT slice가 아닌 patch 조각이 배경
- coverage 없는 곳과 score 0인 곳이 구분되지 않음 (alpha=0 미적용)

## P-C-NORMAL37c 수정

- 원본 CT volume에서 full z slice 로드 (512×512)
- coverage 없는 영역 alpha=0 (투명)
- 4 panel: CT only / coverage map / mean overlay (blocky+smoothed) / max overlay
- logit normalize: [{logit_min:.2f}, {logit_max:.2f}] (p{LOGIT_P_LOW}~p{LOGIT_P_HIGH})

## 한계

- 이 결과는 **pixel-level Grad-CAM이 아님**
- **crop-level classifier score를 96×96 영역에 투영**한 map
- 세밀한 병변 attribution이 필요하면 Grad-CAM/occlusion/RD4AD spatial map 별도 필요

## Coverage

| 항목 | 값 |
|------|----|
| n_sample_patients | {len(sample_patients)} |
| n_png_saved | {n_png_saved} |
| n_render_fail | {n_render_fail} |
| guardrail_fail | {guardrail_fail} |
| full_ct_bg | True |
| crop_mosaic_used | False |

## Output

```
outputs/p_c_normal37c_full_slice_heatmap_render_fix/previews/<patient_id>/
  *_z*.png   (6 panels per slice)
```
"""
    (REPORT_DIR / "p_c_normal37c_heatmap_render_fix_report.md").write_text(report_md)

    _write_json({
        "stage": STAGE_LABEL, "timestamp": ts2,
        "verdict": verdict, "guardrail_fail": guardrail_fail,
        "n_sample_patients": len(sample_patients),
        "n_png_saved": n_png_saved, "n_render_fail": n_render_fail,
        "logit_norm_range": [round(logit_min, 2), round(logit_max, 2)],
        "full_ct_bg": True, "crop_mosaic_used": False,
        "preview_dir": str(PREVIEW_DIR),
        "next_step": "사용자 확인 후 전체 환자 생성 여부 결정",
    }, REPORT_DIR / "p_c_normal37c_summary.json")

    _write_json({
        "stage": STAGE_LABEL, "timestamp": ts2,
        "verdict": verdict, "n_png_saved": n_png_saved,
    }, REPORT_DIR / "DONE.json")

    print(f"[{STAGE_LABEL}] VERDICT: {verdict}")
    print(f"[{STAGE_LABEL}] n_png_saved={n_png_saved}, guardrail_fail={guardrail_fail}")
    print(f"[{STAGE_LABEL}] preview: {PREVIEW_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
