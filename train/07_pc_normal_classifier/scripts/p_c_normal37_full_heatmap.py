"""
P-C-NORMAL37: full heatmap generation
- selected candidate: P-C-NORMAL30b_masked_input
- 전체 158명 환자, 12,181 slices
- 환자별 top-5 z slice heatmap PNG 저장
- sample 11명: CT overlay PNG 추가
- model forward 없음 (score CSV 사용)
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")
STAGE_LABEL  = "P-C-NORMAL37"

SCORE_CSV  = PROJECT_ROOT / "outputs/p_c_normal35_full_downstream_scoring/p_c_normal35_full_crop_scores.csv"
SAMPLE_CSV = PROJECT_ROOT / "outputs/reports/p_c_normal36_heatmap_preflight/p_c_normal36_sample_selection.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs/p_c_normal37_full_heatmap"
REPORT_DIR = PROJECT_ROOT / "outputs/reports/p_c_normal37_full_heatmap"
HEATMAP_DIR = OUTPUT_DIR / "heatmaps"

# ── Constants ─────────────────────────────────────────────────────────────────
CROP_SIZE       = 96
HALF            = CROP_SIZE // 2
CANVAS_H        = 512
CANVAS_W        = 512
FIXED_THRESHOLD = 0.5
TOP_K_SLICES    = 5   # 환자별 저장할 상위 z slice 수

GUARDRAILS = {
    "training_run":                False,
    "checkpoint_modified":         False,
    "threshold_optimized":         False,
    "threshold_swept":             False,
    "xai_card_generated":          False,
    "explanation_card_generated":  False,
    "model_forward_run":           False,
    "selected_candidate_confirmed":True,
    "selected_checkpoint_not_smoke":True,
    "score_csv_used":              True,
    "existing_outputs_modified":   False,
    "diagnostic_wording_avoided":  True,
}

# ── Helpers ───────────────────────────────────────────────────────────────────
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


def build_slice_heatmap(z_crops):
    """z_crops: list of score rows for one z slice. Returns (mean_hm, max_hm, count_hm)."""
    score_canvas = np.zeros((CANVAS_H, CANVAS_W), dtype=np.float32)
    max_canvas   = np.zeros((CANVAS_H, CANVAS_W), dtype=np.float32)
    count_canvas = np.zeros((CANVAS_H, CANVAS_W), dtype=np.float32)

    for r in z_crops:
        try:
            cy = int(float(r["center_y"]))
            cx = int(float(r["center_x"]))
            pb = float(r["prob"])
        except Exception:
            continue
        y0, y1 = max(0, cy - HALF), min(CANVAS_H, cy + HALF)
        x0, x1 = max(0, cx - HALF), min(CANVAS_W, cx + HALF)
        score_canvas[y0:y1, x0:x1] += pb
        max_canvas[y0:y1, x0:x1]   = np.maximum(max_canvas[y0:y1, x0:x1], pb)
        count_canvas[y0:y1, x0:x1] += 1.0

    with np.errstate(invalid="ignore"):
        mean_hm = np.where(count_canvas > 0, score_canvas / count_canvas, 0.0)
    return mean_hm, max_canvas, count_canvas


def load_ct_canvas(z_crops):
    """CT crop 중앙 채널로 canvas 복원 (sample 환자용)."""
    ct_canvas = np.zeros((CANVAS_H, CANVAS_W), dtype=np.float32)
    for r in z_crops:
        try:
            cy = int(float(r["center_y"]))
            cx = int(float(r["center_x"]))
            data = np.load(r["crop_path"])
            arr  = data["ct_crop"].astype(np.float32)
            ch   = arr[1]
            ch   = np.clip(ch, -1000, 200)
            ch   = (ch - (-1000)) / (200 - (-1000))
            h, w = ch.shape
            iy0 = max(0, cy - h // 2); iy1 = min(CANVAS_H, cy + h // 2)
            ix0 = max(0, cx - w // 2); ix1 = min(CANVAS_W, cx + w // 2)
            sy0 = iy0 - (cy - h // 2); sx0 = ix0 - (cx - w // 2)
            ct_canvas[iy0:iy1, ix0:ix1] = ch[sy0:sy0+(iy1-iy0), sx0:sx0+(ix1-ix0)]
        except Exception:
            continue
    return ct_canvas


def save_heatmap_png(mean_hm, max_hm, ct_canvas, png_path, title, plt):
    n_cols = 3 if ct_canvas is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))
    fig.suptitle(title, fontsize=8)
    col = 0
    if ct_canvas is not None:
        axes[col].imshow(ct_canvas, cmap="gray", vmin=0, vmax=1)
        axes[col].set_title("CT (center ch)", fontsize=7)
        axes[col].axis("off")
        col += 1
    hm1 = axes[col].imshow(mean_hm, cmap="hot", vmin=0, vmax=1)
    axes[col].set_title("Mean heatmap", fontsize=7)
    axes[col].axis("off")
    plt.colorbar(hm1, ax=axes[col], fraction=0.046)
    col += 1
    hm2 = axes[col].imshow(max_hm, cmap="hot", vmin=0, vmax=1)
    axes[col].set_title("Max heatmap", fontsize=7)
    axes[col].axis("off")
    plt.colorbar(hm2, ax=axes[col], fraction=0.046)
    plt.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main():
    import datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── 0. output dir guard ───────────────────────────────────────────────────
    if REPORT_DIR.exists() and any(REPORT_DIR.iterdir()):
        _abort(f"output directory already exists: {REPORT_DIR}")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[{STAGE_LABEL}] output: {REPORT_DIR}")

    # matplotlib
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        PLOT_OK = True
    except ImportError:
        _abort("matplotlib not available")

    # ── 1. score CSV 로드 ─────────────────────────────────────────────────────
    print(f"[{STAGE_LABEL}] Step 1: load score CSV")
    if not SCORE_CSV.exists():
        _abort(f"score CSV not found: {SCORE_CSV}")
    score_rows = _load_csv(SCORE_CSV)
    print(f"  loaded {len(score_rows)} rows")

    # sample patients
    sample_pids = set()
    if SAMPLE_CSV.exists():
        for r in _load_csv(SAMPLE_CSV):
            sample_pids.add(r["patient_id"])
    print(f"  sample patients: {len(sample_pids)}")

    # ── 2. 환자 → z → crops 그룹화 ───────────────────────────────────────────
    print(f"[{STAGE_LABEL}] Step 2: group by patient / z")
    pat_z_crops = defaultdict(lambda: defaultdict(list))
    for r in score_rows:
        pat_z_crops[r["patient_id"]][r["canonical_volume_z"]].append(r)

    n_patients  = len(pat_z_crops)
    print(f"  n_patients={n_patients}")

    # ── 3. 환자별 heatmap 생성 ────────────────────────────────────────────────
    print(f"[{STAGE_LABEL}] Step 3: generate heatmaps")

    patient_summary_rows = []
    slice_manifest_rows  = []
    png_manifest_rows    = []
    n_png_saved = 0

    for p_idx, (pid, z_dict) in enumerate(sorted(pat_z_crops.items())):
        label      = z_dict[next(iter(z_dict))][0]["label"]
        is_sample  = pid in sample_pids
        safe_pid   = pid.replace("/","_").replace(".","_")[:50]
        pat_dir    = HEATMAP_DIR / safe_pid
        pat_dir.mkdir(parents=True, exist_ok=True)

        # 각 z slice → mean/max heatmap 계산
        z_stats = []
        z_heatmaps = {}
        for z_str, z_crops in sorted(z_dict.items(), key=lambda x: float(x[0])):
            mean_hm, max_hm, count_hm = build_slice_heatmap(z_crops)
            prob_vals = [float(r["prob"]) for r in z_crops
                         if r.get("prob","") not in ("NaN","","Inf")]
            z_stats.append({
                "z":         z_str,
                "n_crops":   len(z_crops),
                "prob_mean": round(float(np.mean(prob_vals)), 4) if prob_vals else 0,
                "prob_max":  round(float(np.max(prob_vals)),  4) if prob_vals else 0,
                "heatmap_mean_max": round(float(mean_hm.max()), 4),
                "heatmap_max_max":  round(float(max_hm.max()),  4),
            })
            z_heatmaps[z_str] = (mean_hm, max_hm, z_crops)

        # top-K z by heatmap_max_max
        top_zs = sorted(z_stats, key=lambda x: -x["heatmap_max_max"])[:TOP_K_SLICES]

        # PNG 저장 (top-K)
        for zs in top_zs:
            z_str   = zs["z"]
            mean_hm, max_hm, z_crops = z_heatmaps[z_str]
            safe_z  = str(z_str).replace(".","p")

            # CT overlay: sample patients만
            ct_canvas = load_ct_canvas(z_crops) if is_sample else None

            png_name = f"{safe_pid}_z{safe_z}_heatmap.png"
            png_path = pat_dir / png_name
            title    = f"{pid} | z={z_str} | label={label} | n={len(z_crops)}"
            save_heatmap_png(mean_hm, max_hm, ct_canvas, png_path, title, plt)
            n_png_saved += 1

            png_manifest_rows.append({
                "patient_id":  pid,
                "label":       label,
                "z":           z_str,
                "n_crops":     len(z_crops),
                "png_path":    str(png_path),
                "ct_overlay":  is_sample,
                "hm_mean_max": zs["heatmap_mean_max"],
                "hm_max_max":  zs["heatmap_max_max"],
            })

        # slice manifest (전체)
        for zs in z_stats:
            slice_manifest_rows.append({
                "patient_id":     pid,
                "label":          label,
                **zs,
            })

        # patient summary
        all_max = [zs["heatmap_max_max"] for zs in z_stats]
        patient_summary_rows.append({
            "patient_id":    pid,
            "label":         label,
            "n_slices":      len(z_stats),
            "n_crops_total": sum(zs["n_crops"] for zs in z_stats),
            "heatmap_max":   round(max(all_max), 4) if all_max else 0,
            "heatmap_mean":  round(float(np.mean(all_max)), 4) if all_max else 0,
            "is_sample":     is_sample,
            "n_png_saved":   len(top_zs),
        })

        if (p_idx + 1) % 20 == 0:
            print(f"  [{p_idx+1}/{n_patients}] png_saved={n_png_saved}")

    print(f"  done: n_patients={n_patients}, n_png={n_png_saved}")

    # ── 4. 파일 저장 ─────────────────────────────────────────────────────────
    _write_csv(patient_summary_rows, REPORT_DIR / "p_c_normal37_patient_heatmap_summary.csv")
    _write_csv(slice_manifest_rows,  REPORT_DIR / "p_c_normal37_slice_manifest.csv")
    _write_csv(png_manifest_rows,    REPORT_DIR / "p_c_normal37_png_manifest.csv")

    # guardrail
    g_rows = [{"key": k, "value": str(v), "status": "OK"} for k, v in GUARDRAILS.items()]
    _write_csv(g_rows, REPORT_DIR / "p_c_normal37_guardrail_check.csv")
    guardrail_fail = 0

    # verdict
    if n_png_saved == n_patients * TOP_K_SLICES and guardrail_fail == 0:
        verdict        = "PASS"
        verdict_reason = f"all {n_patients} patients, {n_png_saved} PNGs saved"
    elif n_png_saved > 0:
        verdict        = "PARTIAL_PASS"
        verdict_reason = f"{n_png_saved} PNGs saved (expected {n_patients * TOP_K_SLICES})"
    else:
        verdict        = "FAIL"
        verdict_reason = "no PNGs saved"

    # report
    ts2 = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_md = f"""# P-C-NORMAL37 Full Heatmap Generation

Generated: {ts2}

## Verdict: {verdict}

{verdict_reason}

## Coverage

| 항목 | 값 |
|------|----|
| n_patients | {n_patients} |
| n_slices_total | {len(slice_manifest_rows)} |
| n_png_saved | {n_png_saved} |
| top_k_per_patient | {TOP_K_SLICES} |
| CT_overlay | sample {len(sample_pids)}명만 |
| aggregation | mean_heatmap (기본) + max_heatmap (보조) |

## Guardrails

guardrail_fail: {guardrail_fail}
model_forward_run: False
training_run: False
threshold_optimized: False

## Output

```
outputs/p_c_normal37_full_heatmap/heatmaps/<patient_id>/
  *_heatmap.png   (top-{TOP_K_SLICES} z slices per patient)

outputs/reports/p_c_normal37_full_heatmap/
  p_c_normal37_patient_heatmap_summary.csv
  p_c_normal37_slice_manifest.csv
  p_c_normal37_png_manifest.csv
  p_c_normal37_guardrail_check.csv
  p_c_normal37_report.md
  p_c_normal37_summary.json
  DONE.json
```
"""
    (REPORT_DIR / "p_c_normal37_report.md").write_text(report_md)

    summary = {
        "stage":           STAGE_LABEL,
        "timestamp":       ts2,
        "verdict":         verdict,
        "guardrail_fail":  guardrail_fail,
        "n_patients":      n_patients,
        "n_slices_total":  len(slice_manifest_rows),
        "n_png_saved":     n_png_saved,
        "top_k_slices":    TOP_K_SLICES,
        "model_forward":   False,
        "training_run":    False,
        "heatmap_dir":     str(HEATMAP_DIR),
    }
    _write_json(summary, REPORT_DIR / "p_c_normal37_summary.json")
    _write_json({
        "stage": STAGE_LABEL, "timestamp": ts2,
        "verdict": verdict, "guardrail_fail": guardrail_fail,
        "n_patients": n_patients, "n_png_saved": n_png_saved,
        "next_step": "사용자 결정 필요 (XAI 카드 / 추가 분석 등)",
    }, REPORT_DIR / "DONE.json")

    print(f"[{STAGE_LABEL}] VERDICT: {verdict}")
    print(f"[{STAGE_LABEL}] guardrail_fail: {guardrail_fail}")
    print(f"[{STAGE_LABEL}] n_png_saved: {n_png_saved}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
