"""
P-C-NORMAL37b: heatmap overlay (logit 기반 + CT overlay)
- prob 대신 logit 사용 (연속값, percentile clip 후 normalize)
- CT grayscale 위에 heatmap alpha overlay
- 기존 37 결과와 별도 output root
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")
STAGE_LABEL  = "P-C-NORMAL37b"

SCORE_CSV   = PROJECT_ROOT / "outputs/p_c_normal35_full_downstream_scoring/p_c_normal35_full_crop_scores.csv"
SAMPLE_CSV  = PROJECT_ROOT / "outputs/reports/p_c_normal36_heatmap_preflight/p_c_normal36_sample_selection.csv"

OUTPUT_DIR  = PROJECT_ROOT / "outputs/p_c_normal37b_heatmap_overlay"
REPORT_DIR  = PROJECT_ROOT / "outputs/reports/p_c_normal37b_heatmap_overlay"
HEATMAP_DIR = OUTPUT_DIR / "heatmaps"

CROP_SIZE       = 96
HALF            = CROP_SIZE // 2
CANVAS_H        = 512
CANVAS_W        = 512
TOP_K_SLICES    = 5
LOGIT_P_LOW     = 5    # percentile for logit normalization
LOGIT_P_HIGH    = 95
HEATMAP_ALPHA   = 0.55  # CT overlay transparency

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


def build_slice_heatmap(z_crops, logit_min, logit_range):
    """logit normalize → mean/max heatmap canvas."""
    score_canvas = np.zeros((CANVAS_H, CANVAS_W), dtype=np.float32)
    max_canvas   = np.zeros((CANVAS_H, CANVAS_W), dtype=np.float32)
    count_canvas = np.zeros((CANVAS_H, CANVAS_W), dtype=np.float32)

    for r in z_crops:
        try:
            cy = int(float(r["center_y"]))
            cx = int(float(r["center_x"]))
            # logit normalize to [0,1]
            lg = float(r["logit"])
            lg_norm = np.clip((lg - logit_min) / logit_range, 0.0, 1.0)
        except Exception:
            continue
        y0, y1 = max(0, cy - HALF), min(CANVAS_H, cy + HALF)
        x0, x1 = max(0, cx - HALF), min(CANVAS_W, cx + HALF)
        score_canvas[y0:y1, x0:x1] += float(lg_norm)
        max_canvas[y0:y1, x0:x1]   = np.maximum(max_canvas[y0:y1, x0:x1], float(lg_norm))
        count_canvas[y0:y1, x0:x1] += 1.0

    with np.errstate(invalid="ignore"):
        mean_hm = np.where(count_canvas > 0, score_canvas / count_canvas, 0.0)
    return mean_hm, max_canvas, count_canvas


def load_ct_canvas(z_crops):
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


def save_overlay_png(mean_hm, max_hm, ct_canvas, png_path, title, plt):
    """CT 위에 heatmap alpha overlay — 2개 subplot (mean / max)."""
    import matplotlib.cm as cm

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle(title, fontsize=8)

    for ax, hm, subtitle in zip(axes, [mean_hm, max_hm], ["Mean heatmap", "Max heatmap"]):
        ax.imshow(ct_canvas, cmap="gray", vmin=0, vmax=1)
        im = ax.imshow(hm, cmap="hot", alpha=HEATMAP_ALPHA, vmin=0, vmax=1)
        ax.set_title(subtitle, fontsize=8)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

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
    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[{STAGE_LABEL}] output: {HEATMAP_DIR}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. score CSV 로드
    print(f"[{STAGE_LABEL}] Step 1: load score CSV")
    score_rows = _load_csv(SCORE_CSV)
    print(f"  {len(score_rows)} rows")

    # logit 전체 percentile 계산
    all_logits = [float(r["logit"]) for r in score_rows if r.get("logit","") not in ("NaN","","Inf")]
    logit_min  = float(np.percentile(all_logits, LOGIT_P_LOW))
    logit_max  = float(np.percentile(all_logits, LOGIT_P_HIGH))
    logit_range = logit_max - logit_min
    print(f"  logit normalize: [{logit_min:.2f}, {logit_max:.2f}] (p{LOGIT_P_LOW}~p{LOGIT_P_HIGH})")

    # sample patients
    sample_pids = set()
    if SAMPLE_CSV.exists():
        for r in _load_csv(SAMPLE_CSV):
            sample_pids.add(r["patient_id"])

    # 2. 환자 → z → crops 그룹화
    print(f"[{STAGE_LABEL}] Step 2: group by patient / z")
    pat_z_crops = defaultdict(lambda: defaultdict(list))
    for r in score_rows:
        pat_z_crops[r["patient_id"]][r["canonical_volume_z"]].append(r)
    n_patients = len(pat_z_crops)
    print(f"  n_patients={n_patients}")

    # 3. 환자별 heatmap 생성
    print(f"[{STAGE_LABEL}] Step 3: generate overlay heatmaps")
    patient_summary_rows = []
    png_manifest_rows    = []
    n_png_saved = 0

    for p_idx, (pid, z_dict) in enumerate(sorted(pat_z_crops.items())):
        label     = z_dict[next(iter(z_dict))][0]["label"]
        safe_pid  = pid.replace("/","_").replace(".","_")[:60]
        pat_dir   = HEATMAP_DIR / safe_pid
        pat_dir.mkdir(parents=True, exist_ok=True)

        # 각 z slice → heatmap 계산
        z_stats = []
        z_heatmaps = {}
        for z_str, z_crops in sorted(z_dict.items(), key=lambda x: float(x[0])):
            mean_hm, max_hm, count_hm = build_slice_heatmap(z_crops, logit_min, logit_range)
            logit_vals = [float(r["logit"]) for r in z_crops
                          if r.get("logit","") not in ("NaN","","Inf")]
            z_stats.append({
                "z":            z_str,
                "n_crops":      len(z_crops),
                "logit_mean":   round(float(np.mean(logit_vals)), 3) if logit_vals else 0,
                "logit_max":    round(float(np.max(logit_vals)),  3) if logit_vals else 0,
                "hm_mean_max":  round(float(mean_hm.max()), 4),
                "hm_max_max":   round(float(max_hm.max()),  4),
            })
            z_heatmaps[z_str] = (mean_hm, max_hm, z_crops)

        # top-K z by hm_max_max
        top_zs = sorted(z_stats, key=lambda x: -x["hm_max_max"])[:TOP_K_SLICES]

        for zs in top_zs:
            z_str = zs["z"]
            mean_hm, max_hm, z_crops = z_heatmaps[z_str]
            ct_canvas = load_ct_canvas(z_crops)   # 모든 환자 CT 로드

            safe_z   = str(z_str).replace(".","p")
            png_name = f"{safe_pid}_z{safe_z}.png"
            png_path = pat_dir / png_name
            title    = f"{pid} | z={z_str} | label={'normal' if label=='0' else 'NSCLC'} | n={len(z_crops)}"
            save_overlay_png(mean_hm, max_hm, ct_canvas, png_path, title, plt)
            n_png_saved += 1

            png_manifest_rows.append({
                "patient_id":  pid,
                "label":       label,
                "z":           z_str,
                "n_crops":     len(z_crops),
                "png_path":    str(png_path),
                "hm_max":      zs["hm_max_max"],
                "logit_max":   zs["logit_max"],
            })

        all_hm_max = [zs["hm_max_max"] for zs in z_stats]
        patient_summary_rows.append({
            "patient_id":   pid,
            "label":        label,
            "n_slices":     len(z_stats),
            "n_png_saved":  len(top_zs),
            "hm_max":       round(max(all_hm_max), 4) if all_hm_max else 0,
            "hm_mean":      round(float(np.mean(all_hm_max)), 4) if all_hm_max else 0,
        })

        if (p_idx + 1) % 20 == 0:
            print(f"  [{p_idx+1}/{n_patients}] png_saved={n_png_saved}")

    print(f"  done: n_patients={n_patients}, n_png={n_png_saved}")

    _write_csv(patient_summary_rows, REPORT_DIR / "p_c_normal37b_patient_summary.csv")
    _write_csv(png_manifest_rows,    REPORT_DIR / "p_c_normal37b_png_manifest.csv")
    _write_json({
        "stage": STAGE_LABEL, "timestamp": ts,
        "verdict": "PASS", "n_patients": n_patients,
        "n_png_saved": n_png_saved,
        "logit_norm_range": [round(logit_min,2), round(logit_max,2)],
        "heatmap_alpha": HEATMAP_ALPHA,
        "heatmap_dir": str(HEATMAP_DIR),
        "model_forward": False, "training_run": False,
    }, REPORT_DIR / "DONE.json")

    ts2 = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{STAGE_LABEL}] DONE: {n_png_saved} PNGs @ {HEATMAP_DIR}")
    print(f"[{STAGE_LABEL}] logit norm: [{logit_min:.2f}, {logit_max:.2f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
