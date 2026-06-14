"""
normal reference bank v4 LUNG1-052 crop preview actual generation
- candidate + top3 normal reference crops (96x96 each)
- 5 PNG: candidate x1, normal_ref x3, montage x1
- CT read-only mmap only
- same-z 표현 금지: "same cell" / "same lung-ROI position cell" 만 사용
"""

import json
import csv
import sys
from pathlib import Path
from datetime import date, datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ──────────────────────────────────────────────
# GUARD FLAGS
# ──────────────────────────────────────────────
ALLOW_CT_LOAD              = True
ALLOW_PREVIEW_PNG_WRITE    = True
ALLOW_STAGE2_HOLDOUT       = False
ALLOW_MODEL_FORWARD        = False
ALLOW_FEATURE_EXTRACTION   = False
ALLOW_CONTRIBUTION_RECALC  = False
ALLOW_FULL300              = False

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
PREFLIGHT_ROOT = Path(
    "outputs/position-aware-padim-v1/reports/"
    "reference_bank_v4_lung1_052_crop_preview_preflight"
)
OUTPUT_ROOT = Path(
    "outputs/position-aware-padim-v1/visualizations/"
    "reference_bank_v4_lung1_052_crop_preview"
)

PREFLIGHT_DONE   = PREFLIGHT_ROOT / "DONE.json"
PLAN_CSV         = PREFLIGHT_ROOT / "crop_preview_plan_lung1_052_v4.csv"

# ──────────────────────────────────────────────
# LUNG WINDOW
# ──────────────────────────────────────────────
LUNG_WINDOW_CENTER = -600
LUNG_WINDOW_WIDTH  = 1500
LUNG_WIN_LO = LUNG_WINDOW_CENTER - LUNG_WINDOW_WIDTH // 2   # -1350
LUNG_WIN_HI = LUNG_WINDOW_CENTER + LUNG_WINDOW_WIDTH // 2   #   150

DISPLAY_PX = 256   # each panel rendered at this size

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def _abort(msg: str, code: int = 2):
    print(f"\nBLOCKED: {msg}", file=sys.stderr)
    sys.exit(code)


def apply_lung_window(arr_2d: np.ndarray) -> np.ndarray:
    """int16 HU → float clipped to lung window → uint8 [0,255]"""
    clipped = np.clip(arr_2d.astype(np.float32), LUNG_WIN_LO, LUNG_WIN_HI)
    normed = (clipped - LUNG_WIN_LO) / (LUNG_WIN_HI - LUNG_WIN_LO)
    return (normed * 255).astype(np.uint8)


def load_crop_mmap(ct_path: str, z: int, y0: int, x0: int, y1: int, x1: int) -> np.ndarray:
    """CT를 read-only mmap으로 열고 지정 slice/crop만 반환"""
    if not ALLOW_CT_LOAD:
        _abort("ALLOW_CT_LOAD=False — CT load not permitted")
    ct = np.load(ct_path, mmap_mode="r")   # read-only
    crop = ct[z, y0:y1, x0:x1].copy()     # 필요한 부분만 메모리로
    del ct                                  # mmap 참조 해제
    return crop


def short_pid(patient_id: str, max_len: int = 22) -> str:
    """긴 patient_id를 표시용으로 축약"""
    if len(patient_id) <= max_len:
        return patient_id
    # LUNA16 형식: subsetN_...long_uid...
    parts = patient_id.split("_", 1)
    prefix = parts[0] if len(parts) > 1 else ""
    uid = parts[1] if len(parts) > 1 else patient_id
    return f"{prefix}_...{uid[-12:]}"


def save_single_png(img_uint8: np.ndarray, out_path: Path, title: str):
    """96×96 crop을 단독 PNG로 저장 (DISPLAY_PX 크기)"""
    if not ALLOW_PREVIEW_PNG_WRITE:
        _abort("ALLOW_PREVIEW_PNG_WRITE=False — PNG write not permitted")
    fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=96)
    ax.imshow(img_uint8, cmap="gray", vmin=0, vmax=255,
              interpolation="nearest", aspect="equal")
    ax.set_title(title, fontsize=7, pad=3)
    ax.axis("off")
    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=96, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("normal reference bank v4 LUNG1-052 crop preview")
    print("=" * 60)

    # ── guard checks ──
    if ALLOW_STAGE2_HOLDOUT:
        _abort("ALLOW_STAGE2_HOLDOUT must be False")
    if ALLOW_MODEL_FORWARD or ALLOW_FEATURE_EXTRACTION or ALLOW_CONTRIBUTION_RECALC:
        _abort("model/feature/contribution guards must be False")

    # ── output collision ──
    if (OUTPUT_ROOT / "DONE.json").exists():
        _abort("existing DONE.json at output root. Archive or use new root.")

    # ── preflight DONE check ──
    if not PREFLIGHT_DONE.exists():
        _abort("preflight DONE.json not found")
    done_pre = json.loads(PREFLIGHT_DONE.read_text())
    if done_pre.get("verdict") != "PASS":
        _abort(f"preflight verdict is not PASS: {done_pre.get('verdict')}")

    # ── load plan ──
    if not PLAN_CSV.exists():
        _abort(f"plan CSV not found: {PLAN_CSV}")
    plan_df = pd.read_csv(PLAN_CSV)
    print(f"\n[plan loaded] {len(plan_df)} rows")

    cand_row = plan_df[plan_df["role"] == "candidate"].iloc[0]
    ref_rows = plan_df[plan_df["role"].str.startswith("normal_ref")].sort_values("role")

    if len(ref_rows) != 3:
        _abort(f"expected 3 normal_ref rows, got {len(ref_rows)}")

    # ── output dir ──
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    errors = []
    generated_pngs = []
    crops_info = []

    # ── load and save candidate crop ──
    print("\n[1/5] candidate crop...")
    try:
        cand_crop_raw = load_crop_mmap(
            cand_row["ct_path"], int(cand_row["local_z"]),
            int(cand_row["crop_y0"]), int(cand_row["crop_x0"]),
            int(cand_row["crop_y1"]), int(cand_row["crop_x1"]),
        )
        if cand_crop_raw.shape != (96, 96):
            errors.append(f"candidate crop shape {cand_crop_raw.shape} != (96,96)")
        cand_img = apply_lung_window(cand_crop_raw)
        title = (
            f"LUNG1-052 (candidate)\n"
            f"z={int(cand_row['local_z'])}  "
            f"y{int(cand_row['crop_y0'])}:{int(cand_row['crop_y1'])} "
            f"x{int(cand_row['crop_x0'])}:{int(cand_row['crop_x1'])}\n"
            f"cell: {cand_row['cell_key']}"
        )
        out_cand = OUTPUT_ROOT / "candidate_crop_lung1_052.png"
        save_single_png(cand_img, out_cand, title)
        generated_pngs.append(out_cand)
        print(f"  → saved: candidate_crop_lung1_052.png  shape={cand_crop_raw.shape}")
        crops_info.append({
            "role": "candidate",
            "patient_id": str(cand_row["patient_id"]),
            "z": int(cand_row["local_z"]),
            "crop_y0": int(cand_row["crop_y0"]),
            "crop_x0": int(cand_row["crop_x0"]),
            "crop_y1": int(cand_row["crop_y1"]),
            "crop_x1": int(cand_row["crop_x1"]),
            "cell_key": str(cand_row["cell_key"]),
            "png": "candidate_crop_lung1_052.png",
        })
    except Exception as e:
        errors.append(f"candidate crop failed: {e}")
        cand_img = np.zeros((96, 96), dtype=np.uint8)
        crops_info.append({"role": "candidate", "error": str(e)})

    # ── load and save normal reference crops ──
    ref_imgs = []
    for _, ref_row in ref_rows.iterrows():
        role = ref_row["role"]
        rank = role.split("_")[-1]
        print(f"\n[{int(rank)+1}/5] {role}...")
        try:
            ref_crop_raw = load_crop_mmap(
                ref_row["ct_path"], int(ref_row["local_z"]),
                int(ref_row["crop_y0"]), int(ref_row["crop_x0"]),
                int(ref_row["crop_y1"]), int(ref_row["crop_x1"]),
            )
            if ref_crop_raw.shape != (96, 96):
                errors.append(f"{role} crop shape {ref_crop_raw.shape} != (96,96)")
            ref_img = apply_lung_window(ref_crop_raw)
            ref_imgs.append(ref_img)
            pid_short = short_pid(str(ref_row["patient_id"]))
            title = (
                f"{role}\n"
                f"{pid_short}\n"
                f"z={int(ref_row['local_z'])}  "
                f"y{int(ref_row['crop_y0'])}:{int(ref_row['crop_y1'])} "
                f"x{int(ref_row['crop_x0'])}:{int(ref_row['crop_x1'])}\n"
                f"cell: {ref_row['cell_key']}"
            )
            out_ref = OUTPUT_ROOT / f"normal_ref_{rank}_v4.png"
            save_single_png(ref_img, out_ref, title)
            generated_pngs.append(out_ref)
            print(f"  → saved: normal_ref_{rank}_v4.png  shape={ref_crop_raw.shape}")
            crops_info.append({
                "role": role,
                "patient_id": str(ref_row["patient_id"]),
                "z": int(ref_row["local_z"]),
                "crop_y0": int(ref_row["crop_y0"]),
                "crop_x0": int(ref_row["crop_x0"]),
                "crop_y1": int(ref_row["crop_y1"]),
                "crop_x1": int(ref_row["crop_x1"]),
                "cell_key": str(ref_row["cell_key"]),
                "reference_quality_score": float(ref_row.get("reference_quality_score", 0)),
                "png": f"normal_ref_{rank}_v4.png",
            })
        except Exception as e:
            errors.append(f"{role} crop failed: {e}")
            ref_imgs.append(np.zeros((96, 96), dtype=np.uint8))
            crops_info.append({"role": role, "error": str(e)})

    # ── montage ──
    print("\n[5/5] montage...")
    all_imgs = [cand_img] + ref_imgs   # 4 panels

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5), dpi=120)
    fig.patch.set_facecolor("#1a1a1a")

    panel_labels = ["LUNG1-052\n(candidate)"] + [
        f"{r['role']}\n{short_pid(r.get('patient_id','?'), 18)}"
        for r in crops_info[1:4]
    ]
    panel_z = [cand_row["local_z"]] + [
        ref_rows.iloc[i]["local_z"] for i in range(3)
    ]
    panel_bbox = [
        (cand_row["crop_y0"], cand_row["crop_x0"],
         cand_row["crop_y1"], cand_row["crop_x1"])
    ] + [
        (ref_rows.iloc[i]["crop_y0"], ref_rows.iloc[i]["crop_x0"],
         ref_rows.iloc[i]["crop_y1"], ref_rows.iloc[i]["crop_x1"])
        for i in range(3)
    ]
    cell_key = str(cand_row["cell_key"])

    for idx, (ax, img) in enumerate(zip(axes, all_imgs)):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255,
                  interpolation="nearest", aspect="equal")
        ax.set_facecolor("#1a1a1a")

        y0, x0, y1, x1 = panel_bbox[idx]
        caption = (
            f"{panel_labels[idx]}\n"
            f"z={int(panel_z[idx])}  "
            f"y{int(y0)}:{int(y1)} x{int(x0)}:{int(x1)}\n"
            f"cell: {cell_key}"
        )
        ax.set_title(caption, color="white", fontsize=6.5, pad=4,
                     fontfamily="monospace")
        ax.tick_params(left=False, bottom=False,
                       labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_edgecolor("#555555")
            spine.set_linewidth(0.8)

        # candidate 강조 테두리
        if idx == 0:
            for spine in ax.spines.values():
                spine.set_edgecolor("#ffcc00")
                spine.set_linewidth(2.0)

    # 제목
    fig.suptitle(
        "LUNG1-052 exact-cell normal reference preview",
        color="white", fontsize=11, fontweight="bold", y=1.01,
    )

    # 주석 (same-z 금지 문구)
    fig.text(
        0.5, -0.02,
        "same cell comparison, not same-z matching  |  "
        "References are matched by lung-ROI position cell; "
        "z-direction alignment is limited.",
        ha="center", va="top", color="#aaaaaa", fontsize=6.5,
        style="italic",
    )

    fig.tight_layout(pad=0.8)
    out_montage = OUTPUT_ROOT / "lung1_052_v4_reference_preview_montage.png"
    fig.savefig(out_montage, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    generated_pngs.append(out_montage)
    print(f"  → saved: lung1_052_v4_reference_preview_montage.png")

    # ── preview_metadata.json ──
    meta_dict = {
        "date": str(date.today()),
        "candidate_patient_id": str(cand_row["patient_id"]),
        "candidate_z": int(cand_row["local_z"]),
        "candidate_crop_y0": int(cand_row["crop_y0"]),
        "candidate_crop_x0": int(cand_row["crop_x0"]),
        "candidate_crop_y1": int(cand_row["crop_y1"]),
        "candidate_crop_x1": int(cand_row["crop_x1"]),
        "cell_key": cell_key,
        "normal_refs": [
            {
                "role": r.get("role"),
                "patient_id": r.get("patient_id"),
                "z": r.get("z"),
                "crop_y0": r.get("crop_y0"),
                "crop_x0": r.get("crop_x0"),
                "crop_y1": r.get("crop_y1"),
                "crop_x1": r.get("crop_x1"),
                "reference_quality_score": r.get("reference_quality_score"),
            }
            for r in crops_info[1:4]
        ],
        "display_size_equal": True,
        "crop_size": 96,
        "window_type": "lung",
        "lung_window_center": LUNG_WINDOW_CENTER,
        "lung_window_width": LUNG_WINDOW_WIDTH,
        "z_orientation_limitation": True,
        "not_same_z_matched": True,
        "stage2_holdout_accessed": False,
        "model_forward_occurred": False,
        "feature_extraction_occurred": False,
        "contribution_recalc_occurred": False,
        "existing_artifact_modified": False,
        "png_count": len(generated_pngs),
        "errors": errors,
    }
    meta_json = OUTPUT_ROOT / "preview_metadata.json"
    meta_json.write_text(json.dumps(meta_dict, indent=2))
    print(f"\n  → saved: preview_metadata.json")

    # ── preview_index.csv ──
    index_rows = []
    for r in crops_info:
        index_rows.append({
            "role": r.get("role", ""),
            "patient_id": r.get("patient_id", ""),
            "z": r.get("z", ""),
            "crop_y0": r.get("crop_y0", ""),
            "crop_x0": r.get("crop_x0", ""),
            "crop_y1": r.get("crop_y1", ""),
            "crop_x1": r.get("crop_x1", ""),
            "cell_key": r.get("cell_key", cell_key),
            "reference_quality_score": r.get("reference_quality_score", ""),
            "png": r.get("png", ""),
        })
    index_df = pd.DataFrame(index_rows)
    index_csv = OUTPUT_ROOT / "preview_index.csv"
    index_df.to_csv(index_csv, index=False)
    print(f"  → saved: preview_index.csv  ({len(index_df)} rows)")

    # ── runtime_summary.json ──
    runtime_dict = {
        "date": str(date.today()),
        "png_generated": [p.name for p in generated_pngs],
        "png_count": len(generated_pngs),
        "ct_load_mode": "mmap_mode=r (read-only)",
        "stage2_holdout_accessed": False,
        "model_forward": False,
        "feature_extraction": False,
        "contribution_recalc": False,
        "existing_artifact_modified": False,
        "errors": errors,
    }
    runtime_json = OUTPUT_ROOT / "runtime_summary.json"
    runtime_json.write_text(json.dumps(runtime_dict, indent=2))
    print(f"  → saved: runtime_summary.json")

    # ── errors.csv ──
    errors_csv = OUTPUT_ROOT / "errors.csv"
    with open(errors_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "message"])
        for e in errors:
            w.writerow(["crop_preview", e])
    print(f"  → saved: errors.csv ({len(errors)} errors)")

    # ── verdict ──
    blocked = []
    if ALLOW_STAGE2_HOLDOUT:
        blocked.append("ALLOW_STAGE2_HOLDOUT=True")
    if len(generated_pngs) < 5:
        blocked.append(f"PNG count {len(generated_pngs)} < 5")

    if blocked:
        verdict = "BLOCKED"
    elif len(errors) > 0:
        verdict = "PARTIAL_PASS" if len(generated_pngs) == 5 else "NEEDS_FIX"
    else:
        verdict = "PASS"

    # ── DONE.json ──
    done_dict = {
        "status": "DONE",
        "verdict": verdict,
        "date": str(date.today()),
        "png_count": len(generated_pngs),
        "errors": len(errors),
        "blockers": len(blocked),
        "outputs": [p.name for p in generated_pngs] + [
            "preview_metadata.json",
            "preview_index.csv",
            "runtime_summary.json",
            "errors.csv",
            "DONE.json",
        ],
    }
    done_json = OUTPUT_ROOT / "DONE.json"
    done_json.write_text(json.dumps(done_dict, indent=2))
    print(f"  → saved: DONE.json")

    # ── final summary ──
    print()
    print("=" * 60)
    print(f"VERDICT: {verdict}")
    print(f"  PNG generated: {len(generated_pngs)}/5")
    for p in generated_pngs:
        print(f"    {p.name}")
    print(f"  errors: {len(errors)}")
    print(f"  CT load mode: mmap read-only")
    print(f"  stage2_holdout: 0 / model/feature/contribution: 0")
    print(f"  not_same_z_matched: True")
    print("=" * 60)

    if verdict == "BLOCKED":
        sys.exit(2)
    elif verdict == "NEEDS_FIX":
        sys.exit(1)


if __name__ == "__main__":
    main()
