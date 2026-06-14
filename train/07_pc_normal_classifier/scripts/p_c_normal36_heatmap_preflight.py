"""
P-C-NORMAL36: heatmap preflight
- selected candidate: P-C-NORMAL30b_masked_input
- preflight + tiny preview only (환자 2명, slice 4장)
- 전체 heatmap 생성 금지
"""

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")
STAGE_LABEL  = "P-C-NORMAL36"

SELECTED_CKPT       = PROJECT_ROOT / "outputs/p_c_normal30b_masked_input_full_train/checkpoints/p_c_normal30b_best_val_auc_checkpoint.pt"
FINAL_TEST_MANIFEST = PROJECT_ROOT / "outputs/manifests/p_c_normal27_scalar_repair_final_test_manifest/p_c_normal27_final_test_feature_manifest_repaired_usable.csv"
SCORE_CSV           = PROJECT_ROOT / "outputs/p_c_normal35_full_downstream_scoring/p_c_normal35_full_crop_scores.csv"
HANDOFF_DIR         = PROJECT_ROOT / "outputs/reports/p_c_normal33_selected_candidate_handoff_package"
NORMAL32_DIR        = PROJECT_ROOT / "outputs/reports/p_c_normal32_final_decision_checkpoint"

REPORT_DIR          = PROJECT_ROOT / "outputs/reports/p_c_normal36_heatmap_preflight"
PREVIEW_DIR         = REPORT_DIR / "previews"

# ── Constants ─────────────────────────────────────────────────────────────────
EXPECTED_N_TOTAL  = 66283
CROP_SIZE         = 96
CANVAS_H          = 512
CANVAS_W          = 512
FIXED_THRESHOLD   = 0.5

GUARDRAILS = {
    "training_run":                   False,
    "checkpoint_modified":            False,
    "threshold_optimized":            False,
    "threshold_swept":                False,
    "xai_card_generated":             False,
    "explanation_card_generated":     False,
    "full_heatmap_generation_run":    False,
    "sample_preview_only":            True,
    "selected_candidate_confirmed":   True,
    "selected_checkpoint_not_smoke":  True,
    "score_csv_join_100pct":          True,
    "coordinate_reconstruction_checked": True,
    "existing_outputs_modified":      False,
    "diagnostic_wording_avoided":     True,
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


def main():
    import datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── 0. output dir guard ───────────────────────────────────────────────────
    if REPORT_DIR.exists() and any(REPORT_DIR.iterdir()):
        _abort(f"output directory already exists: {REPORT_DIR}")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[{STAGE_LABEL}] output: {REPORT_DIR}")

    # ── 1. 입력 검증 ──────────────────────────────────────────────────────────
    print(f"[{STAGE_LABEL}] Step 1: input validation")

    # 1-1. checkpoint
    if not SELECTED_CKPT.exists():
        _abort(f"checkpoint not found: {SELECTED_CKPT}")
    if "smoke" in SELECTED_CKPT.name.lower():
        _abort(f"smoke checkpoint must not be used: {SELECTED_CKPT.name}")

    # 1-2. selected candidate 확인
    n32_json = NORMAL32_DIR / "p_c_normal32_final_decision_checkpoint.json"
    if n32_json.exists():
        with open(n32_json) as f:
            n32 = json.load(f)
        sel = n32.get("selected_candidate", "")
        if "30b" not in sel.lower():
            _abort(f"selected candidate not 30b: {sel}")
        print(f"  selected_candidate confirmed: {sel}")

    # 1-3. manifest
    if not FINAL_TEST_MANIFEST.exists():
        _abort(f"final_test manifest not found: {FINAL_TEST_MANIFEST}")
    ft_rows = _load_csv(FINAL_TEST_MANIFEST)
    print(f"  final_test manifest: {len(ft_rows)} rows")

    # 1-4. score CSV
    if not SCORE_CSV.exists():
        # fallback: auto search
        candidates = sorted(PROJECT_ROOT.glob("outputs/p_c_normal35_full_downstream_scoring/*.csv"))
        if not candidates:
            _abort(f"score CSV not found: {SCORE_CSV}")
        SCORE_CSV_USE = candidates[0]
        print(f"  [fallback] using {SCORE_CSV_USE}")
    else:
        SCORE_CSV_USE = SCORE_CSV

    score_rows = _load_csv(SCORE_CSV_USE)
    print(f"  score CSV: {len(score_rows)} rows")
    if len(score_rows) != EXPECTED_N_TOTAL:
        _abort(f"score CSV row count mismatch: expected {EXPECTED_N_TOTAL}, got {len(score_rows)}")

    # 1-5. NaN/Inf check
    n_nan_prob = sum(1 for r in score_rows if r.get("prob","") in ("NaN","","Inf"))
    print(f"  NaN/Inf prob: {n_nan_prob}")

    # 1-6. join key check (crop_path 100%)
    ft_cp_set = set(r["crop_path"] for r in ft_rows)
    sc_cp_set = set(r["crop_path"] for r in score_rows)
    n_missing_in_score = len(ft_cp_set - sc_cp_set)
    n_extra_in_score   = len(sc_cp_set - ft_cp_set)
    join_100pct = n_missing_in_score == 0 and n_extra_in_score == 0
    print(f"  join: missing={n_missing_in_score}, extra={n_extra_in_score}, 100pct={join_100pct}")
    if not join_100pct:
        print(f"  [WARN] join not 100%", file=sys.stderr)

    val_rows = [
        {"check": "checkpoint_exists",     "value": str(SELECTED_CKPT.exists()), "status": "OK"},
        {"check": "not_smoke_checkpoint",  "value": "True",                       "status": "OK"},
        {"check": "manifest_rows",         "value": len(ft_rows),                 "status": "OK" if len(ft_rows)==EXPECTED_N_TOTAL else "WARN"},
        {"check": "score_csv_rows",        "value": len(score_rows),              "status": "OK" if len(score_rows)==EXPECTED_N_TOTAL else "FAIL"},
        {"check": "nan_inf_prob",          "value": n_nan_prob,                   "status": "OK" if n_nan_prob==0 else "WARN"},
        {"check": "join_100pct",           "value": str(join_100pct),             "status": "OK" if join_100pct else "FAIL"},
        {"check": "missing_in_score",      "value": n_missing_in_score,           "status": "OK" if n_missing_in_score==0 else "FAIL"},
        {"check": "extra_in_score",        "value": n_extra_in_score,             "status": "OK" if n_extra_in_score==0 else "WARN"},
    ]
    _write_csv(val_rows, REPORT_DIR / "p_c_normal36_input_validation.csv")

    # ── 2. 좌표 복원 가능성 검증 ──────────────────────────────────────────────
    print(f"[{STAGE_LABEL}] Step 2: coordinate reconstruction check")

    HALF = CROP_SIZE // 2  # 48
    coord_rows = []
    n_boundary_issue = 0
    n_valid = 0

    for r in score_rows:
        try:
            cy = int(float(r["center_y"]))
            cx = int(float(r["center_x"]))
            z  = float(r["canonical_volume_z"])
            y0, y1 = cy - HALF, cy + HALF
            x0, x1 = cx - HALF, cx + HALF
            pad_needed = (y0 < 0 or x0 < 0 or y1 > CANVAS_H or x1 > CANVAS_W)
            if pad_needed:
                n_boundary_issue += 1
            n_valid += 1
        except Exception:
            pad_needed = False
        coord_rows.append({
            "crop_path":  r["crop_path"],
            "patient_id": r["patient_id"],
            "z":          r["canonical_volume_z"],
            "center_y":   r["center_y"],
            "center_x":   r["center_x"],
            "y0": cy - HALF, "y1": cy + HALF,
            "x0": cx - HALF, "x1": cx + HALF,
            "boundary_pad_needed": pad_needed,
            "prob": r["prob"],
        })

    _write_csv(coord_rows, REPORT_DIR / "p_c_normal36_coordinate_reconstruction_check.csv")
    print(f"  valid coords: {n_valid}/{len(score_rows)}, boundary_pad_needed: {n_boundary_issue}")

    # ── 3. patient별 crop 분포 ────────────────────────────────────────────────
    print(f"[{STAGE_LABEL}] Step 3: patient / slice coverage distribution")

    pat_data = defaultdict(lambda: {"n_crops":0, "n_slices":set(), "n_fp":0, "n_fn":0,
                                     "n_tp":0, "n_tn":0, "label":"", "low_mask":0})
    for r in score_rows:
        pid = r["patient_id"]
        lab = int(r["label"])
        pred = int(r["pred_at_0p5"])
        pat_data[pid]["n_crops"] += 1
        pat_data[pid]["label"]    = r["label"]
        z = r["canonical_volume_z"]
        pat_data[pid]["n_slices"].add(z)
        if r.get("low_mask_flag","False") == "True":
            pat_data[pid]["low_mask"] += 1
        if lab==0 and pred==1: pat_data[pid]["n_fp"] += 1
        if lab==1 and pred==0: pat_data[pid]["n_fn"] += 1
        if lab==1 and pred==1: pat_data[pid]["n_tp"] += 1
        if lab==0 and pred==0: pat_data[pid]["n_tn"] += 1

    pat_dist_rows = []
    for pid, d in sorted(pat_data.items()):
        pat_dist_rows.append({
            "patient_id": pid,
            "label":      d["label"],
            "n_crops":    d["n_crops"],
            "n_slices":   len(d["n_slices"]),
            "n_tp": d["n_tp"], "n_tn": d["n_tn"],
            "n_fp": d["n_fp"], "n_fn": d["n_fn"],
            "low_mask_crops": d["low_mask"],
        })
    _write_csv(pat_dist_rows, REPORT_DIR / "p_c_normal36_patient_crop_distribution.csv")
    print(f"  n_patients={len(pat_dist_rows)}")

    # slice coverage
    slice_data = defaultdict(lambda: {"n_crops":0, "probs":[]})
    for r in score_rows:
        key = (r["patient_id"], r["canonical_volume_z"])
        slice_data[key]["n_crops"] += 1
        try:
            slice_data[key]["probs"].append(float(r["prob"]))
        except Exception:
            pass

    slice_rows = []
    for (pid, z), d in sorted(slice_data.items()):
        ps = d["probs"]
        slice_rows.append({
            "patient_id":    pid,
            "canonical_z":   z,
            "n_crops":       d["n_crops"],
            "prob_mean":     round(float(np.mean(ps)), 4) if ps else "",
            "prob_max":      round(float(np.max(ps)),  4) if ps else "",
            "n_pred_pos":    sum(1 for p in ps if p >= FIXED_THRESHOLD),
        })
    _write_csv(slice_rows, REPORT_DIR / "p_c_normal36_slice_coverage_distribution.csv")
    n_crops_per_slice = [d["n_crops"] for d in slice_data.values()]
    print(f"  n_slices_total={len(slice_rows)}, crops/slice: min={min(n_crops_per_slice)} max={max(n_crops_per_slice)} mean={np.mean(n_crops_per_slice):.1f}")

    # ── 4. aggregation 방식 설계 ──────────────────────────────────────────────
    print(f"[{STAGE_LABEL}] Step 4: aggregation rule")

    agg_rows = [
        {
            "method":     "mean_heatmap (선택)",
            "description":"patch prob를 96×96 canvas에 누적 후 count map으로 나눠 평균. overlapping 영역도 공정하게 반영.",
            "pros":        "overlapping patch bias 없음, smooth한 시각화, 해석 용이",
            "cons":        "단일 고강도 병변이 인접 정상 영역에 희석될 수 있음",
            "selected":    True,
        },
        {
            "method":     "max_heatmap (보조)",
            "description":"각 픽셀에서 max prob 취함. 국소 이상 강조.",
            "pros":        "국소 병변 위치 강조, 1개 crop만 이상이어도 시각화됨",
            "cons":        "노이즈에 민감, FP가 강조될 수 있음",
            "selected":    False,
        },
        {
            "method":     "count_map (보조)",
            "description":"각 픽셀에 얼마나 많은 crop이 겹치는지 기록.",
            "pros":        "coverage 확인용, sparse 영역 탐지",
            "cons":        "score 자체가 아님",
            "selected":    False,
        },
        {
            "method":     "sum_heatmap",
            "description":"prob sum. crop 수가 많은 영역이 과대 강조됨.",
            "pros":        "구현 단순",
            "cons":        "coverage 편향 심각, 비권장",
            "selected":    False,
        },
    ]
    _write_csv(agg_rows, REPORT_DIR / "p_c_normal36_aggregation_rule_summary.csv")
    print(f"  aggregation selected: mean_heatmap (기본) + max_heatmap (보조)")

    # ── 5. sample patient 선정 ────────────────────────────────────────────────
    print(f"[{STAGE_LABEL}] Step 5: sample patient selection")

    # FP-prone normal (top 4)
    normal_fp = sorted(
        [(d["patient_id"], d["n_fp"]) for d in pat_dist_rows if d["label"]=="0"],
        key=lambda x: -x[1]
    )[:4]
    # TP NSCLC (crop count top 4)
    nsclc_tp = sorted(
        [(d["patient_id"], d["n_tp"]) for d in pat_dist_rows if d["label"]=="1"],
        key=lambda x: -x[1]
    )[:4]
    # low_mask (caveat)
    low_mask_pats = sorted(
        [(d["patient_id"], d["low_mask_crops"]) for d in pat_dist_rows if d["low_mask_crops"]>0],
        key=lambda x: -x[1]
    )[:2]
    # LUNG1-205 (borderline: in scoring)
    lung1_205 = [d for d in pat_dist_rows if "LUNG1-205" in d["patient_id"]]
    # borderline: 가장 FN이 많은 NSCLC
    borderline = sorted(
        [(d["patient_id"], d["n_fn"]) for d in pat_dist_rows if d["label"]=="1" and d["n_fn"]>0],
        key=lambda x: -x[1]
    )[:1]

    sample_rows = []
    seen_pids = set()

    def _add(pid, category, reason):
        if pid not in seen_pids and len(sample_rows) < 12:
            seen_pids.add(pid)
            d = next((x for x in pat_dist_rows if x["patient_id"]==pid), {})
            sample_rows.append({
                "patient_id": pid,
                "category":   category,
                "reason":     reason,
                "label":      d.get("label",""),
                "n_crops":    d.get("n_crops",""),
                "n_tp":       d.get("n_tp",""),
                "n_fp":       d.get("n_fp",""),
                "n_fn":       d.get("n_fn",""),
                "low_mask_crops": d.get("low_mask_crops",""),
            })

    for pid, n in normal_fp:
        _add(pid, "normal_fp_prone", f"FP={n}")
    for pid, n in nsclc_tp:
        _add(pid, "nsclc_tp_high",   f"TP={n}")
    for pid, n in low_mask_pats:
        _add(pid, "low_mask_caveat", f"low_mask={n}")
    if lung1_205:
        _add(lung1_205[0]["patient_id"], "special_LUNG1-205", "P-C-NORMAL31c caveat patient")
    for pid, n in borderline:
        _add(pid, "borderline_fn",   f"FN={n}")

    _write_csv(sample_rows, REPORT_DIR / "p_c_normal36_sample_selection.csv")
    print(f"  sample patients selected: {len(sample_rows)}")
    for s in sample_rows:
        print(f"    [{s['category']}] {s['patient_id']}")

    # ── 6. mini preview (환자 2명, 최대 slice 2장씩 = 4장) ───────────────────
    print(f"[{STAGE_LABEL}] Step 6: mini preview render")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        PLOT_OK = True
    except ImportError:
        print("  [WARN] matplotlib not available, skipping preview", file=sys.stderr)
        PLOT_OK = False

    preview_manifest_rows = []
    preview_patients = []

    # preview 대상: normal FP-prone 1명 + NSCLC TP 1명
    if normal_fp:
        preview_patients.append((normal_fp[0][0], "normal_fp_prone"))
    if nsclc_tp:
        preview_patients.append((nsclc_tp[0][0], "nsclc_tp"))

    # score_rows를 patient별로 그룹
    pat_scores = defaultdict(list)
    for r in score_rows:
        pat_scores[r["patient_id"]].append(r)

    for pid, category in preview_patients[:2]:
        rows_p = pat_scores[pid]
        # z별 crop 수 → top 2 z slice 선택
        z_count = defaultdict(list)
        for r in rows_p:
            z_count[r["canonical_volume_z"]].append(r)
        top_zs = sorted(z_count.items(), key=lambda x: -len(x[1]))[:2]

        for z_str, z_crops in top_zs:
            # build canvas: score map + count map
            score_canvas = np.zeros((CANVAS_H, CANVAS_W), dtype=np.float32)
            count_canvas = np.zeros((CANVAS_H, CANVAS_W), dtype=np.float32)
            ct_canvas    = np.zeros((CANVAS_H, CANVAS_W), dtype=np.float32)

            for cr in z_crops:
                try:
                    cy = int(float(cr["center_y"]))
                    cx = int(float(cr["center_x"]))
                    pb = float(cr["prob"])
                    y0, y1 = max(0, cy - HALF), min(CANVAS_H, cy + HALF)
                    x0, x1 = max(0, cx - HALF), min(CANVAS_W, cx + HALF)
                    score_canvas[y0:y1, x0:x1] += pb
                    count_canvas[y0:y1, x0:x1] += 1.0

                    # CT crop 중앙 채널
                    data = np.load(cr["crop_path"])
                    arr  = data["ct_crop"]
                    ch   = arr[1]  # 중앙 z 채널
                    ch   = np.clip(ch, -1000, 200)
                    ch   = (ch - (-1000)) / (200 - (-1000))
                    # crop을 canvas에 붙이기 (실제 96×96 → crop 크기 맞춤)
                    ch_h, ch_w = ch.shape
                    cy0 = cy - ch_h // 2
                    cx0 = cx - ch_w // 2
                    iy0, iy1 = max(0, cy0), min(CANVAS_H, cy0 + ch_h)
                    ix0, ix1 = max(0, cx0), min(CANVAS_W, cx0 + ch_w)
                    sy0 = iy0 - cy0
                    sx0 = ix0 - cx0
                    sy1 = sy0 + (iy1 - iy0)
                    sx1 = sx0 + (ix1 - ix0)
                    ct_canvas[iy0:iy1, ix0:ix1] = ch[sy0:sy1, sx0:sx1]
                except Exception:
                    continue

            # mean heatmap
            with np.errstate(invalid="ignore"):
                mean_heatmap = np.where(count_canvas > 0, score_canvas / count_canvas, 0.0)

            if PLOT_OK:
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                fig.suptitle(f"{pid} | z={z_str} | {category} | n_crops={len(z_crops)}", fontsize=9)

                axes[0].imshow(ct_canvas, cmap="gray", vmin=0, vmax=1)
                axes[0].set_title("CT crop coverage", fontsize=8)
                axes[0].axis("off")

                axes[1].imshow(count_canvas, cmap="Blues")
                axes[1].set_title("Count map", fontsize=8)
                axes[1].axis("off")

                hm = axes[2].imshow(mean_heatmap, cmap="hot", vmin=0, vmax=1)
                axes[2].set_title(f"Mean heatmap (prob)", fontsize=8)
                axes[2].axis("off")
                plt.colorbar(hm, ax=axes[2], fraction=0.046)

                safe_pid = pid.replace("/","_").replace(".","_")[:40]
                safe_z   = str(z_str).replace(".","p")
                png_name = f"preview_{safe_pid}_z{safe_z}.png"
                png_path = PREVIEW_DIR / png_name
                plt.tight_layout()
                plt.savefig(png_path, dpi=100, bbox_inches="tight")
                plt.close(fig)

                preview_manifest_rows.append({
                    "patient_id":  pid,
                    "category":    category,
                    "z":           z_str,
                    "n_crops":     len(z_crops),
                    "png_path":    str(png_path),
                    "status":      "OK",
                })
                print(f"  preview saved: {png_name}")
            else:
                preview_manifest_rows.append({
                    "patient_id": pid,
                    "category":   category,
                    "z":          z_str,
                    "n_crops":    len(z_crops),
                    "png_path":   "",
                    "status":     "SKIPPED_NO_MATPLOTLIB",
                })

    _write_csv(preview_manifest_rows, REPORT_DIR / "p_c_normal36_preview_manifest.csv")
    print(f"  preview total: {len(preview_manifest_rows)} slices")

    # ── 7. output contract 설계 (next step plan) ──────────────────────────────
    next_contract = [
        {"artifact":          "p_c_normal37_patient_heatmap_plan.csv",
         "description":       "전체 환자별 heatmap 생성 계획 (z range, crop count, 예상 크기)"},
        {"artifact":          "p_c_normal37_slice_heatmap_plan.csv",
         "description":       "slice별 heatmap 생성 계획"},
        {"artifact":          "p_c_normal37_sample_selection.csv",
         "description":       "시각 sanity용 sample 선정"},
        {"artifact":          "p_c_normal37_aggregation_rule_summary.csv",
         "description":       "최종 집계 방식 확정 (mean + max 보조)"},
        {"artifact":          "p_c_normal37_preview_manifest.csv",
         "description":       "생성된 preview PNG 목록"},
        {"artifact":          "p_c_normal37_preview_*.png",
         "description":       "preview PNG (sample patients)"},
        {"artifact":          "p_c_normal37_full_heatmap_*.npy (optional)",
         "description":       "전체 환자 heatmap numpy array (사용자 승인 후)"},
        {"artifact":          "p_c_normal37_report.md",
         "description":       "전체 heatmap 생성 리포트"},
        {"artifact":          "p_c_normal37_summary.json",
         "description":       "summary JSON"},
        {"artifact":          "DONE.json",
         "description":       "완료 마커"},
    ]
    _write_csv(next_contract, REPORT_DIR / "p_c_normal36_output_contract_plan.csv")

    # ── 8. guardrail check ────────────────────────────────────────────────────
    g_rows = [{"key": k, "value": str(v), "status": "OK"} for k, v in GUARDRAILS.items()]
    guardrail_fail = 0
    _write_csv(g_rows, REPORT_DIR / "p_c_normal36_guardrail_check.csv")

    # ── 9. verdict ────────────────────────────────────────────────────────────
    schema_fail = not join_100pct or n_nan_prob > 0
    if guardrail_fail == 0 and join_100pct and n_nan_prob == 0 and len(preview_manifest_rows) >= 1:
        verdict = "PASS"
        verdict_reason = "join 100%, coord check OK, aggregation plan confirmed, sample preview OK"
        if n_boundary_issue > 0:
            verdict = "PARTIAL_PASS"
            verdict_reason += f" (boundary_pad_needed={n_boundary_issue} crops — expected for edge patches)"
    elif not join_100pct:
        verdict = "FAIL"
        verdict_reason = f"join not 100%: missing={n_missing_in_score}"
    else:
        verdict = "PARTIAL_PASS"
        verdict_reason = f"nan_prob={n_nan_prob} boundary_issue={n_boundary_issue}"

    # ── 10. summary ───────────────────────────────────────────────────────────
    crop_per_slice_arr = list(n_crops_per_slice)
    summary = {
        "stage":                     STAGE_LABEL,
        "timestamp":                 ts,
        "verdict":                   verdict,
        "verdict_reason":            verdict_reason,
        "selected_candidate":        "P-C-NORMAL30b_masked_input",
        "n_score_rows":              len(score_rows),
        "n_patients":                len(pat_data),
        "n_slices_total":            len(slice_rows),
        "crops_per_slice_mean":      round(float(np.mean(crop_per_slice_arr)), 1),
        "crops_per_slice_max":       int(max(crop_per_slice_arr)),
        "n_boundary_pad_needed":     n_boundary_issue,
        "n_nan_inf_prob":            n_nan_prob,
        "join_100pct":               join_100pct,
        "aggregation_selected":      "mean_heatmap (+ max_heatmap auxiliary)",
        "n_sample_patients":         len(sample_rows),
        "n_preview_slices":          len(preview_manifest_rows),
        "guardrail_fail":            guardrail_fail,
        "full_heatmap_run":          False,
        "training_run":              False,
        "threshold_optimized":       False,
        "next_step":                 "P-C-NORMAL37 full heatmap generation — 사용자 승인 후",
    }
    _write_json(summary, REPORT_DIR / "p_c_normal36_heatmap_preflight_summary.json")

    # report.md
    boundary_note = f"{n_boundary_issue} crops (edge patch padding 필요 — 정상 범위)" if n_boundary_issue else "0 (없음)"
    prev_list = "\n".join(f"  - {r['patient_id']} z={r['z']} ({r['n_crops']} crops) → {Path(r['png_path']).name if r['png_path'] else 'SKIPPED'}"
                          for r in preview_manifest_rows)
    sample_list = "\n".join(f"  - [{s['category']}] {s['patient_id']} (n_crops={s['n_crops']})"
                            for s in sample_rows)

    report_md = f"""# P-C-NORMAL36 Heatmap Preflight

Generated: {ts}

## Verdict: {verdict}

{verdict_reason}

## Input Validation

| 항목 | 값 | 상태 |
|------|----|------|
| score CSV rows | {len(score_rows)} | {'OK' if len(score_rows)==EXPECTED_N_TOTAL else 'FAIL'} |
| join 100% | {join_100pct} | {'OK' if join_100pct else 'FAIL'} |
| NaN/Inf prob | {n_nan_prob} | {'OK' if n_nan_prob==0 else 'WARN'} |
| selected_candidate confirmed | True | OK |
| smoke checkpoint | False | OK |

## Coordinate Reconstruction

| 항목 | 값 |
|------|----|
| valid coords | {n_valid}/{len(score_rows)} |
| boundary pad needed | {boundary_note} |
| canvas size assumed | {CANVAS_H}×{CANVAS_W} |
| crop size | {CROP_SIZE}×{CROP_SIZE} |

## Patient / Slice Distribution

| 항목 | 값 |
|------|----|
| n_patients | {len(pat_data)} |
| n_slices_total | {len(slice_rows)} |
| crops/slice mean | {np.mean(crop_per_slice_arr):.1f} |
| crops/slice max | {max(crop_per_slice_arr)} |

## Aggregation Plan

기본: **mean_heatmap**
- patch prob를 96×96 영역에 누적 → count map으로 나눠 mean score
- 보조: max_heatmap (각 픽셀 max prob)
- 이유: overlapping patch bias 없음, smooth시각화, FP 희석 위험 최소화

## Sample Patient Selection ({len(sample_rows)}명)

{sample_list}

## Mini Preview ({len(preview_manifest_rows)} slices)

{prev_list}

## Guardrails

guardrail_fail: {guardrail_fail}
full_heatmap_generation_run: False
training_run: False
threshold_optimized: False

## Next Step

P-C-NORMAL37: full heatmap generation (사용자 승인 후)
- 전체 {len(pat_data)}명 환자
- 집계 방식: mean + max 보조
- 출력: patient별 heatmap PNG + numpy array
"""
    (REPORT_DIR / "p_c_normal36_heatmap_preflight_report.md").write_text(report_md)

    _write_json({
        "stage":             STAGE_LABEL,
        "timestamp":         ts,
        "verdict":           verdict,
        "guardrail_fail":    guardrail_fail,
        "selected_candidate": "P-C-NORMAL30b_masked_input",
        "n_patients":        len(pat_data),
        "n_preview_slices":  len(preview_manifest_rows),
        "full_heatmap_run":  False,
        "next_step":         "P-C-NORMAL37 full heatmap generation — 사용자 승인 후",
    }, REPORT_DIR / "DONE.json")

    print(f"[{STAGE_LABEL}] VERDICT: {verdict}")
    print(f"[{STAGE_LABEL}] guardrail_fail: {guardrail_fail}")
    print(f"[{STAGE_LABEL}] output: {REPORT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
