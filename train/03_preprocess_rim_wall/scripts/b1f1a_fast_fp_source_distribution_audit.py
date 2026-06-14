#!/usr/bin/env python3
"""
B1-F1a Fast FP Source Distribution Audit

변경점 (B1-F1 대비):
  - vessel mask / Frangi 계산 비활성화
  - CT 로드 비활성화
  - lesion mask 로드 비활성화 → score CSV 기반 근사
  - PNG 생성 비활성화
  - component / lesion overlap / ROI boundary / position_bin / hilar proxy / stage2 handoff 만 계산

안전 조건:
  - stage2_holdout 접근 금지
  - score/model/threshold/ROI/CT/mask 수정 금지
  - threshold p95=13.231265 고정 (JSON 로드, 재계산 금지)

실행:
  --dry-run / --plan   계획 출력만 (파일 생성 없음)
  --real               실제 분석 및 파일 생성
"""
import sys

if __name__ == "__main__" and len(sys.argv) < 2:
    print(
        "[ERROR] bare-run guard: 인수 없이 실행 금지. "
        "--dry-run 또는 --real 을 사용하세요.",
        file=sys.stderr,
    )
    sys.exit(2)

import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import label as nd_label, binary_erosion

ALLOW_REAL = "--real" in sys.argv

# ─── 경로 ─────────────────────────────────────────────────────────────────────
PROJECT = Path("/home/jinhy/project/lung-ct-anomaly")

SCORE_ROOT = (
    PROJECT
    / "experiments/efficientnet_b0_imagenet_chestwall_removed_roi_v1"
    / "outputs/scores/lesion_stage1_dev_by_patient"
)
THRESHOLD_JSON = (
    PROJECT
    / "experiments/efficientnet_b0_imagenet_chestwall_removed_roi_v1"
    / "outputs/reports/normal_val/p_b9_normal_val_threshold.json"
)
STAGE_SPLIT_CSV = (
    PROJECT / "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1.csv"
)
STAGE2_MANIFEST_CSV = (
    PROJECT
    / "outputs/second-stage-lesion-refiner-v1/candidates"
    / "rule_s6a_gs2_selected_candidate_manifest_dryrun.csv"
)
ROI_BASE = (
    PROJECT
    / "outputs/mip-postprocess-research-v1/masks/refined_roi_v4_20_modeB_all_v1/lesion"
)
OUT_ROOT = (
    PROJECT
    / "outputs/position-aware-padim-v1"
    / "efficientnet_v4_20_fp_source_audit"
    / "b1f1a_fast_fp_source_distribution_v1"
)

# ─── 파라미터 ──────────────────────────────────────────────────────────────────
THRESHOLD_P95      = 13.231265   # 고정값 (JSON 재확인 후 검증)
PATCH_STRIDE       = 16
BOUNDARY_ERODE_PX  = 3
BOUNDARY_TOUCH_THR = 0.10
BOUNDARY_BBOX_PX   = 3
VESSEL_DOMINANT_RATIO = 0.25     # vessel 계산 없으므로 label 미사용


# ══════════════════════════════════════════════════════════════════════════════
# 로드 유틸
# ══════════════════════════════════════════════════════════════════════════════

def verify_threshold():
    with open(THRESHOLD_JSON, encoding="utf-8") as f:
        d = json.load(f)
    json_val = float(d["threshold_p95"])
    assert abs(json_val - THRESHOLD_P95) < 1e-4, (
        f"threshold mismatch: JSON={json_val:.6f}, fixed={THRESHOLD_P95}"
    )
    assert d.get("v4_20_branch_specific_threshold", False) is True
    return json_val


def load_holdout():
    holdout = set()
    with open(STAGE_SPLIT_CSV, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["stage_split"] == "stage2_holdout":
                holdout.add(r["patient_id"])
    return holdout


def load_stage1_dev():
    patients = []
    with open(STAGE_SPLIT_CSV, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["stage_split"] == "stage1_dev":
                patients.append({
                    "patient_id": r["patient_id"],
                    "safe_id":    r["safe_id"],
                    "group":      r["group"],
                })
    return patients


def load_stage2_manifest():
    """key=(patient_id, local_z, y0, x0) → sampling_label"""
    data = {}
    if not STAGE2_MANIFEST_CSV.exists():
        return data
    with open(STAGE2_MANIFEST_CSV, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r.get("stage_split", "") == "stage2_holdout":
                continue
            key = (r["patient_id"], int(r["local_z"]), int(r["y0"]), int(r["x0"]))
            data[key] = r.get("sampling_label", "unknown")
    return data


def load_score_csv(patient_id):
    rows = []
    with open(SCORE_ROOT / f"{patient_id}.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Component 생성
# ══════════════════════════════════════════════════════════════════════════════

def build_components(rows):
    hot = [r for r in rows if float(r["padim_score"]) >= THRESHOLD_P95]
    if not hot:
        return [], len(rows), 0

    coords = np.array([
        [int(r["local_z"]), int(r["y0"]) // PATCH_STRIDE, int(r["x0"]) // PATCH_STRIDE]
        for r in hot
    ], dtype=np.int64)

    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)
    shape = (z_max - z_min + 1, y_max - y_min + 1, x_max - x_min + 1)
    grid = np.zeros(shape, dtype=np.uint8)
    shifted = coords - np.array([z_min, y_min, x_min])
    for sz, sy, sx in shifted:
        grid[sz, sy, sx] = 1

    struct = np.ones((3, 3, 3), dtype=np.uint8)
    labeled, n_comp = nd_label(grid, structure=struct)

    comp_map = {}
    for idx, (sz, sy, sx) in enumerate(shifted):
        cid = int(labeled[sz, sy, sx])
        comp_map.setdefault(cid, []).append(idx)

    components = []
    for cid, idxs in comp_map.items():
        cr     = [hot[i] for i in idxs]
        scores = [float(r["padim_score"]) for r in cr]
        zvals  = [int(r["local_z"])       for r in cr]
        y0s    = [int(r["y0"])            for r in cr]
        x0s    = [int(r["x0"])            for r in cr]
        y1s    = [int(r["y1"])            for r in cr]
        x1s    = [int(r["x1"])            for r in cr]
        lpx    = [int(r["lesion_pixels"]) for r in cr]
        lpr    = [float(r["lesion_patch_ratio"]) for r in cr]
        hlp    = [int(r["has_lesion_patch"])     for r in cr]
        pbins  = [r["position_bin"]       for r in cr]

        pbin_cnt = {}
        for p in pbins:
            pbin_cnt[p] = pbin_cnt.get(p, 0) + 1
        pbin_maj = max(pbin_cnt, key=pbin_cnt.get)

        top_i  = int(np.argmax(scores))

        # lesion_near_flag 근사: component 인접 z(±1) 범위에서 lesion_pixels>0 존재
        z_lo = min(zvals) - 1
        z_hi = max(zvals) + 1
        near_flag = any(
            int(r["lesion_pixels"]) > 0
            for r in rows
            if z_lo <= int(r["local_z"]) <= z_hi
        )

        components.append({
            "component_id":          cid,
            "rows":                  cr,
            "max_score":             float(np.max(scores)),
            "mean_score":            float(np.mean(scores)),
            "patch_count":           len(cr),
            "z_span":                int(max(zvals) - min(zvals) + 1),
            "local_z_min":           int(min(zvals)),
            "local_z_max":           int(max(zvals)),
            "bbox_y0":               int(min(y0s)),
            "bbox_x0":               int(min(x0s)),
            "bbox_y1":               int(max(y1s)),
            "bbox_x1":               int(max(x1s)),
            "top_patch_local_z":     int(zvals[top_i]),
            "top_patch_y0":          int(y0s[top_i]),
            "top_patch_x0":          int(x0s[top_i]),
            "position_bin_majority": pbin_maj,
            "central_peripheral":    cr[0].get("central_peripheral", "unknown"),
            "lesion_pixels_sum":     int(sum(lpx)),
            "lesion_overlap_ratio":  float(np.mean(lpr)),
            "lesion_hit_flag":       any(h == 1 for h in hlp),
            "lesion_near_flag":      near_flag,
            # vessel: 비활성화
            "vessel_overlap_ratio":  -1.0,
            "vessel_dominant_flag":  False,
            "vessel_touch_flag":     False,
            # boundary: ROI 로드 후 업데이트
            "boundary_touch_ratio":  0.0,
            "chestwall_boundary_flag": False,
            "peripheral_flag":       "peripheral" in pbin_maj,
            "hilar_proxy_flag":      False,
            "component_source_label": "pending",
            "stage2_handoff_label":  "not_in_manifest",
            "note":                  "",
        })

    return components, len(rows), len(hot)


# ══════════════════════════════════════════════════════════════════════════════
# ROI Boundary 계산 (CT 없음, frangi 없음)
# ══════════════════════════════════════════════════════════════════════════════

def compute_boundary(roi_vol, comp):
    try:
        n = roi_vol.shape[0]
        z_lo = max(0, comp["local_z_min"])
        z_hi = min(n, comp["local_z_max"] + 1)
        slab = np.array(roi_vol[z_lo:z_hi]).astype(bool)
        if not slab.any():
            return 0.0, False

        boundary_proj = np.zeros(slab.shape[1:], dtype=bool)
        for zi in range(slab.shape[0]):
            sl = slab[zi]
            eroded = binary_erosion(sl, iterations=BOUNDARY_ERODE_PX)
            boundary_proj |= (sl & ~eroded)

        h, w = boundary_proj.shape
        by0 = max(0, comp["bbox_y0"])
        bx0 = max(0, comp["bbox_x0"])
        by1 = min(h, comp["bbox_y1"])
        bx1 = min(w, comp["bbox_x1"])
        if by1 <= by0 or bx1 <= bx0:
            return 0.0, False

        comp_mask = np.zeros((h, w), dtype=bool)
        comp_mask[by0:by1, bx0:bx1] = True
        comp_pixels = int(comp_mask.sum())
        touch = int((boundary_proj & comp_mask).sum())
        ratio = touch / comp_pixels if comp_pixels > 0 else 0.0

        # bbox가 ROI boundary에 가까운지
        roi_proj = np.any(slab, axis=0)
        dist = 999
        if roi_proj.any():
            iy, ix = np.where(roi_proj)
            dist = min(
                abs(comp["bbox_y0"] - iy.min()), abs(comp["bbox_y1"] - iy.max()),
                abs(comp["bbox_x0"] - ix.min()), abs(comp["bbox_x1"] - ix.max()),
            )

        flag = (ratio >= BOUNDARY_TOUCH_THR) or (int(dist) <= BOUNDARY_BBOX_PX)
        return float(ratio), bool(flag)
    except Exception:
        return 0.0, False


# ══════════════════════════════════════════════════════════════════════════════
# Source label (vessel 없으므로 vessel 항목 제외)
# ══════════════════════════════════════════════════════════════════════════════

def assign_source_label(comp):
    if comp["lesion_hit_flag"]:
        return "lesion_hit"
    if comp["lesion_near_flag"] and not comp["lesion_hit_flag"]:
        return "lesion_near"
    if comp["chestwall_boundary_flag"]:
        return "boundary_chestwall"
    if comp["hilar_proxy_flag"]:
        return "hilar_mediastinal_proxy"
    if comp["peripheral_flag"]:
        return "peripheral_fp"
    return "other_fp"


# ══════════════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    # threshold 검증
    thr_val = verify_threshold()
    print(f"[INFO] threshold_p95 = {thr_val:.6f}  (고정값 검증 OK)")

    holdout  = load_holdout()
    patients = load_stage1_dev()
    s2_map   = load_stage2_manifest()
    print(f"[INFO] stage2_holdout={len(holdout)}, stage1_dev={len(patients)}, s2_manifest={len(s2_map)}")

    # dry-run: 샘플 1명으로 component 수 추정
    sample = patients[0]
    sample_rows = load_score_csv(sample["patient_id"])
    sample_comps, total_p, hot_p = build_components(sample_rows)
    print(f"[DRY-SAMPLE] {sample['patient_id']}: total={total_p}, hot={hot_p}, components={len(sample_comps)}")

    est_comps = len(sample_comps) * len(patients)
    est_sec   = len(patients) * 2.5   # score CSV + component + boundary per patient
    print(f"\n[PLAN]")
    print(f"  분석 대상: stage1_dev {len(patients)}명")
    print(f"  threshold_p95: {THRESHOLD_P95} (고정)")
    print(f"  vessel/Frangi: 비활성화")
    print(f"  CT 로드: 비활성화")
    print(f"  lesion mask 로드: 비활성화 (score CSV 근사)")
    print(f"  ROI boundary: 활성화")
    print(f"  PNG: 비활성화")
    print(f"  예상 component 수: ~{est_comps:,} (샘플 기준 추정)")
    print(f"  예상 실행시간: ~{est_sec/60:.0f}분 (ROI boundary 포함)")
    print(f"  출력 루트: {OUT_ROOT}")
    print(f"  stage2_holdout 접근: False")

    if not ALLOW_REAL:
        print("\n[READY] --real 로 실행하세요.")
        return

    # ── real 실행 ───────────────────────────────────────────────────────────────
    if OUT_ROOT.exists():
        print(f"[ERROR] 출력 루트 이미 존재: {OUT_ROOT}", file=sys.stderr)
        sys.exit(1)
    OUT_ROOT.mkdir(parents=True)

    all_comp_rows     = []
    patient_summaries = []
    error_rows        = []
    holdout_access    = 0

    for pat in patients:
        pid = pat["patient_id"]
        sid = pat["safe_id"]
        grp = pat["group"]

        if pid in holdout:
            holdout_access += 1
            error_rows.append({"patient_id": pid, "error": "stage2_holdout 차단"})
            continue

        # score CSV 로드
        try:
            rows = load_score_csv(pid)
        except Exception as e:
            error_rows.append({"patient_id": pid, "error": f"score CSV: {e}"})
            continue

        # component 생성
        components, total_p, hot_p = build_components(rows)

        # ROI 로드 (boundary 계산용)
        roi_path = ROI_BASE / sid / "refined_roi.npy"
        roi_vol  = None
        if roi_path.exists():
            try:
                roi_vol = np.load(roi_path, mmap_mode="r")
            except Exception as e:
                error_rows.append({"patient_id": pid, "error": f"ROI 로드: {e}"})

        # component별 계산
        for c in components:
            # boundary
            if roi_vol is not None:
                c["boundary_touch_ratio"], c["chestwall_boundary_flag"] = compute_boundary(roi_vol, c)

            # hilar proxy: central이고 peripheral_flag 아닌 경우
            c["hilar_proxy_flag"] = (
                "central" in c["position_bin_majority"] and not c["peripheral_flag"]
            )

            # source label
            c["component_source_label"] = assign_source_label(c)

            # stage2 handoff (top patch 기준, 이후 전체 row fallback)
            top_key = (pid, c["top_patch_local_z"], c["top_patch_y0"], c["top_patch_x0"])
            if top_key in s2_map:
                c["stage2_handoff_label"] = s2_map[top_key]
            else:
                for row in c["rows"]:
                    k = (pid, int(row["local_z"]), int(row["y0"]), int(row["x0"]))
                    if k in s2_map:
                        c["stage2_handoff_label"] = s2_map[k]
                        break

            note_parts = []
            if roi_vol is None:
                note_parts.append("roi_missing")
            c["note"] = ";".join(note_parts)

            all_comp_rows.append({
                "patient_id":               pid,
                "group":                    grp,
                "component_id":             c["component_id"],
                "max_score":                round(c["max_score"], 4),
                "mean_score":               round(c["mean_score"], 4),
                "patch_count":              c["patch_count"],
                "z_span":                   c["z_span"],
                "position_bin_majority":    c["position_bin_majority"],
                "bbox_y0":                  c["bbox_y0"],
                "bbox_x0":                  c["bbox_x0"],
                "bbox_y1":                  c["bbox_y1"],
                "bbox_x1":                  c["bbox_x1"],
                "local_z_min":              c["local_z_min"],
                "local_z_max":              c["local_z_max"],
                "top_patch_local_z":        c["top_patch_local_z"],
                "lesion_overlap_ratio":     round(c["lesion_overlap_ratio"], 4),
                "lesion_hit_flag":          int(c["lesion_hit_flag"]),
                "lesion_near_flag":         int(c["lesion_near_flag"]),
                "boundary_touch_ratio":     round(c["boundary_touch_ratio"], 4),
                "chestwall_boundary_flag":  int(c["chestwall_boundary_flag"]),
                "peripheral_flag":          int(c["peripheral_flag"]),
                "hilar_proxy_flag":         int(c["hilar_proxy_flag"]),
                "component_source_label":   c["component_source_label"],
                "stage2_handoff_label":     c["stage2_handoff_label"],
                "note":                     c["note"],
            })

        # 환자 요약
        label_cnt = {}
        for c in components:
            lb = c["component_source_label"]
            label_cnt[lb] = label_cnt.get(lb, 0) + 1

        top3 = [c["component_source_label"]
                for c in sorted(components, key=lambda x: -x["max_score"])[:3]]
        patient_summaries.append({
            "patient_id":                pid,
            "group":                     grp,
            "n_components":              len(components),
            "n_hot_patches":             hot_p,
            "n_lesion_hit":              label_cnt.get("lesion_hit", 0),
            "n_lesion_near":             label_cnt.get("lesion_near", 0),
            "n_boundary_chestwall":      label_cnt.get("boundary_chestwall", 0),
            "n_hilar_proxy":             label_cnt.get("hilar_mediastinal_proxy", 0),
            "n_peripheral_fp":           label_cnt.get("peripheral_fp", 0),
            "n_other_fp":                label_cnt.get("other_fp", 0),
            "max_score_comp_label":      sorted(components, key=lambda x: -x["max_score"])[0]["component_source_label"] if components else "none",
            "top3_comp_labels":          "|".join(top3),
        })

    assert holdout_access == 0, f"stage2_holdout 접근 감지: {holdout_access}"

    # stage2 handoff 요약
    hs = {}
    for row in all_comp_rows:
        hl = row["stage2_handoff_label"]
        sl = row["component_source_label"]
        if hl not in hs:
            hs[hl] = {"n": 0, "lesion_hit": 0, "boundary": 0, "hilar": 0, "peripheral": 0, "other": 0, "score_sum": 0.0}
        d = hs[hl]
        d["n"] += 1
        if sl == "lesion_hit":              d["lesion_hit"] += 1
        elif sl == "boundary_chestwall":    d["boundary"] += 1
        elif sl == "hilar_mediastinal_proxy": d["hilar"] += 1
        elif sl == "peripheral_fp":         d["peripheral"] += 1
        else:                               d["other"] += 1
        d["score_sum"] += row["max_score"]

    s2_rows = []
    for hl, d in hs.items():
        n = d["n"]
        s2_rows.append({
            "handoff_label":          hl,
            "n_components":           n,
            "lesion_hit_ratio":       round(d["lesion_hit"] / n, 4) if n else 0,
            "boundary_ratio":         round(d["boundary"] / n, 4) if n else 0,
            "hilar_ratio":            round(d["hilar"] / n, 4) if n else 0,
            "peripheral_fp_ratio":    round(d["peripheral"] / n, 4) if n else 0,
            "other_fp_ratio":         round(d["other"] / n, 4) if n else 0,
            "mean_max_score":         round(d["score_sum"] / n, 4) if n else 0,
        })

    # 전체 분포
    total_c = len(all_comp_rows)
    label_dist = {}
    for row in all_comp_rows:
        lb = row["component_source_label"]
        label_dist[lb] = label_dist.get(lb, 0) + 1

    elapsed = time.time() - t0
    summary = {
        "step": "B1-F1a",
        "threshold_p95": THRESHOLD_P95,
        "threshold_source": "p_b9_normal_val_threshold.json",
        "n_patients": len(patients),
        "total_components": total_c,
        "elapsed_seconds": round(elapsed, 1),
        "vessel_mask": "disabled",
        "frangi": "disabled",
        "ct_load": "disabled",
        "lesion_mask_load": "disabled (score CSV approximation)",
        "roi_boundary": "enabled",
        "png": "disabled",
        "stage2_holdout_access": 0,
        "n_errors": len(error_rows),
        "label_distribution": dict(sorted(label_dist.items(), key=lambda x: -x[1])),
        "label_pct": {
            k: round(v / total_c * 100, 2) if total_c > 0 else 0
            for k, v in label_dist.items()
        },
        "note": (
            "vessel_overlap은 비활성화 (proxy 없음). "
            "lesion_near_flag는 인접 z slice score CSV 기반 근사. "
            "component_source_label은 heuristic proxy이며 GT 레이블 아님."
        ),
        "guardrails": {
            "score_modified": False,
            "model_rerun": False,
            "threshold_recalculated": False,
            "stage2_holdout_accessed": False,
            "gpu_used": False,
        },
    }

    # ── 파일 저장 ───────────────────────────────────────────────────────────────
    def write_csv(path, rows, fields):
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    comp_fields = [
        "patient_id", "group", "component_id",
        "max_score", "mean_score", "patch_count", "z_span",
        "position_bin_majority", "bbox_y0", "bbox_x0", "bbox_y1", "bbox_x1",
        "local_z_min", "local_z_max", "top_patch_local_z",
        "lesion_overlap_ratio", "lesion_hit_flag", "lesion_near_flag",
        "boundary_touch_ratio", "chestwall_boundary_flag",
        "peripheral_flag", "hilar_proxy_flag",
        "component_source_label", "stage2_handoff_label", "note",
    ]
    pat_fields = [
        "patient_id", "group", "n_components", "n_hot_patches",
        "n_lesion_hit", "n_lesion_near", "n_boundary_chestwall",
        "n_hilar_proxy", "n_peripheral_fp", "n_other_fp",
        "max_score_comp_label", "top3_comp_labels",
    ]
    s2_fields = [
        "handoff_label", "n_components",
        "lesion_hit_ratio", "boundary_ratio", "hilar_ratio",
        "peripheral_fp_ratio", "other_fp_ratio", "mean_max_score",
    ]
    err_fields = ["patient_id", "error"]

    write_csv(OUT_ROOT / "b1f1a_component_source_audit.csv",      all_comp_rows,     comp_fields)
    write_csv(OUT_ROOT / "b1f1a_patient_source_summary.csv",      patient_summaries, pat_fields)
    write_csv(OUT_ROOT / "b1f1a_stage2_handoff_label_summary.csv", s2_rows,          s2_fields)
    write_csv(OUT_ROOT / "b1f1a_errors.csv",                      error_rows,        err_fields)

    with open(OUT_ROOT / "b1f1a_source_distribution_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ── 간략 보고서 ─────────────────────────────────────────────────────────────
    fp_labels = {k: v for k, v in label_dist.items() if k not in ("lesion_hit", "lesion_near")}
    fp_total  = sum(fp_labels.values())

    lines = [
        "# B1-F1a Fast FP Source Distribution Report",
        "",
        "## 설정",
        f"- threshold_p95: {THRESHOLD_P95}  (출처: p_b9, 재계산 없음)",
        f"- vessel/Frangi: **비활성화**",
        f"- lesion_near_flag: 인접 z score CSV 근사",
        f"- ROI boundary: 활성화",
        f"- stage2_holdout 접근: **0회**",
        f"- 분석 환자: {len(patients)}명 / 오류: {len(error_rows)}명",
        f"- 전체 component: {total_c}",
        f"- 실행 시간: {elapsed:.1f}초",
        "",
        "## 전체 Component Source 분포",
        "",
    ]
    for lb, cnt in sorted(label_dist.items(), key=lambda x: -x[1]):
        pct = cnt / total_c * 100 if total_c else 0
        lines.append(f"- **{lb}**: {cnt} ({pct:.1f}%)")

    lines += [
        "",
        "> vessel_overlap은 비활성화 상태. component_source_label은 heuristic proxy (GT 아님).",
        "",
        "## FP Component 원인 분포 (lesion 제외)",
        "",
    ]
    for lb, cnt in sorted(fp_labels.items(), key=lambda x: -x[1]):
        pct = cnt / fp_total * 100 if fp_total else 0
        lines.append(f"- **{lb}**: {cnt} ({pct:.1f}%)")

    lines += [
        "",
        "## Stage2 Handoff Label 분포",
        "",
    ]
    for r in sorted(s2_rows, key=lambda x: -x["n_components"]):
        lines.append(
            f"- {r['handoff_label']}: {r['n_components']}개 "
            f"(lesion_hit {r['lesion_hit_ratio']*100:.1f}%, "
            f"boundary {r['boundary_ratio']*100:.1f}%, "
            f"peripheral {r['peripheral_fp_ratio']*100:.1f}%)"
        )

    lines += [
        "",
        "## Position_bin별 FP 분포",
        "",
    ]
    pbin_fp = {}
    for row in all_comp_rows:
        if row["component_source_label"] in ("lesion_hit", "lesion_near"):
            continue
        pb = row["position_bin_majority"]
        sl = row["component_source_label"]
        pbin_fp.setdefault(pb, {})
        pbin_fp[pb][sl] = pbin_fp[pb].get(sl, 0) + 1
    for pb in sorted(pbin_fp):
        lines.append(f"**{pb}**:")
        for sl, cnt in sorted(pbin_fp[pb].items(), key=lambda x: -x[1]):
            lines.append(f"  - {sl}: {cnt}")

    with open(OUT_ROOT / "b1f1a_fp_source_distribution_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    (OUT_ROOT / "DONE").write_text("B1-F1a complete\n")

    print(f"\n[DONE] elapsed={elapsed:.1f}s")
    print(f"  component_source_audit.csv: {len(all_comp_rows)} rows")
    print(f"  patient_source_summary.csv: {len(patient_summaries)} rows")
    print(f"  stage2_handoff_label_summary.csv: {len(s2_rows)} rows")
    print(f"  errors.csv: {len(error_rows)} rows")
    print(f"  stage2_holdout_access: 0  ✓")

    # 주요 분포 요약 출력
    print("\n[SOURCE DISTRIBUTION]")
    for lb, cnt in sorted(label_dist.items(), key=lambda x: -x[1]):
        pct = cnt / total_c * 100 if total_c else 0
        print(f"  {lb}: {cnt} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
