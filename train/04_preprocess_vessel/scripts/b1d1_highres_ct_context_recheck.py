#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B1-D1.7_wall_mediastinum_fp_high_resolution_ct_context_recheck

B1-D1.5/1.6 에서 low/medium confidence 였던 핵심 후보 15개를 고해상도 6-panel CT-context
overlay 로 재확인한다. (AD_wall_med 10 + B_true_boundary_hard_case 2 + lesion_risk_unclear 3)

- bare-run 즉시 중단(exit 2).
- --dry-run            : 입력/경로/shape/z±1/좌표/대상 선정 검증만. 파일/폴더 생성 없음.
- --real --confirm-write : ALLOW_REAL_PROCESSING=True 일 때만 6-panel PNG 생성 (B1-D1.7b).
- CT 는 np.load(mmap_mode='r') 후 z-1,z,z+1 slice 만 asarray. full volume copy 금지.
- 기존 score/mask/ROI/labels/PNG 수정 없음. PNG 제목은 ASCII/English (한글 폰트 회피).
- A/D 를 확정하지 않는다(재확인 자료 생성용). D_keep 을 재확인 없이 확정 근거로 쓰지 않는다.
"""
import argparse
import csv
import sys
from pathlib import Path
from collections import Counter

ALLOW_REAL_PROCESSING = False

BASE = Path("/home/jinhy/project/lung-ct-anomaly")
DIR = BASE / "outputs/b1d1_wall_mediastinum_fp_cause_diagnostic_v1"

LABELS_CSV = DIR / "b1d1_overlay_visual_review_labels.csv"   # 대상 선정(visual_label)
TARGET_CSV = DIR / "b1d1_overlay_target_selection.csv"       # 좌표
BRIDGE = BASE / "qa/dev_safe_mixed_error_visual_qa/b0_vessel_pleura_visual_bridge_manifest.csv"  # holdout 재검사
REFINED_ROI_ROOT = BASE / "outputs/mip-postprocess-research-v1/masks/refined_roi_v4_20_modeB_all_v1"
NROOT = Path("/mnt/c/Users/jinhy/Desktop/Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1/volumes_npy")
LROOT = Path("/mnt/c/Users/jinhy/Desktop/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1/volumes_npy")
CT_FILENAME = "ct_hu.npy"

OUT_FOLDER = DIR / "highres_ct_context_recheck_v1"  # B1-D1.7b 에서만 생성

PATCH_SIZE = 32
WL, WW = -600.0, 1500.0
N_EXPECT = 15

RECHECK_Q = {
    "AD_wall_med_inside": "A/D recheck: removable wall/mediastinum residue, or boundary structure to keep?",
    "B_true_boundary_hard_case": "Boundary hard-case recheck: can overlap rule suppress this safely?",
    "lesion_risk_unclear": "Lesion-risk recheck: would stronger ROI trimming cut lesion-adjacent tissue?",
}


def _fail(msg, code=2):
    print(f"[B1-D1.7][중단] {msg}", file=sys.stderr)
    sys.exit(code)


def resolve_mask(patient_id):
    c = sorted((REFINED_ROI_ROOT / "normal").glob(f"{patient_id}*/refined_roi.npy"))
    c += sorted((REFINED_ROI_ROOT / "lesion").glob(f"*{patient_id}*/refined_roi.npy"))
    if len(c) == 1:
        return c[0], "ok"
    return None, ("not_found" if not c else f"ambiguous({len(c)})")


def ct_path_from_mask(mask_path):
    group = mask_path.parent.parent.name
    voldir = mask_path.parent.name
    root = NROOT if group == "normal" else LROOT
    return root / voldir / CT_FILENAME, group


def load_holdout_map():
    m = {}
    if BRIDGE.exists():
        for r in csv.DictReader(open(BRIDGE, encoding="utf-8-sig")):
            m[r["review_id"]] = str(r.get("stage2_holdout_flag", "0")).strip()
    return m


def select_targets():
    """labels(대상 선정) + target(좌표) join. AD_wall_med 전부 + hard_case + lesion_risk_unclear."""
    labels = {r["selection_id"]: r for r in csv.DictReader(open(LABELS_CSV, encoding="utf-8"))}
    tgt = {r["selection_id"]: r for r in csv.DictReader(open(TARGET_CSV, encoding="utf-8"))}
    sel = []
    for sid, r in labels.items():
        if r["selection_group"] == "AD_wall_med_inside" or r["visual_label"] in ("B_true_boundary_hard_case", "lesion_risk_unclear"):
            t = tgt[sid]
            sel.append({"selection_id": sid, "review_id": r["review_id"], "patient_id": r["patient_id"],
                        "cause_class": r["cause_class"], "selection_group": r["selection_group"],
                        "human_label": r["human_label"], "candidate_score": r["candidate_score"],
                        "roi_0_0_patch_ratio": r["roi_0_0_patch_ratio"], "refined_roi_ratio": r["refined_roi_ratio"],
                        "previous_visual_label": r["visual_label"], "previous_confidence": r["visual_confidence"],
                        "candidate_local_z": t["candidate_local_z"], "candidate_y0": t["candidate_y0"],
                        "candidate_x0": t["candidate_x0"]})
    sel.sort(key=lambda r: r["selection_id"])
    return sel


def recheck_q(row):
    if row["selection_group"] == "AD_wall_med_inside":
        return RECHECK_Q["AD_wall_med_inside"]
    if row["previous_visual_label"] == "B_true_boundary_hard_case":
        return RECHECK_Q["B_true_boundary_hard_case"]
    return RECHECK_Q["lesion_risk_unclear"]


def validate(sel):
    import numpy as np
    res = {"fails": [], "n": len(sel), "unique_patients": 0, "holdout_access": 0,
           "mask_ok": 0, "ct_ok": 0, "shape_ok": 0, "zpm1_ok": 0, "coord_ok": 0,
           "by_group": {}, "by_prev_label": {}}
    if not LABELS_CSV.exists() or not TARGET_CSV.exists():
        _fail("입력 labels/target CSV 없음")
    if len(sel) != N_EXPECT:
        res["fails"].append(("n_count", "-", "-", f"{len(sel)} != {N_EXPECT}"))
    res["by_group"] = dict(Counter(r["selection_group"] for r in sel))
    res["by_prev_label"] = dict(Counter(r["previous_visual_label"] for r in sel))

    hmap = load_holdout_map()
    bad = [r["review_id"] for r in sel if hmap.get(r["review_id"], "0") not in ("0", "", "False", "false")]
    if bad:
        res["holdout_access"] = len(bad)
        for rid in bad:
            res["fails"].append(("stage2_holdout", rid, "-", "flag!=0"))
        return res

    seen = {}
    for p in sorted(set(r["patient_id"] for r in sel)):
        mp, st = resolve_mask(p)
        if st == "ok":
            cp, grp = ct_path_from_mask(mp)
            seen[p] = (mp, cp, st)
        else:
            seen[p] = (None, None, st)
    res["unique_patients"] = len(seen)

    shp = {}
    for p, (mp, cp, st) in seen.items():
        if st != "ok":
            res["fails"].append(("mask_map", "-", p, st)); continue
        if not (cp and cp.exists()):
            res["fails"].append(("ct_map", "-", p, f"CT 없음:{cp}")); continue
        try:
            ct = np.load(cp, mmap_mode="r"); mk = np.load(mp, mmap_mode="r")
            shp[p] = (tuple(ct.shape), tuple(mk.shape))
        except Exception as e:
            res["fails"].append(("load", "-", p, repr(e)))

    for r in sel:
        p = r["patient_id"]; sid = r["selection_id"]
        mp, cp, st = seen.get(p, (None, None, "na"))
        if st != "ok":
            continue
        res["mask_ok"] += 1
        if not (cp and cp.exists()):
            continue
        res["ct_ok"] += 1
        if p not in shp:
            continue
        cts, mks = shp[p]
        if not (len(cts) == 3 and cts == mks and cts[1] == 512 and cts[2] == 512):
            res["fails"].append(("shape", sid, p, f"ct{cts} mk{mks}")); continue
        res["shape_ok"] += 1
        Z = cts[0]
        z = int(float(r["candidate_local_z"])); y0 = int(float(r["candidate_y0"])); x0 = int(float(r["candidate_x0"]))
        if z - 1 >= 0 and z + 1 < Z:
            res["zpm1_ok"] += 1
        else:
            res["fails"].append(("zpm1", sid, p, f"z{z}/Z{Z} (z±1 범위밖)"))
        cy, cx = y0 + PATCH_SIZE // 2, x0 + PATCH_SIZE // 2
        if 0 <= y0 and 0 <= x0 and y0 + PATCH_SIZE <= 512 and x0 + PATCH_SIZE <= 512 and 0 <= cy < 512 and 0 <= cx < 512:
            res["coord_ok"] += 1
        else:
            res["fails"].append(("coord", sid, p, f"y{y0} x{x0}"))
    return res


def verdict_of(res):
    if res["holdout_access"] > 0 or any(f[0] in ("mask_map", "ct_map", "load") for f in res["fails"]):
        return "BLOCKED"
    return "NEEDS_FIX" if res["fails"] else "PASS"


def _crop(img, cy, cx, half):
    ys, ye = max(0, cy - half), min(512, cy + half)
    xs, xe = max(0, cx - half), min(512, cx + half)
    return img[ys:ye, xs:xe], xs, ys


def main():
    ap = argparse.ArgumentParser(description="B1-D1.7 high-res CT-context recheck (기본 차단)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--confirm-write", action="store_true")
    a = ap.parse_args()
    if not a.dry_run and not a.real:
        _fail("인자 없이 실행됨. --dry-run 또는 --real 필요(기본 실행 안 함).")

    sel = select_targets()
    res = validate(sel)
    print("=" * 60)
    print("B1-D1.7 high-res CT-context recheck - 검증")
    print("=" * 60)
    for k in ["n", "unique_patients", "holdout_access", "mask_ok", "ct_ok", "shape_ok", "zpm1_ok", "coord_ok"]:
        print(f"  {k:<16}: {res[k]}")
    print(f"  by_group        : {res['by_group']}")
    print(f"  by_prev_label   : {res['by_prev_label']}")
    if res["fails"]:
        print("  FAILS:")
        for c, sid, pid, d in res["fails"]:
            print(f"   - [{c}] {sid}/{pid}: {d}")
    v = verdict_of(res)
    print(f"  [검증 판정] {v}")

    if a.dry_run:
        print("\n[dry-run] 검증만 수행. 파일/폴더 생성 없음.")
        return

    # ---- real (B1-D1.7b) ----
    if not (ALLOW_REAL_PROCESSING and a.real and a.confirm_write):
        _fail("real 차단: ALLOW_REAL_PROCESSING=True 이고 --real --confirm-write 가 모두 필요.")
    if v != "PASS":
        _fail(f"검증 {v} 상태에서 PNG 생성 불가.")
    if OUT_FOLDER.exists():
        _fail(f"출력 폴더가 이미 존재함(덮어쓰기 금지): {OUT_FOLDER}")
    OUT_FOLDER.mkdir(parents=True, exist_ok=False)

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    lo, hi = WL - WW / 2, WL + WW / 2

    def win(sl):
        return np.clip((np.asarray(sl).astype(float) - lo) / (hi - lo), 0, 1)

    for r in sel:
        p = r["patient_id"]
        mp, st = resolve_mask(p)
        cp, grp = ct_path_from_mask(mp)
        ct = np.load(cp, mmap_mode="r")
        mk = np.load(mp, mmap_mode="r")
        z = int(float(r["candidate_local_z"])); y0 = int(float(r["candidate_y0"])); x0 = int(float(r["candidate_x0"]))
        cy, cx = y0 + PATCH_SIZE // 2, x0 + PATCH_SIZE // 2
        ct_z = win(ct[z]); mk_z = np.asarray(mk[z]); ct_m1 = win(ct[z - 1]); ct_p1 = win(ct[z + 1])
        fig, ax = plt.subplots(2, 3, figsize=(18, 12))
        # P1 full
        ax[0, 0].imshow(ct_z, cmap="gray"); ax[0, 0].set_title("P1 CT slice (full)"); ax[0, 0].axis("off")
        ax[0, 0].add_patch(Rectangle((x0, y0), PATCH_SIZE, PATCH_SIZE, fill=False, edgecolor="yellow", lw=1))
        # P2 CT+ROI+bbox
        ax[0, 1].imshow(ct_z, cmap="gray"); ax[0, 1].imshow(mk_z, alpha=0.3, cmap="Reds")
        ax[0, 1].add_patch(Rectangle((x0, y0), PATCH_SIZE, PATCH_SIZE, fill=False, edgecolor="yellow", lw=1.5))
        ax[0, 1].plot(cx, cy, "c+", ms=8); ax[0, 1].set_title("P2 CT + refined ROI + patch"); ax[0, 1].axis("off")
        # P3 160 crop
        c, xs, ys = _crop(ct_z, cy, cx, 80)
        ax[0, 2].imshow(c, cmap="gray"); ax[0, 2].add_patch(Rectangle((x0 - xs, y0 - ys), PATCH_SIZE, PATCH_SIZE, fill=False, edgecolor="yellow", lw=1.5))
        ax[0, 2].set_title("P3 zoom 160"); ax[0, 2].axis("off")
        # P4 96 crop
        c, xs, ys = _crop(ct_z, cy, cx, 48)
        ax[1, 0].imshow(c, cmap="gray"); ax[1, 0].add_patch(Rectangle((x0 - xs, y0 - ys), PATCH_SIZE, PATCH_SIZE, fill=False, edgecolor="yellow", lw=1.5))
        ax[1, 0].set_title("P4 zoom 96"); ax[1, 0].axis("off")
        # P5 z-1 160
        c, xs, ys = _crop(ct_m1, cy, cx, 80)
        ax[1, 1].imshow(c, cmap="gray"); ax[1, 1].add_patch(Rectangle((x0 - xs, y0 - ys), PATCH_SIZE, PATCH_SIZE, fill=False, edgecolor="orange", lw=1))
        ax[1, 1].set_title(f"P5 z-1 ({z - 1}) zoom160"); ax[1, 1].axis("off")
        # P6 z+1 160
        c, xs, ys = _crop(ct_p1, cy, cx, 80)
        ax[1, 2].imshow(c, cmap="gray"); ax[1, 2].add_patch(Rectangle((x0 - xs, y0 - ys), PATCH_SIZE, PATCH_SIZE, fill=False, edgecolor="orange", lw=1))
        ax[1, 2].set_title(f"P6 z+1 ({z + 1}) zoom160"); ax[1, 2].axis("off")
        fig.suptitle(f"{r['selection_id']} {r['review_id']} {p[:26]} | {r['cause_class']} | "
                     f"prev={r['previous_visual_label']}({r['previous_confidence']}) | "
                     f"roi0={r['roi_0_0_patch_ratio']} refined={r['refined_roi_ratio']} sc={r['candidate_score']} "
                     f"z={z} y={y0} x={x0}\nQ: {recheck_q(r)}", fontsize=9)
        out = OUT_FOLDER / f"{r['selection_id']}_{r['review_id']}_{r['previous_visual_label']}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
    print(f"[real] high-res recheck PNG {len(sel)}장 생성: {OUT_FOLDER}")


if __name__ == "__main__":
    main()
