#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B1-D1.4_wall_mediastinum_fp_overlay_generation

선정된 24개 overlay target(B1-D1.3)에 대해 CT + refined ROI + patch overlay PNG 를 생성한다.
- 기본 실행(bare-run)은 즉시 중단(exit 2).
- --dry-run            : 입력/mask/CT 경로/shape/좌표 검증만. 파일/폴더 생성 없음.
- --real --confirm-write : ALLOW_REAL_PROCESSING=True 일 때만 PNG 생성 (B1-D1.4b).
- CT 는 np.load(mmap_mode='r') 후 필요한 slice 1장만 asarray. full volume copy 금지.
- 기존 score/mask/ROI/CSV/JSON 수정 없음. AD_wall_med_inside 를 A/D 로 단정하지 않음.
"""
import argparse
import csv
import sys
from pathlib import Path
from collections import Counter

# =========================================================
# 안전 가드 (사용자 승인 전에는 True 로 바꾸지 않는다)
# =========================================================
ALLOW_REAL_PROCESSING = False

BASE = Path("/home/jinhy/project/lung-ct-anomaly")
DIR = BASE / "outputs/b1d1_wall_mediastinum_fp_cause_diagnostic_v1"

# ---- 입력 (read-only) ----
TARGET_CSV = DIR / "b1d1_overlay_target_selection.csv"
TARGET_JSON = DIR / "b1d1_overlay_target_selection_summary.json"
BRIDGE = BASE / "qa/dev_safe_mixed_error_visual_qa/b0_vessel_pleura_visual_bridge_manifest.csv"  # holdout 재검사용
REFINED_ROI_ROOT = BASE / "outputs/mip-postprocess-research-v1/masks/refined_roi_v4_20_modeB_all_v1"
NROOT = Path("/mnt/c/Users/jinhy/Desktop/Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1/volumes_npy")
LROOT = Path("/mnt/c/Users/jinhy/Desktop/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1/volumes_npy")
CT_FILENAME = "ct_hu.npy"

# ---- 출력 (B1-D1.4b 에서만 생성, exist_ok=False) ----
OUT_FOLDER = DIR / "overlay_png_selected_v1"

PATCH_SIZE = 32
N_EXPECT = 24
# lung window
WL, WW = -600.0, 1500.0

REQ_COLS = ["selection_id", "review_id", "patient_id", "safety_role", "human_label",
            "candidate_local_z", "candidate_y0", "candidate_x0", "candidate_score",
            "roi_0_0_patch_ratio", "refined_roi_ratio", "center_in_refined_roi",
            "cause_class", "selection_group", "selection_reason", "overlay_question"]


def _fail(msg, code=2):
    print(f"[B1-D1.4][중단] {msg}", file=sys.stderr)
    sys.exit(code)


def resolve_mask(patient_id):
    c = sorted((REFINED_ROI_ROOT / "normal").glob(f"{patient_id}*/refined_roi.npy"))
    c += sorted((REFINED_ROI_ROOT / "lesion").glob(f"*{patient_id}*/refined_roi.npy"))
    if len(c) == 1:
        return c[0], "ok"
    return None, ("not_found" if not c else f"ambiguous({len(c)})")


def ct_path_from_mask(mask_path):
    """mask 디렉토리명(__hash) == CT 디렉토리명. group 에 따라 root 선택."""
    group = mask_path.parent.parent.name   # normal / lesion
    voldir = mask_path.parent.name
    root = NROOT if group == "normal" else LROOT
    return root / voldir / CT_FILENAME, group


def load_holdout_map():
    m = {}
    if BRIDGE.exists():
        for r in csv.DictReader(open(BRIDGE, encoding="utf-8-sig")):
            m[r["review_id"]] = str(r.get("stage2_holdout_flag", "0")).strip()
    return m


def validate_targets():
    """read-only 검증. 결과 dict + rows 반환. CT/mask 는 mmap shape 만 확인."""
    import numpy as np
    res = {"fails": [], "rows": 0, "unique_patients": 0, "holdout_access": 0,
           "mask_unique_ok": 0, "mask_unique_fail": 0, "mask_row_ok": 0, "mask_row_fail": 0,
           "ct_unique_ok": 0, "ct_unique_fail": 0, "ct_row_ok": 0, "ct_row_fail": 0,
           "shape_ok": 0, "shape_fail": 0, "coord_ok": 0, "coord_fail": 0,
           "by_group": {}}
    if not TARGET_CSV.exists():
        _fail(f"target CSV 없음: {TARGET_CSV}")
    rows = list(csv.DictReader(open(TARGET_CSV, encoding="utf-8")))
    res["rows"] = len(rows)
    if len(rows) != N_EXPECT:
        res["fails"].append(("row_count", "-", "-", f"{len(rows)} != {N_EXPECT}"))
    ids = [r["selection_id"] for r in rows]
    if ids != [f"SEL{i:03d}" for i in range(1, len(rows) + 1)]:
        res["fails"].append(("selection_id_seq", "-", "-", "불연속"))
    miss = [c for c in REQ_COLS if rows and c not in rows[0]]
    if miss:
        res["fails"].append(("missing_cols", "-", "-", str(miss)))
    res["by_group"] = dict(Counter(r["selection_group"] for r in rows))

    # stage2_holdout 재검사 (bridge manifest 의 flag 를 review_id 로 join)
    hmap = load_holdout_map()
    bad = [r["review_id"] for r in rows if hmap.get(r["review_id"], "0") not in ("0", "", "False", "false")]
    if bad:
        res["holdout_access"] = len(bad)
        for rid in bad:
            res["fails"].append(("stage2_holdout", rid, "-", "holdout flag != 0"))
        return res, rows  # holdout 발견 시 즉시 반환 (BLOCKED)

    # patient 별 mask/CT 경로 매핑
    seen = {}
    upat = sorted(set(r["patient_id"] for r in rows))
    res["unique_patients"] = len(upat)
    for p in upat:
        mp, st = resolve_mask(p)
        if st == "ok":
            cp, grp = ct_path_from_mask(mp)
            seen[p] = (mp, cp, grp, st)
        else:
            seen[p] = (None, None, None, st)

    # unique 기준 mask/CT
    for p in upat:
        mp, cp, grp, st = seen[p]
        if st != "ok":
            res["mask_unique_fail"] += 1
            res["fails"].append(("mask_map", "-", p, st))
            continue
        res["mask_unique_ok"] += 1
        if cp and cp.exists():
            res["ct_unique_ok"] += 1
        else:
            res["ct_unique_fail"] += 1
            res["fails"].append(("ct_map", "-", p, f"CT 없음: {cp}"))

    # shape 캐시 (unique, mmap)
    shp = {}
    for p in upat:
        mp, cp, grp, st = seen[p]
        if st != "ok" or not (cp and cp.exists()):
            continue
        try:
            ct = np.load(cp, mmap_mode="r")
            mk = np.load(mp, mmap_mode="r")
            shp[p] = (tuple(ct.shape), tuple(mk.shape))
        except Exception as e:
            res["fails"].append(("load", "-", p, repr(e)))

    # row 단위 shape/좌표
    for r in rows:
        p = r["patient_id"]; sid = r["selection_id"]; rid = r["review_id"]
        mp, cp, grp, st = seen[p]
        if st != "ok":
            res["mask_row_fail"] += 1
            continue
        res["mask_row_ok"] += 1
        if not (cp and cp.exists()):
            res["ct_row_fail"] += 1
            continue
        res["ct_row_ok"] += 1
        if p not in shp:
            res["shape_fail"] += 1
            continue
        cts, mks = shp[p]
        if not (len(cts) == 3 and len(mks) == 3 and cts == mks and cts[1] == 512 and cts[2] == 512):
            res["shape_fail"] += 1
            res["fails"].append(("shape", sid, p, f"ct{cts} mk{mks}"))
            continue
        res["shape_ok"] += 1
        Z = cts[0]
        z = int(float(r["candidate_local_z"])); y0 = int(float(r["candidate_y0"])); x0 = int(float(r["candidate_x0"]))
        cy, cx = y0 + PATCH_SIZE // 2, x0 + PATCH_SIZE // 2
        if (0 <= z < Z and y0 >= 0 and x0 >= 0 and y0 + PATCH_SIZE <= 512 and x0 + PATCH_SIZE <= 512
                and 0 <= cy < 512 and 0 <= cx < 512):
            res["coord_ok"] += 1
        else:
            res["coord_fail"] += 1
            res["fails"].append(("coord", sid, rid, f"z{z}/Z{Z} y{y0} x{x0}"))
    return res, rows


def overall_verdict(res):
    if res["holdout_access"] > 0 or any(f[0] in ("mask_map", "ct_map", "load") for f in res["fails"]):
        return "BLOCKED"
    if res["fails"]:
        return "NEEDS_FIX"
    return "PASS"


def main():
    ap = argparse.ArgumentParser(description="B1-D1.4 overlay generation (기본 차단)")
    ap.add_argument("--dry-run", action="store_true", help="검증만, 파일/폴더 생성 없음")
    ap.add_argument("--real", action="store_true", help="PNG 생성 (ALLOW_REAL_PROCESSING + --confirm-write 필요)")
    ap.add_argument("--confirm-write", action="store_true", help="PNG 생성 확인 플래그")
    a = ap.parse_args()
    if not a.dry_run and not a.real:
        _fail("인자 없이 실행됨. --dry-run 또는 --real 필요(기본 실행 안 함).")

    res, rows = validate_targets()
    print("=" * 60)
    print("B1-D1.4 overlay generation - 검증")
    print("=" * 60)
    for k in ["rows", "unique_patients", "holdout_access",
              "mask_unique_ok", "mask_unique_fail", "ct_unique_ok", "ct_unique_fail",
              "mask_row_ok", "mask_row_fail", "ct_row_ok", "ct_row_fail",
              "shape_ok", "shape_fail", "coord_ok", "coord_fail"]:
        print(f"  {k:<18}: {res[k]}")
    print(f"  by_group          : {res['by_group']}")
    if res["fails"]:
        print("  FAILS:")
        for c, sid, pid, d in res["fails"]:
            print(f"   - [{c}] {sid}/{pid}: {d}")
    verdict = overall_verdict(res)
    print(f"  [검증 판정] {verdict}")

    if a.dry_run:
        print("\n[dry-run] 검증만 수행. 파일/폴더 생성 없음.")
        return

    # ---- real (B1-D1.4b) ----
    if not (ALLOW_REAL_PROCESSING and a.real and a.confirm_write):
        _fail("real 차단: ALLOW_REAL_PROCESSING=True 이고 --real --confirm-write 가 모두 필요.")
    if verdict != "PASS":
        _fail(f"검증 {verdict} 상태에서는 PNG 생성 불가.")
    if OUT_FOLDER.exists():
        _fail(f"출력 폴더가 이미 존재함(덮어쓰기 금지): {OUT_FOLDER}")
    OUT_FOLDER.mkdir(parents=True, exist_ok=False)

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    lo, hi = WL - WW / 2, WL + WW / 2
    pad = 48
    for r in rows:
        p = r["patient_id"]
        mp, st = resolve_mask(p)
        cp, grp = ct_path_from_mask(mp)
        ct = np.load(cp, mmap_mode="r")
        mk = np.load(mp, mmap_mode="r")
        z = int(float(r["candidate_local_z"])); y0 = int(float(r["candidate_y0"])); x0 = int(float(r["candidate_x0"]))
        ct_sl = np.asarray(ct[z]).astype(float)   # 해당 slice 1장만 (full volume copy 아님)
        mk_sl = np.asarray(mk[z])
        img = np.clip((ct_sl - lo) / (hi - lo), 0, 1)
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(img, cmap="gray"); ax[0].set_title("CT slice"); ax[0].axis("off")
        ax[1].imshow(img, cmap="gray"); ax[1].imshow(mk_sl, alpha=0.3, cmap="Reds")
        ax[1].add_patch(Rectangle((x0, y0), PATCH_SIZE, PATCH_SIZE, fill=False, edgecolor="yellow", lw=1.5))
        ax[1].plot(x0 + PATCH_SIZE // 2, y0 + PATCH_SIZE // 2, "c+", ms=8)
        ax[1].set_title("CT + refined ROI + patch"); ax[1].axis("off")
        ys, ye = max(0, y0 - pad), min(512, y0 + PATCH_SIZE + pad)
        xs, xe = max(0, x0 - pad), min(512, x0 + PATCH_SIZE + pad)
        ax[2].imshow(img[ys:ye, xs:xe], cmap="gray")
        ax[2].add_patch(Rectangle((x0 - xs, y0 - ys), PATCH_SIZE, PATCH_SIZE, fill=False, edgecolor="yellow", lw=1.5))
        ax[2].set_title("zoom"); ax[2].axis("off")
        fig.suptitle(f"{r['selection_id']} {r['review_id']} {r['patient_id'][:30]} | {r['cause_class']} "
                     f"| roi0={r['roi_0_0_patch_ratio']} refined={r['refined_roi_ratio']} sc={r['candidate_score']}\n"
                     f"Q: {r['overlay_question']}", fontsize=8)
        out = OUT_FOLDER / f"{r['selection_id']}_{r['review_id']}_{r['cause_class']}.png"
        fig.savefig(out, dpi=100, bbox_inches="tight")
        plt.close(fig)
    print(f"[real] overlay PNG {len(rows)}장 생성: {OUT_FOLDER}")


if __name__ == "__main__":
    main()
