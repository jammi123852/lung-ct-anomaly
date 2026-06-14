"""
generate_dynamic_normal_reference_bank_three_patients_v1.py

목적 (STEP 1):
정상 환자 3명(LUNG1-052__c3 final card 의 same-cell normal ref 3명)의 폐 실질 slice 전체를
동일 조건 lung-window PNG 로 export 하고, slice-level metadata index 를 생성한다.
이후 retrieval 단계에서 candidate 의 폐 내부 상대 위치(lung_z_pct + lung bbox 상대 y/x)로
가장 비슷한 normal slice/patch 를 동적으로 고를 수 있게 한다.

원칙:
- read-only CT load (mmap) 만 사용. model forward / feature extraction / score recompute / training 금지.
- raw CT 는 절대 output 으로 copy 하지 않는다 (PNG + metadata 만 생성).
- stage2_holdout 접근 금지. 병변/MSD candidate 금지 (정상 LUNA16 normal bank 만).
- roi_0_0.npy = 이진 lung mask (uint8 0/1), ct_hu.npy = int16 HU.

guard:
- 기본값 전부 False. 실제 생성은 env ALLOW_CT_LOAD=1 + ALLOW_PNG_WRITE=1 + `--run-generate --confirm-generate`
  3개 조건이 모두 충족될 때만 수행. 아니면 exit 2 (BLOCKED).
- --selftest / --plan-only / --dry-run / --static-drycheck 는 CT load / PNG write 없이 동작.

window: WL=-600, WW=1500 (S5 card 와 동일). 해상도 512x512 유지, stretch 금지, clean PNG.
"""

import os
import sys
import csv
import json
import argparse
import traceback

# --------------------------------------------------------------------------------------
# 상수 / 경로
# --------------------------------------------------------------------------------------
PROJECT_ROOT = "/home/jinhy/project/lung-ct-anomaly"
OUT_ROOT = os.path.join(
    PROJECT_ROOT,
    "outputs/position-aware-padim-v1/reports/dynamic_normal_reference_bank_three_patients_v1",
)

WL = -600
WW = 1500
CROP_SIZE = 96  # retrieval 단계 참고용 (STEP1 에서는 미사용)
LOW_QUALITY_AREA_RATIO = 0.02  # 이 미만이면 apex/base 극단부로 slice_quality=low (그래도 valid)

# 정상 환자 3명 (LUNG1-052__c3 final card v4_reference_selection 와 동일)
NORMAL_BANK_BASE = (
    "/mnt/c/Users/jinhy/Desktop/"
    "Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1/volumes_npy"
)
PATIENTS = [
    {
        "alias": "normal_patient_1",
        "role": "normal_ref_1",
        "patient_id": "subset1_1.3.6.1.4.1.14519.5.2.1.6279.6001.100684836163890911914061745866",
        "volume_id": "subset1_1.3.6.1.4.1.14519.5.2.1.6279.6001.100684836163890911914061745866__179f88da02",
    },
    {
        "alias": "normal_patient_2",
        "role": "normal_ref_3",
        "patient_id": "subset9_1.3.6.1.4.1.14519.5.2.1.6279.6001.109882169963817627559804568094",
        "volume_id": "subset9_1.3.6.1.4.1.14519.5.2.1.6279.6001.109882169963817627559804568094__b0c943be3c",
    },
    {
        "alias": "normal_patient_3",
        "role": "normal_ref_2",
        "patient_id": "subset2_1.3.6.1.4.1.14519.5.2.1.6279.6001.311236942972970815890902714604",
        "volume_id": "subset2_1.3.6.1.4.1.14519.5.2.1.6279.6001.311236942972970815890902714604__5917898e8b",
    },
]

SLICE_INDEX_COLUMNS = [
    "patient_alias", "patient_id", "volume_id", "local_z", "png_path",
    "ct_shape_z", "image_h", "image_w",
    "lung_area_px", "lung_area_ratio",
    "lung_bbox_y0", "lung_bbox_x0", "lung_bbox_y1", "lung_bbox_x1",
    "lung_center_y", "lung_center_x",
    "lung_z_min", "lung_z_max", "lung_z_pct",
    "valid_lung_slice", "slice_quality", "image_lung_side_available",
    "source_ct_path_hash_or_basename", "source_roi_path_hash_or_basename",
    "stage2_holdout_flag", "notes",
]


# --------------------------------------------------------------------------------------
# 순수 함수 (selftest 대상, CT 불필요)
# --------------------------------------------------------------------------------------
def patient_paths(p):
    d = os.path.join(NORMAL_BANK_BASE, p["volume_id"])
    return os.path.join(d, "ct_hu.npy"), os.path.join(d, "roi_0_0.npy")


def lung_z_pct(local_z, z_min, z_max):
    """폐 z 범위 기준 상대 위치. 절대 local_z 사용 금지."""
    return (local_z - z_min) / max(z_max - z_min, 1)


def window_hu_to_uint8(arr, wl=WL, ww=WW):
    """HU -> uint8 lung window. stretch 없음(고정 window)."""
    import numpy as np
    lo = wl - ww / 2.0
    hi = wl + ww / 2.0
    a = (np.asarray(arr, dtype="float32") - lo) / max(hi - lo, 1e-6)
    a = np.clip(a, 0.0, 1.0)
    return (a * 255.0 + 0.5).astype("uint8")


def lung_bbox_from_mask(mask2d):
    """이진 lung mask(2D) -> (y0,x0,y1,x1, cy, cx, area, side_avail).
    lung 없으면 area=0 반환."""
    import numpy as np
    m = (np.asarray(mask2d) > 0)
    area = int(m.sum())
    if area == 0:
        return (0, 0, 0, 0, 0.0, 0.0, 0, "none")
    ys, xs = np.where(m)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    cy = float(ys.mean())
    cx = float(xs.mean())
    h, w = m.shape
    mid = w // 2
    left_has = bool(m[:, :mid].any())   # image_left = 이미지 왼쪽 절반
    right_has = bool(m[:, mid:].any())
    if left_has and right_has:
        side = "both"
    elif left_has:
        side = "left"
    elif right_has:
        side = "right"
    else:
        side = "none"
    return (y0, x0, y1, x1, cy, cx, area, side)


def slice_quality_label(area_ratio):
    return "low" if area_ratio < LOW_QUALITY_AREA_RATIO else "ok"


def png_name(local_z):
    return f"z_{local_z:03d}.png"


# --------------------------------------------------------------------------------------
# 모드: selftest / plan / dry-run / static-drycheck
# --------------------------------------------------------------------------------------
def run_selftest():
    import numpy as np
    checks = []

    def ck(name, cond):
        checks.append((name, bool(cond)))

    # lung_z_pct
    ck("z_pct_min", abs(lung_z_pct(10, 10, 50) - 0.0) < 1e-9)
    ck("z_pct_max", abs(lung_z_pct(50, 10, 50) - 1.0) < 1e-9)
    ck("z_pct_mid", abs(lung_z_pct(30, 10, 50) - 0.5) < 1e-9)
    ck("z_pct_div0_guard", abs(lung_z_pct(10, 10, 10) - 0.0) < 1e-9)

    # window
    w = window_hu_to_uint8(np.array([[-1350, 150], [-600, 1112]], dtype="int16"))
    ck("window_dtype_uint8", w.dtype == np.uint8)
    ck("window_lo_is_0", int(w[0, 0]) == 0)
    ck("window_hi_is_255", int(w[0, 1]) == 255)
    ck("window_shape_keep", w.shape == (2, 2))

    # bbox
    m = np.zeros((512, 512), dtype="uint8")
    m[100:200, 50:150] = 1   # left half
    m[100:200, 360:460] = 1  # right half
    y0, x0, y1, x1, cy, cx, area, side = lung_bbox_from_mask(m)
    ck("bbox_y0", y0 == 100)
    ck("bbox_x0", x0 == 50)
    ck("bbox_y1", y1 == 200)
    ck("bbox_x1", x1 == 460)
    ck("bbox_area", area == 100 * 100 * 2)
    ck("bbox_side_both", side == "both")
    # empty mask
    y0, x0, y1, x1, cy, cx, area, side = lung_bbox_from_mask(np.zeros((10, 10), "uint8"))
    ck("empty_area0", area == 0)
    ck("empty_side_none", side == "none")
    # left only
    ml = np.zeros((100, 100), "uint8"); ml[10:20, 5:15] = 1
    ck("side_left", lung_bbox_from_mask(ml)[7] == "left")
    mr = np.zeros((100, 100), "uint8"); mr[10:20, 80:90] = 1
    ck("side_right", lung_bbox_from_mask(mr)[7] == "right")

    # quality
    ck("quality_low", slice_quality_label(0.005) == "low")
    ck("quality_ok", slice_quality_label(0.30) == "ok")

    # naming
    ck("png_name_pad", png_name(7) == "z_007.png")
    ck("png_name_pad3", png_name(123) == "z_123.png")

    # column set
    ck("columns_count", len(SLICE_INDEX_COLUMNS) == 26)

    npass = sum(1 for _, c in checks if c)
    for name, c in checks:
        print(f"  [{'PASS' if c else 'FAIL'}] {name}")
    print(f"SELFTEST: {npass}/{len(checks)} PASS")
    return npass == len(checks)


def run_plan_only():
    print("=== PLAN-ONLY (CT load / PNG write 없음) ===")
    print(f"OUT_ROOT: {OUT_ROOT}")
    print(f"window: WL={WL} WW={WW}  | 해상도 512x512 유지, stretch 금지")
    print(f"정상 환자 {len(PATIENTS)}명:")
    for p in PATIENTS:
        ctp, roip = patient_paths(p)
        print(f"  - {p['alias']} ({p['role']}) {p['patient_id'][:24]}...")
        print(f"      ct : {ctp}")
        print(f"      roi: {roip}")
    print("출력 예정: normal_patient_{1,2,3}/slices_png/z_*.png + patient_slice_index.csv")
    print("           dynamic_reference_slice_index.csv, dynamic_reference_patient_inventory.csv")
    print("           generation_summary.json, safety_check.json, errors.csv, DONE.json")
    print("실제 생성 조건: ALLOW_CT_LOAD=1 ALLOW_PNG_WRITE=1 ... --run-generate --confirm-generate")
    return True


def run_dry_run():
    print("=== DRY-RUN (경로 존재만 read-only 확인, CT 미load / PNG 미생성) ===")
    ok = True
    for p in PATIENTS:
        ctp, roip = patient_paths(p)
        ce, re_ = os.path.exists(ctp), os.path.exists(roip)
        print(f"  {p['alias']}: ct_exists={ce} roi_exists={re_}")
        ok = ok and ce and re_
    print(f"DRY-RUN paths_ok={ok}")
    return ok


def run_static_drycheck():
    print("=== STATIC-DRYCHECK (소스 스캔, 미실행) ===")
    full = open(__file__, encoding="utf-8").read()
    # 자기 검사 함수 본문은 forbidden 토큰 리터럴을 포함하므로 self-match 방지 위해 제외
    start = full.find("def run_static_drycheck")
    end = full.find("\ndef run_generate", start)
    src = full[:start] + (full[end:] if end != -1 else "")
    checks = []

    def ck(name, cond):
        checks.append((name, bool(cond)))

    ck("guard_ALLOW_CT_LOAD", 'ALLOW_CT_LOAD' in src)
    ck("guard_ALLOW_PNG_WRITE", 'ALLOW_PNG_WRITE' in src)
    ck("guard_confirm_generate", '--confirm-generate' in src or 'confirm_generate' in src)
    ck("no_model_forward", '.forward(' not in src and 'model(' not in src)
    ck("no_feature_extract", 'extract_feature' not in src and 'featuremap' not in src)
    ck("no_grad_cam", 'grad_cam' not in src.lower())
    ck("no_score_recompute", 'mahalanobis' not in src.lower() and 'cov_inv' not in src.lower())
    ck("no_train", 'optimizer' not in src.lower() and '.backward(' not in src)
    ck("readonly_mmap", "mmap_mode='r'" in src or 'mmap_mode="r"' in src)
    ck("window_consts", 'WL = -600' in src and 'WW = 1500' in src)
    ck("no_raw_ct_copy", 'shutil.copy' not in src and 'copyfile' not in src)
    npass = sum(1 for _, c in checks if c)
    for name, c in checks:
        print(f"  [{'PASS' if c else 'FAIL'}] {name}")
    print(f"STATIC-DRYCHECK: {npass}/{len(checks)} PASS")
    return npass == len(checks)


# --------------------------------------------------------------------------------------
# 실제 생성 (guard 통과 시에만)
# --------------------------------------------------------------------------------------
def run_generate():
    import numpy as np
    from PIL import Image

    os.makedirs(OUT_ROOT, exist_ok=True)
    errors = []
    all_rows = []
    inv_rows = []
    H = W = 512

    for p in PATIENTS:
        alias = p["alias"]
        ctp, roip = patient_paths(p)
        png_dir = os.path.join(OUT_ROOT, alias, "slices_png")
        os.makedirs(png_dir, exist_ok=True)

        try:
            ct = np.load(ctp, mmap_mode="r")
            roi = np.load(roip, mmap_mode="r")
        except Exception as e:
            errors.append({"patient_alias": alias, "stage": "load", "error": repr(e)})
            continue

        Z = int(ct.shape[0])
        if roi.shape != ct.shape:
            errors.append({"patient_alias": alias, "stage": "shape",
                           "error": f"shape mismatch ct{ct.shape} roi{roi.shape}"})
            continue

        # lung z 범위 먼저 계산 (area>0 인 z)
        areas = np.array([int((np.asarray(roi[z]) > 0).sum()) for z in range(Z)], dtype="int64")
        lung_zs = np.where(areas > 0)[0]
        if lung_zs.size == 0:
            errors.append({"patient_alias": alias, "stage": "lung_range", "error": "no lung voxels"})
            continue
        z_min, z_max = int(lung_zs.min()), int(lung_zs.max())

        n_valid = 0
        n_low = 0
        for z in range(Z):
            area = int(areas[z])
            if area == 0:
                continue  # 폐 없는 slice 는 export/index 안 함
            mask2d = np.asarray(roi[z])
            y0, x0, y1, x1, cy, cx, _a, side = lung_bbox_from_mask(mask2d)
            area_ratio = area / float(H * W)
            quality = slice_quality_label(area_ratio)
            zpct = lung_z_pct(z, z_min, z_max)
            name = png_name(z)
            rel_png = os.path.join(alias, "slices_png", name)
            abs_png = os.path.join(OUT_ROOT, rel_png)

            # PNG export (resume: 있으면 skip)
            if not os.path.exists(abs_png):
                try:
                    img8 = window_hu_to_uint8(np.asarray(ct[z]))
                    Image.fromarray(img8, mode="L").save(abs_png)
                except Exception as e:
                    errors.append({"patient_alias": alias, "stage": f"png_z{z}", "error": repr(e)})
                    continue

            row = {
                "patient_alias": alias, "patient_id": p["patient_id"], "volume_id": p["volume_id"],
                "local_z": z, "png_path": rel_png,
                "ct_shape_z": Z, "image_h": H, "image_w": W,
                "lung_area_px": area, "lung_area_ratio": round(area_ratio, 6),
                "lung_bbox_y0": y0, "lung_bbox_x0": x0, "lung_bbox_y1": y1, "lung_bbox_x1": x1,
                "lung_center_y": round(cy, 2), "lung_center_x": round(cx, 2),
                "lung_z_min": z_min, "lung_z_max": z_max, "lung_z_pct": round(zpct, 6),
                "valid_lung_slice": True, "slice_quality": quality,
                "image_lung_side_available": side,
                "source_ct_path_hash_or_basename": p["volume_id"],
                "source_roi_path_hash_or_basename": p["volume_id"],
                "stage2_holdout_flag": False, "notes": "",
            }
            all_rows.append(row)
            n_valid += 1
            if quality == "low":
                n_low += 1

        # per-patient index
        pidx = os.path.join(OUT_ROOT, alias, "patient_slice_index.csv")
        with open(pidx, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SLICE_INDEX_COLUMNS)
            w.writeheader()
            for r in all_rows:
                if r["patient_alias"] == alias:
                    w.writerow(r)

        inv_rows.append({
            "patient_alias": alias, "role": p["role"], "patient_id": p["patient_id"],
            "volume_id": p["volume_id"], "ct_shape_z": Z,
            "n_valid_slices": n_valid, "n_low_quality": n_low,
            "lung_z_min": z_min, "lung_z_max": z_max,
            "png_dir": os.path.join(alias, "slices_png"),
            "stage2_holdout_flag": False,
        })

    # combined slice index
    with open(os.path.join(OUT_ROOT, "dynamic_reference_slice_index.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SLICE_INDEX_COLUMNS)
        w.writeheader()
        w.writerows(all_rows)

    # patient inventory
    inv_cols = ["patient_alias", "role", "patient_id", "volume_id", "ct_shape_z",
                "n_valid_slices", "n_low_quality", "lung_z_min", "lung_z_max",
                "png_dir", "stage2_holdout_flag"]
    with open(os.path.join(OUT_ROOT, "dynamic_reference_patient_inventory.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=inv_cols)
        w.writeheader()
        w.writerows(inv_rows)

    # errors.csv
    with open(os.path.join(OUT_ROOT, "errors.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["patient_alias", "stage", "error"])
        w.writeheader()
        w.writerows(errors)

    total_png = len(all_rows)
    summary = {
        "out_root": OUT_ROOT, "window": {"WL": WL, "WW": WW},
        "resolution": "512x512", "stretch": False,
        "n_patients": len(PATIENTS), "n_patients_with_slices": len(inv_rows),
        "total_valid_slice_png": total_png,
        "per_patient": inv_rows,
        "ct_load": "read-only mmap", "model_forward": False, "feature_extraction": False,
        "score_recompute": False, "training": False, "raw_ct_copied_to_output": False,
    }
    with open(os.path.join(OUT_ROOT, "generation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    safety = {
        "raw_ct_copied_to_output": False, "model_weights_written": False,
        "feature_files_written": False, "stage2_holdout_accessed": False,
        "lesion_or_msd_candidate_included": False, "model_forward": False,
        "feature_extraction": False, "score_recompute": False, "training": False,
        "ct_load": "read-only mmap (slice PNG + lung bbox/position metadata only)",
        "patients_all_normal_holdout_false": True,
    }
    with open(os.path.join(OUT_ROOT, "safety_check.json"), "w") as f:
        json.dump(safety, f, indent=2)

    # PNG 수 vs index 수 검증
    disk_png = 0
    for p in PATIENTS:
        d = os.path.join(OUT_ROOT, p["alias"], "slices_png")
        if os.path.isdir(d):
            disk_png += len([x for x in os.listdir(d) if x.endswith(".png")])
    conditions_ok = (
        len(inv_rows) == len(PATIENTS)
        and all(r["n_valid_slices"] > 0 for r in inv_rows)
        and disk_png == total_png
        and len(errors) == 0
    )
    done = {
        "conditions_ok": bool(conditions_ok),
        "n_patients": len(PATIENTS), "n_patients_with_slices": len(inv_rows),
        "total_valid_slice_png_index_rows": total_png,
        "total_png_on_disk": disk_png,
        "png_eq_index": disk_png == total_png,
        "errors": len(errors),
    }
    with open(os.path.join(OUT_ROOT, "DONE.json"), "w") as f:
        json.dump(done, f, indent=2)

    print(json.dumps(done, indent=2))
    print(f"total_valid_slice_png={total_png} disk_png={disk_png} errors={len(errors)}")
    return conditions_ok


# --------------------------------------------------------------------------------------
# main / guard
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--static-drycheck", action="store_true")
    ap.add_argument("--run-generate", action="store_true")
    ap.add_argument("--confirm-generate", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if run_selftest() else 1)
    if args.plan_only:
        sys.exit(0 if run_plan_only() else 1)
    if args.dry_run:
        sys.exit(0 if run_dry_run() else 1)
    if args.static_drycheck:
        sys.exit(0 if run_static_drycheck() else 1)

    if args.run_generate:
        allow_ct = os.environ.get("ALLOW_CT_LOAD") == "1"
        allow_png = os.environ.get("ALLOW_PNG_WRITE") == "1"
        if not (allow_ct and allow_png and args.confirm_generate):
            print("BLOCKED: 실제 생성에는 ALLOW_CT_LOAD=1 + ALLOW_PNG_WRITE=1 + --confirm-generate 필요.")
            print(f"  ALLOW_CT_LOAD={os.environ.get('ALLOW_CT_LOAD')} "
                  f"ALLOW_PNG_WRITE={os.environ.get('ALLOW_PNG_WRITE')} "
                  f"confirm={args.confirm_generate}")
            sys.exit(2)
        try:
            ok = run_generate()
            sys.exit(0 if ok else 1)
        except Exception:
            traceback.print_exc()
            sys.exit(1)

    print("모드 미지정. --selftest / --plan-only / --dry-run / --static-drycheck / "
          "--run-generate --confirm-generate 중 하나를 지정하세요. (기본 BLOCKED)")
    sys.exit(2)


if __name__ == "__main__":
    main()
