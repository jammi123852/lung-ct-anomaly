#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B1-D1_wall_mediastinum_fp_cause_diagnostic_preflight

목적:
    PatchCore 구현 전, 현재 PaDiM(v2 roi_0_0) 흉벽/종격동 false positive 가
    "왜 남는지" 를 read-only 로 원인 분리 진단한다.

원인 분류:
    A = ROI/mask 미흡          : 흉벽/종격동인데 refined ROI 제거 후에도 patch 가 ROI 안에 거의 다 남음 (더 깎을 여지)
    B = patch 경계 걸침         : patch 가 refined ROI 경계를 걸침 (중간 overlap)
    C = ROI 밖 ranking 포함     : patch 가 refined ROI 밖인데 후보로 들어옴
    D = 경계 정상구조 고점수    : refined ROI 안에 남길 수밖에 없는(더 깎으면 폐실질/병변 손실) 정상 구조가 고점수
                                  (A 와 1차로 같이 묶고 overlay 눈검증으로 세분)
    => D 가 충분히 많을 때만 PatchCore 검토 가치가 있다.

이 스크립트가 하지 않는 것 (절대 금지):
    - training / model forward / scoring 재실행 / threshold 재계산
    - stage2_holdout 접근 (holdout flag != 0 발견 시 즉시 중단)
    - 기존 score CSV / mask / ROI 수정·삭제·이동·덮어쓰기
    - adjusted_score / suppression_weight / refined score 생성
    - PatchCore 구현
    - full run

기본값은 "실행되지 않음" 이다.
    - 인자 없이 실행하면 즉시 중단.
    - --dry-run  : read-only 입력/매핑/shape 점검만. 파일을 생성하지 않음.
    - --real     : ALLOW_REAL_PROCESSING=True 이고 --confirm-write 가 함께 있을 때만
                   진단 CSV/JSON 을 새 폴더에 생성. (overlay PNG 는 --make-overlay 추가 플래그)
"""

import argparse
import csv
import json
import sys
from pathlib import Path

# =========================================================
# 안전 가드 (사용자 승인 전에는 절대 True 로 바꾸지 않는다)
# =========================================================
ALLOW_REAL_PROCESSING = False

BASE = Path("/home/jinhy/project/lung-ct-anomaly")

# ---- 입력 (read-only) ----
BRIDGE_MANIFEST = BASE / "qa/dev_safe_mixed_error_visual_qa/b0_vessel_pleura_visual_bridge_manifest.csv"
REFINED_ROI_ROOT = BASE / "outputs/mip-postprocess-research-v1/masks/refined_roi_v4_20_modeB_all_v1"  # 하위 normal/ , lesion/

# ---- 출력 (새 폴더, exist_ok=False) ----
OUT_DIR = BASE / "outputs/b1d1_wall_mediastinum_fp_cause_diagnostic_v1"

# ---- 진단 파라미터 ----
PATCH_SIZE = 32
WALL_MED_LABELS = {"pleura_or_chest_wall", "hilar_or_mediastinal"}
# refined ROI overlap 임계 (분류용, 눈검증으로 최종 확정)
RATIO_INSIDE = 0.90   # 이상이면 ROI 안에 거의 다 남음 (A/D)
RATIO_OUTSIDE = 0.10  # 이하이면 ROI 밖 (C)

# bridge manifest 컬럼명 (실제 헤더 기준)
COL_PID = "patient_id"
COL_RID = "review_id"
COL_ROLE = "safety_role"            # fp_candidate / lesion_protect
COL_LABEL = "human_label"
COL_Z = "candidate_local_z"
COL_Y0 = "candidate_y0"
COL_X0 = "candidate_x0"
COL_SCORE = "candidate_score"
COL_ROI0 = "roi_0_0_patch_ratio"    # 흉벽 제거 전(roi_0_0) overlap (이미 backfill 되어 있음)
COL_HOLDOUT = "stage2_holdout_flag"


def _fail(msg, code=2):
    print(f"[B1-D1][중단] {msg}", file=sys.stderr)
    sys.exit(code)


def load_manifest():
    if not BRIDGE_MANIFEST.exists():
        _fail(f"bridge manifest 없음: {BRIDGE_MANIFEST}")
    with open(BRIDGE_MANIFEST, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        _fail("bridge manifest 가 비어 있음")
    return rows


def assert_no_holdout(rows):
    """stage2_holdout 차단: flag != 0 인 행이 하나라도 있으면 즉시 중단."""
    bad = [r[COL_RID] for r in rows if str(r.get(COL_HOLDOUT, "0")).strip() not in ("0", "", "False", "false")]
    if bad:
        _fail(f"stage2_holdout 행 포함 발견({len(bad)}개): {bad[:10]} ... -> 진단 중단(차단)")
    return 0  # holdout 접근 0 보장


def resolve_mask_path(patient_id):
    """
    review patient_id -> refined ROI npy 경로 매핑.
    normal: subset*_<SUID>          -> REFINED_ROI_ROOT/normal/<patient_id>*/refined_roi.npy
    lesion: MSD_lung_xxx / LUNG1-x  -> REFINED_ROI_ROOT/lesion/*<patient_id>*/refined_roi.npy
    glob 로 prefix/substring 매칭. 0개 또는 2개 이상이면 매핑 실패로 표시.
    """
    candidates = []
    # normal: SUID prefix 매칭
    candidates += sorted((REFINED_ROI_ROOT / "normal").glob(f"{patient_id}*/refined_roi.npy"))
    # lesion: substring 매칭 (MSD_Lung_MSD_lung_065__hash 형태이므로 substring)
    candidates += sorted((REFINED_ROI_ROOT / "lesion").glob(f"*{patient_id}*/refined_roi.npy"))
    if len(candidates) == 1:
        return candidates[0], "ok"
    if len(candidates) == 0:
        return None, "not_found"
    return None, f"ambiguous({len(candidates)})"


def classify(role, label, refined_ratio, center_in_roi):
    """A/B/C/D + lesion safety 1차 분류 (눈검증으로 최종 확정)."""
    if role == "lesion_protect":
        # 병변 보호 후보: refined ROI 가 병변을 잘라내는지 점검
        if refined_ratio <= RATIO_OUTSIDE:
            return "LESION_RISK_cut_out"   # 흉벽 제거가 병변을 거의 잘라냄 (위험)
        if refined_ratio < RATIO_INSIDE:
            return "LESION_RISK_partial"   # 병변 일부가 경계에서 잘림
        return "lesion_kept"               # 병변 보존 OK
    # fp_candidate
    if refined_ratio <= RATIO_OUTSIDE:
        return "C_outside_roi"             # refined ROI 밖인데 후보 -> ranking 포함 문제
    if refined_ratio < RATIO_INSIDE:
        return "B_boundary"                # 경계 걸침
    # refined_ratio >= RATIO_INSIDE : ROI 안에 거의 다 남음
    if label in WALL_MED_LABELS:
        return "AD_wall_med_inside"        # A(더 깎을 여지) / D(못 깎음) -> overlay 로 세분
    return "AD_other_inside"               # vessel/diaphragm 등 ROI 안 고점수


def validate_dry_run(rows, seen, holdout_access):
    """--dry-run 전용 read-only 검증 (B1-D1.1 보강). 파일 생성 없음. mmap_mode='r'만 사용."""
    import numpy as np
    from collections import Counter

    fails = []  # (category, review_id, patient_id, detail)

    # ---- (1) unique patient mask 매핑 ----
    unique_patients = list(seen.keys())
    mapped = [p for p in unique_patients if seen[p][1] == "ok"]
    failed_map = [p for p in unique_patients if seen[p][1] != "ok"]
    for p in failed_map:
        fails.append(("mask_mapping", "-", p, seen[p][1]))
    print("-" * 60)
    print("[1] unique patient mask 매핑")
    print(f"    unique_patient_count        : {len(unique_patients)}")
    print(f"    mapped_unique_patient_count : {len(mapped)}")
    print(f"    failed_unique_patient_count : {len(failed_map)}")
    if failed_map:
        print(f"    failed_patient_id           : {failed_map}")

    # ---- (2) mask shape 검증 (매핑 성공 patient 전수) ----
    print("[2] mask shape 검증 (mmap_mode='r')")
    mask_z = {}  # patient -> Z (local_z 검증용)
    for p in mapped:
        path = seen[p][0]
        try:
            arr = np.load(path, mmap_mode="r")
        except Exception as e:
            fails.append(("mask_load", "-", p, repr(e)))
            print(f"    [FAIL load] {p}: {e!r}")
            continue
        shp = arr.shape
        is3d = (arr.ndim == 3)
        hw_ok = bool(is3d and shp[1] == 512 and shp[2] == 512)
        if is3d:
            mask_z[p] = shp[0]
            if not hw_ok:
                fails.append(("mask_hw", "-", p, f"H,W={shp[1]},{shp[2]} != 512,512"))
        else:
            fails.append(("mask_not_3d", "-", p, f"shape={shp}"))
        # dtype 은 출력만, 조건 실패로 막지 않음
        print(f"    {p[:46]:<46} shape={str(shp):<18} dtype={arr.dtype} 3D={is3d} HW512={hw_ok}")

    # ---- (3) patient별 local_z 범위 검증 ----
    print("[3] local_z 범위 검증 (patient별)")
    pz = {}  # patient -> [zmin, zmax]
    for r in rows:
        p = r[COL_PID]; z = int(float(r[COL_Z]))
        if p not in pz:
            pz[p] = [z, z]
        pz[p][0] = min(pz[p][0], z); pz[p][1] = max(pz[p][1], z)
    for p in mapped:
        zmin, zmax = pz[p]
        Z = mask_z.get(p)
        if Z is None:
            print(f"    {p[:46]:<46} local_z[{zmin},{zmax}] vs Z=N/A (shape fail)")
            continue
        ok = (zmin >= 0) and (zmax < Z)
        if not ok:
            bad = [r[COL_RID] for r in rows if r[COL_PID] == p and not (0 <= int(float(r[COL_Z])) < Z)]
            fails.append(("local_z_range", ",".join(bad), p, f"local_z[{zmin},{zmax}] vs Z={Z}"))
        print(f"    {p[:46]:<46} local_z[{zmin},{zmax}] vs Z={Z}  {'OK' if ok else 'FAIL'}")

    # ---- (4) patch 좌표 범위 검증 (전 행) ----
    print("[4] patch 좌표 범위 검증 (y0,x0>=0, +PATCH_SIZE<=512)")
    coord_fail = 0
    for r in rows:
        y0 = int(float(r[COL_Y0])); x0 = int(float(r[COL_X0]))
        ok = (y0 >= 0 and x0 >= 0 and y0 + PATCH_SIZE <= 512 and x0 + PATCH_SIZE <= 512)
        if not ok:
            coord_fail += 1
            fails.append(("patch_coord", r[COL_RID], r[COL_PID], f"y0={y0},x0={x0},+{PATCH_SIZE}"))
            print(f"    [FAIL] review_id={r[COL_RID]} patient_id={r[COL_PID][:40]} y0={y0} x0={x0}")
    print(f"    좌표 범위 벗어난 행 수: {coord_fail}/{len(rows)}")

    # ---- (5) stage2_holdout 검증 (기존 assert 유지 + 분포 출력) ----
    dist = dict(Counter(str(r.get(COL_HOLDOUT, "0")).strip() for r in rows))
    print("[5] stage2_holdout 검증")
    print(f"    stage2_holdout_access : {holdout_access}  (반드시 0)")
    print(f"    holdout_flag 분포     : {dist}")

    # ---- (6) 종합 판정 ----
    print("-" * 60)
    if not fails and holdout_access == 0:
        verdict = "PASS"
        print("[dry-run 판정] PASS")
    else:
        verdict = "NEEDS_FIX"
        print(f"[dry-run 판정] {verdict}  (실패 {len(fails)}건)")
        for cat, rid, pid, detail in fails:
            print(f"    - [{cat}] review_id={rid} patient_id={pid} :: {detail}")
    return verdict, fails


def main():
    ap = argparse.ArgumentParser(description="B1-D1 흉벽/종격동 FP 원인 진단 (read-only preflight)")
    ap.add_argument("--dry-run", action="store_true", help="입력/매핑/shape 점검만, 파일 생성 없음")
    ap.add_argument("--real", action="store_true", help="진단 CSV/JSON 생성 (ALLOW_REAL_PROCESSING + --confirm-write 필요)")
    ap.add_argument("--confirm-write", action="store_true", help="실제 파일 생성 확인 플래그")
    ap.add_argument("--make-overlay", action="store_true", help="overlay PNG 생성 (real 모드에서만, 추가 승인 대상)")
    args = ap.parse_args()

    if not args.dry_run and not args.real:
        _fail("인자 없이 실행됨. --dry-run 또는 --real 을 명시해야 한다. (기본 실행 안 함)")

    rows = load_manifest()
    holdout_access = assert_no_holdout(rows)

    # 매핑/shape 점검 (numpy 는 shape 확인 시에만 import)
    map_report = []
    seen = {}
    for r in rows:
        pid = r[COL_PID]
        if pid not in seen:
            path, status = resolve_mask_path(pid)
            seen[pid] = (path, status)
        path, status = seen[pid]
        map_report.append({"review_id": r[COL_RID], "patient_id": pid,
                           "role": r[COL_ROLE], "label": r[COL_LABEL],
                           "mask_path": str(path) if path else None, "map_status": status})

    n_ok = sum(1 for m in map_report if m["map_status"] == "ok")
    n_fail = len(map_report) - n_ok
    fail_pids = sorted({m["patient_id"] for m in map_report if m["map_status"] != "ok"})

    print("=" * 60)
    print("B1-D1 흉벽/종격동 FP 원인 진단 - preflight")
    print("=" * 60)
    print(f"manifest 행수            : {len(rows)}")
    print(f"unique 환자              : {len(seen)}")
    print(f"stage2_holdout 접근      : {holdout_access}  (반드시 0)")
    print(f"refined ROI mask 매핑 OK : {n_ok}/{len(map_report)}")
    if n_fail:
        print(f"매핑 실패 환자({len(fail_pids)}) : {fail_pids}")
    print(f"refined ROI root         : {REFINED_ROI_ROOT}")
    print(f"출력 폴더(미생성)        : {OUT_DIR}")

    if args.dry_run:
        verdict, fails = validate_dry_run(rows, seen, holdout_access)
        print("\n[dry-run] read-only 점검 완료. 파일 생성 없음.")
        print("[dry-run] 실제 진단 CSV 생성은 --real --confirm-write 와 사용자 승인 필요.")
        return

    # ----- real 모드 -----
    if not (ALLOW_REAL_PROCESSING and args.real and args.confirm_write):
        _fail("real 모드 차단: ALLOW_REAL_PROCESSING=True 이고 --real --confirm-write 가 모두 필요. (현재 미충족)")

    if OUT_DIR.exists():
        _fail(f"출력 폴더가 이미 존재함(덮어쓰기 금지): {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=False)  # collision guard

    import numpy as np
    mask_cache = {}
    out_rows = []
    for r in rows:
        pid = r[COL_PID]
        path, status = seen[pid]
        if status != "ok":
            out_rows.append({**{k: r.get(k) for k in (COL_RID, COL_PID, COL_ROLE, COL_LABEL,
                                                       COL_Z, COL_Y0, COL_X0, COL_SCORE, COL_ROI0)},
                             "refined_roi_ratio": "", "center_in_refined_roi": "",
                             "cause_class": f"MASK_{status}"})
            continue
        if pid not in mask_cache:
            mask_cache[pid] = np.load(path, mmap_mode="r")
        m = mask_cache[pid]
        z = int(float(r[COL_Z])); y0 = int(float(r[COL_Y0])); x0 = int(float(r[COL_X0]))
        patch = np.asarray(m[z, y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE])
        refined_ratio = float((patch > 0).mean()) if patch.size else 0.0
        cy, cx = y0 + PATCH_SIZE // 2, x0 + PATCH_SIZE // 2
        center_in = bool(m[z, cy, cx] > 0)
        cls = classify(r[COL_ROLE], r[COL_LABEL], refined_ratio, center_in)
        out_rows.append({**{k: r.get(k) for k in (COL_RID, COL_PID, COL_ROLE, COL_LABEL,
                                                   COL_Z, COL_Y0, COL_X0, COL_SCORE, COL_ROI0)},
                         "refined_roi_ratio": round(refined_ratio, 4),
                         "center_in_refined_roi": center_in,
                         "cause_class": cls})

    # 진단 CSV
    csv_path = OUT_DIR / "b1d1_fp_cause_diagnostic.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)

    # 원인 분류 요약
    from collections import Counter
    summary = {
        "manifest": str(BRIDGE_MANIFEST),
        "refined_roi_root": str(REFINED_ROI_ROOT),
        "n_rows": len(out_rows),
        "stage2_holdout_access": holdout_access,
        "cause_class_counts": dict(Counter(o["cause_class"] for o in out_rows)),
        "fp_candidate_counts": dict(Counter(o["cause_class"] for o in out_rows if o[COL_ROLE] == "fp_candidate")),
        "lesion_protect_counts": dict(Counter(o["cause_class"] for o in out_rows if o[COL_ROLE] == "lesion_protect")),
    }
    with open(OUT_DIR / "b1d1_cause_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[real] 진단 CSV : {csv_path}")
    print(f"[real] 요약 JSON: {OUT_DIR / 'b1d1_cause_summary.json'}")
    print(f"[real] 원인 분류: {summary['fp_candidate_counts']}")

    if args.make_overlay:
        _fail("overlay PNG 생성은 별도 승인 대상이다. 본 스크립트에서는 미구현(차단).")


if __name__ == "__main__":
    main()
