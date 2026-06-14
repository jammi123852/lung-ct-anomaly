#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
b1d3j_patchcore_anatomy_conditioned_feature_smoke

Gate-P2 anatomy-conditioned feature smoke 스크립트.
B1-D3g 대비 핵심 변경: memory 패치를 roi_0_0_patch_ratio 기반으로 필터링하여
boundary anatomy(흉벽/종격동 경계)를 명시적으로 타겟팅.

memory condition 3종:
  C1 (anatomy_boundary):    pos_bin + cdr±0.10 + roi_0_0_patch_ratio < 0.85
  C2 (anatomy_near_inside): pos_bin + cdr±0.10 + 0.85 <= roi_0_0_patch_ratio
  C3 (mixed_boundary_inside): C1 50% + C2 50% (per-bin)

ratio_source: normal score CSV 컬럼 roi_0_0_patch_ratio
             (original roi_0_0 mask 기준 32x32 패치 내 비율 — NOT v4 refined mask)

★ ALLOW_REAL_PROCESSING = False (기본 차단). bare-run 즉시 exit 2.
★ --dry-run: 입력/경로/shape/ratio/coverage 검증만. feature/torch 0.
★ --run --confirm-feature-smoke: B1-D3j1 별도 승인 후에만 feature 추출.
★ --device cuda 는 --confirm-gpu 없으면 차단. 이 plan 은 cpu 만.
★ score/threshold/ROI 무수정. output exist_ok=False (collision guard).
★ stage2_holdout 접근 금지.
"""
import argparse
import csv
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

ALLOW_REAL_PROCESSING = False  # ★ 기본 차단. importlib runtime override 로만 real 허용.

BASE = Path("/home/jinhy/project/lung-ct-anomaly")
DIR = BASE / "outputs/b1d1_wall_mediastinum_fp_cause_diagnostic_v1"
NSCORE = BASE / "outputs/position-aware-padim-v1/scores/padim_v2_roi0_0/normal_by_patient"
MROOT = BASE / "outputs/mip-postprocess-research-v1/masks/refined_roi_v4_20_modeB_all_v1"
NROOT = Path("/mnt/c/Users/jinhy/Desktop/Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1/volumes_npy")
SEL_IDX_NPZ = BASE / "outputs/position-aware-padim-v1/models/padim_v2_roi0_0/distributions/position_bin_stats.npz"

# stage2_holdout 경로 패턴 — 접근 시 fail
STAGE2_HOLDOUT_PATTERNS = ["stage2", "holdout", "test_patient", "lesion_score"]

B3F_CAND = DIR / "b1d3f_gate_p2_position_conditioned_candidates_summary.csv"
B3I_SUMMARY = DIR / "b1d3i_patchcore_anatomy_conditioned_memory_preflight_summary.json"

PATCH = 32
RAW_FEATURE_DIM = 448
REDUCED_DIM = 100
CDR_TOL = 0.10
RATIO_BOUNDARY_THR = 0.85  # C1: < 0.85 = boundary, C2: >= 0.85 = near_inside

CONDITION_NAMES = {
    "c1": "anatomy_boundary",
    "c2": "anatomy_near_inside",
    "c3": "mixed_boundary_inside",
}

RECOMMENDED_NORMAL_PATIENTS = [
    "normal004", "normal013", "normal014", "normal016", "normal017"
]


def fail(msg, code=2):
    print(f"[b1d3j][중단] {msg}", file=sys.stderr)
    sys.exit(code)


def guard_stage2(path_str):
    """stage2/holdout 경로 접근 차단."""
    for pat in STAGE2_HOLDOUT_PATTERNS:
        if pat in str(path_str).lower():
            fail(f"stage2_holdout 접근 차단: {path_str}")


def load_rows(p, enc="utf-8"):
    with open(p, encoding=enc) as f:
        return list(csv.DictReader(f))


def resolve_from_safe_id(safe_id):
    """normal safe_id → (mask_path, ct_path). stage2 차단."""
    guard_stage2(safe_id)
    return (MROOT / "normal" / safe_id / "refined_roi.npy",
            NROOT / safe_id / "ct_hu.npy")


def resolve_candidate_dirs(patient_id, gcid, review_id):
    """normal candidate 의 mask/ct npy 매핑. lesion root 매칭 시 fail."""
    guard_stage2(patient_id)
    hits = sorted((MROOT / "normal").glob(f"{patient_id}__*"))
    if len(hits) != 1:
        fail(f"candidate mask dir 매칭 {len(hits)} != 1: gc={gcid} review={review_id} "
             f"patient={patient_id}")
    lhits = sorted((MROOT / "lesion").glob(f"{patient_id}__*"))
    if lhits:
        fail(f"candidate 가 lesion root 에 매칭됨: gc={gcid} review={review_id} "
             f"patient={patient_id} lesion_hits={len(lhits)}")
    voldir = hits[0].name
    return hits[0] / "refined_roi.npy", NROOT / voldir / "ct_hu.npy", voldir


def check_ratio_column(rows, patient_id):
    """score CSV에 roi_0_0_patch_ratio 컬럼 존재 확인."""
    if not rows:
        fail(f"score CSV 빈 파일: {patient_id}")
    if "roi_0_0_patch_ratio" not in rows[0]:
        fail(f"roi_0_0_patch_ratio 컬럼 없음: {patient_id} (ratio_source 불명확)")
    return True


def count_by_condition(rows, pb, cdr_target, cdr_tol, condition):
    """조건별 패치 수 계산 (feature 없음)."""
    filtered = []
    for r in rows:
        if r.get("position_bin") != pb:
            continue
        try:
            cdr = float(r.get("central_distance_ratio_mean", "nan"))
            ratio = float(r.get("roi_0_0_patch_ratio", "nan"))
        except ValueError:
            continue
        if abs(cdr - cdr_target) > cdr_tol:
            continue
        if condition == "c1" and ratio < RATIO_BOUNDARY_THR:
            filtered.append(r)
        elif condition == "c2" and ratio >= RATIO_BOUNDARY_THR:
            filtered.append(r)
        elif condition == "c3":  # 전부 포함 (혼합은 sampling 시 50/50)
            filtered.append(r)
    return len(filtered)


def validate(args):
    """read-only 검증. feature/torch 없음. dry/real 공통."""

    # B1-D3i summary PASS 확인
    if not B3I_SUMMARY.exists():
        fail(f"B1-D3i summary 없음: {B3I_SUMMARY}")
    with open(B3I_SUMMARY, encoding="utf-8") as f:
        b3i = json.load(f)
    if b3i.get("verdict") != "PASS":
        fail(f"B1-D3i verdict != PASS: {b3i.get('verdict')}")
    if b3i.get("stage2_holdout_access", 1) != 0:
        fail("B1-D3i stage2_holdout_access != 0")
    if b3i.get("gate_candidate_count") != 6:
        fail(f"B1-D3i gate_candidate_count != 6: {b3i.get('gate_candidate_count')}")

    # candidates 로드
    if not B3F_CAND.exists():
        fail(f"B1-D3f candidates 없음: {B3F_CAND}")
    cands_raw = load_rows(B3F_CAND)
    if len(cands_raw) != 6:
        fail(f"candidates {len(cands_raw)} != 6")
    if len(cands_raw) > args.candidate_limit:
        fail(f"candidate {len(cands_raw)} > limit {args.candidate_limit}")

    cand_pos = []
    for c in cands_raw:
        cand_pos.append({
            "gate_candidate_id": c["gate_candidate_id"],
            "review_id": c["review_id"],
            "patient_id": c["patient_id"],
            "z": int(c["candidate_local_z"]),
            "y0": int(c["candidate_y0"]),
            "x0": int(c["candidate_x0"]),
            "position_bin": c["position_bin"],
            "z_level": c["z_level_bin"],
            "central_peripheral": c["central_peripheral"],
            "cdr": float(c["central_distance_ratio_mean"]),
            "candidate_score": c["candidate_score"],
            "roi_0_0_patch_ratio": float(c["roi_0_0_patch_ratio"]),
        })

    cand_patients = set(c["patient_id"] for c in cand_pos)
    need_bins = sorted(set(c["position_bin"] for c in cand_pos))

    # candidate CT/mask 매핑/shape/range 검증
    cand_path_status = {}
    for c in cand_pos:
        guard_stage2(c["patient_id"])
        md, cd, voldir = resolve_candidate_dirs(
            c["patient_id"], c["gate_candidate_id"], c["review_id"])
        cstat = "ok" if cd.exists() else "missing"
        mstat = "ok" if md.exists() else "missing"
        shape_ok = range_ok = False
        shp = None
        if cstat == "ok" and mstat == "ok":
            m = np.load(md, mmap_mode="r")
            cc = np.load(cd, mmap_mode="r")
            shp = m.shape
            shape_ok = (m.shape == cc.shape and len(m.shape) == 3
                        and m.shape[1:] == (512, 512))
            z, y0, x0 = c["z"], c["y0"], c["x0"]
            range_ok = (shape_ok and 0 <= z < m.shape[0]
                        and 0 <= y0 <= 512 - PATCH and 0 <= x0 <= 512 - PATCH)
            del m, cc
        if cstat != "ok" or mstat != "ok":
            fail(f"candidate CT/mask 없음: gc={c['gate_candidate_id']} "
                 f"review={c['review_id']} ct={cstat} mask={mstat}")
        if not shape_ok:
            fail(f"candidate shape 이상: gc={c['gate_candidate_id']} shape={shp}")
        if not range_ok:
            fail(f"candidate 좌표 범위 이상: gc={c['gate_candidate_id']} "
                 f"z{c['z']} y{c['y0']} x{c['x0']} shape={shp}")
        cand_path_status[c["gate_candidate_id"]] = {
            "voldir": voldir, "ct": cstat, "mask": mstat,
            "shape": str(shp), "shape_ok": shape_ok, "range_ok": range_ok,
        }

    # memory patients: recommended 5명 우선 (candidate 환자 제외 확인)
    mem_patients_all = [p for p in NSCORE.glob("*.csv")]
    mem_patient_ids = sorted(
        p.stem for p in mem_patients_all
        if p.stem not in cand_patients
    )
    # recommended 환자 우선
    mem_patients_pref = [p for p in RECOMMENDED_NORMAL_PATIENTS
                         if p not in cand_patients]
    extra = [p for p in mem_patient_ids if p not in mem_patients_pref]
    mem_patients_ordered = mem_patients_pref + extra
    mem_patients = mem_patients_ordered[:args.memory_patient_limit]

    if len(mem_patients) < 5:
        fail(f"memory 환자 {len(mem_patients)} < 5")

    # bin별 cdr 대표값 (candidate 평균)
    bin_cdr = defaultdict(list)
    for c in cand_pos:
        bin_cdr[c["position_bin"]].append(c["cdr"])
    bin_cdr = {b: float(np.mean(x)) for b, x in bin_cdr.items()}

    per_bin_per_patient = max(1, args.per_patient_patch_cap // max(len(need_bins), 1))

    # 환자별 score CSV 검증 + ratio_source + coverage count
    path_status = {}
    coverage = {cond: defaultdict(lambda: defaultdict(int))
                for cond in ["c1", "c2", "c3"]}
    ratio_source_ok = True
    ratio_source_note = "ok: roi_0_0_patch_ratio from normal score CSV (original roi_0_0 mask)"

    for pid in mem_patients:
        guard_stage2(pid)
        f = NSCORE / f"{pid}.csv"
        if not f.exists():
            fail(f"memory score CSV 없음: {pid}")
        rows = load_rows(f, enc="utf-8-sig")
        check_ratio_column(rows, pid)

        safe_id = rows[0].get("safe_id", "")
        guard_stage2(safe_id)
        if safe_id == "" or not (MROOT / "normal" / safe_id).is_dir():
            fail(f"normal mask dir 없음(lesion/holdout 의심): {pid} safe_id={safe_id}")

        md, cd = resolve_from_safe_id(safe_id)
        cstat = "ok" if cd.exists() else "missing"
        mstat = "ok" if md.exists() else "missing"
        sstat = "unchecked"
        if cstat == "ok" and mstat == "ok":
            m = np.load(md, mmap_mode="r")
            c = np.load(cd, mmap_mode="r")
            sstat = (f"ok {m.shape}" if (m.shape == c.shape and m.shape[1:] == (512, 512))
                     else f"mismatch {m.shape} vs {c.shape}")
            del m, c
        path_status[pid] = {"safe_id": safe_id, "ct": cstat, "mask": mstat, "shape": sstat}
        if cstat != "ok" or mstat != "ok":
            fail(f"memory 환자 CT/mask 없음: {pid} ct={cstat} mask={mstat}")

        for pb in need_bins:
            cdr_target = bin_cdr.get(pb, 0.0)
            for cond in ["c1", "c2", "c3"]:
                cnt = count_by_condition(rows, pb, cdr_target, CDR_TOL, cond)
                coverage[cond][pid][pb] = cnt

    # coverage gap 검사
    coverage_status = {}
    for cond in ["c1", "c2", "c3"]:
        gaps = []
        complete = True
        for pid in mem_patients:
            for pb in need_bins:
                cnt = coverage[cond][pid][pb]
                if cnt < per_bin_per_patient:
                    complete = False
                    gaps.append(f"{pid}/{pb}/{cond}={cnt}<{per_bin_per_patient}")
        coverage_status[cond] = {
            "complete": complete, "gaps": gaps,
            "total": sum(coverage[cond][pid][pb]
                        for pid in mem_patients for pb in need_bins),
        }

    # selected_feature_indices 존재 확인 (forward 없음)
    sel_ok = SEL_IDX_NPZ.exists()
    sel_shape_ok = False
    if sel_ok:
        try:
            sel = np.load(SEL_IDX_NPZ, allow_pickle=True)["selected_feature_indices"].astype(int)
            sel_shape_ok = (sel.shape[0] == REDUCED_DIM
                            and sel.min() >= 0 and sel.max() < RAW_FEATURE_DIM)
        except Exception:
            pass

    return {
        "cand_pos": cand_pos,
        "need_bins": need_bins,
        "bin_cdr": bin_cdr,
        "mem_patients": mem_patients,
        "per_bin_per_patient": per_bin_per_patient,
        "path_status": path_status,
        "cand_path_status": cand_path_status,
        "candidate_ct_mask_ok": sum(
            1 for s in cand_path_status.values()
            if s["ct"] == "ok" and s["mask"] == "ok"),
        "candidate_shape_ok": sum(
            1 for s in cand_path_status.values() if s["shape_ok"]),
        "candidate_range_ok": sum(
            1 for s in cand_path_status.values() if s["range_ok"]),
        "memory_ct_mask_ok": sum(
            1 for p in mem_patients
            if path_status[p]["ct"] == "ok" and path_status[p]["mask"] == "ok"),
        "coverage": coverage_status,
        "ratio_source_ok": ratio_source_ok,
        "ratio_source_note": ratio_source_note,
        "sel_ok": sel_ok,
        "sel_shape_ok": sel_shape_ok,
        "device": args.device,
        "per_patient_cap": args.per_patient_patch_cap,
        "total_cap": args.memory_patch_cap,
        "conditions": args.conditions.split(","),
    }


def run_real(v, args):
    """
    ★ B1-D3j1 별도 승인 후에만 실행 가능.
    ALLOW_REAL_PROCESSING=False 이면 여기 도달 불가.
    """
    import time
    t0 = time.time()

    if v["device"] != "cpu" and not args.confirm_gpu:
        fail("GPU 사용은 --confirm-gpu 필요 (이 plan 은 cpu 권장)")

    conds = v["conditions"]
    valid_conds = {"c1", "c2", "c3"}
    for cond in conds:
        if cond not in valid_conds:
            fail(f"알 수 없는 condition: {cond}")

    # --- output directory (collision guard) ---
    out_name = "b1d3j1_anatomy_conditioned_feature_smoke_" + "_".join(conds) + "_v1"
    OUT_DIR = DIR / out_name
    OUT_DIR_TMP = DIR / (out_name + "_tmp")
    if OUT_DIR.exists():
        fail(f"output 폴더 이미 존재 (덮어쓰기 금지): {OUT_DIR}")
    if OUT_DIR_TMP.exists():
        fail(f"tmp 폴더 이미 존재 (이전 실패 잔재 확인 필요): {OUT_DIR_TMP}")
    OUT_DIR_TMP.mkdir(parents=True, exist_ok=False)

    import torch  # noqa: F401
    sys.path.insert(0, str(BASE / "src"))
    from position_aware_padim.feature_extractor import FeatureExtractor
    from position_aware_padim.preprocessing import preprocess_ct_slice

    sel = np.load(SEL_IDX_NPZ, allow_pickle=True)["selected_feature_indices"].astype(int)
    if sel.shape[0] != REDUCED_DIM or sel.min() < 0 or sel.max() >= RAW_FEATURE_DIM:
        fail(f"selected_feature_indices 비정상 shape={sel.shape}")

    fe = FeatureExtractor(device=v["device"])
    if v["device"] == "cpu" and str(fe.device) != "cpu":
        fail(f"device cpu 아님: {fe.device}")

    feat_nan = feat_inf = 0
    total_cap = v["total_cap"]
    per_bin_pp = v["per_bin_per_patient"]

    rng = np.random.RandomState(42)

    # --- per-condition memory bank ---
    mem_by_cond_bin = {cond: defaultdict(list) for cond in conds}
    mem_meta_by_id = {}  # mpid → {normal_patient_id, roi_ratio, refined_roi_ratio_v4, position_bin, condition_name}
    mem_rows = []
    midx = 1
    n_mem_by_cond = {cond: 0 for cond in conds}

    for cond in conds:
        n_mem = 0
        for pid in v["mem_patients"]:
            if n_mem >= total_cap:
                break
            guard_stage2(pid)
            safe_id = v["path_status"][pid]["safe_id"]
            md, cd = resolve_from_safe_id(safe_id)
            rows = load_rows(NSCORE / f"{pid}.csv", enc="utf-8-sig")
            ct = np.load(cd, mmap_mode="r")
            mask = np.load(md, mmap_mode="r")

            by_bin = defaultdict(list)
            for r in rows:
                pb = r.get("position_bin", "")
                if pb not in v["need_bins"]:
                    continue
                try:
                    cdr_val = float(r.get("central_distance_ratio_mean", "nan"))
                    ratio = float(r.get("roi_0_0_patch_ratio", "nan"))
                except ValueError:
                    continue
                cdr_target = v["bin_cdr"].get(pb, 0.0)
                if abs(cdr_val - cdr_target) > CDR_TOL:
                    continue
                # condition 필터
                if cond == "c1" and ratio >= RATIO_BOUNDARY_THR:
                    continue
                if cond == "c2" and ratio < RATIO_BOUNDARY_THR:
                    continue
                # c3: 전부 포함
                by_bin[pb].append(r)

            for pb in v["need_bins"]:
                cands_b = by_bin.get(pb, [])
                if not cands_b:
                    continue
                if n_mem >= total_cap:
                    break
                cdr_target = v["bin_cdr"].get(pb, 0.0)

                if cond == "c3":
                    # 50% boundary + 50% near_inside
                    boundary = [r for r in cands_b
                                 if float(r["roi_0_0_patch_ratio"]) < RATIO_BOUNDARY_THR]
                    inside = [r for r in cands_b
                              if float(r["roi_0_0_patch_ratio"]) >= RATIO_BOUNDARY_THR]
                    half = max(1, per_bin_pp // 2)
                    rng.shuffle(boundary)
                    rng.shuffle(inside)
                    chosen = boundary[:half] + inside[:half]
                else:
                    rng.shuffle(cands_b)
                    chosen = cands_b[:per_bin_pp]

                byz = defaultdict(list)
                for r in chosen:
                    byz[int(r["local_z"])].append(r)
                for z, rs in byz.items():
                    if n_mem >= total_cap:
                        break
                    sl = preprocess_ct_slice(np.asarray(ct[z]).astype(np.float32))
                    mz = np.asarray(mask[z])
                    coords = [(int(r["y0"]), int(r["x0"]),
                               int(r["y0"]) + PATCH, int(r["x0"]) + PATCH) for r in rs]
                    feats = fe.extract_patch_features(sl, coords)[:, sel]
                    for r, fr in zip(rs, feats):
                        if n_mem >= total_cap:
                            break
                        nanc = int(np.isnan(fr).sum())
                        infc = int(np.isinf(fr).sum())
                        feat_nan += nanc
                        feat_inf += infc
                        if nanc or infc:
                            fail(f"memory feature NaN/Inf: {pid} {pb} cond={cond}")
                        if fr.shape[0] != REDUCED_DIM:
                            fail(f"feature_dim {fr.shape[0]} != {REDUCED_DIM}")
                        y0, x0 = int(r["y0"]), int(r["x0"])
                        v4ratio = float((mz[y0:y0 + PATCH, x0:x0 + PATCH] > 0).mean())
                        src_ratio = float(r["roi_0_0_patch_ratio"])
                        mpid = f"ACMEM{cond.upper()}{midx:05d}"
                        mem_by_cond_bin[cond][pb].append((fr, mpid, pid))
                        mem_meta_by_id[mpid] = {
                            "normal_patient_id": pid,
                            "roi_ratio": round(src_ratio, 4),
                            "refined_roi_ratio_v4": round(v4ratio, 4),
                            "position_bin": pb,
                            "condition_name": CONDITION_NAMES[cond],
                        }
                        mem_rows.append({
                            "memory_row_id": mpid,
                            "normal_patient_id": pid,
                            "position_bin": pb,
                            "condition_name": CONDITION_NAMES[cond],
                            "cdr": round(float(r.get("central_distance_ratio_mean", "nan")), 4),
                            "roi_ratio": round(src_ratio, 4),
                            "ratio_source": "score_csv_roi_0_0_patch_ratio",
                            "local_z": int(r["local_z"]),
                            "y0": y0,
                            "x0": x0,
                            "feature_dim": REDUCED_DIM,
                            "refined_roi_ratio_v4": round(v4ratio, 4),
                            "feature_status": "ok" if (nanc == 0 and infc == 0) else "nan_inf",
                        })
                        midx += 1
                        n_mem += 1
            del ct, mask
        n_mem_by_cond[cond] = n_mem
        if n_mem == 0:
            fail(f"condition {cond}: memory feature 0")

    # --- per-condition per-bin NN ---
    cand_rows = []
    for c in v["cand_pos"]:
        pb = c["position_bin"]
        md_c, cd_c, voldir_c = resolve_candidate_dirs(
            c["patient_id"], c["gate_candidate_id"], c["review_id"])
        ct = np.load(cd_c, mmap_mode="r")
        sl = preprocess_ct_slice(np.asarray(ct[c["z"]]).astype(np.float32))
        feat = fe.extract_patch_features(
            sl, [(c["y0"], c["x0"], c["y0"] + PATCH, c["x0"] + PATCH)])[0][sel]
        del ct
        nanc = int(np.isnan(feat).sum())
        infc = int(np.isinf(feat).sum())
        feat_nan += nanc
        feat_inf += infc
        if nanc or infc:
            fail(f"candidate feature NaN/Inf: {c['review_id']}")
        if feat.shape[0] != REDUCED_DIM:
            fail(f"candidate feature_dim {feat.shape[0]} != {REDUCED_DIM}")

        for cond in conds:
            items = mem_by_cond_bin[cond].get(pb, [])
            if not items:
                fail(f"candidate {c['review_id']} bin {pb} cond={cond} memory 0")
            mat = np.asarray([it[0] for it in items], dtype=np.float32)
            d = np.linalg.norm(mat - feat[None, :], axis=1)
            j = int(np.argmin(d))
            dist = float(d[j])
            if not np.isfinite(dist):
                fail(f"distance NaN/Inf: {c['review_id']} cond={cond}")
            # self-NN 분포
            self_nn = np.empty(mat.shape[0])
            for i in range(mat.shape[0]):
                d2 = np.linalg.norm(mat - mat[i][None, :], axis=1)
                d2[i] = np.inf
                self_nn[i] = d2.min()
            pct = round(float((self_nn < dist).mean() * 100), 1)
            p50 = float(np.percentile(self_nn, 50))
            p90 = float(np.percentile(self_nn, 90))
            flag = "suspicious" if dist > p90 else ("borderline" if dist > p50 else "normal")
            feat_arr, nearest_mpid, nearest_pid = items[j]
            meta = mem_meta_by_id.get(nearest_mpid, {})
            nearest_ratio = meta.get("roi_ratio", -1.0)
            nearest_refined_ratio = meta.get("refined_roi_ratio_v4", -1.0)

            cand_rows.append({
                "gate_candidate_id": c["gate_candidate_id"],
                "review_id": c["review_id"],
                "patient_id": c["patient_id"],
                "position_bin": pb,
                "condition_name": CONDITION_NAMES[cond],
                "candidate_score": c["candidate_score"],
                "nearest_dist": round(dist, 4),
                "within_bin_percentile": pct,
                "flag": flag,
                "nearest_memory_patient": nearest_pid,
                "nearest_memory_mpid": nearest_mpid,
                "nearest_memory_ratio": nearest_ratio,
                "nearest_memory_refined_roi_ratio_v4": nearest_refined_ratio,
                "ratio_source": "score_csv_roi_0_0_patch_ratio",
                "memory_condition": cond,
                "memory_pool_size": len(items),
                "mem_p50": round(p50, 4),
                "mem_p90": round(p90, 4),
                "interpretation_note": (
                    f"dist={dist:.3f} > mem_p90={p90:.3f} → suspicious"
                    if flag == "suspicious"
                    else f"dist={dist:.3f} vs mem_p50={p50:.3f}/p90={p90:.3f}"
                ),
            })

    elapsed = time.time() - t0

    # --- 출력 저장 ---
    mem_fields = [
        "memory_row_id", "normal_patient_id", "position_bin", "condition_name",
        "cdr", "roi_ratio", "ratio_source", "local_z", "y0", "x0",
        "feature_dim", "refined_roi_ratio_v4", "feature_status",
    ]
    cand_fields = [
        "gate_candidate_id", "review_id", "patient_id", "position_bin",
        "condition_name", "candidate_score", "nearest_dist", "within_bin_percentile",
        "flag", "nearest_memory_patient", "nearest_memory_mpid",
        "nearest_memory_ratio", "nearest_memory_refined_roi_ratio_v4",
        "ratio_source", "memory_condition", "memory_pool_size",
        "mem_p50", "mem_p90", "interpretation_note",
    ]

    mem_csv = OUT_DIR_TMP / "b1d3j1_anatomy_conditioned_memory_feature_preview.csv"
    cand_csv = OUT_DIR_TMP / "b1d3j1_anatomy_conditioned_candidate_distance_preview.csv"
    summary_json = OUT_DIR_TMP / "b1d3j1_anatomy_conditioned_feature_smoke_summary.json"
    report_md = OUT_DIR_TMP / "b1d3j1_anatomy_conditioned_feature_smoke_report.md"

    with open(mem_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=mem_fields)
        w.writeheader()
        w.writerows(mem_rows)

    with open(cand_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cand_fields)
        w.writeheader()
        w.writerows(cand_rows)

    summary = {
        "step": "B1-D3j1_anatomy_conditioned_feature_smoke",
        "conditions": conds,
        "candidate_count": len(v["cand_pos"]),
        "memory_rows_total": len(mem_rows),
        "n_mem_by_cond": n_mem_by_cond,
        "feat_nan": feat_nan,
        "feat_inf": feat_inf,
        "stage2_holdout_access": 0,
        "score_modified": False,
        "threshold_recomputed": False,
        "adjusted_score_created": False,
        "suppression_weight_created": False,
        "refined_score_created": False,
        "ratio_source": "score_csv_roi_0_0_patch_ratio",
        "feature_dim": REDUCED_DIM,
        "device": v["device"],
        "elapsed_sec": round(elapsed, 1),
        "verdict": "completed",
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines = [f"# B1-D3j1 anatomy-conditioned feature smoke report",
             f"conditions: {conds}",
             f"memory rows: {len(mem_rows)} | candidates: {len(v['cand_pos'])}",
             f"feat_nan: {feat_nan} | feat_inf: {feat_inf}",
             f"elapsed: {elapsed:.1f}s",
             ""]
    suspicious_count = sum(1 for r in cand_rows if r["flag"] == "suspicious")
    lines.append(f"suspicious candidates: {suspicious_count}/{len(cand_rows)}")
    for r in cand_rows:
        lines.append(
            f"  {r['gate_candidate_id']} ({r['condition_name']}): "
            f"dist={r['nearest_dist']} pct={r['within_bin_percentile']}% flag={r['flag']}")
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    OUT_DIR_TMP.rename(OUT_DIR)
    print(f"[b1d3j1] 완료. elapsed={elapsed:.1f}s "
          f"mem={len(mem_rows)} cand={len(cand_rows)}")
    return summary


def dry_run(v, args):
    """read-only preflight 요약 출력. feature 없음."""
    print("[b1d3j][dry-run] 검증 결과:")
    print(f"  B1-D3i verdict: PASS")
    print(f"  gate_candidate_count: {len(v['cand_pos'])}")
    print(f"  need_bins: {v['need_bins']}")
    print(f"  memory_patients: {v['mem_patients']}")
    print(f"  candidate_ct_mask_ok: {v['candidate_ct_mask_ok']}/6")
    print(f"  memory_ct_mask_ok: {v['memory_ct_mask_ok']}/{len(v['mem_patients'])}")
    print(f"  ratio_source_ok: {v['ratio_source_ok']}")
    print(f"  ratio_source: {v['ratio_source_note']}")
    print(f"  sel_ok: {v['sel_ok']}, sel_shape_ok: {v['sel_shape_ok']}")
    print(f"  per_bin_per_patient: {v['per_bin_per_patient']}")
    print(f"  per_patient_cap: {v['per_patient_cap']}, total_cap: {v['total_cap']}")
    print(f"  device: {v['device']}")
    print(f"  conditions: {v['conditions']}")
    for cond in ["c1", "c2", "c3"]:
        cs = v["coverage"][cond]
        note = ""
        if cond == "c3":
            note = " [C3 real: boundary/inside split 별도 확인 필요 — 이번 Plan-A 실행 범위 아님]"
        print(f"  coverage[{cond}]: complete={cs['complete']} "
              f"total={cs['total']} gaps={cs['gaps'][:3]}{note}")
    print("[b1d3j][dry-run] PASS — feature extraction 없음.")


def main():
    if not ALLOW_REAL_PROCESSING and len(sys.argv) == 1:
        print("[b1d3j][차단] bare-run 금지: ALLOW_REAL_PROCESSING=False. "
              "사용: --dry-run 또는 B1-D3j1 승인 후 --run --confirm-feature-smoke",
              file=sys.stderr)
        sys.exit(2)

    p = argparse.ArgumentParser(description="b1d3j anatomy-conditioned feature smoke")
    p.add_argument("--dry-run", action="store_true",
                   help="read-only preflight (feature 없음)")
    p.add_argument("--run", action="store_true",
                   help="실제 feature smoke (ALLOW_REAL_PROCESSING=True + --confirm 필요)")
    p.add_argument("--confirm-feature-smoke", action="store_true",
                   help="B1-D3j1 실행 확인 플래그")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="feature extraction device (권장: cpu)")
    p.add_argument("--confirm-gpu", action="store_true",
                   help="GPU 사용 명시 승인")
    p.add_argument("--conditions", default="c1",
                   help="실행 조건 (c1,c2,c3 또는 조합, 예: c1,c3)")
    p.add_argument("--candidate-limit", type=int, default=6)
    p.add_argument("--memory-patient-limit", type=int, default=5)
    p.add_argument("--per-patient-patch-cap", type=int, default=100)
    p.add_argument("--memory-patch-cap", type=int, default=600)
    args = p.parse_args()

    if args.run and not ALLOW_REAL_PROCESSING and not args.confirm_feature_smoke:
        fail("--run 은 ALLOW_REAL_PROCESSING=True + --confirm-feature-smoke 필요 "
             "(B1-D3j1 승인 후에만 실행)")

    if args.run and not ALLOW_REAL_PROCESSING:
        fail("ALLOW_REAL_PROCESSING=False: --run 차단. B1-D3j1 승인 후 importlib override 필요")

    v = validate(args)

    if args.dry_run:
        dry_run(v, args)
        return

    if args.run:
        if not args.confirm_feature_smoke:
            fail("--run 은 --confirm-feature-smoke 필요")
        run_real(v, args)
        return

    print("[b1d3j] 옵션 없음: --dry-run 또는 --run --confirm-feature-smoke 사용",
          file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
