#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
b1d3g_gate_p2_position_conditioned_feature_smoke

Gate-P2 position-conditioned minimal feature smoke 실행 스크립트 (초안).
B1-D3d 대비 차이: normal memory 를 candidate 와 **동일 position_bin**(canonical: z_level×central/peripheral)
으로 위치 정합하여 per-position pool 을 만들고, candidate 를 같은 bin pool 과만 NN 비교한다.

★ 기본 차단(ALLOW_REAL_PROCESSING=False). bare-run 즉시 중단(exit 2).
★ --dry-run: 입력/범위/경로/shape/cap/position coverage 검증만. feature/torch/파일 0.
★ --real --confirm-feature-smoke: 별도 승인(B1-D3g1) 후에만 feature 추출.
★ --device cuda 는 --confirm-gpu 없으면 차단. 이 plan 은 cpu 만.
★ feature space = v2 selected 100D. preprocessing = v2 동일. score 무수정. output exist_ok=False.
"""
import argparse
import csv
import json
import sys
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np

ALLOW_REAL_PROCESSING = False  # ★ 기본 차단. 런타임 override 로만 real 허용.

BASE = Path("/home/jinhy/project/lung-ct-anomaly")
DIR = BASE / "outputs/b1d1_wall_mediastinum_fp_cause_diagnostic_v1"
NSCORE = BASE / "outputs/position-aware-padim-v1/scores/padim_v2_roi0_0/normal_by_patient"
MROOT = BASE / "outputs/mip-postprocess-research-v1/masks/refined_roi_v4_20_modeB_all_v1"
NROOT = Path("/mnt/c/Users/jinhy/Desktop/Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1/volumes_npy")
SEL_IDX_NPZ = BASE / "outputs/position-aware-padim-v1/models/padim_v2_roi0_0/distributions/position_bin_stats.npz"

B3F_CAND = DIR / "b1d3f_gate_p2_position_conditioned_candidates_summary.csv"
B3F_POOL = DIR / "b1d3f_gate_p2_position_conditioned_memory_pool_preview.csv"
OUT_DIR = DIR / "b1d3g1_gate_p2_position_conditioned_feature_smoke_plan_pc_s_cpu_v1"

PATCH = 32
RAW_FEATURE_DIM = 448
REDUCED_DIM = 100
CDR_TOL = 0.10


def fail(msg, code=2):
    print(f"[b1d3g][중단] {msg}", file=sys.stderr)
    sys.exit(code)


def load_rows(p, enc="utf-8"):
    with open(p, encoding=enc) as f:
        return list(csv.DictReader(f))


def resolve_from_safe_id(safe_id):
    return (MROOT / "normal" / safe_id / "refined_roi.npy", NROOT / safe_id / "ct_hu.npy")


def resolve_candidate_dirs(patient_id, gcid, review_id):
    """normal candidate 의 mask/ct npy 안전 해석. hits 정확히 1개 아니면 fail, lesion 매칭 차단."""
    hits = sorted((MROOT / "normal").glob(f"{patient_id}__*"))
    if len(hits) != 1:
        fail(f"candidate mask dir 매칭 {len(hits)} != 1: gc={gcid} review={review_id} "
             f"patient={patient_id} hit_count={len(hits)}")
    lhits = sorted((MROOT / "lesion").glob(f"{patient_id}__*"))
    if lhits:
        fail(f"candidate 가 lesion root 에 매칭됨(normal 아님): gc={gcid} review={review_id} "
             f"patient={patient_id} lesion_hits={len(lhits)}")
    voldir = hits[0].name
    return hits[0] / "refined_roi.npy", NROOT / voldir / "ct_hu.npy", voldir


def validate(args):
    """read-only 검증. feature 없음. dry/real 공통."""
    if not B3F_CAND.exists() or not B3F_POOL.exists():
        fail("B1-D3f candidates/pool CSV 없음")
    cands = load_rows(B3F_CAND)
    if len(cands) != 6:
        fail(f"candidates {len(cands)} != 6")
    if len(cands) > args.candidate_limit:
        fail(f"candidate {len(cands)} > limit {args.candidate_limit}")

    # candidate 위치 메타
    cand_pos = []
    for c in cands:
        cand_pos.append({"gate_candidate_id": c["gate_candidate_id"], "review_id": c["review_id"],
                         "patient_id": c["patient_id"], "z": int(c["candidate_local_z"]),
                         "y0": int(c["candidate_y0"]), "x0": int(c["candidate_x0"]),
                         "position_bin": c["position_bin"], "z_level": c["z_level_bin"],
                         "central_peripheral": c["central_peripheral"],
                         "cdr": float(c["central_distance_ratio_mean"]),
                         "candidate_score": c["candidate_score"]})
    cand_patients = set(c["patient_id"] for c in cand_pos)
    need_bins = sorted(set(c["position_bin"] for c in cand_pos))

    # candidate CT/mask 매핑/shape/range 검증 (dry/real 공통)
    cand_path_status = {}
    for c in cand_pos:
        md, cd, voldir = resolve_candidate_dirs(c["patient_id"], c["gate_candidate_id"], c["review_id"])
        cstat = "ok" if cd.exists() else "missing"
        mstat = "ok" if md.exists() else "missing"
        if cstat != "ok" or mstat != "ok":
            fail(f"candidate CT/mask npy 없음: gc={c['gate_candidate_id']} review={c['review_id']} "
                 f"ct={cstat} mask={mstat}")
        m = np.load(md, mmap_mode="r")
        cc = np.load(cd, mmap_mode="r")
        shape_ok = (m.shape == cc.shape and len(m.shape) == 3 and m.shape[1:] == (512, 512))
        z, y0, x0 = c["z"], c["y0"], c["x0"]
        range_ok = (shape_ok and 0 <= z < m.shape[0]
                    and 0 <= y0 <= 512 - PATCH and 0 <= x0 <= 512 - PATCH)
        shp = m.shape
        del m, cc
        if not shape_ok:
            fail(f"candidate shape mismatch: gc={c['gate_candidate_id']} review={c['review_id']} shape={shp}")
        if not range_ok:
            fail(f"candidate 좌표 범위 이상: gc={c['gate_candidate_id']} review={c['review_id']} "
                 f"z{z} y{y0} x{x0} shape={shp}")
        cand_path_status[c["gate_candidate_id"]] = {
            "voldir": voldir, "ct": cstat, "mask": mstat, "shape_ok": shape_ok, "range_ok": range_ok}

    # memory 환자: B1-D3f preview 환자 중 candidate 환자 제외, 앞 N
    pool = load_rows(B3F_POOL)
    preview_patients = []
    for r in pool:
        if r["preview_patient_id"] not in preview_patients:
            preview_patients.append(r["preview_patient_id"])
    mem_patients = [p for p in preview_patients if p not in cand_patients][:args.memory_patient_limit]
    if len(mem_patients) < 5:
        fail(f"memory 환자 {len(mem_patients)} < 5")
    if len(mem_patients) > args.memory_patient_limit:
        fail(f"memory 환자 {len(mem_patients)} > limit")

    per_bin_per_patient = max(1, args.per_patient_patch_cap // len(need_bins))

    # 환자별 score CSV → position coverage + 경로/shape
    coverage = defaultdict(lambda: defaultdict(int))  # patient -> bin -> count
    path_status = {}
    for pid in mem_patients:
        f = NSCORE / f"{pid}.csv"
        if not f.exists():
            fail(f"memory 환자 score CSV 없음: {pid}")
        rows = load_rows(f, enc="utf-8-sig")
        # lesion 차단: normal score 폴더만 사용(normal only) — 경로상 normal_by_patient
        safe_id = rows[0].get("safe_id", "")
        if safe_id == "" or not (MROOT / "normal" / safe_id).is_dir():
            fail(f"normal mask dir 없음(lesion/holdout 의심): {pid} safe_id={safe_id}")
        md, cd = resolve_from_safe_id(safe_id)
        cstat = "ok" if cd.exists() else "missing"
        mstat = "ok" if md.exists() else "missing"
        sstat = "unchecked"
        if cstat == "ok" and mstat == "ok":
            m = np.load(md, mmap_mode="r")
            c = np.load(cd, mmap_mode="r")
            sstat = f"ok {m.shape}" if (m.shape == c.shape and m.shape[1:] == (512, 512)) else "mismatch"
            del m, c
        path_status[pid] = {"safe_id": safe_id, "ct": cstat, "mask": mstat, "shape": sstat}
        for r in rows:
            pb = r["position_bin"]
            if pb in need_bins:
                coverage[pid][pb] += 1

    # position coverage complete: 모든 (환자, need_bin) 에 per_bin_per_patient 이상 후보
    coverage_complete = True
    coverage_gaps = []
    for pid in mem_patients:
        for pb in need_bins:
            if coverage[pid][pb] < per_bin_per_patient:
                coverage_complete = False
                coverage_gaps.append(f"{pid}/{pb}={coverage[pid][pb]}<{per_bin_per_patient}")

    memory_ct_mask_ok = sum(1 for p in mem_patients
                            if path_status[p]["ct"] == "ok" and path_status[p]["mask"] == "ok")

    return {"cand_pos": cand_pos, "need_bins": need_bins, "mem_patients": mem_patients,
            "per_bin_per_patient": per_bin_per_patient, "path_status": path_status,
            "cand_path_status": cand_path_status,
            "candidate_ct_mask_mapping": sum(1 for s in cand_path_status.values()
                                             if s["ct"] == "ok" and s["mask"] == "ok"),
            "candidate_shape_ok": sum(1 for s in cand_path_status.values() if s["shape_ok"]),
            "candidate_range_ok": sum(1 for s in cand_path_status.values() if s["range_ok"]),
            "memory_ct_mask_ok": memory_ct_mask_ok,
            "coverage": {p: dict(coverage[p]) for p in mem_patients},
            "coverage_complete": coverage_complete, "coverage_gaps": coverage_gaps,
            "device": args.device, "per_patient_cap": args.per_patient_patch_cap,
            "total_cap": args.memory_patch_cap}


def run_real(v):
    """★ 승인(B1-D3g1) 후에만. per-position 100D feature + per-bin NN + 3단계 flag. score 무수정."""
    import time
    t0 = time.time()
    if v["device"] != "cpu":
        fail("이 plan 은 device=cpu 만 허용(GPU 금지).")
    OUT_DIR.mkdir(parents=True, exist_ok=False)

    import torch  # noqa: F401 (real 분기에서만)
    sys.path.insert(0, str(BASE / "src"))
    from position_aware_padim.feature_extractor import FeatureExtractor
    from position_aware_padim.preprocessing import preprocess_ct_slice

    sel = np.load(SEL_IDX_NPZ, allow_pickle=True)["selected_feature_indices"].astype(int)
    if sel.shape[0] != REDUCED_DIM or sel.min() < 0 or sel.max() >= RAW_FEATURE_DIM:
        fail(f"selected_feature_indices 비정상 shape={sel.shape}")
    fe = FeatureExtractor(device="cpu")
    if str(fe.device) != "cpu":
        fail(f"device cpu 아님: {fe.device}")

    feat_nan = feat_inf = 0
    per_bin_pp = v["per_bin_per_patient"]
    total_cap = v["total_cap"]

    # --- per-position memory bank ---
    mem_by_bin = defaultdict(list)       # bin -> list of (feat, mem_patch_id, patient)
    mem_rows = []
    midx = 1
    n_mem = 0
    # bin 별 대표 cdr (candidate 평균)
    bin_cdr = defaultdict(list)
    for c in v["cand_pos"]:
        bin_cdr[c["position_bin"]].append(c["cdr"])
    bin_cdr = {b: float(np.mean(x)) for b, x in bin_cdr.items()}

    rng = np.random.RandomState(0)
    for pid in v["mem_patients"]:
        if n_mem >= total_cap:
            break
        safe_id = v["path_status"][pid]["safe_id"]
        md, cd = resolve_from_safe_id(safe_id)
        rows = load_rows(NSCORE / f"{pid}.csv", enc="utf-8-sig")
        ct = np.load(cd, mmap_mode="r")
        mask = np.load(md, mmap_mode="r")  # v4 refined ROI mask (실제 patch ratio 계산용)
        by_bin = defaultdict(list)
        for r in rows:
            if r["position_bin"] in v["need_bins"]:
                by_bin[r["position_bin"]].append(r)
        for pb in v["need_bins"]:
            cands_b = by_bin.get(pb, [])
            if not cands_b:
                continue
            target = bin_cdr.get(pb, 0.0)

            def keyf(r):
                try:
                    return abs(float(r.get("central_distance_ratio_mean", "nan")) - target)
                except Exception:
                    return 9.9
            chosen = sorted(cands_b, key=keyf)[:per_bin_pp]  # tight cdr 우선
            # 같은 slice 끼리 묶어 forward 1회
            byz = defaultdict(list)
            for r in chosen:
                byz[int(r["local_z"])].append(r)
            for z, rs in byz.items():
                if n_mem >= total_cap:
                    break
                sl = preprocess_ct_slice(np.asarray(ct[z]).astype(np.float32))
                mz = np.asarray(mask[z])  # v4 refined mask slice
                coords = [(int(r["y0"]), int(r["x0"]), int(r["y0"]) + PATCH, int(r["x0"]) + PATCH) for r in rs]
                feats = fe.extract_patch_features(sl, coords)[:, sel]
                for r, fr in zip(rs, feats):
                    if n_mem >= total_cap:
                        break
                    nanc, infc = int(np.isnan(fr).sum()), int(np.isinf(fr).sum())
                    feat_nan += nanc
                    feat_inf += infc
                    y0, x0 = int(r["y0"]), int(r["x0"])
                    v4ratio = float((mz[y0:y0 + PATCH, x0:x0 + PATCH] > 0).mean())  # 실제 v4 refined ROI patch ratio
                    mpid = f"PCMEM{midx:04d}"
                    mem_by_bin[pb].append((fr, mpid, pid))
                    mem_rows.append({
                        "memory_patch_id": mpid, "memory_patient_id": pid,
                        "matched_gate_candidate_id": ",".join(c["gate_candidate_id"] for c in v["cand_pos"] if c["position_bin"] == pb),
                        "position_bin": pb, "z_level": r["z_level"], "y0": y0, "x0": x0,
                        "refined_roi_ratio": round(v4ratio, 4),
                        "source_roi_0_0_patch_ratio": round(float(r["roi_0_0_patch_ratio"]), 4),
                        "feature_status": "ok" if (nanc == 0 and infc == 0) else "nan_inf",
                        "feature_dim": REDUCED_DIM, "used_in_memory": "true",
                        "sampling_reason": f"position_bin=={pb} tight_cdr~{target:.3f}; refined_roi_ratio=v4_mask계산",
                        "exclusion_reason": "",
                    })
                    midx += 1
                    n_mem += 1
        del ct, mask
    if n_mem == 0:
        fail("memory feature 0")

    # --- per-bin self-NN 분포(임계) ---
    bin_thr = {}
    for pb, items in mem_by_bin.items():
        mat = np.asarray([it[0] for it in items], dtype=np.float32)
        if mat.shape[0] < 2:
            bin_thr[pb] = (None, None, mat)
            continue
        self_nn = np.empty(mat.shape[0])
        for i in range(mat.shape[0]):
            d = np.linalg.norm(mat - mat[i][None, :], axis=1)
            d[i] = np.inf
            self_nn[i] = d.min()
        bin_thr[pb] = (float(np.percentile(self_nn, 50)), float(np.percentile(self_nn, 90)), self_nn)

    # --- candidate per-bin NN + flag ---
    cand_rows = []
    flag_counts = Counter()
    dist_nan = dist_inf = 0
    for c in v["cand_pos"]:
        pb = c["position_bin"]
        items = mem_by_bin.get(pb, [])
        if not items:
            fail(f"candidate {c['review_id']} position_bin {pb} memory 0")
        md_c, cd_c, voldir_c = resolve_candidate_dirs(c["patient_id"], c["gate_candidate_id"], c["review_id"])
        ct = np.load(cd_c, mmap_mode="r")
        sl = preprocess_ct_slice(np.asarray(ct[c["z"]]).astype(np.float32))
        feat = fe.extract_patch_features(sl, [(c["y0"], c["x0"], c["y0"] + PATCH, c["x0"] + PATCH)])[0][sel]
        del ct
        nanc, infc = int(np.isnan(feat).sum()), int(np.isinf(feat).sum())
        feat_nan += nanc
        feat_inf += infc
        if nanc or infc:
            fail(f"candidate feature NaN/Inf: {c['review_id']}")
        mat = np.asarray([it[0] for it in items], dtype=np.float32)
        d = np.linalg.norm(mat - feat[None, :], axis=1)
        j = int(np.argmin(d))
        dist = float(d[j])
        if not np.isfinite(dist):
            dist_inf += 1
            fail(f"distance NaN/Inf: {c['review_id']}")
        p50, p90, self_nn = bin_thr[pb]
        if p50 is None:
            flag, pct = "uncertain", "n/a(pool<2)"
        else:
            pct = round(float((self_nn < dist).mean() * 100), 1)
            flag = "normal_like" if dist <= p50 else ("suspicious" if dist > p90 else "uncertain")
            pct = f"{pct}%ile_in_{pb}(p50={p50:.2f}/p90={p90:.2f})"
        flag_counts[flag] += 1
        cand_rows.append({
            "gate_candidate_id": c["gate_candidate_id"], "review_id": c["review_id"],
            "patient_id": c["patient_id"], "position_bin": pb, "candidate_score": c["candidate_score"],
            "feature_status": "ok", "feature_dim": REDUCED_DIM, "nearest_distance": round(dist, 4),
            "nearest_memory_patient": items[j][2], "nearest_memory_patch_id": items[j][1],
            "matched_position_bin": pb, "distance_percentile_within_position_pool": pct,
            "gate_p2_flag": flag,
            "flag_reason": f"per-position NN in {pb}; <=p50->normal_like,>p90->suspicious (임시 smoke 기준)",
            "score_modified": "false", "safety_note": "preview only; PaDiM score 무수정; 성능지표 아님",
        })

    runtime = round(time.time() - t0, 1)
    try:
        import resource
        peak_mem = f"{round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1)} MB"
    except Exception:
        peak_mem = "n/a"

    with open(OUT_DIR / "b1d3g1_position_conditioned_memory_feature_preview.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(mem_rows[0].keys()))
        w.writeheader()
        w.writerows(mem_rows)
    with open(OUT_DIR / "b1d3g1_position_conditioned_candidate_distance_preview.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cand_rows[0].keys()))
        w.writeheader()
        w.writerows(cand_rows)

    summary = {
        "step": "B1-D3g1_Gate_P2_position_conditioned_minimal_feature_smoke_CPU_Plan_PC_S",
        "stage2_holdout_access": 0, "plan_used": "Plan-PC-S",
        "device_used": "cpu", "gpu_used": False, "cuda_used": False,
        "memory_feature_rows": int(n_mem), "candidate_feature_rows": len(cand_rows),
        "memory_by_position_bin": {pb: len(items) for pb, items in mem_by_bin.items()},
        "memory_by_patient": dict(Counter(r["memory_patient_id"] for r in mem_rows)),
        "feature_dim": REDUCED_DIM, "raw_feature_dim": RAW_FEATURE_DIM,
        "feature_nan_count": feat_nan, "feature_inf_count": feat_inf,
        "distance_nan_count": dist_nan, "distance_inf_count": dist_inf,
        "per_bin_thresholds": {pb: {"p50": bin_thr[pb][0], "p90": bin_thr[pb][1]} for pb in mem_by_bin},
        "gate_p2_flag_counts": dict(flag_counts), "candidate_distances": cand_rows,
        "score_modified": False, "adjusted_score_created": False,
        "suppression_weight_created": False, "refined_score_created": False,
        "preprocessing_match_status": "match v2 (preprocess_ct_slice default hu -1000/200, ImageNet, layer1+2+3)",
        "selected_feature_index_status": "matched v2 (position_bin_stats.npz selected_feature_indices 100D)",
        "memory_conditioning": "position-conditioned per candidate position_bin (canonical z_level×central/peripheral)",
        "refined_roi_ratio_source": "v4 refined mask 실제 patch ratio (mask[z][y0:y1,x0:x1]>0).mean(). source_roi_0_0_patch_ratio=score CSV roi_0_0값(별도 컬럼).",
        "runtime_seconds": runtime, "peak_memory_if_available": peak_mem,
        "patchcore_implemented": False,
        "limitations": ["normal 5명/per-patient 100/total 500 minimal preview",
                        "임시 smoke flag threshold(per-bin p50/p90)", "성능지표/threshold 아님",
                        "stage2_holdout 미사용"],
    }
    with open(OUT_DIR / "b1d3g1_position_conditioned_feature_smoke_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    cand_tbl = "\n".join(
        f"| {r['gate_candidate_id']} | {r['review_id']} | {r['position_bin']} | {float(r['candidate_score']):.1f} | "
        f"{r['nearest_distance']} | {r['distance_percentile_within_position_pool']} | **{r['gate_p2_flag']}** |"
        for r in cand_rows)
    md = f"""# B1-D3g1 Gate-P2 Position-conditioned Minimal Feature Smoke (CPU, Plan-PC-S) — Report

position-conditioned memory(candidate 와 동일 position_bin) 기반 minimal feature smoke. CPU only, GPU 미사용.
v2 100D selected feature. score 무수정. 성능 개선 실험 아님.

## 판정: PASS
- device cpu, gpu_used False, holdout 0, memory {n_mem}(5명, per-bin pool), candidate {len(cand_rows)}
- feature_dim {REDUCED_DIM}, NaN/Inf feature {feat_nan}/{feat_inf}, distance {dist_nan}/{dist_inf}
- memory_by_position_bin: {{ {', '.join(f'{pb}:{len(items)}' for pb,items in mem_by_bin.items())} }}
- runtime {runtime}s, peak {peak_mem}

## candidate distance preview (per-position pool)
| GC | review | position_bin | score | nearest_dist | percentile_in_bin | flag |
|---|---|---|---|---|---|---|
{cand_tbl}

- gate_p2_flag_counts: {dict(flag_counts)}

## 한계
- normal 5명/500 cap, per-bin p50/p90 임시 기준, 성능지표 아님. score 무수정.

## 다음 단계
B1-D3h Gate-P2 position-conditioned smoke result interpretation.
"""
    with open(OUT_DIR / "b1d3g1_position_conditioned_feature_smoke_report.md", "w", encoding="utf-8") as f:
        f.write(md)

    print(f"[b1d3g][REAL] PASS. memory={n_mem}, candidates={len(cand_rows)}, flags={dict(flag_counts)}, runtime={runtime}s")
    print(f"  out={OUT_DIR.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--confirm-feature-smoke", action="store_true")
    ap.add_argument("--confirm-gpu", action="store_true")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--memory-patient-limit", type=int, default=5)
    ap.add_argument("--per-patient-patch-cap", type=int, default=100)
    ap.add_argument("--memory-patch-cap", type=int, default=500)
    ap.add_argument("--candidate-limit", type=int, default=6)
    args = ap.parse_args()

    if not args.dry_run and not args.real:
        fail("bare-run 금지. --dry-run 또는 (--real --confirm-feature-smoke) 필요.", code=2)
    if args.device == "cuda" and not args.confirm_gpu:
        fail("device=cuda 는 --confirm-gpu(승인) 없으면 차단.", code=2)
    if args.real:
        if not args.confirm_feature_smoke:
            fail("--real 은 --confirm-feature-smoke 필요.", code=2)
        if not ALLOW_REAL_PROCESSING:
            fail("ALLOW_REAL_PROCESSING=False. real 차단(런타임 override 승인 필요).", code=2)

    v = validate(args)

    if args.dry_run:
        result = {
            "mode": "dry-run", "feature_extracted": False, "files_created": 0, "gpu_used": False,
            "memory_bank_created": False, "nearest_neighbor_computed": False, "score_modified": False,
            "candidates": len(v["cand_pos"]), "planned_memory_patients": len(v["mem_patients"]),
            "per_patient_patch_cap": v["per_patient_cap"], "total_patch_cap": v["total_cap"],
            "per_bin_per_patient": v["per_bin_per_patient"],
            "candidate_ct_mask_mapping": v["candidate_ct_mask_mapping"],
            "candidate_shape_ok": v["candidate_shape_ok"],
            "candidate_range_ok": v["candidate_range_ok"],
            "memory_ct_mask_mapping": v["memory_ct_mask_ok"],
            "position_coverage_complete": v["coverage_complete"],
            "coverage_gaps": v["coverage_gaps"][:5], "stage2_holdout_access": 0,
        }
        print("DRYRUN_RESULT " + json.dumps(result, ensure_ascii=False))
        return

    run_real(v)


if __name__ == "__main__":
    main()
