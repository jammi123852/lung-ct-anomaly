#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
b1d3d_gate_p2_minimal_feature_smoke

Gate-P2 PatchCore/gated filter 의 minimal feature smoke 실행 스크립트 (초안).

★ 기본 차단(ALLOW_REAL_PROCESSING=False). bare-run 즉시 중단(exit 2).
★ --dry-run: 입력/범위/경로/shape/cap 검증만. feature 추출/torch import/파일 생성 0.
★ --real --confirm-feature-smoke: 별도 승인 후에만 feature 추출(기본 차단 유지).
★ --device gpu 는 --confirm-gpu 없으면 차단(과금 방지). 기본 cpu.
★ score/mask/ROI 수정 금지. output folder exist_ok=False. stage2_holdout 접근 0(normal only).

이번 단계(B1-D3d0)에서는 py_compile + bare-run(exit 2) + --dry-run 까지만 허용한다.
real feature smoke 는 B1-D3d1 별도 승인 후 진행한다.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ALLOW_REAL_PROCESSING = False  # ★ 기본 차단. 런타임 override 로만 real 허용.

BASE = Path("/home/jinhy/project/lung-ct-anomaly")
DIR = BASE / "outputs/b1d1_wall_mediastinum_fp_cause_diagnostic_v1"
MROOT = BASE / "outputs/mip-postprocess-research-v1/masks/refined_roi_v4_20_modeB_all_v1"
NROOT = Path("/mnt/c/Users/jinhy/Desktop/Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1/volumes_npy")

CAND_CSV = DIR / "b1d3c_gate_p2_feature_preflight_candidates.csv"
POOL_CSV = DIR / "b1d3c_gate_p2_memory_pool_preview.csv"
OUT_DIR = DIR / "b1d3d1_gate_p2_minimal_feature_smoke_plan_s_cpu_v1"
SEL_IDX_NPZ = BASE / "outputs/position-aware-padim-v1/models/padim_v2_roi0_0/distributions/position_bin_stats.npz"

PATCH = 32
STRIDE = 16
RAW_FEATURE_DIM = 448   # resnet18 layer1+2+3 concat (64+128+256)
REDUCED_DIM = 100       # v2 selected_feature_indices 로 100차원 축소 (v2 PaDiM과 동일 feature 공간)
# v2 score_patient 은 preprocess_ct_slice(slice_2d) (인자 없음) → default hu_min=-1000/hu_max=200 사용. smoke 도 동일.
HU_MIN, HU_MAX = -1000, 200


def fail(msg, code=2):
    print(f"[b1d3d][중단] {msg}", file=sys.stderr)
    sys.exit(code)


def load_rows(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resolve_normal_dirs(patient_voldir):
    """preview_patient_id(voldir 명) → (mask_npy, ct_npy)."""
    md = MROOT / "normal" / patient_voldir / "refined_roi.npy"
    cd = NROOT / patient_voldir / "ct_hu.npy"
    return md, cd


def validate(args):
    """read-only 검증. feature 없음. dry/real 공통."""
    if not CAND_CSV.exists():
        fail(f"candidates CSV 없음: {CAND_CSV}")
    if not POOL_CSV.exists():
        fail(f"memory pool CSV 없음: {POOL_CSV}")

    cands = load_rows(CAND_CSV)
    if len(cands) != 6:
        fail(f"candidates row {len(cands)} != 6")
    if len(cands) > args.candidate_limit:
        fail(f"candidate 수 {len(cands)} > candidate_limit {args.candidate_limit}")

    pool = [r for r in load_rows(POOL_CSV) if r["usable_for_memory_pool"] == "true"]
    if len(pool) == 0:
        fail("usable memory pool 환자 0")
    mem = pool[:args.memory_patient_limit]
    if len(mem) > args.memory_patient_limit:
        fail(f"memory patient {len(mem)} > limit {args.memory_patient_limit}")

    # memory 는 normal only (lesion 경로가 섞이면 차단)
    for r in mem:
        if not (MROOT / "normal" / r["preview_patient_id"]).is_dir():
            fail(f"memory 환자가 normal root 에 없음(또는 lesion 혼입): {r['preview_patient_id']}")

    # 경로/shape 검증 (mmap, feature 없음)
    checked = {"candidates_ok": 0, "memory_ok": 0}
    for c in cands:
        md = MROOT / "normal" / (c["patient_id"])
        hits = sorted((MROOT / "normal").glob(f"{c['patient_id']}__*"))
        if not hits:
            fail(f"candidate mask dir 없음: {c['patient_id']}")
        mnpy = hits[0] / "refined_roi.npy"
        cnpy = NROOT / hits[0].name / "ct_hu.npy"
        if not mnpy.exists() or not cnpy.exists():
            fail(f"candidate CT/mask npy 없음: {hits[0].name}")
        m = np.load(mnpy, mmap_mode="r")
        cc = np.load(cnpy, mmap_mode="r")
        z, y0, x0 = int(c["candidate_local_z"]), int(c["candidate_y0"]), int(c["candidate_x0"])
        if not (m.shape == cc.shape and m.shape[1:] == (512, 512)
                and 0 <= z < m.shape[0] and 0 <= y0 <= 512 - PATCH and 0 <= x0 <= 512 - PATCH):
            fail(f"candidate shape/좌표 오류: {c['review_id']} {m.shape} z{z} y{y0} x{x0}")
        del m, cc
        checked["candidates_ok"] += 1

    for r in mem:
        mnpy, cnpy = resolve_normal_dirs(r["preview_patient_id"])
        if not mnpy.exists() or not cnpy.exists():
            fail(f"memory CT/mask npy 없음: {r['preview_patient_id']}")
        m = np.load(mnpy, mmap_mode="r")
        cc = np.load(cnpy, mmap_mode="r")
        if not (m.shape == cc.shape and m.shape[1:] == (512, 512)):
            fail(f"memory shape 오류: {r['preview_patient_id']} {m.shape}")
        del m, cc
        checked["memory_ok"] += 1

    return {"candidates": cands, "memory_patients": mem,
            "memory_patch_cap": args.memory_patch_cap, "checked": checked,
            "device": args.device}


def patch_grid_centers(mask2d):
    """refined ROI 안에 center 가 있는 patch (y0,x0,y1,x1) 목록."""
    coords = []
    for y in range(0, 512 - PATCH + 1, STRIDE):
        for x in range(0, 512 - PATCH + 1, STRIDE):
            if mask2d[y + PATCH // 2, x + PATCH // 2] > 0:
                coords.append((y, x, y + PATCH, x + PATCH))
    return coords


def run_real(v):
    """★ 승인 후에만 호출. feature 추출(v2 100차원 selected) + nearest distance + 3단계 flag. score 무수정."""
    import time
    from collections import Counter
    t0 = time.time()

    if v["device"] != "cpu":
        fail("이 smoke 는 device=cpu 만 허용(GPU 금지).")

    OUT_DIR.mkdir(parents=True, exist_ok=False)  # collision guard

    import torch  # noqa: F401  (real 분기에서만 import)
    sys.path.insert(0, str(BASE / "src"))
    from position_aware_padim.feature_extractor import FeatureExtractor
    from position_aware_padim.preprocessing import preprocess_ct_slice

    # --- v2 selected_feature_indices (100차원 축소; v2 PaDiM 과 동일 feature 공간) ---
    if not SEL_IDX_NPZ.exists():
        fail(f"selected_feature_indices npz 없음: {SEL_IDX_NPZ}")
    sel = np.load(SEL_IDX_NPZ, allow_pickle=True)["selected_feature_indices"].astype(int)
    if sel.shape[0] != REDUCED_DIM or sel.min() < 0 or sel.max() >= RAW_FEATURE_DIM:
        fail(f"selected_feature_indices 비정상: shape={sel.shape} range[{sel.min()},{sel.max()}]")

    fe = FeatureExtractor(device="cpu")
    if str(fe.device) != "cpu":
        fail(f"FeatureExtractor device가 cpu 아님: {fe.device}")

    feat_nan = feat_inf = 0

    # --- memory bank (normal only, cap, 100차원) ---
    mem_feats, mem_owner, mem_rows = [], [], []
    cap = int(v["memory_patch_cap"])
    rng = np.random.RandomState(0)
    midx = 1
    for r in v["memory_patients"]:
        if len(mem_feats) >= cap:
            break
        mnpy, cnpy = resolve_normal_dirs(r["preview_patient_id"])
        m = np.load(mnpy, mmap_mode="r")
        c = np.load(cnpy, mmap_mode="r")
        Z = m.shape[0]
        zs = np.linspace(int(Z * 0.3), int(Z * 0.7), 5).astype(int)  # 중간부 위주
        for zi in zs:
            if len(mem_feats) >= cap:
                break
            mask2d = np.asarray(m[zi])
            coords = patch_grid_centers(mask2d)
            if not coords:
                continue
            if len(coords) > 60:
                idx = rng.choice(len(coords), 60, replace=False)
                coords = [coords[i] for i in idx]
            sl = preprocess_ct_slice(np.asarray(c[zi]).astype(np.float32))  # default hu == v2
            feats = fe.extract_patch_features(sl, coords)[:, sel]  # (n, 100)
            for (y0, x0, y1, x1), fr in zip(coords, feats):
                if len(mem_feats) >= cap:
                    break
                nanc, infc = int(np.isnan(fr).sum()), int(np.isinf(fr).sum())
                feat_nan += nanc
                feat_inf += infc
                ratio = float((mask2d[y0:y1, x0:x1] > 0).mean())
                mid = f"MEM{midx:04d}"
                mem_feats.append(fr)
                mem_owner.append(r["preview_patient_id"])
                mem_rows.append({
                    "memory_patch_id": mid, "memory_patient_id": r["preview_patient_id"],
                    "source": "normal_stage1_dev", "z": int(zi), "y0": int(y0), "x0": int(x0),
                    "refined_roi_ratio": round(ratio, 4),
                    "feature_status": "ok" if (nanc == 0 and infc == 0) else "nan_inf",
                    "feature_dim": REDUCED_DIM, "used_in_memory": "true", "exclusion_reason": "",
                })
                midx += 1
        del m, c
    mem_mat = np.asarray(mem_feats, dtype=np.float32)
    if mem_mat.shape[0] == 0:
        fail("memory feature 0")
    if mem_mat.shape[1] != REDUCED_DIM:
        fail(f"memory feature_dim {mem_mat.shape[1]} != {REDUCED_DIM}")
    if not np.isfinite(mem_mat).all():
        fail("memory feature 에 NaN/Inf")

    # --- memory 내부 NN 거리 분포(임시 smoke 임계 기준) ---
    self_nn = np.empty(mem_mat.shape[0], dtype=np.float64)
    for i in range(mem_mat.shape[0]):
        d = np.linalg.norm(mem_mat - mem_mat[i][None, :], axis=1)
        d[i] = np.inf
        self_nn[i] = d.min()
    dist_nan = int(np.isnan(self_nn).sum())
    dist_inf = int(np.isinf(self_nn).sum())
    p50, p90 = float(np.percentile(self_nn, 50)), float(np.percentile(self_nn, 90))

    # --- candidate features(100차원) + nearest distance + 3단계 flag ---
    cand_rows = []
    flag_counts = Counter()
    for c in v["candidates"]:
        hits = sorted((MROOT / "normal").glob(f"{c['patient_id']}__*"))
        cnpy = NROOT / hits[0].name / "ct_hu.npy"
        z, y0, x0 = int(c["candidate_local_z"]), int(c["candidate_y0"]), int(c["candidate_x0"])
        ct = np.load(cnpy, mmap_mode="r")
        sl = preprocess_ct_slice(np.asarray(ct[z]).astype(np.float32))
        feat = fe.extract_patch_features(sl, [(y0, x0, y0 + PATCH, x0 + PATCH)])[0][sel]  # (100,)
        del ct
        nanc, infc = int(np.isnan(feat).sum()), int(np.isinf(feat).sum())
        feat_nan += nanc
        feat_inf += infc
        if nanc or infc:
            fail(f"candidate feature NaN/Inf: {c['review_id']}")
        d = np.linalg.norm(mem_mat - feat[None, :], axis=1)
        j = int(np.argmin(d))
        dist = float(d[j])
        if not np.isfinite(dist):
            fail(f"distance NaN/Inf: {c['review_id']}")
        pct = round(float((self_nn < dist).mean() * 100.0), 1)  # memory self-NN 분포 대비 percentile
        flag = "normal_like" if dist <= p50 else ("suspicious" if dist > p90 else "uncertain")
        flag_counts[flag] += 1
        cand_rows.append({
            "gate_candidate_id": c["gate_candidate_id"], "review_id": c["review_id"],
            "patient_id": c["patient_id"], "candidate_score": c["candidate_score"],
            "feature_status": "ok", "feature_dim": REDUCED_DIM,
            "nearest_distance": round(dist, 4), "nearest_memory_patient": mem_owner[j],
            "nearest_memory_patch_id": mem_rows[j]["memory_patch_id"],
            "distance_rank_or_percentile": f"{pct}%ile_vs_memNN(p50={p50:.2f}/p90={p90:.2f})",
            "gate_p2_flag": flag,
            "flag_reason": f"nn_dist={dist:.2f}; <=p50->normal_like, >p90->suspicious (임시 smoke 기준)",
            "score_modified": "false",
            "safety_note": "preview only; PaDiM score 무수정; 성능지표 아님",
        })

    runtime = round(time.time() - t0, 1)
    try:
        import resource
        peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_mem = f"{round(peak_kb/1024, 1)} MB (ru_maxrss)"
    except Exception:
        peak_mem = "n/a"

    # --- 출력 CSV ---
    with open(OUT_DIR / "b1d3d1_gate_p2_memory_feature_preview.csv", "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(mem_rows[0].keys()))
        w.writeheader()
        w.writerows(mem_rows)
    with open(OUT_DIR / "b1d3d1_gate_p2_candidate_distance_preview.csv", "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cand_rows[0].keys()))
        w.writeheader()
        w.writerows(cand_rows)

    summary = {
        "step": "B1-D3d1_Gate_P2_minimal_feature_smoke_CPU_Plan_S",
        "stage2_holdout_access": 0,
        "plan_used": "Plan-S",
        "device_used": "cpu", "gpu_used": False, "cuda_used": False,
        "memory_patient_limit": len(v["memory_patients"]),
        "memory_patch_cap": cap,
        "candidate_limit": len(v["candidates"]),
        "memory_feature_rows": int(mem_mat.shape[0]),
        "candidate_feature_rows": len(cand_rows),
        "feature_dim": REDUCED_DIM, "raw_feature_dim": RAW_FEATURE_DIM,
        "feature_nan_count": feat_nan, "feature_inf_count": feat_inf,
        "distance_nan_count": dist_nan, "distance_inf_count": dist_inf,
        "memNN_p50": p50, "memNN_p90": p90,
        "gate_p2_flag_counts": dict(flag_counts),
        "candidate_distances": cand_rows,
        "memory_patches_by_patient": dict(Counter(mem_owner)),
        "score_modified": False, "adjusted_score_created": False,
        "suppression_weight_created": False, "refined_score_created": False,
        "preprocessing_match_status": ("match: v2 score_patient=preprocess_ct_slice(slice_2d) "
                                       "default hu_min=-1000/hu_max=200, ImageNet mean/std, 3ch, "
                                       "resnet18 layer1+2+3 concat 448"),
        "selected_feature_index_status": ("matched v2: position_bin_stats.npz selected_feature_indices "
                                          "(100,), padim_v1 npy와 identical. raw448→100 축소 적용"),
        "runtime_seconds": runtime, "peak_memory_if_available": peak_mem,
        "patchcore_implemented": False,
        "limitations": ["normal memory 3명/cap500 minimal preview", "임시 smoke flag threshold(p50/p90)",
                        "성능지표/threshold 아님", "stage2_holdout 미사용"],
    }
    with open(OUT_DIR / "b1d3d1_gate_p2_minimal_feature_smoke_summary.json", "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    cand_tbl = "\n".join(
        f"| {r['gate_candidate_id']} | {r['review_id']} | {float(r['candidate_score']):.1f} | "
        f"{r['nearest_distance']} | {r['nearest_memory_patient'][:18]} | {r['distance_rank_or_percentile']} | "
        f"**{r['gate_p2_flag']}** |" for r in cand_rows)
    md = f"""# B1-D3d1 Gate-P2 Minimal Feature Smoke (CPU, Plan-S) — Report

PatchCore/gated filter minimal feature smoke. **CPU only, GPU 미사용.** v2 100차원 selected feature 공간.
score 무수정. 성능 개선 실험 아님(distance/flag 분리 동작 preview).

## 판정: PASS
- device=cpu, gpu_used=False, stage2_holdout=0
- memory {int(mem_mat.shape[0])} patch (normal {len(v['memory_patients'])}명, cap {cap}), candidate {len(cand_rows)}
- feature_dim={REDUCED_DIM} (raw {RAW_FEATURE_DIM}→v2 selected 100), NaN/Inf feature={feat_nan}/{feat_inf}, distance NaN/Inf={dist_nan}/{dist_inf}
- preprocessing match: v2와 동일(hu default -1000/200, ImageNet, layer1+2+3)
- selected_feature_index: v2 identical 적용
- runtime={runtime}s, peak_mem={peak_mem}

## candidate distance preview
| GC | review | score | nearest_dist | nearest_mem_patient | percentile | flag |
|---|---|---|---|---|---|---|
{cand_tbl}

- gate_p2_flag_counts: {dict(flag_counts)}
- memNN p50={p50:.3f}, p90={p90:.3f} (임시 smoke 기준)

## 한계
- normal memory 3명/500 cap minimal preview. flag threshold 는 임시(p50/p90), 성능지표/threshold 아님.
- score 무수정, stage2_holdout 미사용. 결과를 성능 개선으로 단정하지 않음.

## 다음 단계
B1-D3e Gate-P2 smoke result interpretation.

---
device=cpu, gpu_used=False, score_modified=False, adjusted_score_created=False, patchcore_implemented=False, stage2_holdout_access=0
"""
    with open(OUT_DIR / "b1d3d1_gate_p2_minimal_feature_smoke_report.md", "w", encoding="utf-8") as f:
        f.write(md)

    print(f"[b1d3d][REAL] PASS. memory={mem_mat.shape[0]}, candidates={len(cand_rows)}, "
          f"flags={dict(flag_counts)}, feature_dim={REDUCED_DIM}, runtime={runtime}s")
    print(f"  out={OUT_DIR.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--confirm-feature-smoke", action="store_true")
    ap.add_argument("--confirm-gpu", action="store_true")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--memory-patient-limit", type=int, default=3)
    ap.add_argument("--memory-patch-cap", type=int, default=500)
    ap.add_argument("--candidate-limit", type=int, default=6)
    args = ap.parse_args()

    # bare-run 차단
    if not args.dry_run and not args.real:
        fail("bare-run 금지. --dry-run 또는 (--real --confirm-feature-smoke) 필요.", code=2)

    # device gate
    if args.device == "cuda" and not args.confirm_gpu:
        fail("device=cuda 는 --confirm-gpu(사용자 승인) 없으면 차단.", code=2)

    if args.real:
        if not args.confirm_feature_smoke:
            fail("--real 은 --confirm-feature-smoke 필요.", code=2)
        if not ALLOW_REAL_PROCESSING:
            fail("ALLOW_REAL_PROCESSING=False. real feature smoke 차단(런타임 override 승인 필요).", code=2)

    v = validate(args)

    if args.dry_run:
        result = {
            "mode": "dry-run", "feature_extracted": False, "files_created": 0,
            "gpu_used": False, "candidates": len(v["candidates"]),
            "memory_patients": len(v["memory_patients"]),
            "memory_patch_cap": v["memory_patch_cap"], "device": v["device"],
            "checked": v["checked"], "stage2_holdout_access": 0,
        }
        print("DRYRUN_RESULT " + json.dumps(result, ensure_ascii=False))
        return

    # real (gated; ALLOW_REAL_PROCESSING 또는 런타임 override 통과 시에만 도달)
    run_real(v)


if __name__ == "__main__":
    main()
