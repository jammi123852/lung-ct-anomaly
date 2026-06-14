"""
P-B5: v4_20 ROI EfficientNet-B0 normal train smoke limit=5

normal train 5명(normal001~005)으로 v4_20 흉벽제거 ROI PaDiM train 파이프라인 확장 검증.
- CT: roi_0_0 ct_hu.npy (C드라이브, v4_20 refined ROI와 동일 좌표계)
- ROI: refined_roi_v4_20_modeB_all_v1/normal/<safe_id>/refined_roi.npy (v4_20 lock)
- patch 재필터링: v4_20 ROI ratio >= 0.5 (기존 roi_0_0 patch threshold 동일, 흉벽 patch 제외)
- P-B4 limit1 결과(train_limit1)는 보존, train_limit5에만 저장

금지: full train / limit>5 / val·test scoring / lesion scoring / threshold / metrics / stage2_holdout
미사용: model_roi.npy / E드라이브 / lesion 파일
"""
from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT  = Path(__file__).resolve().parents[1]
ROI0_BRANCH = PROJ_ROOT / "experiments" / "efficientnet_b0_imagenet_v1"
SRC_DIR   = PROJ_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

BACKBONE            = "efficientnet_b0"
RAW_FEATURE_DIM     = 144
REDUCED_FEATURE_DIM = 100
MASK_TYPE           = "roi_0_0"
PATHS_CONFIG        = "paths.local.v2_roi0_0.yaml"
LIMIT               = 5
SCRIPT_NAME         = "p_b5_train_smoke_limit5.py"

V4_20_PATCH_RATIO_THRESHOLD = 0.5
V4_20_NORMAL_ROOT = PROJ_ROOT / "outputs" / "mip-postprocess-research-v1" / "masks" / "refined_roi_v4_20_modeB_all_v1" / "normal"

SELECTED_INDICES_PATH = EXP_ROOT / "outputs" / "models" / "distributions" / "selected_feature_indices.npy"
SELECTED_INDICES_SRC  = ROI0_BRANCH / "outputs" / "models" / "distributions" / "selected_feature_indices.npy"
SMOKE_NPZ             = EXP_ROOT / "outputs" / "smoke" / "train_limit5" / "position_bin_stats.npz"
SMOKE_NPZ_LIMIT1      = EXP_ROOT / "outputs" / "smoke" / "train_limit1" / "position_bin_stats.npz"
FULL_MODEL_NPZ        = EXP_ROOT / "outputs" / "models" / "distributions" / "position_bin_stats.npz"
REPORTS_DIR           = EXP_ROOT / "outputs" / "reports" / "smoke"
ERROR_CSV             = REPORTS_DIR / "error.csv"
RUNTIME_CSV           = REPORTS_DIR / "p_b5_runtime_summary.csv"
PATCH_FILTER_CSV      = REPORTS_DIR / "p_b5_patch_filtering_summary.csv"

SPLIT_JSON  = PROJ_ROOT / "outputs" / "position-aware-padim-v1" / "splits" / "normal_v1.json"
P_B4_JSON   = REPORTS_DIR / "p_b4_train_smoke_limit1.json"
P_B3_JSON   = EXP_ROOT / "outputs" / "reports" / "p_b3_lesion_safety_validation" / "p_b3_lesion_safety_validation.json"
P_B2_6_JSON = EXP_ROOT / "outputs" / "reports" / "p_b2_6_v4_20_source_lock" / "p_b2_6_v4_20_source_lock.json"

RUNTIME_COLUMNS = ["timestamp", "script", "metric", "value"]
PATCH_FILTER_COLUMNS = ["patient_id", "safe_id", "ct_shape", "roi_shape", "shape_match",
                        "roi_voxel", "patch_before", "patch_after", "patch_removed", "removed_ratio"]


def record_runtime(rows):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not RUNTIME_CSV.exists() or RUNTIME_CSV.stat().st_size == 0
    with open(RUNTIME_CSV, "a", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RUNTIME_COLUMNS)
        if write_header:
            w.writeheader()
        w.writerows(rows)


def save_patch_filter_csv(rows):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(PATCH_FILTER_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PATCH_FILTER_COLUMNS)
        w.writeheader()
        w.writerows(rows)


def main():
    import numpy as np

    # ── 가드 1: P-B4 verdict=통과 ────────────────────────────────────────
    assert P_B4_JSON.exists(), f"P-B4 보고서 없음: {P_B4_JSON}"
    p_b4 = json.load(open(P_B4_JSON, encoding="utf-8"))
    assert p_b4.get("verdict") == "통과", f"P-B4 verdict != 통과: {p_b4.get('verdict')}"
    assert p_b4.get("next_step_p_b5_train_smoke_limit5_feasible") is True
    print(f"[guard1] P-B4 verdict=통과, P-B5 feasible=True ✅")

    # ── 가드 1b: P-B4 limit1 결과 보존 확인 ──────────────────────────────
    assert SMOKE_NPZ_LIMIT1.exists(), f"P-B4 limit1 결과 없음(보존 위반): {SMOKE_NPZ_LIMIT1}"
    print(f"[guard1b] P-B4 limit1 smoke npz 보존 확인 ✅")

    # ── 가드 2: P-B3 lesion safety ───────────────────────────────────────
    assert P_B3_JSON.exists(), f"P-B3 보고서 없음: {P_B3_JSON}"
    p_b3 = json.load(open(P_B3_JSON, encoding="utf-8"))
    assert p_b3.get("verdict") in ("통과", "부분통과"), f"P-B3 verdict 불충족: {p_b3.get('verdict')}"
    print(f"[guard2] P-B3 verdict={p_b3.get('verdict')}, complete_loss={p_b3.get('complete_lesion_loss')} ✅")

    # ── 가드 3: P-B2.6 source lock ───────────────────────────────────────
    p_b26 = json.load(open(P_B2_6_JSON, encoding="utf-8"))
    assert p_b26.get("user_correction_applied", {}).get("official_roi_source") == "refined_roi_v4_20_modeB_all_v1"
    print("[guard3] P-B2.6 source lock = v4_20 ✅")

    # ── 가드 4: branch-local selected index ──────────────────────────────
    SELECTED_INDICES_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not SELECTED_INDICES_PATH.exists():
        src_idx = np.load(str(SELECTED_INDICES_SRC))
        np.save(str(SELECTED_INDICES_PATH), src_idx)
        print(f"[guard4] selected index 복사: → {SELECTED_INDICES_PATH}")
    sidx = np.load(str(SELECTED_INDICES_PATH))
    assert sidx.shape == (REDUCED_FEATURE_DIM,), f"selected index shape: {sidx.shape}"
    assert len(np.unique(sidx)) == REDUCED_FEATURE_DIM
    assert sidx.min() >= 0 and sidx.max() <= 143
    print(f"[guard4] selected index OK: shape={sidx.shape}, unique={len(np.unique(sidx))}, range=[{sidx.min()},{sidx.max()}]")

    # ── 가드 5: stage2_holdout 접근 없음 ─────────────────────────────────
    print("[guard5] stage2_holdout 접근 없음 (normal train smoke) ✅")

    # ── 가드 6: normal train split ───────────────────────────────────────
    split_data = json.load(open(SPLIT_JSON, encoding="utf-8"))
    train_patients = list(split_data["train"])
    assert len(train_patients) == 290, f"train split 수: {len(train_patients)}"
    p2s = split_data.get("patient_to_safe_id", {})
    print(f"[guard6] normal train split 290명 확인 ✅")

    # ── 가드 7: smoke output 경로 분리/비어있음 ──────────────────────────
    if SMOKE_NPZ.exists():
        print(f"[guard7][ABORT] smoke output 이미 존재: {SMOKE_NPZ}")
        sys.exit(1)
    assert SMOKE_NPZ != FULL_MODEL_NPZ and SMOKE_NPZ != SMOKE_NPZ_LIMIT1, "smoke 경로 충돌"
    print(f"[guard7] smoke output 경로 비어있음/분리 OK")

    smoke_patients = train_patients[:LIMIT]
    print(f"\n[train] smoke limit={LIMIT}, 환자={smoke_patients}")
    print(f"[train] ROI: v4_20 refined_roi (patch ratio>={V4_20_PATCH_RATIO_THRESHOLD} 재필터링)")

    from position_aware_padim.config_manager import ConfigManager
    from position_aware_padim.data_loader import DataLoader
    from position_aware_padim.path_resolver import PathResolver
    from position_aware_padim.padim_model import PaDiMModel
    from position_aware_padim.feature_extractor_effnet_b0_scaffold import FeatureExtractorEffNetB0

    cfg = ConfigManager(str(PROJ_ROOT))
    cfg.load_config(paths_yaml=PATHS_CONFIG)
    normal_training_ready = cfg.get("paths", "normal_training_ready", "")
    manifest_path = Path(normal_training_ready) / "manifests" / "patient_manifest.csv"
    assert manifest_path.exists(), f"manifest 없음: {manifest_path}"

    path_resolver = PathResolver(str(manifest_path), normal_training_ready)
    loader = DataLoader(str(manifest_path), path_resolver, str(ERROR_CSV), use_mmap=True)

    print("\n[train] FeatureExtractorEffNetB0 초기화...")
    feature_extractor = FeatureExtractorEffNetB0()
    print(f"[train] device: {feature_extractor.device}")
    assert feature_extractor.raw_feature_dim == RAW_FEATURE_DIM
    print(f"[train] raw_feature_dim={feature_extractor.raw_feature_dim} ✅")

    import torch
    gpu_avail = (feature_extractor.device == "cuda")
    if gpu_avail:
        torch.cuda.reset_peak_memory_stats()

    model = PaDiMModel(
        selected_feature_indices_path=str(SELECTED_INDICES_PATH),
        feature_dim=REDUCED_FEATURE_DIM, eps=1e-5,
    )
    print(f"[train] PaDiMModel 초기화: feature_dim={model.feature_dim}")

    per_patient_rows = []
    shape_mismatch_count = 0

    def patient_stream():
        nonlocal shape_mismatch_count
        for pid in smoke_patients:
            safe_id = p2s.get(pid, pid)
            data = loader.load_patient_data(pid, mask_type=MASK_TYPE)
            if data is None:
                print(f"  [SKIP] {pid}: 로드 실패")
                continue
            ct_hu = data["ct_hu"]
            roi_path = V4_20_NORMAL_ROOT / safe_id / "refined_roi.npy"
            if not roi_path.exists():
                print(f"  [SKIP] {pid}: v4_20 ROI 없음")
                continue
            refined_roi = np.load(str(roi_path), mmap_mode='r')

            shape_match = (refined_roi.shape == ct_hu.shape)
            if not shape_match:
                shape_mismatch_count += 1
                print(f"  [SKIP] {pid}: shape mismatch ct={ct_hu.shape} roi={refined_roi.shape}")
                per_patient_rows.append({
                    "patient_id": pid, "safe_id": safe_id,
                    "ct_shape": str(tuple(ct_hu.shape)), "roi_shape": str(tuple(refined_roi.shape)),
                    "shape_match": False, "roi_voxel": None,
                    "patch_before": None, "patch_after": None,
                    "patch_removed": None, "removed_ratio": None,
                })
                continue

            roi_voxel = int(np.asarray(refined_roi).sum())
            patch_df = data["patch_df"]
            n_before = len(patch_df)
            keep = []
            for r in patch_df.itertuples(index=False):
                z = int(r.local_z)
                if z < 0 or z >= refined_roi.shape[0]:
                    keep.append(False); continue
                sub = np.asarray(refined_roi[z, int(r.y0):int(r.y1), int(r.x0):int(r.x1)])
                ratio = float(sub.mean()) if sub.size > 0 else 0.0
                keep.append(ratio >= V4_20_PATCH_RATIO_THRESHOLD)
            keep = np.array(keep, dtype=bool)
            patch_df_v4 = patch_df[keep].reset_index(drop=True)
            n_after = len(patch_df_v4)
            removed = n_before - n_after
            removed_ratio = round(removed / n_before, 6) if n_before > 0 else 0.0

            per_patient_rows.append({
                "patient_id": pid, "safe_id": safe_id,
                "ct_shape": str(tuple(ct_hu.shape)), "roi_shape": str(tuple(refined_roi.shape)),
                "shape_match": True, "roi_voxel": roi_voxel,
                "patch_before": n_before, "patch_after": n_after,
                "patch_removed": removed, "removed_ratio": removed_ratio,
            })
            print(f"  [OK] {pid}: ct={tuple(ct_hu.shape)}, roi_voxel={roi_voxel:,}, "
                  f"patch {n_before:,}→{n_after:,} (제외 {removed:,}, {removed_ratio:.4f})")

            data["mask"] = np.asarray(refined_roi)
            data["patch_df"] = patch_df_v4
            yield data

    start = time.time()
    print("\n[train] smoke 학습 시작...")
    model.train(feature_extractor, patient_stream(), split=None)
    elapsed = time.time() - start

    peak_gpu_gb = 0.0
    if gpu_avail:
        peak_gpu_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"[train] peak GPU memory: {peak_gpu_gb:.3f} GB")

    summary = model.train_summary
    print(f"[train] 완료: {summary['n_patients_success']}명 성공, "
          f"{summary['n_patients_failed']}명 실패, {elapsed:.1f}초")
    print(f"[train] used patch={summary['n_patches_used']:,}, skipped={summary['n_patches_skipped']:,}")

    assert model.stats, "stats 비어있음"
    SMOKE_NPZ.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(SMOKE_NPZ))
    print(f"[train] 저장: {SMOKE_NPZ}")

    # ── 통계/무결성 ──────────────────────────────────────────────────────
    n_bins_with_data = 0
    total_nan = total_inf = 0
    sample_mean_shape = sample_cov_shape = None
    for key in model._all_keys():
        s = model.stats.get(key, {})
        if s.get("count", 0) > 0 and "mean" in s:
            mean = s["mean"]; cov = s.get("cov", np.array([]))
            total_nan += int(np.isnan(mean).sum()) + (int(np.isnan(cov).sum()) if cov.size else 0)
            total_inf += int(np.isinf(mean).sum()) + (int(np.isinf(cov).sum()) if cov.size else 0)
            n_bins_with_data += 1
            if sample_mean_shape is None:
                sample_mean_shape = mean.shape; sample_cov_shape = cov.shape

    print(f"\n[검증] mean shape={sample_mean_shape}, cov shape={sample_cov_shape}")
    print(f"[검증] position_bin with data={n_bins_with_data}, NaN={total_nan}, Inf={total_inf}")

    assert sample_mean_shape == (REDUCED_FEATURE_DIM,), f"mean shape: {sample_mean_shape}"
    assert sample_cov_shape == (REDUCED_FEATURE_DIM, REDUCED_FEATURE_DIM), f"cov shape: {sample_cov_shape}"
    assert total_nan == 0 and total_inf == 0, f"NaN={total_nan} Inf={total_inf}"

    # ── 집계 ─────────────────────────────────────────────────────────────
    ok_rows = [r for r in per_patient_rows if r["shape_match"] and r["patch_before"] is not None]
    total_before = sum(r["patch_before"] for r in ok_rows)
    total_after  = sum(r["patch_after"] for r in ok_rows)
    total_removed = total_before - total_after
    total_removed_ratio = round(total_removed / total_before, 6) if total_before > 0 else 0.0
    print(f"[집계] 전체 patch {total_before:,} → {total_after:,} (제외 {total_removed:,}, {total_removed_ratio:.4f})")

    save_patch_filter_csv(per_patient_rows)

    ts = datetime.now().isoformat(timespec="seconds")
    record_runtime([
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "smoke_patients", "value": ",".join(smoke_patients)},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "n_patients_processed", "value": summary["n_patients_success"]},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "shape_mismatch", "value": shape_mismatch_count},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "total_patch_before", "value": total_before},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "total_patch_after", "value": total_after},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "total_patch_removed", "value": total_removed},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "total_removed_ratio", "value": total_removed_ratio},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "n_patches_used", "value": summary["n_patches_used"]},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "n_patches_skipped", "value": summary["n_patches_skipped"]},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "position_bins_with_data", "value": n_bins_with_data},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "elapsed_seconds", "value": round(elapsed, 2)},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "peak_gpu_gb", "value": round(peak_gpu_gb, 3)},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "v4_patch_ratio_threshold", "value": V4_20_PATCH_RATIO_THRESHOLD},
    ])

    verdict = "통과" if (shape_mismatch_count == 0 and summary["n_patients_success"] == LIMIT) else "부분통과"

    report = {
        "step": "P-B5", "verdict": verdict, "timestamp": ts,
        "branch": "efficientnet_b0_imagenet_chestwall_removed_roi_v1",
        "backbone": BACKBONE, "official_roi_source": "refined_roi_v4_20_modeB_all_v1",
        "smoke_patients": smoke_patients, "limit": LIMIT,
        "p_b4_verdict": p_b4.get("verdict"),
        "p_b4_limit1_preserved": True,
        "p_b3_verdict": p_b3.get("verdict"),
        "p_b3_complete_lesion_loss": p_b3.get("complete_lesion_loss"),
        "p_b3_aggregate_preservation": p_b3.get("preservation_distribution", {}).get("aggregate_preservation"),
        "p_b2_6_source_locked": True,
        "roi_connection": {
            "ct_source": "roi_0_0 ct_hu.npy (C드라이브, v4_20와 동일 좌표계)",
            "roi_source": "v4_20 refined_roi.npy",
            "patch_refilter": f"v4_20 ROI ratio >= {V4_20_PATCH_RATIO_THRESHOLD}",
            "model_train_uses_mask": False,
        },
        "n_patients_processed": summary["n_patients_success"],
        "n_patients_failed": summary["n_patients_failed"],
        "shape_mismatch": shape_mismatch_count,
        "per_patient": per_patient_rows,
        "total_patch_before": total_before,
        "total_patch_after": total_after,
        "total_patch_removed": total_removed,
        "total_removed_ratio": total_removed_ratio,
        "raw_feature_dim": RAW_FEATURE_DIM,
        "raw_feature_dim_verified": feature_extractor.raw_feature_dim == RAW_FEATURE_DIM,
        "selected_feature_dim": REDUCED_FEATURE_DIM,
        "selected_feature_dim_verified": model.feature_dim == REDUCED_FEATURE_DIM,
        "selected_index_shape": list(sidx.shape),
        "selected_index_range": [int(sidx.min()), int(sidx.max())],
        "mean_shape": list(sample_mean_shape) if sample_mean_shape else None,
        "cov_shape": list(sample_cov_shape) if sample_cov_shape else None,
        "position_bins_with_data": n_bins_with_data,
        "n_patches_used": summary["n_patches_used"],
        "n_patches_skipped": summary["n_patches_skipped"],
        "total_nan": total_nan, "total_inf": total_inf,
        "elapsed_seconds": round(elapsed, 2), "peak_gpu_gb": round(peak_gpu_gb, 3),
        "output_npz": str(SMOKE_NPZ),
        "safety": {
            "full_train": False, "val_test_scoring": False, "lesion_scoring": False,
            "threshold_calculated": False, "metrics_calculated": False,
            "stage2_holdout_accessed": False, "model_roi_used": False,
            "e_drive_used": False, "lesion_file_used": False,
            "existing_results_modified": False, "pip_install": False, "additional_download": False,
        },
        "next_step_p_b6_full_train_preflight_feasible": (verdict == "통과"),
    }
    with open(REPORTS_DIR / "p_b5_train_smoke_limit5.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ── MD ───────────────────────────────────────────────────────────────
    md = [
        "# P-B5 v4_20 ROI EfficientNet-B0 Train Smoke Limit=5\n",
        f"**판정: {verdict}**\n",
        f"- 생성일시: {ts}",
        f"- branch: efficientnet_b0_imagenet_chestwall_removed_roi_v1",
        f"- 백본: {BACKBONE} / ROI source: refined_roi_v4_20_modeB_all_v1\n",
        "## P-B4 입력 검증\n",
        f"- P-B4 verdict: {p_b4.get('verdict')}",
        f"- P-B4 limit1 결과 보존: True\n",
        "## P-B3 lesion safety 요약\n",
        f"- P-B3 verdict: {p_b3.get('verdict')}",
        f"- complete lesion loss: {p_b3.get('complete_lesion_loss')}건",
        f"- aggregate preservation: {p_b3.get('preservation_distribution',{}).get('aggregate_preservation')}\n",
        "## ROI 연결 방식\n",
        "- CT: roi_0_0 ct_hu.npy (C드라이브, v4_20와 동일 좌표계)",
        "- ROI: v4_20 refined_roi.npy",
        f"- patch 재필터링: v4_20 ROI ratio >= {V4_20_PATCH_RATIO_THRESHOLD} (기존 roi_0_0 threshold 동일)\n",
        "## 처리 환자 (5명) — patch 재필터링\n",
        "| patient | CT shape | ROI voxel | patch 전 | patch 후 | 제외 | 제외율 |",
        "|---------|----------|-----------|----------|----------|------|--------|",
    ]
    for r in per_patient_rows:
        if r["shape_match"] and r["patch_before"] is not None:
            md.append(f"| {r['patient_id']} | {r['ct_shape']} | {r['roi_voxel']:,} | "
                      f"{r['patch_before']:,} | {r['patch_after']:,} | {r['patch_removed']:,} | {r['removed_ratio']:.4f} |")
        else:
            md.append(f"| {r['patient_id']} | {r['ct_shape']} | shape_mismatch | - | - | - | - |")
    md += [
        "",
        f"- **전체 patch: {total_before:,} → {total_after:,} (제외 {total_removed:,}, 제외율 {total_removed_ratio:.4f})**\n",
        "## CT/ROI shape\n",
        f"- shape mismatch: {shape_mismatch_count}건\n",
        "## Feature 차원\n",
        f"- raw_feature_dim=144: {feature_extractor.raw_feature_dim == 144}",
        f"- selected_dim=100: {model.feature_dim == 100}",
        f"- selected index: shape={list(sidx.shape)}, unique={len(np.unique(sidx))}, range=[{int(sidx.min())},{int(sidx.max())}]",
        f"- mean shape: {sample_mean_shape} / cov shape: {sample_cov_shape}\n",
        "## Patch 통계 / 무결성\n",
        f"- used patch: {summary['n_patches_used']:,}",
        f"- skipped patch: {summary['n_patches_skipped']:,}",
        f"- position_bin with data: {n_bins_with_data}",
        f"- NaN: {total_nan} / Inf: {total_inf}\n",
        "## 실행 정보\n",
        f"- elapsed: {round(elapsed,2)}초",
        f"- peak GPU: {round(peak_gpu_gb,3)} GB",
        f"- smoke output: {SMOKE_NPZ}\n",
        "## 미사용 / 무수정 확인\n",
        "- model_roi.npy / E드라이브 / lesion 파일 / stage2_holdout: 미사용",
        "- full train / scoring / threshold / metrics: 미실행",
        "- 기존 roi_0_0 / EfficientNet-B0 / P-B1~P-B4 결과: 무수정",
        "- P-B4 limit1 smoke npz 보존\n",
        "## 학습 범위\n",
        "- **normal train smoke limit=5** (5명만). full train 아님.\n",
        "## 다음 단계\n",
        f"- P-B6 full train preflight 진행 가능: {verdict == '통과'}",
    ]
    with open(REPORTS_DIR / "p_b5_train_smoke_limit5.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(f"\n=== P-B5 smoke limit5 완료: {verdict} ===")
    print(f"[보고서] {REPORTS_DIR}")


if __name__ == "__main__":
    main()
