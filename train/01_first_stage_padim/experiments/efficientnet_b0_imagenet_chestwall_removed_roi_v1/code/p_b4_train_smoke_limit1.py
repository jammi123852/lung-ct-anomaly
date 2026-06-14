"""
P-B4: v4_20 ROI EfficientNet-B0 normal train smoke limit=1

normal train 1명(normal001)으로 v4_20 흉벽제거 ROI PaDiM train 파이프라인 검증.
- CT: roi_0_0 ct_hu.npy (C드라이브, v4_20 refined ROI와 동일 좌표계)
- ROI: refined_roi_v4_20_modeB_all_v1/normal/<safe_id>/refined_roi.npy (v4_20 lock)
- 핵심: model.train()은 patch_df 좌표만 사용 → patch_df를 v4_20 ROI ratio>=0.5로 재필터링
        (기존 roi_0_0 patch 생성 threshold와 동일, 흉벽 patch 자동 제외)

금지: full train / limit>1 / val·test scoring / lesion scoring / threshold / metrics / stage2_holdout
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

# 백본 고정값 (기존 EfficientNet-B0 branch와 동일)
BACKBONE            = "efficientnet_b0"
RAW_FEATURE_DIM     = 144
REDUCED_FEATURE_DIM = 100
MASK_TYPE           = "roi_0_0"   # DataLoader로 CT+roi0 로드 (CT 좌표계 확보용). 실제 ROI는 v4_20로 교체.
PATHS_CONFIG        = "paths.local.v2_roi0_0.yaml"
LIMIT               = 1
SCRIPT_NAME         = "p_b4_train_smoke_limit1.py"

# v4_20 ROI patch 재필터링 threshold (기존 roi_0_0 patch 생성과 동일)
V4_20_PATCH_RATIO_THRESHOLD = 0.5

# v4_20 ROI source
V4_20_NORMAL_ROOT = PROJ_ROOT / "outputs" / "mip-postprocess-research-v1" / "masks" / "refined_roi_v4_20_modeB_all_v1" / "normal"

# 경로 (새 branch 내부)
SELECTED_INDICES_PATH = EXP_ROOT / "outputs" / "models" / "distributions" / "selected_feature_indices.npy"
SELECTED_INDICES_SRC  = ROI0_BRANCH / "outputs" / "models" / "distributions" / "selected_feature_indices.npy"
SMOKE_NPZ             = EXP_ROOT / "outputs" / "smoke" / "train_limit1" / "position_bin_stats.npz"
FULL_MODEL_NPZ        = EXP_ROOT / "outputs" / "models" / "distributions" / "position_bin_stats.npz"
REPORTS_DIR           = EXP_ROOT / "outputs" / "reports" / "smoke"
ERROR_CSV             = REPORTS_DIR / "error.csv"
RUNTIME_CSV           = REPORTS_DIR / "p_b4_runtime_summary.csv"

# split / 입력 검증
SPLIT_JSON = PROJ_ROOT / "outputs" / "position-aware-padim-v1" / "splits" / "normal_v1.json"
P_B3_JSON  = EXP_ROOT / "outputs" / "reports" / "p_b3_lesion_safety_validation" / "p_b3_lesion_safety_validation.json"
P_B2_6_JSON = EXP_ROOT / "outputs" / "reports" / "p_b2_6_v4_20_source_lock" / "p_b2_6_v4_20_source_lock.json"

RUNTIME_COLUMNS = ["timestamp", "script", "metric", "value"]


def record_runtime(rows):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not RUNTIME_CSV.exists() or RUNTIME_CSV.stat().st_size == 0
    with open(RUNTIME_CSV, "a", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RUNTIME_COLUMNS)
        if write_header:
            w.writeheader()
        w.writerows(rows)


def main():
    import numpy as np

    # ── 가드 1: P-B3 verdict ─────────────────────────────────────────────
    assert P_B3_JSON.exists(), f"P-B3 보고서 없음: {P_B3_JSON}"
    p_b3 = json.load(open(P_B3_JSON, encoding="utf-8"))
    assert p_b3.get("verdict") in ("통과", "부분통과"), f"P-B3 verdict 불충족: {p_b3.get('verdict')}"
    assert p_b3.get("p_b4_readiness", {}).get("can_proceed") is True, "P-B3 p_b4 can_proceed != True"
    print(f"[guard1] P-B3 verdict={p_b3.get('verdict')}, p_b4_can_proceed=True ✅")

    # ── 가드 2: P-B2.6 source lock ───────────────────────────────────────
    assert P_B2_6_JSON.exists(), f"P-B2.6 보고서 없음: {P_B2_6_JSON}"
    p_b26 = json.load(open(P_B2_6_JSON, encoding="utf-8"))
    assert p_b26.get("user_correction_applied", {}).get("official_roi_source") == "refined_roi_v4_20_modeB_all_v1", \
        "P-B2.6 official ROI source != v4_20"
    print("[guard2] P-B2.6 source lock = v4_20 ✅")

    # ── 가드 3: selected index 준비 (기존 branch 값 복사, 동일 정책) ──────
    SELECTED_INDICES_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not SELECTED_INDICES_PATH.exists():
        assert SELECTED_INDICES_SRC.exists(), f"기존 selected index 없음: {SELECTED_INDICES_SRC}"
        src_idx = np.load(str(SELECTED_INDICES_SRC))
        np.save(str(SELECTED_INDICES_PATH), src_idx)
        print(f"[guard3] selected index 복사: {SELECTED_INDICES_SRC} → {SELECTED_INDICES_PATH}")
    sidx = np.load(str(SELECTED_INDICES_PATH))
    assert sidx.shape == (REDUCED_FEATURE_DIM,), f"selected index shape 불일치: {sidx.shape}"
    assert len(np.unique(sidx)) == REDUCED_FEATURE_DIM, "selected index unique 불일치"
    assert sidx.min() >= 0 and sidx.max() <= 143, f"selected index range 오류: {sidx.min()}~{sidx.max()}"
    print(f"[guard3] selected index OK: shape={sidx.shape}, unique={len(np.unique(sidx))}, range=[{sidx.min()},{sidx.max()}]")

    # ── 가드 4: stage2_holdout 접근 없음 (normal train만) ────────────────
    print("[guard4] stage2_holdout 접근 없음 (normal train smoke) ✅")

    # ── 가드 5: normal train split 290 ───────────────────────────────────
    split_data = json.load(open(SPLIT_JSON, encoding="utf-8"))
    train_patients = list(split_data["train"])
    assert len(train_patients) == 290, f"train split 수 불일치: {len(train_patients)}"
    p2s = split_data.get("patient_to_safe_id", {})
    print(f"[guard5] normal train split 290명 확인 ✅")

    # ── 가드 6: smoke output 경로 비어있음 ───────────────────────────────
    if SMOKE_NPZ.exists():
        print(f"[guard6][ABORT] smoke output 이미 존재: {SMOKE_NPZ}")
        sys.exit(1)
    assert SMOKE_NPZ != FULL_MODEL_NPZ, "smoke/full 경로 충돌"
    print(f"[guard6] smoke output 경로 비어있음 OK")

    # ── limit=1 환자 (기존 branch와 동일 첫 환자) ────────────────────────
    smoke_patients = train_patients[:LIMIT]
    pid = smoke_patients[0]
    safe_id = p2s.get(pid, pid)
    print(f"\n[train] smoke limit={LIMIT}, 환자={pid} (safe_id={safe_id})")
    print(f"[train] ROI source: v4_20 refined_roi (patch ratio>={V4_20_PATCH_RATIO_THRESHOLD} 재필터링)")

    # ── 모듈 import ──────────────────────────────────────────────────────
    from position_aware_padim.config_manager import ConfigManager
    from position_aware_padim.data_loader import DataLoader
    from position_aware_padim.path_resolver import PathResolver
    from position_aware_padim.padim_model import PaDiMModel
    from position_aware_padim.feature_extractor_effnet_b0_scaffold import FeatureExtractorEffNetB0

    cfg = ConfigManager(str(PROJ_ROOT))
    cfg.load_config(paths_yaml=PATHS_CONFIG)
    normal_training_ready = cfg.get("paths", "normal_training_ready", "")
    assert normal_training_ready, "paths config에 normal_training_ready 없음"
    manifest_path = Path(normal_training_ready) / "manifests" / "patient_manifest.csv"
    assert manifest_path.exists(), f"manifest 없음: {manifest_path}"

    path_resolver = PathResolver(str(manifest_path), normal_training_ready)
    loader = DataLoader(str(manifest_path), path_resolver, str(ERROR_CSV), use_mmap=True)

    print("\n[train] FeatureExtractorEffNetB0 초기화...")
    feature_extractor = FeatureExtractorEffNetB0()
    print(f"[train] device: {feature_extractor.device}")
    assert feature_extractor.raw_feature_dim == RAW_FEATURE_DIM, \
        f"raw_feature_dim 불일치: {feature_extractor.raw_feature_dim}"
    print(f"[train] raw_feature_dim={feature_extractor.raw_feature_dim} 확인 ✅")

    import torch
    gpu_avail = (feature_extractor.device == "cuda")
    if gpu_avail:
        torch.cuda.reset_peak_memory_stats()

    model = PaDiMModel(
        selected_feature_indices_path=str(SELECTED_INDICES_PATH),
        feature_dim=REDUCED_FEATURE_DIM,
        eps=1e-5,
    )
    print(f"[train] PaDiMModel 초기화: feature_dim={model.feature_dim}")

    # ── v4_20 ROI 로드 + shape 검증 + patch 재필터링 정보 ────────────────
    roi_path = V4_20_NORMAL_ROOT / safe_id / "refined_roi.npy"
    assert roi_path.exists(), f"v4_20 normal ROI 없음: {roi_path}"

    stats_holder = {}

    def patient_stream():
        data = loader.load_patient_data(pid, mask_type=MASK_TYPE)
        if data is None:
            raise RuntimeError(f"DataLoader가 None 반환: {pid}")
        ct_hu = data["ct_hu"]
        refined_roi = np.load(str(roi_path), mmap_mode='r')

        # CT/ROI shape 일치 확인
        assert refined_roi.shape == ct_hu.shape, \
            f"CT/ROI shape 불일치: ct={ct_hu.shape} roi={refined_roi.shape}"

        ct_shape = tuple(ct_hu.shape)
        roi_voxel = int(np.asarray(refined_roi).sum())

        # patch_df를 v4_20 ROI ratio>=threshold 로 재필터링
        patch_df = data["patch_df"]
        n_before = len(patch_df)
        keep_mask = []
        for r in patch_df.itertuples(index=False):
            z = int(r.local_z)
            if z < 0 or z >= refined_roi.shape[0]:
                keep_mask.append(False); continue
            sub = np.asarray(refined_roi[z, int(r.y0):int(r.y1), int(r.x0):int(r.x1)])
            ratio = float(sub.mean()) if sub.size > 0 else 0.0
            keep_mask.append(ratio >= V4_20_PATCH_RATIO_THRESHOLD)
        keep_mask = np.array(keep_mask, dtype=bool)
        patch_df_v4 = patch_df[keep_mask].reset_index(drop=True)
        n_after = len(patch_df_v4)

        stats_holder.update({
            "ct_shape": ct_shape,
            "roi_shape": tuple(refined_roi.shape),
            "roi_voxel": roi_voxel,
            "patch_before": n_before,
            "patch_after": n_after,
            "patch_removed_by_v4": n_before - n_after,
        })
        print(f"  [v4_20] ct={ct_shape}, roi_voxel={roi_voxel:,}")
        print(f"  [v4_20] patch {n_before:,} → {n_after:,} (흉벽제거 {n_before-n_after:,} skip)")

        data["mask"] = np.asarray(refined_roi)   # ROI 교체 (참고용, train은 patch_df만 사용)
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

    # ── position_bin 통계 / NaN/Inf 검증 ─────────────────────────────────
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

    assert sample_mean_shape == (REDUCED_FEATURE_DIM,), f"mean shape 불일치: {sample_mean_shape}"
    assert sample_cov_shape == (REDUCED_FEATURE_DIM, REDUCED_FEATURE_DIM), f"cov shape 불일치: {sample_cov_shape}"
    assert total_nan == 0, f"NaN 발생: {total_nan}"
    assert total_inf == 0, f"Inf 발생: {total_inf}"

    # ── runtime CSV ──────────────────────────────────────────────────────
    ts = datetime.now().isoformat(timespec="seconds")
    record_runtime([
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "smoke_patient", "value": pid},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "n_patients_processed", "value": summary["n_patients_success"]},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "ct_shape", "value": str(stats_holder.get("ct_shape"))},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "roi_voxel", "value": stats_holder.get("roi_voxel")},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "patch_before_v4", "value": stats_holder.get("patch_before")},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "patch_after_v4", "value": stats_holder.get("patch_after")},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "patch_removed_by_v4", "value": stats_holder.get("patch_removed_by_v4")},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "n_patches_used", "value": summary["n_patches_used"]},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "n_patches_skipped", "value": summary["n_patches_skipped"]},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "position_bins_with_data", "value": n_bins_with_data},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "elapsed_seconds", "value": round(elapsed, 2)},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "peak_gpu_gb", "value": round(peak_gpu_gb, 3)},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "v4_patch_ratio_threshold", "value": V4_20_PATCH_RATIO_THRESHOLD},
    ])

    verdict = "통과"
    report = {
        "step": "P-B4",
        "verdict": verdict,
        "timestamp": ts,
        "branch": "efficientnet_b0_imagenet_chestwall_removed_roi_v1",
        "backbone": BACKBONE,
        "official_roi_source": "refined_roi_v4_20_modeB_all_v1",
        "smoke_patient": pid,
        "safe_id": safe_id,
        "limit": LIMIT,
        "p_b3_verdict": p_b3.get("verdict"),
        "p_b2_6_source_locked": True,
        "roi_connection": {
            "ct_source": "roi_0_0 ct_hu.npy (C드라이브, v4_20와 동일 좌표계)",
            "roi_source": "v4_20 refined_roi.npy",
            "patch_refilter": f"v4_20 ROI ratio >= {V4_20_PATCH_RATIO_THRESHOLD} (기존 roi_0_0 patch threshold와 동일)",
            "model_train_uses_mask": False,
            "note": "model.train()은 patch_df 좌표만 사용. v4_20 효과는 patch 재필터링으로 구현.",
        },
        "ct_shape": list(stats_holder.get("ct_shape")) if stats_holder.get("ct_shape") else None,
        "roi_shape": list(stats_holder.get("roi_shape")) if stats_holder.get("roi_shape") else None,
        "ct_roi_shape_match": stats_holder.get("ct_shape") == stats_holder.get("roi_shape"),
        "roi_voxel": stats_holder.get("roi_voxel"),
        "patch_before_v4": stats_holder.get("patch_before"),
        "patch_after_v4": stats_holder.get("patch_after"),
        "patch_removed_by_v4": stats_holder.get("patch_removed_by_v4"),
        "raw_feature_dim": RAW_FEATURE_DIM,
        "raw_feature_dim_verified": feature_extractor.raw_feature_dim == RAW_FEATURE_DIM,
        "selected_feature_dim": REDUCED_FEATURE_DIM,
        "selected_feature_dim_verified": model.feature_dim == REDUCED_FEATURE_DIM,
        "selected_index_source": "기존 EfficientNet-B0 branch 값 복사 (seed=42 random100 동일)",
        "mean_shape": list(sample_mean_shape) if sample_mean_shape else None,
        "cov_shape": list(sample_cov_shape) if sample_cov_shape else None,
        "position_bins_with_data": n_bins_with_data,
        "n_patches_used": summary["n_patches_used"],
        "n_patches_skipped": summary["n_patches_skipped"],
        "total_nan": total_nan,
        "total_inf": total_inf,
        "elapsed_seconds": round(elapsed, 2),
        "peak_gpu_gb": round(peak_gpu_gb, 3),
        "output_npz": str(SMOKE_NPZ),
        "safety": {
            "full_train": False, "val_test_scoring": False, "lesion_scoring": False,
            "threshold_calculated": False, "metrics_calculated": False,
            "stage2_holdout_accessed": False, "model_roi_used": False,
            "e_drive_used": False, "lesion_file_used": False,
            "existing_results_modified": False, "pip_install": False, "additional_download": False,
        },
        "next_step_p_b5_train_smoke_limit5_feasible": True,
    }
    with open(REPORTS_DIR / "p_b4_train_smoke_limit1.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    md = [
        "# P-B4 v4_20 ROI EfficientNet-B0 Train Smoke Limit=1\n",
        f"**판정: {verdict}**\n",
        f"- 생성일시: {ts}",
        f"- branch: efficientnet_b0_imagenet_chestwall_removed_roi_v1",
        f"- 백본: {BACKBONE} / ROI source: refined_roi_v4_20_modeB_all_v1\n",
        "## P-B3 lesion safety 요약\n",
        f"- P-B3 verdict: {p_b3.get('verdict')}",
        f"- complete lesion loss: {p_b3.get('complete_lesion_loss')}건",
        f"- aggregate preservation: {p_b3.get('preservation_distribution',{}).get('aggregate_preservation')}\n",
        "## ROI 연결 방식\n",
        "- CT: roi_0_0 ct_hu.npy (C드라이브, v4_20와 동일 좌표계)",
        "- ROI: v4_20 refined_roi.npy",
        f"- patch 재필터링: v4_20 ROI ratio >= {V4_20_PATCH_RATIO_THRESHOLD} (기존 roi_0_0 threshold와 동일)",
        "- ⚠ model.train()은 patch_df 좌표만 사용 → v4_20 효과는 patch 재필터링으로 구현\n",
        "## 처리 결과\n",
        f"- 처리 환자: {pid} (safe_id={safe_id})",
        f"- CT shape: {stats_holder.get('ct_shape')}",
        f"- ROI shape: {stats_holder.get('roi_shape')}",
        f"- CT/ROI shape 일치: {stats_holder.get('ct_shape') == stats_holder.get('roi_shape')}",
        f"- ROI voxel: {stats_holder.get('roi_voxel'):,}",
        f"- patch (v4 재필터링 전): {stats_holder.get('patch_before'):,}",
        f"- patch (v4 재필터링 후): {stats_holder.get('patch_after'):,}",
        f"- 흉벽 제거로 skip된 patch: {stats_holder.get('patch_removed_by_v4'):,}\n",
        "## Feature 차원\n",
        f"- raw_feature_dim=144: {feature_extractor.raw_feature_dim == 144}",
        f"- selected_dim=100: {model.feature_dim == 100}",
        f"- selected index: 기존 branch 값 복사 (seed=42 random100)",
        f"- mean shape: {sample_mean_shape}",
        f"- cov shape: {sample_cov_shape}\n",
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
        "- model_roi.npy 사용: False",
        "- E드라이브 사용: False",
        "- lesion 파일 사용: False",
        "- stage2_holdout 접근: False",
        "- full train / scoring / threshold / metrics: False",
        "- 기존 roi_0_0 / EfficientNet-B0 / P-B1~P-B3 결과 수정: False\n",
        "## 학습 범위\n",
        "- **normal train smoke limit=1** (1명만). full train 아님.\n",
        "## 다음 단계\n",
        "- P-B5 train smoke limit5 진행 가능: True",
    ]
    with open(REPORTS_DIR / "p_b4_train_smoke_limit1.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(f"\n=== P-B4 smoke limit1 완료: {verdict} ===")
    print(f"[보고서] {REPORTS_DIR}")


if __name__ == "__main__":
    main()
