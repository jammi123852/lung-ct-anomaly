"""
P-A67b: EfficientNet-B0 weight/local-cache check + 1-slice forward smoke.

- weight 캐시 확인 → 없으면 torchvision 공식 weight 1회 다운로드
- normal train 1명 / 1 slice forward (FeatureExtractorEffNetB0)
- tap point feature map shape 실측
- raw_feature_dim=144 검증
- selected_feature_indices 생성 금지 / train/scoring/metrics 금지
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# --- 프로젝트 루트를 sys.path에 추가 ---
PROJ_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJ_ROOT))

from src.position_aware_padim.preprocessing import preprocess_ct_slice

# --- 경로 설정 ---
NORMAL_TRAIN_ROOT = Path(
    "/mnt/c/Users/jinhy/Desktop/"
    "Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1"
)
SMOKE_PATIENT_ID  = "normal001__104e7cb873"
SMOKE_SLICE_IDX   = 122  # 중간 slice 1장

OUTPUT_ROOT = PROJ_ROOT / "experiments/efficientnet_b0_imagenet_v1/outputs/reports/p_a67b_weight_and_1slice_smoke"

# --- EfficientNet-B0 weight 설정 ---
from torchvision.models import EfficientNet_B0_Weights
EFFNET_WEIGHTS_ENUM = EfficientNet_B0_Weights.IMAGENET1K_V1
CACHE_PATH = Path(torch.hub.get_dir()) / "checkpoints" / os.path.basename(EFFNET_WEIGHTS_ENUM.url)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    output_dir = OUTPUT_ROOT
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 가드 1: P-A67a verdict 확인 ---
    p67a_json = PROJ_ROOT / "experiments/efficientnet_b0_imagenet_v1/outputs/reports/p_a67a_effnet_b0_scaffold_preflight.json"
    assert p67a_json.exists(), f"P-A67a JSON 없음: {p67a_json}"
    with open(p67a_json) as f:
        p67a = json.load(f)
    assert p67a.get("verdict") == "pass", f"P-A67a verdict != pass: {p67a.get('verdict')}"
    print("[가드1] P-A67a verdict: pass ✅")

    # --- 가드 2: 기존 P-A67b 결과 덮어쓰기 방지 ---
    existing = list(output_dir.glob("p_a67b_*.md")) + list(output_dir.glob("p_a67b_*.json"))
    assert len(existing) == 0, f"기존 P-A67b 결과 존재 — 덮어쓰기 방지: {existing}"
    print("[가드2] 기존 P-A67b 결과 없음 ✅")

    # --- Step 1: weight 캐시 확인 / 다운로드 ---
    print(f"\n[Step1] EfficientNet-B0 weight cache: {CACHE_PATH}")
    download_performed = False
    if not CACHE_PATH.exists():
        print("  캐시 없음 → 공식 torchvision weight 1회 다운로드 중...")
        t0 = time.time()
        from torchvision.models import efficientnet_b0
        efficientnet_b0(weights=EFFNET_WEIGHTS_ENUM)  # 다운로드 트리거
        elapsed = time.time() - t0
        print(f"  다운로드 완료: {elapsed:.1f}초")
        download_performed = True
    else:
        print("  캐시 존재 ✅")

    assert CACHE_PATH.exists(), "다운로드 후에도 캐시 없음"
    weight_size_bytes = CACHE_PATH.stat().st_size
    weight_sha256     = sha256_file(CACHE_PATH)
    print(f"  파일 크기: {weight_size_bytes / 1e6:.2f} MB")
    print(f"  sha256: {weight_sha256}")

    # weight manifest 저장
    manifest = {
        "weight_source": "torchvision_official",
        "weight_enum": "EfficientNet_B0_Weights.IMAGENET1K_V1",
        "weight_url": EFFNET_WEIGHTS_ENUM.url,
        "weight_cache_path": str(CACHE_PATH),
        "weight_size_bytes": weight_size_bytes,
        "weight_size_mb": round(weight_size_bytes / 1e6, 2),
        "weight_sha256": weight_sha256,
        "download_performed": download_performed,
        "unofficial_mirror_used": False,
    }
    with open(output_dir / "effnet_b0_weight_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("  weight manifest 저장 완료")

    # --- Step 2: FeatureExtractorEffNetB0 instantiation ---
    print("\n[Step2] FeatureExtractorEffNetB0 instantiation")
    from src.position_aware_padim.feature_extractor_effnet_b0_scaffold import (
        FeatureExtractorEffNetB0, RAW_FEATURE_DIM, EFFNET_B0_LAYER_CHANNELS, EFFNET_B0_STRIDES,
    )
    t0 = time.time()
    extractor = FeatureExtractorEffNetB0(device=None)
    elapsed = time.time() - t0
    print(f"  instantiation 완료: {elapsed:.2f}초, device={extractor.device}")
    print(f"  raw_feature_dim (설계): {RAW_FEATURE_DIM}")

    # --- Step 3: normal train 1명 / 1 slice 로드 ---
    print(f"\n[Step3] normal train 1 slice 로드: {SMOKE_PATIENT_ID}, slice_idx={SMOKE_SLICE_IDX}")
    ct_path = NORMAL_TRAIN_ROOT / "volumes_npy" / SMOKE_PATIENT_ID / "ct_hu.npy"
    assert ct_path.exists(), f"ct_hu.npy 없음: {ct_path}"
    ct_vol = np.load(str(ct_path), mmap_mode="r")
    print(f"  ct volume shape: {ct_vol.shape}, dtype: {ct_vol.dtype}")

    slice_2d = ct_vol[SMOKE_SLICE_IDX].astype(np.float32)
    print(f"  slice_2d shape: {slice_2d.shape}")

    # preprocess: HU windowing → 3ch ImageNet normalize
    slice_3ch = preprocess_ct_slice(slice_2d)
    print(f"  slice_3ch shape: {slice_3ch.shape}, dtype: {slice_3ch.dtype}")
    assert slice_3ch.shape[0] == 3, f"3ch 아님: {slice_3ch.shape}"

    # --- Step 4: 1 slice forward smoke ---
    print("\n[Step4] 1 slice forward smoke")
    t0 = time.time()

    # GPU memory 기록 (before)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / 1e6

    slice_features = extractor.extract_slice_features(slice_3ch)

    elapsed = time.time() - t0
    if torch.cuda.is_available():
        mem_peak = torch.cuda.max_memory_allocated() / 1e6
    else:
        mem_peak = None

    print(f"  forward 완료: {elapsed:.3f}초")
    if mem_peak is not None:
        print(f"  GPU peak memory: {mem_peak:.1f} MB")

    # --- Step 5: feature shape 확인 ---
    print("\n[Step5] tap point feature shape 확인")
    H, W = slice_2d.shape
    results = []
    for tap_name, tap_key, expected_ch, expected_stride in [
        ("early", "early", EFFNET_B0_LAYER_CHANNELS[0], EFFNET_B0_STRIDES[0]),
        ("mid",   "mid",   EFFNET_B0_LAYER_CHANNELS[1], EFFNET_B0_STRIDES[1]),
        ("late",  "late",  EFFNET_B0_LAYER_CHANNELS[2], EFFNET_B0_STRIDES[2]),
    ]:
        fmap = slice_features[tap_key]
        actual_ch, actual_h, actual_w = fmap.shape
        expected_h = H // expected_stride
        expected_w = W // expected_stride

        ch_ok     = (actual_ch == expected_ch)
        shape_ok  = (actual_h == expected_h and actual_w == expected_w)
        stride_ok = shape_ok  # stride 확인은 feature map 크기로 간접 검증

        print(f"  [{tap_name}] shape={fmap.shape}, expected=({expected_ch}, {expected_h}, {expected_w}), ch_ok={ch_ok}, shape_ok={shape_ok}")
        results.append({
            "tap_point": tap_name,
            "expected_channel": expected_ch,
            "actual_channel": int(actual_ch),
            "expected_stride": expected_stride,
            "expected_h": expected_h,
            "expected_w": expected_w,
            "actual_h": int(actual_h),
            "actual_w": int(actual_w),
            "channel_match": ch_ok,
            "shape_match": shape_ok,
            "stride_verified": stride_ok,
        })

    # --- Step 6: raw_feature_dim 실측 ---
    print("\n[Step6] raw_feature_dim 실측")
    concat_dim = sum(r["actual_channel"] for r in results)
    design_dim = RAW_FEATURE_DIM
    dim_match = (concat_dim == design_dim)
    print(f"  concat_dim={concat_dim}, design_dim={design_dim}, match={dim_match}")

    all_ok = all(r["channel_match"] and r["shape_match"] for r in results) and dim_match
    verdict = "pass" if all_ok else "fail"
    print(f"\n  최종 판정: {verdict.upper()}")

    # --- 출력 파일 저장 ---
    import csv
    csv_path = output_dir / "effnet_b0_1slice_feature_shape_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  CSV 저장: {csv_path}")

    # JSON 결과
    result_json = {
        "report_id": "p_a67b",
        "generated": "2026-06-02",
        "verdict": verdict,
        "guards": {
            "p_a67a_verdict_pass": True,
            "no_existing_p_a67b_overwrite": True,
            "official_weight_only": True,
            "unofficial_mirror_used": False,
            "normal_train_1_patient_1_slice_only": True,
            "lesion_data_accessed": False,
            "stage2_holdout_accessed": False,
            "selected_feature_indices_generated": False,
            "train_executed": False,
            "scoring_executed": False,
            "threshold_calculated": False,
            "metrics_calculated": False,
            "existing_results_modified": False,
            "pip_conda_install": False,
        },
        "weight": manifest,
        "smoke_input": {
            "patient_id": SMOKE_PATIENT_ID,
            "split": "train",
            "slice_idx": SMOKE_SLICE_IDX,
            "ct_volume_shape": list(ct_vol.shape),
            "slice_2d_shape": list(slice_2d.shape),
            "slice_3ch_shape": list(slice_3ch.shape),
        },
        "tap_points": results,
        "raw_feature_dim": {
            "design": design_dim,
            "measured": concat_dim,
            "match": dim_match,
        },
        "runtime": {
            "forward_seconds": round(elapsed, 3),
            "gpu_peak_memory_mb": round(mem_peak, 1) if mem_peak else None,
            "device": extractor.device,
        },
        "next_step": {
            "p_a68_proceed_allowed": all_ok,
            "reason": "raw_feature_dim=144 verified" if all_ok else "shape mismatch — check tap points",
        },
    }
    json_path = output_dir / "p_a67b_effnet_b0_weight_and_1slice_smoke.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)
    print(f"  JSON 저장: {json_path}")

    print("\n[완료] P-A67b smoke test 종료")
    return result_json, results


if __name__ == "__main__":
    result_json, results = main()
    print("\n--- 요약 ---")
    print(f"verdict : {result_json['verdict']}")
    print(f"weight  : {result_json['weight']['weight_size_mb']} MB, sha256: {result_json['weight']['weight_sha256'][:16]}...")
    print(f"raw_dim : design={result_json['raw_feature_dim']['design']}, measured={result_json['raw_feature_dim']['measured']}, match={result_json['raw_feature_dim']['match']}")
