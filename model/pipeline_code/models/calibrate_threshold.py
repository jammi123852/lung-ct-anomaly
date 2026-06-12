"""
calibrate_threshold.py
======================
LUNA16(.mhd/.raw) 정상 데이터로 PaDiM patch score를 집계하고
지정한 분위수(기본 p95)를 threshold로 저장합니다.

사용법:
    python calibrate_threshold.py \
        --dicom_dir C:\LUNA16 \
        --percentile 95
"""

import os
import sys
import glob
import argparse
import numpy as np
import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from position_aware_padim.padim_model import PaDiMModel
from position_aware_padim.feature_extractor_effnet_b0_scaffold import FeatureExtractorEffNetB0
from position_aware_padim.preprocessing import preprocess_ct_slice
from padim import _hu_lung_mask, assign_position_bin

WEIGHTS_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
NPZ_PATH     = os.path.join(WEIGHTS_DIR, "position_bin_stats.npz")
INDICES_PATH = os.path.join(WEIGHTS_DIR, "selected_feature_indices.npy")


def collect_all_scores(data_dir: str, padim_model, feature_extractor) -> list:
    mhd_files = glob.glob(os.path.join(data_dir, "**", "*.mhd"), recursive=True)
    if not mhd_files:
        mhd_files = glob.glob(os.path.join(data_dir, "*.mhd"))

    print(f"[INFO] Found {len(mhd_files)} MHD files")

    all_scores = []

    for path in mhd_files:
        try:
            sitk_img = sitk.ReadImage(path)
            vol      = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
            # vol shape: (Z, H, W)
            print(f"[INFO] {os.path.basename(path)} → shape={vol.shape}")

            for z in range(vol.shape[0]):
                hu_slice = vol[z]
                H, W     = hu_slice.shape

                pure_lung = _hu_lung_mask(hu_slice)
                if pure_lung.sum() < 100:
                    continue

                preprocessed = preprocess_ct_slice(hu_slice)

                patch_size, stride = 32, 16
                patch_coords = []
                for y0 in range(0, H - patch_size + 1, stride):
                    for x0 in range(0, W - patch_size + 1, stride):
                        cy = (y0 * 2 + patch_size) // 2
                        cx = (x0 * 2 + patch_size) // 2
                        if not pure_lung[cy, cx]:
                            continue
                        pm = pure_lung[y0:y0+patch_size, x0:x0+patch_size]
                        if pm.sum() >= patch_size * patch_size * 0.5:
                            patch_coords.append((y0, x0, y0+patch_size, x0+patch_size))

                if not patch_coords:
                    continue

                features_144 = feature_extractor.extract_patch_features(preprocessed, patch_coords)
                features_100 = features_144[:, padim_model.selected_feature_indices]

                for i, (y0, x0, y1, x1) in enumerate(patch_coords):
                    cy = (y0 + y1) / 2
                    cx = (x0 + x1) / 2
                    pb = assign_position_bin(cy, cx, H, W)
                    try:
                        score = padim_model.score_patch(features_100[i], pb)
                        all_scores.append(float(score))
                    except Exception:
                        continue

        except Exception as e:
            print(f"[WARN] 처리 실패 {path}: {e}")
            continue

    return all_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom_dir",  required=True,
                        help="LUNA16 데이터 폴더 (subset 포함 상위 폴더도 가능)")
    parser.add_argument("--percentile", type=float, default=95)
    parser.add_argument("--out_path",   default=os.path.join(WEIGHTS_DIR, "padim_threshold.npy"))
    args = parser.parse_args()

    print(f"[INFO] 모델 로드: {NPZ_PATH}")
    padim_model = PaDiMModel(
        selected_feature_indices_path=INDICES_PATH,
        feature_dim=100,
        eps=1e-5,
    )
    padim_model.load(NPZ_PATH)
    feature_extractor = FeatureExtractorEffNetB0()

    all_scores = collect_all_scores(args.dicom_dir, padim_model, feature_extractor)

    if not all_scores:
        print("[ERROR] 수집된 score가 없습니다. 경로를 확인하세요.")
        sys.exit(1)

    all_scores    = np.array(all_scores)
    new_threshold = float(np.percentile(all_scores, args.percentile))

    print(f"\n[RESULT] 총 패치 수 : {len(all_scores)}")
    print(f"[RESULT] Score 범위  : min={all_scores.min():.4f}, max={all_scores.max():.4f}")
    print(f"[RESULT] Score 평균  : {all_scores.mean():.4f}")
    print(f"[RESULT] P{int(args.percentile):02d} threshold: {new_threshold:.6f}")

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    np.save(args.out_path, np.array(new_threshold))
    print(f"\n[SAVED] threshold 저장 완료: {args.out_path}")
    print(f"[INFO] 서버를 재시작하면 새 threshold가 자동 적용됩니다.")


if __name__ == "__main__":
    main()