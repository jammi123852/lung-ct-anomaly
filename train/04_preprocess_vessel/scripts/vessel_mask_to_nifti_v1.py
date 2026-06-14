#!/usr/bin/env python
"""
vessel_mask_to_nifti_v1.py

목적:
  슬라이스별 2D 혈관 마스크(.npz: key 'vessel_mask', 512x512 uint8 binary)를
  z축으로 쌓아 3D NIfTI(.nii.gz)로 저장한다.
  3D Slicer 등에서 입체(surface/volume rendering)로 혈관 구조를 보기 위함.

입력 (read-only):
  --src   환자 폴더 (slice_XXXX_vessel_candidate_softmask.npz 들이 들어있음)

출력:
  --out   3D NIfTI 파일 1개 (.nii.gz)

주의:
  - slice 번호가 불연속이면, min~max 전체 z grid를 만들고
    마스크가 없는 z는 0(빈 슬라이스)으로 채운다 -> 기하가 어긋나지 않음.
  - metadata에 실제 mm voxel spacing이 없어 spacing은 (1,1,1) 기본값.
    실제 비율은 3D Slicer 'Volumes' 모듈에서 Image Spacing으로 조정 가능.
"""

import argparse
import glob
import os
import re

import numpy as np
import SimpleITK as sitk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="환자 폴더 경로")
    ap.add_argument("--out", required=True, help="출력 .nii.gz 경로")
    ap.add_argument("--key", default="vessel_mask", help="npz 안의 array key")
    ap.add_argument("--spacing", default="1,1,1",
                    help="voxel spacing 'sx,sy,sz' (mm). 기본 1,1,1")
    args = ap.parse_args()

    fs = sorted(glob.glob(os.path.join(args.src, "*_vessel_candidate_softmask.npz")))
    if not fs:
        # binary 형식 폴더도 지원
        fs = sorted(glob.glob(os.path.join(args.src, "*_vessel_binary.npy")))
        is_npy = True
    else:
        is_npy = False

    if not fs:
        raise SystemExit(f"[ERR] 혈관 마스크 파일을 찾지 못함: {args.src}")

    def load(f):
        if is_npy:
            return np.load(f)
        return np.load(f)[args.key]

    # slice 번호 추출
    def snum(f):
        m = re.search(r"slice_(\d+)", os.path.basename(f))
        return int(m.group(1))

    pairs = sorted(((snum(f), f) for f in fs), key=lambda x: x[0])
    zmin, zmax = pairs[0][0], pairs[-1][0]
    sample = load(pairs[0][1])
    H, W = sample.shape
    depth = zmax - zmin + 1

    print(f"[INFO] 파일수={len(fs)}  z범위={zmin}~{zmax}  full depth={depth}  (H,W)=({H},{W})")

    # (Z, H, W) grid, 빈 z는 0
    vol = np.zeros((depth, H, W), dtype=np.uint8)
    filled = 0
    for z, f in pairs:
        a = load(f).astype(np.uint8)
        vol[z - zmin] = (a > 0).astype(np.uint8)
        filled += 1
    print(f"[INFO] 채운 슬라이스={filled} / 전체 z={depth} (빈 z={depth-filled})")
    print(f"[INFO] 혈관 voxel 총합={int(vol.sum())}")

    # SimpleITK: numpy (Z,H,W) -> image (x=W,y=H,z=Z)
    img = sitk.GetImageFromArray(vol)
    sx, sy, sz = [float(v) for v in args.spacing.split(",")]
    img.SetSpacing((sx, sy, sz))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sitk.WriteImage(img, args.out)
    print(f"[OK] 저장 완료: {args.out}")
    print(f"     spacing(mm)=({sx},{sy},{sz})  -> 3D Slicer에서 조정 가능")


if __name__ == "__main__":
    main()
