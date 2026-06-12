# ============================================================
# run.py
# ------------------------------------------------------------
# 진입점 함수:
#   preprocess_to_arrays(input_path, ...) -> (ct_hu, roi_0_0, meta)
#   run_full_preprocess(input_path, out_dir, ...) -> dict(경로들)
#
# 파이프라인 순서 (노트북 process_one_patient 기준):
#   1. volume load (DICOM or MHD)
#   2. orient LPS
#   3. save native CT NIfTI (TotalSegmentator 입력용 임시)
#   4. run TotalSegmentator on native CT
#   5. z-only 1mm resample
#   6. organ masks native -> 1mm (in-memory array)
#   7. TS lung guard (union 5 lobes + dilate x2)
#   8. body guard (HU 기반)
#   9. refined_lung (HU 기반) + TS lung guard 교집합
#   10. organ exclusion mask + pure_lung
#   11. lung z-range 찾기
#   12. z-range crop: ct_1mm_lung_range, pure_lung_lung_range
#   13. build_roi_0_0 on lung_range
#   14. meta dict 생성
# ============================================================

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from . import config as C
from .pipeline import (
    ensure_dir,
    safe_name,
    log_print,
    load_input_volume,
    orient_image,
    resample_z_only,
    array_to_sitk_like,
    run_totalsegmentator_native,
    resample_organ_masks_to_1mm,
    build_union_mask_from_organ_arrays,
    build_organ_exclusion_mask,
    build_body_guard_mask_3d,
    build_refined_lung_mask_3d,
    find_lung_z_range_from_pure_lung,
    crop_sitk_z_range,
    dilate_mask,
    build_roi_0_0,
)


# ============================================================
# 내부 헬퍼
# ============================================================

def _make_safe_id(patient_id: str) -> str:
    """patient_id + sha1 앞 10자리로 safe_id 생성."""
    h = hashlib.sha1(patient_id.encode("utf-8")).hexdigest()[:10]
    return f"{safe_name(patient_id)}__{h}"


def _infer_patient_id(input_path: Path) -> str:
    """input_path에서 patient_id를 추론."""
    input_path = Path(input_path)
    if input_path.is_dir():
        return safe_name(input_path.name)
    else:
        # .mhd 파일: parent_name + stem
        subset_name = safe_name(input_path.parent.name)
        case_name = safe_name(input_path.stem)
        return safe_name(f"{subset_name}_{case_name}")


# ============================================================
# 핵심 파이프라인
# ============================================================

def preprocess_to_arrays(
    input_path,
    patient_id: str = None,
    group: str = "Unknown",
    label: str = "unknown",
    totalseg_kwargs: dict = None,
    verbose: bool = False,
    _work_dir: Path = None,
) -> tuple:
    """
    한 환자 CT를 ct_hu.npy + roi_0_0.npy + meta dict으로 변환.
    TotalSegmentator 실행이 필요하므로 임시 디렉토리를 사용.

    Args:
        input_path : DICOM 폴더 경로 또는 .mhd 파일 경로
        patient_id : 환자 ID. None이면 input_path에서 추론
        group      : 그룹 이름 (meta.json group 필드)
        label      : 레이블 (meta.json label 필드, 예: "normal", "lesion_test")
        totalseg_kwargs : run_totalsegmentator_native에 전달할 옵션 dict
                          None이면 config.py 기본값 사용
        verbose    : 진행 로그 출력 여부
        _work_dir  : TotalSegmentator 임시 파일 저장 디렉토리.
                     None이면 tempfile.TemporaryDirectory 사용.

    Returns:
        (ct_hu, roi_0_0, meta)
          ct_hu   : np.ndarray int16, shape (Z,H,W)
          roi_0_0 : np.ndarray uint8 (0/1), shape (Z,H,W)
          meta    : dict (meta.json 포맷)
    """
    input_path = Path(input_path)

    if patient_id is None:
        patient_id = _infer_patient_id(input_path)

    if totalseg_kwargs is None:
        totalseg_kwargs = {}

    # --------------------------------------------------
    # 1. volume load
    # --------------------------------------------------
    log_print(f"[1] load volume: {input_path}", verbose, force=True)
    ct_raw = load_input_volume(input_path, verbose=verbose)

    raw_spacing = ct_raw.GetSpacing()
    raw_size = ct_raw.GetSize()
    log_print(f"    raw size: {raw_size}, spacing: {raw_spacing}", verbose)

    # --------------------------------------------------
    # 2. orient LPS
    # --------------------------------------------------
    log_print("[2] orient LPS", verbose)
    ct_native = orient_image(ct_raw, C.ORIENTATION)

    # --------------------------------------------------
    # 3. save native CT NIfTI (TotalSegmentator 입력용)
    #    작업 완료 후 삭제하거나 _work_dir에 유지
    # --------------------------------------------------
    use_temp = _work_dir is None
    if use_temp:
        import tempfile
        _tmp_ctx = tempfile.TemporaryDirectory(prefix="preprocess_ts_")
        work_dir = Path(_tmp_ctx.name)
    else:
        _tmp_ctx = None
        work_dir = Path(_work_dir)
        ensure_dir(work_dir)

    try:
        ct_native_path = work_dir / f"{safe_name(patient_id)}_native_lps.nii.gz"
        log_print(f"[3] save native NIfTI: {ct_native_path}", verbose)
        sitk.WriteImage(ct_native, str(ct_native_path))

        # --------------------------------------------------
        # 4. TotalSegmentator on native CT
        # --------------------------------------------------
        totalseg_out_dir = work_dir / "totalseg_native" / safe_name(patient_id)

        # log_dir 기본값: work_dir/logs
        if "log_dir" not in totalseg_kwargs:
            totalseg_kwargs = dict(totalseg_kwargs)
            totalseg_kwargs["log_dir"] = work_dir / "logs" / safe_name(patient_id)

        log_print(f"[4] run TotalSegmentator -> {totalseg_out_dir}", verbose, force=True)
        run_totalsegmentator_native(
            ct_native_path=ct_native_path,
            out_dir=totalseg_out_dir,
            totalseg_kwargs=totalseg_kwargs,
            verbose=verbose,
        )

        # --------------------------------------------------
        # 5. z-only 1mm resample
        # --------------------------------------------------
        log_print("[5] z-only 1mm resample", verbose)
        ct_1mm = resample_z_only(
            img=ct_native,
            target_z=C.TARGET_Z,
            interpolator=sitk.sitkLinear,
            default_value=-1024.0,
        )
        log_print(f"    1mm size: {ct_1mm.GetSize()}, spacing: {ct_1mm.GetSpacing()}", verbose)

        # --------------------------------------------------
        # 6. organ masks native -> 1mm (in-memory array)
        # --------------------------------------------------
        log_print("[6] resample organ masks to 1mm (in-memory)", verbose)
        organ_arrays = resample_organ_masks_to_1mm(
            totalseg_dir=totalseg_out_dir,
            ct_1mm=ct_1mm,
        )
        log_print(f"    loaded organs: {sorted(organ_arrays.keys())}", verbose)

        # --------------------------------------------------
        # 7. TS lung guard (union 5 lobes + dilate x2)
        # --------------------------------------------------
        log_print("[7] build TS lung guard", verbose)
        ct_arr = sitk.GetArrayFromImage(ct_1mm).astype(np.float32)

        ts_lung_guard, used_ts_lung_names = build_union_mask_from_organ_arrays(
            organ_arrays=organ_arrays,
            target_names=C.TS_LUNG_ROI_NAMES,
            reference_shape=ct_arr.shape,
            dilate_iter=int(C.TS_LUNG_GUARD_DILATE_ITER),
        )

        if ts_lung_guard is None or ts_lung_guard.sum() == 0:
            message = f"TS lung guard 생성 실패: {patient_id}"
            if C.STRICT_TS_LUNG_GUARD:
                raise RuntimeError(message)
            print("[WARN]", message)
            ts_lung_guard = np.ones(ct_arr.shape, dtype=bool)
            used_ts_lung_names = []

        # --------------------------------------------------
        # 8. body guard (HU 기반)
        # --------------------------------------------------
        log_print("[8] build body guard", verbose)
        body_guard = build_body_guard_mask_3d(
            ct_arr=ct_arr,
            hu_threshold=float(C.BODY_GUARD_HU_THRESHOLD),
        )

        # --------------------------------------------------
        # 9. refined_lung (HU 기반) + TS lung guard 교집합
        # --------------------------------------------------
        log_print("[9] build refined lung mask", verbose)
        refined_lung_raw = build_refined_lung_mask_3d(
            ct_arr=ct_arr,
            hu_min=C.HU_MIN,
            hu_max=C.HU_MAX,
            verbose=verbose,
        )

        if C.USE_TS_LUNG_GUARD:
            refined_lung = refined_lung_raw & ts_lung_guard
        else:
            refined_lung = refined_lung_raw

        # --------------------------------------------------
        # 10. organ exclusion + pure_lung
        # --------------------------------------------------
        log_print("[10] build organ exclusion + pure lung", verbose)
        organ_exclusion, used_organs = build_organ_exclusion_mask(
            organ_arrays=organ_arrays,
            organ_exclusion_names=C.ORGAN_EXCLUSION_ROI_NAMES,
            dilate_iter=int(C.ORGAN_EXCLUSION_DILATE_ITER),
        )

        pure_lung = refined_lung & (~organ_exclusion)

        # --------------------------------------------------
        # 11. lung z-range 찾기
        # --------------------------------------------------
        log_print("[11] find lung z-range", verbose)
        lung_range_info = find_lung_z_range_from_pure_lung(
            pure_lung=pure_lung,
            min_area_ratio=C.LUNG_RANGE_MIN_PURE_LUNG_AREA_RATIO,
            margin_slices=C.LUNG_RANGE_MARGIN_SLICES,
            max_gap_slices=C.LUNG_RANGE_MAX_GAP_SLICES,
            min_segment_slices=C.LUNG_RANGE_MIN_SEGMENT_SLICES,
        )

        lung_z_start = int(lung_range_info["z_start"])
        lung_z_end = int(lung_range_info["z_end"])

        log_print(
            f"    z_start={lung_z_start}, z_end={lung_z_end}, "
            f"found={lung_range_info['found_lung_range']}, reason={lung_range_info['reason']}",
            verbose,
        )

        # --------------------------------------------------
        # 12. z-range crop: ct_1mm_lung_range
        # --------------------------------------------------
        log_print("[12] crop to lung z-range", verbose)
        crop_enabled = bool(getattr(C, "LUNG_CROP_ENABLED", True))
        if crop_enabled:
            ct_lung_range_img = crop_sitk_z_range(ct_1mm, lung_z_start, lung_z_end)
        else:
            log_print("    LUNG_CROP_ENABLED=False → 전체 1mm 볼륨 사용(크롭 안 함, 폐 손실 0)", verbose, force=True)
            ct_lung_range_img = ct_1mm

        # ct_hu: int16 저장 (노트북 기준: sitkLinear 보간 소수점 HU는 반올림되어 저장)
        ct_hu = sitk.GetArrayFromImage(ct_lung_range_img).astype(np.int16)

        # pure_lung (roi_0_0 입력용; 크롭 비활성 시 전체)
        if crop_enabled:
            pure_lung_lung_range = pure_lung[lung_z_start:lung_z_end + 1, :, :]
        else:
            pure_lung_lung_range = pure_lung

        # --------------------------------------------------
        # 13. build_roi_0_0
        # ------------------------------------------------
        # ref_02 기준 (no_dilate 버전):
        #   - raw_ts_lung: 폐엽 5개 합침, dilation 없음 (TS_LUNG_DILATE_ITER=0)
        #   - use_body_guard=False
        #   - use_organ_exclusion=True, dilate=1
        # --------------------------------------------------
        log_print("[13] build roi_0_0 (no_dilate)", verbose)

        # TS lung 5 lobes union, dilation 없음 (ROI_TS_LUNG_DILATE_ITER=0)
        raw_ts_lung_full, _ = build_union_mask_from_organ_arrays(
            organ_arrays=organ_arrays,
            target_names=C.TS_LUNG_ROI_NAMES,
            reference_shape=ct_arr.shape,
            dilate_iter=int(C.ROI_TS_LUNG_DILATE_ITER),  # 0 — no_dilate
        )

        if raw_ts_lung_full is None:
            raise RuntimeError(f"폐엽 마스크를 하나도 못 찾음: {patient_id}")

        # organ exclusion for roi_0_0 (dilate=1)
        organ_exclusion_roi, _ = build_organ_exclusion_mask(
            organ_arrays=organ_arrays,
            organ_exclusion_names=C.ORGAN_EXCLUSION_ROI_NAMES,
            dilate_iter=int(C.ROI_ORGAN_EXCLUSION_DILATE_ITER),
        )

        roi_0_0_full = build_roi_0_0(
            raw_ts_lung=raw_ts_lung_full,
            body_guard=None,                    # use_body_guard=False (no_dilate 기준)
            organ_exclusion=organ_exclusion_roi,
            use_body_guard=C.ROI_USE_BODY_GUARD,
            use_organ_exclusion=C.ROI_USE_ORGAN_EXCLUSION,
        )

        # lung_range로 crop (크롭 비활성 시 전체)
        if crop_enabled:
            roi_0_0_crop = roi_0_0_full[lung_z_start:lung_z_end + 1, :, :]
        else:
            roi_0_0_crop = roi_0_0_full

        # roi_0_0: uint8
        roi_0_0 = roi_0_0_crop.astype(np.uint8)

        # --------------------------------------------------
        # 14. meta dict 생성
        # --------------------------------------------------
        safe_id = _make_safe_id(f"{group}_{patient_id}")

        # spacing / origin / direction은 crop된 ct_lung_range_img에서 추출
        spacing_xyz = list(ct_lung_range_img.GetSpacing())   # (sx, sy, sz)
        origin_xyz = list(ct_lung_range_img.GetOrigin())     # (ox, oy, oz)
        direction = list(ct_lung_range_img.GetDirection())   # 9-element flat

        meta = {
            "group": str(group),
            "patient_id": str(patient_id),
            "safe_id": str(safe_id),
            "label": str(label),
            "shape_zyx": list(ct_hu.shape),    # [Z, H, W]
            "ct_dtype": "int16",
            "mask_dtype": "uint8",
            "spacing_xyz": spacing_xyz,
            "origin_xyz": origin_xyz,
            "direction": direction,
        }

        log_print(f"[14] meta: {meta}", verbose)
        log_print("[DONE] preprocess_to_arrays 완료", verbose, force=True)

        return ct_hu, roi_0_0, meta

    finally:
        if _tmp_ctx is not None:
            _tmp_ctx.cleanup()


# ============================================================
# 디스크 저장 진입점
# ============================================================

def run_full_preprocess(
    input_path,
    out_dir,
    patient_id: str = None,
    group: str = "Unknown",
    label: str = "unknown",
    totalseg_kwargs: dict = None,
    verbose: bool = False,
) -> dict:
    """
    preprocess_to_arrays를 호출한 뒤 ct_hu.npy, roi_0_0.npy, meta.json을
    out_dir/<safe_id>/ 아래에 저장.

    Args:
        input_path : DICOM 폴더 또는 .mhd 파일
        out_dir    : npy/json 저장 루트 디렉토리
        patient_id : None이면 input_path에서 추론
        group      : meta.json group 필드
        label      : meta.json label 필드
        totalseg_kwargs : TotalSegmentator 옵션 (None이면 config 기본값)
        verbose    : 로그 출력 여부

    Returns:
        dict with keys:
            ct_hu_npy    : Path
            roi_0_0_npy  : Path
            meta_json    : Path
            safe_id      : str
            meta         : dict
    """
    input_path = Path(input_path)
    out_dir = Path(out_dir)

    # TotalSegmentator 임시 파일은 out_dir/ts_work/ 에 두어 재사용 가능하게 함
    work_dir = out_dir / "_ts_work"

    ct_hu, roi_0_0, meta = preprocess_to_arrays(
        input_path=input_path,
        patient_id=patient_id,
        group=group,
        label=label,
        totalseg_kwargs=totalseg_kwargs,
        verbose=verbose,
        _work_dir=work_dir,
    )

    safe_id = meta["safe_id"]
    patient_out = out_dir / safe_id
    ensure_dir(patient_out)

    ct_hu_npy = patient_out / "ct_hu.npy"
    roi_0_0_npy = patient_out / "roi_0_0.npy"
    meta_json = patient_out / "meta.json"

    np.save(str(ct_hu_npy), ct_hu)
    np.save(str(roi_0_0_npy), roi_0_0)

    # meta에 npy 경로 추가 (reference meta.json 포맷 참고)
    meta_to_save = dict(meta)
    meta_to_save["ct_hu_npy"] = str(ct_hu_npy)
    meta_to_save["roi_0_0_npy"] = str(roi_0_0_npy)

    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta_to_save, f, indent=2, ensure_ascii=False)

    print(f"[SAVED] {patient_out}")
    print(f"  ct_hu.npy   : {ct_hu.shape} {ct_hu.dtype}")
    print(f"  roi_0_0.npy : {roi_0_0.shape} {roi_0_0.dtype}")
    print(f"  meta.json   : {meta_json}")

    return {
        "ct_hu_npy": ct_hu_npy,
        "roi_0_0_npy": roi_0_0_npy,
        "meta_json": meta_json,
        "safe_id": safe_id,
        "meta": meta_to_save,
    }


# ============================================================
# CLI 진입점
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="폐 CT 전처리: DICOM/MHD -> ct_hu.npy + roi_0_0.npy + meta.json"
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="DICOM 폴더 경로 또는 .mhd 파일 경로",
    )
    parser.add_argument(
        "out_dir",
        type=str,
        help="출력 디렉토리 (npy, json 저장)",
    )
    parser.add_argument(
        "--patient_id",
        type=str,
        default=None,
        help="환자 ID (기본값: input_path에서 자동 추론)",
    )
    parser.add_argument(
        "--group",
        type=str,
        default="Unknown",
        help="그룹 이름 (meta.json group 필드, 기본: Unknown)",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="unknown",
        help="레이블 (meta.json label 필드, 기본: unknown)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="상세 로그 출력",
    )

    args = parser.parse_args()

    result = run_full_preprocess(
        input_path=args.input_path,
        out_dir=args.out_dir,
        patient_id=args.patient_id,
        group=args.group,
        label=args.label,
        verbose=args.verbose,
    )

    print("\n[RESULT]")
    for k, v in result.items():
        if k != "meta":
            print(f"  {k}: {v}")
