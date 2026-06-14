"""
DataValidator: patient_manifest 기반으로 training-ready 데이터 구조 무결성을 검증한다.

설계 원칙:
- PathResolver 인스턴스를 생성자 인자로 받는다 (내부 생성 금지).
- {patient_id} 문자열 직접 조립 금지. 모든 경로는 PathResolver.resolve() 경유.
- patient_manifest.csv는 encoding='utf-8-sig'로 읽는다.
- duplicate patient_id 발견 시 조용히 덮어쓰지 않고 error로 기록한다.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .path_resolver import PathResolver


# 검증 대상 파일의 logical key
_FILE_LOGICAL_KEYS = ["ct_hu", "pure_lung", "meta", "patch_index"]

# data_validation_summary.csv 컬럼
SUMMARY_COLUMNS = [
    "patient_id",
    "safe_id",
    "ct_hu_exists",
    "pure_lung_exists",
    "meta_exists",
    "patch_csv_exists",
    "shape_match",
    "error_msg",
]

# error.csv 컬럼
ERROR_COLUMNS = [
    "patient_id",
    "error_type",
    "error_msg",
    "file_logical",
]

# ---------------------------------------------------------------------------
# lesion 테스트 데이터셋 검증용 상수 — v1 (model_roi 기반)
# ---------------------------------------------------------------------------

# 환자 폴더 내 검증 대상 파일.
# ct_hu / pure_lung / meta 는 PathResolver.resolve 경유.
# model_roi / lesion_mask 는 PathResolver 무수정 정책에 따라 volume_dir + 고정 파일명으로 확인.
_LESION_VOLUME_FILES = {
    "ct_hu": "ct_hu.npy",
    "model_roi": "model_roi.npy",
    "lesion_mask": "lesion_mask_model_roi.npy",
    "pure_lung": "pure_lung.npy",
    "meta": "meta.json",
}

# shape 일치 확인 대상 (volume npy) — v1
_LESION_SHAPE_KEYS = ["ct_hu", "model_roi", "lesion_mask", "pure_lung"]

# patch CSV 필수 컬럼 (id 컬럼은 patient_id 또는 safe_id 중 하나 이상)
_LESION_PATCH_REQUIRED = ["local_z", "y0", "x0", "y1", "x1", "position_bin"]
_LESION_PATCH_ID_ANY = ["patient_id", "safe_id"]

# patch CSV lesion 평가용 권장 컬럼 — v1
_LESION_PATCH_RECOMMENDED = [
    "lesion_pixels",
    "lesion_patch_ratio",
    "has_lesion_patch",
    "lesion_zone_type",
    "zone_type",
    "pure_lung_patch_ratio",
    "central_distance_ratio_mean",
]

# lesion_path_validation_summary.csv 컬럼 — v1
LESION_SUMMARY_COLUMNS = [
    "patient_id",
    "safe_id",
    "group",
    "ct_hu_exists",
    "model_roi_exists",
    "lesion_mask_exists",
    "pure_lung_exists",
    "meta_exists",
    "patch_csv_exists",
    "shape_checked",
    "shape_match",
    "error_msg",
]

# ---------------------------------------------------------------------------
# lesion 테스트 데이터셋 검증용 상수 — v2 (roi_0_0 기반)
# ---------------------------------------------------------------------------

# v2: model_roi/pure_lung 없음. roi_0_0 + lesion_mask_roi_0_0 사용.
_LESION_VOLUME_FILES_V2 = {
    "ct_hu": "ct_hu.npy",
    "roi_0_0": "roi_0_0.npy",
    "lesion_mask": "lesion_mask_roi_0_0.npy",
    "meta": "meta.json",
}

# shape 일치 확인 대상 (volume npy) — v2
_LESION_SHAPE_KEYS_V2 = ["ct_hu", "roi_0_0", "lesion_mask"]

# patch CSV lesion 평가용 권장 컬럼 — v2 (pure_lung_patch_ratio 없음, roi_0_0_patch_ratio 사용)
_LESION_PATCH_RECOMMENDED_V2 = [
    "lesion_pixels",
    "lesion_patch_ratio",
    "has_lesion_patch",
    "lesion_zone_type",
    "roi_0_0_patch_ratio",
    "slice_roi_0_0_ratio",
    "central_distance_ratio_mean",
]

# lesion_path_validation_summary.csv 컬럼 — v2
LESION_SUMMARY_COLUMNS_V2 = [
    "patient_id",
    "safe_id",
    "group",
    "ct_hu_exists",
    "roi_0_0_exists",
    "lesion_mask_exists",
    "meta_exists",
    "patch_csv_exists",
    "shape_checked",
    "shape_match",
    "error_msg",
]


class DataValidator:
    """Training-ready 데이터 구조의 무결성을 검증한다."""

    def __init__(self, path_resolver: PathResolver) -> None:
        """
        Parameters
        ----------
        path_resolver : PathResolver
            sample_check 완료 후 확정된 PathResolver 인스턴스.
        """
        self.path_resolver = path_resolver

    def _load_manifest_rows(
        self, manifest_path: str
    ) -> Tuple[List[dict], List[dict]]:
        """
        patient_manifest.csv를 읽어 행 목록을 반환한다.
        duplicate patient_id는 error로 기록하고 첫 번째 행만 유지한다.

        Returns
        -------
        (rows, dup_errors)
            rows       : duplicate를 제거한 행 목록
            dup_errors : duplicate 발생 정보 (error.csv용)
        """
        seen: dict[str, int] = {}  # patient_id → 첫 등장 행 번호
        rows: List[dict] = []
        dup_errors: List[dict] = []

        with open(manifest_path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for line_num, row in enumerate(reader, start=2):  # 헤더가 1행
                pid = (row.get("patient_id") or "").strip()
                if not pid:
                    continue
                if pid in seen:
                    dup_errors.append(
                        {
                            "patient_id": pid,
                            "error_type": "duplicate_patient_id",
                            "error_msg": (
                                f"patient_id '{pid}'가 manifest에 중복됨 "
                                f"(첫 등장 행={seen[pid]}, 현재 행={line_num})"
                            ),
                            "file_logical": "",
                        }
                    )
                else:
                    seen[pid] = line_num
                    rows.append(dict(row))

        return rows, dup_errors

    def validate_structure(
        self,
        limit: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Training-ready 데이터 구조를 검증한다.

        Parameters
        ----------
        limit : int | None
            검증할 최대 환자 수. None이면 전체.

        Returns
        -------
        (summary_df, error_df)
            summary_df : data_validation_summary.csv에 저장될 DataFrame
            error_df   : error.csv에 저장될 DataFrame
        """
        manifest_path = str(
            Path(self.path_resolver.base_path) / "manifests" / "patient_manifest.csv"
        )

        rows, dup_errors = self._load_manifest_rows(manifest_path)

        if limit is not None:
            rows = rows[:limit]

        summary_records: List[dict] = []
        error_records: List[dict] = list(dup_errors)

        for row in rows:
            pid = (row.get("patient_id") or "").strip()
            safe_id = (row.get("safe_id") or "").strip()

            record: dict = {
                "patient_id": pid,
                "safe_id": safe_id,
                "ct_hu_exists": False,
                "pure_lung_exists": False,
                "meta_exists": False,
                "patch_csv_exists": False,
                "shape_match": None,
                "error_msg": "",
            }

            # 각 파일 경로 획득 및 존재 확인
            resolved: dict[str, Optional[str]] = {}
            local_errors: List[str] = []

            key_to_field = {
                "ct_hu": "ct_hu_exists",
                "pure_lung": "pure_lung_exists",
                "meta": "meta_exists",
                "patch_index": "patch_csv_exists",
            }

            for logical_key, field in key_to_field.items():
                try:
                    path_str = self.path_resolver.resolve(pid, logical_key)
                    resolved[logical_key] = path_str
                    record[field] = Path(path_str).exists()
                except (KeyError, ValueError) as exc:
                    resolved[logical_key] = None
                    record[field] = False
                    local_errors.append(f"{logical_key}: {exc}")
                    error_records.append(
                        {
                            "patient_id": pid,
                            "error_type": "path_resolve_error",
                            "error_msg": str(exc),
                            "file_logical": logical_key,
                        }
                    )

            # ct_hu.npy 와 pure_lung.npy shape 일치 확인
            ct_path = resolved.get("ct_hu")
            lung_path = resolved.get("pure_lung")
            if ct_path and lung_path and Path(ct_path).exists() and Path(lung_path).exists():
                try:
                    ct_shape = np.load(ct_path, mmap_mode="r").shape
                    lung_shape = np.load(lung_path, mmap_mode="r").shape
                    record["shape_match"] = ct_shape == lung_shape
                    if not record["shape_match"]:
                        msg = f"shape mismatch: ct_hu={ct_shape} pure_lung={lung_shape}"
                        local_errors.append(msg)
                        error_records.append(
                            {
                                "patient_id": pid,
                                "error_type": "shape_mismatch",
                                "error_msg": msg,
                                "file_logical": "ct_hu+pure_lung",
                            }
                        )
                except Exception as exc:
                    record["shape_match"] = False
                    msg = f"shape 확인 실패: {exc}"
                    local_errors.append(msg)
                    error_records.append(
                        {
                            "patient_id": pid,
                            "error_type": "shape_load_error",
                            "error_msg": msg,
                            "file_logical": "ct_hu+pure_lung",
                        }
                    )
            else:
                record["shape_match"] = None  # 파일 자체가 없어서 확인 불가

            record["error_msg"] = "; ".join(local_errors)
            summary_records.append(record)

        summary_df = pd.DataFrame(summary_records, columns=SUMMARY_COLUMNS)
        error_df = pd.DataFrame(error_records, columns=ERROR_COLUMNS)

        return summary_df, error_df

    def validate_lesion_paths(
        self,
        sample_n: int = 5,
        dataset_profile: str = "v1_model_roi",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
        """
        lesion 테스트 데이터셋의 Phase 8 평가 전 구조 무결성을 검증한다.

        정상 학습셋용 validate_structure와 별개의 메서드이며, 기존 로직은 건드리지 않는다.

        Parameters
        ----------
        sample_n : int
            shape 확인 및 patch CSV 컬럼 확인을 수행할 샘플 환자 수.
        dataset_profile : str
            'v1_model_roi' (기본값): model_roi.npy / lesion_mask_model_roi.npy / pure_lung.npy 검증.
            'v2_roi_0_0': roi_0_0.npy / lesion_mask_roi_0_0.npy 검증. pure_lung 없음.

        경로 해석 정책:
        - ct_hu / meta / patch_index : PathResolver.resolve() 경유.
        - v1: pure_lung → PathResolver.resolve() 경유. model_roi / lesion_mask → volume_dir + 고정 파일명.
        - v2: roi_0_0 / lesion_mask → volume_dir + 고정 파일명. pure_lung 확인 없음.

        검증 범위:
        - 파일/patch CSV 존재 확인: 전체 환자.
        - shape 일치 확인: 샘플 sample_n명만 mmap.
        - patch CSV 필수/권장 컬럼: 샘플 sample_n명의 헤더.

        Returns
        -------
        (summary_df, error_df, stats)
            summary_df : 환자별 검증 결과
            error_df   : error.csv용 (ERROR_COLUMNS)
            stats      : 데이터셋 레벨 집계 dict
        """
        # profile 선택
        is_v2 = (dataset_profile == "v2_roi_0_0")
        volume_files = _LESION_VOLUME_FILES_V2 if is_v2 else _LESION_VOLUME_FILES
        shape_keys = _LESION_SHAPE_KEYS_V2 if is_v2 else _LESION_SHAPE_KEYS
        patch_recommended = _LESION_PATCH_RECOMMENDED_V2 if is_v2 else _LESION_PATCH_RECOMMENDED
        summary_columns = LESION_SUMMARY_COLUMNS_V2 if is_v2 else LESION_SUMMARY_COLUMNS
        base = Path(self.path_resolver.base_path)
        error_records: List[dict] = []

        stats: dict = {
            "root_exists": base.exists(),
            "manifests_dir_exists": (base / "manifests").exists(),
            "volumes_npy_dir_exists": (base / "volumes_npy").exists(),
            "patch_index_dir_exists": (base / "patch_index_by_patient").exists(),
            "manifest_csv_exists": (base / "manifests" / "patient_manifest.csv").exists(),
            "volumes_npy_count": None,
            "patch_csv_count": None,
            "manifest_case_count": None,
            "group_counts": {},
            "count_match": None,
            "dataset_profile": dataset_profile,
            "file_names": dict(volume_files),
            "sample_n": sample_n,
            "sample_shape_match_all": None,
            "sample_shapes": [],
            "patch_required_ok": None,
            "patch_required_missing": [],
            "patch_recommended_present": [],
            "patch_recommended_missing": [],
        }

        # 폴더/CSV 개수
        vol_dir = base / "volumes_npy"
        pidx_dir = base / "patch_index_by_patient"
        if vol_dir.exists():
            stats["volumes_npy_count"] = sum(1 for p in vol_dir.iterdir() if p.is_dir())
        if pidx_dir.exists():
            stats["patch_csv_count"] = sum(1 for _ in pidx_dir.glob("*.csv"))
        if stats["volumes_npy_count"] is not None and stats["patch_csv_count"] is not None:
            stats["count_match"] = stats["volumes_npy_count"] == stats["patch_csv_count"]

        manifest_path = str(base / "manifests" / "patient_manifest.csv")
        if not stats["manifest_csv_exists"]:
            error_records.append({
                "patient_id": "",
                "error_type": "manifest_not_found",
                "error_msg": f"patient_manifest.csv 없음: {manifest_path}",
                "file_logical": "manifest",
            })
            summary_df = pd.DataFrame([], columns=LESION_SUMMARY_COLUMNS)
            error_df = pd.DataFrame(error_records, columns=ERROR_COLUMNS)
            return summary_df, error_df, stats

        rows, dup_errors = self._load_manifest_rows(manifest_path)
        error_records.extend(dup_errors)
        stats["manifest_case_count"] = len(rows)

        # group 집계 (NSCLC/MSD)
        group_counts: dict = {}
        for row in rows:
            g = (row.get("group") or "").strip() or "UNKNOWN"
            group_counts[g] = group_counts.get(g, 0) + 1
        stats["group_counts"] = group_counts

        sample_ids = {(row.get("patient_id") or "").strip() for row in rows[:sample_n]}

        summary_records: List[dict] = []
        shape_match_flags: List[bool] = []

        for row in rows:
            pid = (row.get("patient_id") or "").strip()
            safe_id = (row.get("safe_id") or "").strip()
            group = (row.get("group") or "").strip()
            is_sample = pid in sample_ids

            # profile에 따른 record 초기화
            if is_v2:
                record: dict = {
                    "patient_id": pid,
                    "safe_id": safe_id,
                    "group": group,
                    "ct_hu_exists": False,
                    "roi_0_0_exists": False,
                    "lesion_mask_exists": False,
                    "meta_exists": False,
                    "patch_csv_exists": False,
                    "shape_checked": is_sample,
                    "shape_match": None,
                    "error_msg": "",
                }
            else:
                record: dict = {
                    "patient_id": pid,
                    "safe_id": safe_id,
                    "group": group,
                    "ct_hu_exists": False,
                    "model_roi_exists": False,
                    "lesion_mask_exists": False,
                    "pure_lung_exists": False,
                    "meta_exists": False,
                    "patch_csv_exists": False,
                    "shape_checked": is_sample,
                    "shape_match": None,
                    "error_msg": "",
                }
            local_errors: List[str] = []
            resolved_paths: dict = {}

            # volume_dir resolve (roi npy / lesion_mask join 기준)
            try:
                vol_path: Optional[Path] = Path(self.path_resolver.resolve(pid, "volume_dir"))
            except (KeyError, ValueError) as exc:
                vol_path = None
                local_errors.append(f"volume_dir: {exc}")
                error_records.append({
                    "patient_id": pid,
                    "error_type": "path_resolve_error",
                    "error_msg": str(exc),
                    "file_logical": "volume_dir",
                })

            # ct_hu / meta : resolve 경유 (v1/v2 공통)
            for logical, field in [
                ("ct_hu", "ct_hu_exists"),
                ("meta", "meta_exists"),
            ]:
                try:
                    p = self.path_resolver.resolve(pid, logical)
                    resolved_paths[logical] = p
                    record[field] = Path(p).exists()
                except (KeyError, ValueError) as exc:
                    resolved_paths[logical] = None
                    record[field] = False
                    local_errors.append(f"{logical}: {exc}")
                    error_records.append({
                        "patient_id": pid,
                        "error_type": "path_resolve_error",
                        "error_msg": str(exc),
                        "file_logical": logical,
                    })

            # v1 전용: pure_lung resolve 경유
            if not is_v2:
                try:
                    p = self.path_resolver.resolve(pid, "pure_lung")
                    resolved_paths["pure_lung"] = p
                    record["pure_lung_exists"] = Path(p).exists()
                except (KeyError, ValueError) as exc:
                    resolved_paths["pure_lung"] = None
                    record["pure_lung_exists"] = False
                    local_errors.append(f"pure_lung: {exc}")
                    error_records.append({
                        "patient_id": pid,
                        "error_type": "path_resolve_error",
                        "error_msg": str(exc),
                        "file_logical": "pure_lung",
                    })

            # volume_dir + 고정 파일명으로 roi npy / lesion_mask 확인
            if vol_path is not None:
                if is_v2:
                    roi_f = vol_path / volume_files["roi_0_0"]
                    lm = vol_path / volume_files["lesion_mask"]
                    resolved_paths["roi_0_0"] = str(roi_f)
                    resolved_paths["lesion_mask"] = str(lm)
                    record["roi_0_0_exists"] = roi_f.exists()
                    record["lesion_mask_exists"] = lm.exists()
                else:
                    mr = vol_path / volume_files["model_roi"]
                    lm = vol_path / volume_files["lesion_mask"]
                    resolved_paths["model_roi"] = str(mr)
                    resolved_paths["lesion_mask"] = str(lm)
                    record["model_roi_exists"] = mr.exists()
                    record["lesion_mask_exists"] = lm.exists()

            # patch CSV : resolve(patch_index) 경유
            try:
                patch_p = self.path_resolver.resolve(pid, "patch_index")
                resolved_paths["patch_index"] = patch_p
                record["patch_csv_exists"] = Path(patch_p).exists()
            except (KeyError, ValueError) as exc:
                resolved_paths["patch_index"] = None
                record["patch_csv_exists"] = False
                local_errors.append(f"patch_index: {exc}")
                error_records.append({
                    "patient_id": pid,
                    "error_type": "path_resolve_error",
                    "error_msg": str(exc),
                    "file_logical": "patch_index",
                })

            # 누락 파일 error 기록
            file_field_pairs = [
                ("ct_hu", "ct_hu_exists"),
                ("lesion_mask", "lesion_mask_exists"),
                ("meta", "meta_exists"),
                ("patch_index", "patch_csv_exists"),
            ]
            if is_v2:
                file_field_pairs.insert(1, ("roi_0_0", "roi_0_0_exists"))
            else:
                file_field_pairs.insert(1, ("model_roi", "model_roi_exists"))
                file_field_pairs.insert(3, ("pure_lung", "pure_lung_exists"))
            for logical, field in file_field_pairs:
                if not record[field]:
                    error_records.append({
                        "patient_id": pid,
                        "error_type": "file_not_found",
                        "error_msg": f"{logical} 파일 없음",
                        "file_logical": logical,
                    })

            # 샘플 shape 확인
            if is_sample:
                shapes: dict = {}
                ok = True
                for k in shape_keys:
                    p = resolved_paths.get(k)
                    if p and Path(p).exists():
                        try:
                            shapes[k] = tuple(np.load(p, mmap_mode="r").shape)
                        except Exception as exc:
                            ok = False
                            shapes[k] = None
                            local_errors.append(f"{k} shape 로드 실패: {exc}")
                            error_records.append({
                                "patient_id": pid,
                                "error_type": "shape_load_error",
                                "error_msg": str(exc),
                                "file_logical": k,
                            })
                    else:
                        ok = False
                        shapes[k] = None
                distinct = {s for s in shapes.values() if s is not None}
                match = ok and len(distinct) == 1
                record["shape_match"] = match
                shape_match_flags.append(match)
                stats["sample_shapes"].append({
                    "patient_id": pid,
                    "shapes": {k: shapes.get(k) for k in shape_keys},
                })
                if not match:
                    error_records.append({
                        "patient_id": pid,
                        "error_type": "shape_mismatch",
                        "error_msg": f"shape 불일치/누락: {shapes}",
                        "file_logical": "+".join(shape_keys),
                    })

            record["error_msg"] = "; ".join(local_errors)
            summary_records.append(record)

        if shape_match_flags:
            stats["sample_shape_match_all"] = all(shape_match_flags)

        # patch CSV 필수/권장 컬럼 확인 (샘플)
        req_missing_union: set = set()
        rec_present_union: set = set()
        rec_missing_union: set = set()
        required_ok_all = True
        for row in rows[:sample_n]:
            pid = (row.get("patient_id") or "").strip()
            try:
                patch_p = self.path_resolver.resolve(pid, "patch_index")
            except (KeyError, ValueError):
                continue
            if not Path(patch_p).exists():
                continue
            try:
                with open(patch_p, encoding="utf-8-sig", newline="") as f:
                    header = next(csv.reader(f))
            except Exception as exc:
                error_records.append({
                    "patient_id": pid,
                    "error_type": "patch_header_read_error",
                    "error_msg": str(exc),
                    "file_logical": "patch_index",
                })
                continue
            cols = {h.strip() for h in header}
            missing = [c for c in _LESION_PATCH_REQUIRED if c not in cols]
            if not any(idc in cols for idc in _LESION_PATCH_ID_ANY):
                missing.append("patient_id|safe_id")
            if missing:
                required_ok_all = False
                req_missing_union.update(missing)
                error_records.append({
                    "patient_id": pid,
                    "error_type": "patch_required_column_missing",
                    "error_msg": f"필수 컬럼 누락: {missing}",
                    "file_logical": "patch_index",
                })
            for c in patch_recommended:
                if c in cols:
                    rec_present_union.add(c)
                else:
                    rec_missing_union.add(c)

        stats["patch_required_ok"] = required_ok_all
        stats["patch_required_missing"] = sorted(req_missing_union)
        stats["patch_recommended_present"] = sorted(rec_present_union)
        stats["patch_recommended_missing"] = sorted(rec_missing_union)

        summary_df = pd.DataFrame(summary_records, columns=summary_columns)
        error_df = pd.DataFrame(error_records, columns=ERROR_COLUMNS)

        return summary_df, error_df, stats
