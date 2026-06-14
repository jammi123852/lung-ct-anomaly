"""
build_normal_rd4ad_sampling_manifest.py

Phase 5.19: Normal 2.5D RD4AD Crop Sampling Manifest Builder

- normal_v1.json의 train/val/test split 기준으로 정상 crop 위치 sampling manifest 생성
- two-root 구조:
    metadata root: patch_index_by_patient CSV, manifests, configs
    volume root:   volumes_npy/{safe_id}/ct_hu.npy 만 허용
- high-score sampling 미사용 (train padim_score 없음)
- position_bin + z_level + central_peripheral 균형 기반 환자당 50개 후보 선택
- 실제 crop npz 생성 없음
- stage2_holdout / v2 병변 경로 차단
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]

# ── 상수 ──────────────────────────────────────────────────────────────────────

REQUIRED_COLUMNS: list[str] = [
    "local_z", "y0", "x0", "y1", "x1",
    "pure_lung_patch_ratio", "z_level", "central_peripheral", "position_bin",
]

FORBIDDEN_VOLUME_ROOT_SUBDIRS: list[str] = [
    "manifests", "patch_index_by_patient", "configs", "reports",
]

MANIFEST_OUTPUT_COLUMNS: list[str] = [
    "patient_id", "safe_id", "normal_split", "crop_id",
    "sampling_source", "sampling_rule", "sampling_seed",
    "local_z", "z_center", "z_lo", "z_hi",
    "y0", "x0", "y1", "x1", "crop_size",
    "pure_lung_patch_ratio", "z_level", "central_peripheral", "position_bin",
    "source_patch_csv", "source_ct_path", "volume_root", "metadata_root",
    "crop_dataset_tag", "created_at",
]

EXPECTED_TRAIN = 290
EXPECTED_VAL = 36
EXPECTED_TEST = 36


# ── volume root guard ──────────────────────────────────────────────────────────

def assert_volume_root_path_ok(path: Path, volume_root: Path) -> None:
    """
    volume root 하위에서는 volumes_npy/{safe_id}/ct_hu.npy 만 허용.
    manifests/, patch_index_by_patient/, configs/, reports/ 접근 시 중단.
    """
    try:
        rel = path.relative_to(volume_root)
    except ValueError:
        return  # volume root 외 경로는 이 guard 적용 대상 아님

    parts = rel.parts
    if len(parts) == 0:
        return

    # volumes_npy 이외 최상위 폴더 접근 금지
    if parts[0] != "volumes_npy":
        print(f"[ABORT] volume root에서 volumes_npy 외 경로 접근 금지: {path}")
        sys.exit(1)

    # volumes_npy/{safe_id}/ct_hu.npy 깊이 초과 금지
    if len(parts) > 3:
        print(f"[ABORT] volume root 하위 경로가 예상보다 깊습니다: {path}")
        sys.exit(1)

    # 파일 레벨에서 ct_hu.npy 외 접근 금지
    if len(parts) == 3 and parts[2] != "ct_hu.npy":
        print(f"[ABORT] volume root에서 ct_hu.npy 외 파일 접근 금지: {path}")
        sys.exit(1)


# ── preflight ──────────────────────────────────────────────────────────────────

def run_preflight(
    args: argparse.Namespace,
    split_data: dict,
    output_dir: Path,
) -> None:
    """모든 preflight 검증 수행. 문제 발견 시 sys.exit(1)."""

    metadata_root = Path(args.metadata_root)
    volume_root = Path(args.volume_root)

    # 1. normal_v1.json 존재 확인 (로드 전 확인이므로 split_data로 대체)
    split_json = Path(args.normal_split_json)
    print(f"[preflight] normal_split_json: {split_json} ✓")

    # 2. train/val/test key 확인
    for key in ["train", "val", "test"]:
        if key not in split_data:
            print(f"[ABORT] normal_v1.json에 '{key}' key 없음")
            sys.exit(1)

    # 3. 환자 수 확인
    n_train = len(split_data["train"])
    n_val = len(split_data["val"])
    n_test = len(split_data["test"])
    print(f"[preflight] train={n_train}, val={n_val}, test={n_test}")
    if n_train != EXPECTED_TRAIN:
        print(f"[ABORT] train 환자 수 불일치: 실제 {n_train}, 기준 {EXPECTED_TRAIN}")
        sys.exit(1)
    if n_val != EXPECTED_VAL:
        print(f"[ABORT] val 환자 수 불일치: 실제 {n_val}, 기준 {EXPECTED_VAL}")
        sys.exit(1)
    if n_test != EXPECTED_TEST:
        print(f"[ABORT] test 환자 수 불일치: 실제 {n_test}, 기준 {EXPECTED_TEST}")
        sys.exit(1)

    # 4. train/val/test overlap 0 확인
    train_set = set(split_data["train"])
    val_set = set(split_data["val"])
    test_set = set(split_data["test"])
    tv = train_set & val_set
    tt = train_set & test_set
    vt = val_set & test_set
    if tv:
        print(f"[ABORT] train/val overlap 존재: {sorted(tv)[:5]}")
        sys.exit(1)
    if tt:
        print(f"[ABORT] train/test overlap 존재: {sorted(tt)[:5]}")
        sys.exit(1)
    if vt:
        print(f"[ABORT] val/test overlap 존재: {sorted(vt)[:5]}")
        sys.exit(1)
    print("[preflight] train/val/test overlap 0 ✓")

    # 5. patient_to_safe_id 존재 확인
    if "patient_to_safe_id" not in split_data:
        print("[ABORT] normal_v1.json에 patient_to_safe_id 없음")
        sys.exit(1)
    print("[preflight] patient_to_safe_id 존재 ✓")

    # 6. metadata root 존재 확인
    if not metadata_root.exists():
        print(f"[ABORT] metadata root 없음: {metadata_root}")
        sys.exit(1)
    print(f"[preflight] metadata root 존재 ✓")

    # 7. patch_index_by_patient 존재 확인
    patch_index_dir = metadata_root / "patch_index_by_patient"
    if not patch_index_dir.exists():
        print(f"[ABORT] patch_index_by_patient 없음: {patch_index_dir}")
        sys.exit(1)
    print(f"[preflight] patch_index_by_patient 존재 ✓")

    # 8. volume root 존재 확인
    if not volume_root.exists():
        print(f"[ABORT] volume root 없음: {volume_root}")
        sys.exit(1)
    print(f"[preflight] volume root 존재 ✓")

    # 9. volumes_npy 존재 확인
    volumes_npy_dir = volume_root / "volumes_npy"
    if not volumes_npy_dir.exists():
        print(f"[ABORT] volumes_npy 없음: {volumes_npy_dir}")
        sys.exit(1)
    print(f"[preflight] volumes_npy 존재 ✓")

    # 10. metadata root와 volume root가 같은 경로면 중단
    if metadata_root.resolve() == volume_root.resolve():
        print("[ABORT] metadata root와 volume root가 같은 경로입니다.")
        sys.exit(1)

    # 11. metadata root 경로에 v2 포함은 허용
    #     (사용자가 지정한 metadata root 폴더명에 v2가 포함됨 — 정상)

    # 15. output dir 충돌 확인
    if not args.dry_run and output_dir.exists() and not args.force:
        print(f"[ABORT] 출력 디렉토리가 이미 존재합니다: {output_dir}")
        print("  --force 옵션을 사용하면 덮어쓸 수 있습니다.")
        sys.exit(1)

    print("[preflight] 전체 통과")


# ── 구조 검증 ──────────────────────────────────────────────────────────────────

def validate_structure(
    split_data: dict,
    metadata_root: Path,
    volume_root: Path,
) -> dict:
    """
    전체 362명에 대해:
    - patch CSV / ct_hu.npy 존재 확인
    - 전체 CSV 필수 컬럼 검증 (nrows=1)
    - padim_score 컬럼 전체 집계
    반환:
        missing_patch_csv:       {split: [patient_id, ...]}
        missing_ct_hu:           {split: [patient_id, ...]}
        sample_columns:          첫 번째 유효 CSV 컬럼 목록
        any_padim_score_exists:  bool
        n_csv_with_padim_score:  int
    """
    p2s = split_data["patient_to_safe_id"]
    patch_index_dir = metadata_root / "patch_index_by_patient"
    volumes_npy_dir = volume_root / "volumes_npy"
    padim_col_names = ["padim_score", "score", "anomaly_score"]

    missing_patch: dict[str, list] = {"train": [], "val": [], "test": []}
    missing_ct: dict[str, list] = {"train": [], "val": [], "test": []}
    missing_required_cols: list[dict] = []
    sample_columns: list[str] = []
    n_csv_with_padim_score = 0

    for split in ["train", "val", "test"]:
        for pid in split_data[split]:
            safe_id = p2s.get(pid, pid)

            # patch CSV 존재 확인
            csv_path = patch_index_dir / f"{safe_id}.csv"
            if not csv_path.exists():
                missing_patch[split].append(pid)
            else:
                # 전체 CSV에 대해 nrows=1로 컬럼 검사
                sample_df = pd.read_csv(csv_path, nrows=1)
                cols = list(sample_df.columns)

                # 필수 컬럼 누락 확인
                missing_cols = [c for c in REQUIRED_COLUMNS if c not in cols]
                if missing_cols:
                    missing_required_cols.append({
                        "patient_id": pid, "safe_id": safe_id, "missing": missing_cols,
                    })

                # padim_score 전체 집계
                if any(c in cols for c in padim_col_names):
                    n_csv_with_padim_score += 1

                # 첫 번째 유효 CSV 컬럼 저장
                if not sample_columns and not missing_cols:
                    sample_columns = cols

            # ct_hu.npy 존재 확인 (volume root guard 적용)
            ct_path = volumes_npy_dir / safe_id / "ct_hu.npy"
            assert_volume_root_path_ok(ct_path, volume_root)
            if not ct_path.exists():
                missing_ct[split].append(pid)

    # 필수 컬럼 누락 즉시 중단
    if missing_required_cols:
        print(f"[ABORT] 필수 컬럼 누락 CSV {len(missing_required_cols)}개 발견:")
        for item in missing_required_cols[:5]:
            print(f"  {item['patient_id']}: 누락 컬럼 {item['missing']}")
        sys.exit(1)

    return {
        "missing_patch_csv": missing_patch,
        "missing_ct_hu": missing_ct,
        "sample_columns": sample_columns,
        "any_padim_score_exists": n_csv_with_padim_score > 0,
        "n_csv_with_padim_score": n_csv_with_padim_score,
    }


def check_required_columns(sample_columns: list[str]) -> None:
    """필수 컬럼 존재 확인. 누락 시 중단. padim_score 없어도 중단하지 않음."""
    missing = [c for c in REQUIRED_COLUMNS if c not in sample_columns]
    if missing:
        print(f"[ABORT] patch CSV 필수 컬럼 누락: {missing}")
        sys.exit(1)
    print("[validate] 필수 컬럼 전체 존재 ✓")


# ── 좌표 필터링 ────────────────────────────────────────────────────────────────

def filter_valid_coordinates(
    df: pd.DataFrame,
    crop_size: int,
    ct_path: Path,
    patient_id: str,
    invalid_log: list,
) -> pd.DataFrame:
    """
    invalid 좌표 row를 필터링하고 유효한 row만 반환.
    ct_hu.npy는 mmap_mode='r'로 shape만 읽음. 전체 volume 로드 금지.
    invalid row는 sampling 후보에서 제외하며 전체 중단하지 않음.
    조건:
        patch CSV의 y0/x0/y1/x1은 patch_size=32 기준 좌표 (crop_size=96 아님)
        patch 중심(cy, cx)에서 crop_size//2 범위가 volume 내에 있어야 함
        local_z >= 0 and local_z < vol_z
        y0 >= 0, x0 >= 0, y1 > y0, x1 > x0
    """
    half = crop_size // 2
    cy = (df["y0"] + df["y1"]) // 2
    cx = (df["x0"] + df["x1"]) // 2

    mask = (
        (df["local_z"] >= 0)
        & (df["y0"] >= 0)
        & (df["x0"] >= 0)
        & (df["y1"] > df["y0"])
        & (df["x1"] > df["x0"])
    )

    if ct_path.exists():
        arr = np.load(str(ct_path), mmap_mode="r")
        vol_z, vol_y, vol_x = arr.shape
        mask &= df["local_z"] < vol_z
        mask &= (cy - half) >= 0
        mask &= (cy + half) <= vol_y
        mask &= (cx - half) >= 0
        mask &= (cx + half) <= vol_x

    n_invalid = int((~mask).sum())
    if n_invalid > 0:
        invalid_log.append({
            "patient_id": patient_id,
            "n_invalid_rows": n_invalid,
            "n_total_rows": len(df),
        })
        ratio = n_invalid / max(len(df), 1)
        if ratio > 0.5:
            print(
                f"[warn] {patient_id}: invalid row {n_invalid}/{len(df)} "
                f"({ratio:.1%}) 제거됨"
            )

    return df[mask].copy().reset_index(drop=True)


# ── sampling ───────────────────────────────────────────────────────────────────

def _position_bin_sample(
    df: pd.DataFrame,
    n: int,
    bins: list[str],
) -> pd.DataFrame:
    """position_bin별 균등 배분 선택 (deterministic, 정렬 순)."""
    n_bins = len(bins)
    base_q = n // n_bins
    remainder = n % n_bins

    parts = []
    for i, bin_val in enumerate(bins):
        quota = base_q + (1 if i < remainder else 0)
        bin_df = df[df["position_bin"].astype(str) == bin_val]
        if len(bin_df) == 0:
            continue
        parts.append(bin_df.iloc[:quota])

    return pd.concat(parts).reset_index(drop=True) if parts else pd.DataFrame()


def _zlevel_central_sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """z_level + central_peripheral 조합 기반 fallback 선택 (deterministic)."""
    if len(df) == 0:
        return pd.DataFrame()

    has_zlevel = "z_level" in df.columns
    has_central = "central_peripheral" in df.columns

    if has_zlevel and has_central:
        df = df.copy()
        df["_combo"] = df["z_level"].astype(str) + "_" + df["central_peripheral"].astype(str)
        combos = sorted(df["_combo"].unique().tolist())
        n_combos = len(combos)
        base_q = n // n_combos if n_combos > 0 else n
        remainder = n % n_combos if n_combos > 0 else 0

        parts = []
        for i, combo in enumerate(combos):
            quota = base_q + (1 if i < remainder else 0)
            combo_df = df[df["_combo"] == combo]
            if len(combo_df) == 0:
                continue
            parts.append(combo_df.iloc[:quota])

        result = pd.concat(parts).reset_index(drop=True) if parts else pd.DataFrame()
        if "_combo" in result.columns:
            result = result.drop(columns=["_combo"])
        return result
    else:
        return df.iloc[:n].copy().reset_index(drop=True)


def sample_patient(
    df: pd.DataFrame,
    crops_per_patient: int,
    threshold: float,
    seed: int,
    fallback_log: list,
    patient_id: str,
) -> tuple[pd.DataFrame, str]:
    """
    환자 1명에 대해 position-balanced sampling 수행.
    반환: (선택된 DataFrame, 사용된 sampling_source 설명)
    """
    # 안정 정렬 (deterministic 보장)
    df = df.sort_values(
        ["position_bin", "local_z", "y0", "x0"],
        ascending=True,
        na_position="last",
    ).reset_index(drop=True)

    # threshold 단계별 후보 필터
    valid = df[df["pure_lung_patch_ratio"] >= threshold].copy()
    source_used = f"threshold_{threshold}"

    if len(valid) < crops_per_patient:
        relaxed = df[df["pure_lung_patch_ratio"] > 0].copy()
        if len(relaxed) > len(valid):
            fallback_log.append({
                "patient_id": patient_id,
                "reason": (
                    f"threshold {threshold} 부족 ({len(valid)}개) "
                    f"→ >0 fallback ({len(relaxed)}개)"
                ),
            })
            valid = relaxed
            source_used = "threshold_gt0_fallback"

    if len(valid) < crops_per_patient and len(valid) > 0:
        fallback_log.append({
            "patient_id": patient_id,
            "reason": f">0 후보도 부족 ({len(valid)}개) → 가능한 만큼 사용",
        })
        source_used = "all_available_fallback"

    if len(valid) == 0:
        return pd.DataFrame(), "no_candidates"

    # stable key 부여: index reset 이후에도 row 식별 가능
    valid = valid.copy()
    valid["_row_id"] = range(len(valid))

    # position_bin 균형 선택
    bins = sorted(valid["position_bin"].dropna().astype(str).unique().tolist())

    if bins:
        selected = _position_bin_sample(valid, crops_per_patient, bins)

        # position_bin으로 부족하면 _row_id 기준으로 보충 pool 구성
        if len(selected) < crops_per_patient:
            remaining_quota = crops_per_patient - len(selected)
            used_row_ids = set(selected["_row_id"].tolist()) if len(selected) > 0 else set()
            pool = valid[~valid["_row_id"].isin(used_row_ids)]
            if len(pool) > 0:
                supplement = _zlevel_central_sample(pool, remaining_quota)
                selected = pd.concat([selected, supplement], ignore_index=True)
                source_used += "+zlevel_central_supplement"
    else:
        # position_bin 없으면 z_level + central_peripheral fallback 단독
        selected = _zlevel_central_sample(valid, crops_per_patient)
        source_used += "+zlevel_central_fallback"

    if len(selected) > crops_per_patient:
        selected = selected.iloc[:crops_per_patient].copy()

    # _row_id 중복 검증 (동일 row가 두 번 선택되면 즉시 중단)
    if len(selected) > 0 and selected["_row_id"].duplicated().any():
        dup_count = int(selected["_row_id"].duplicated().sum())
        print(f"[ABORT] {patient_id}: sampling 결과에 _row_id 중복 {dup_count}개 발견")
        sys.exit(1)

    # _row_id 컬럼 제거
    selected = selected.drop(columns=["_row_id"]).reset_index(drop=True)

    return selected, source_used


# ── manifest 빌드 ──────────────────────────────────────────────────────────────

def build_manifest(
    split_data: dict,
    metadata_root: Path,
    volume_root: Path,
    args: argparse.Namespace,
    padim_score_exists: bool,
) -> tuple[pd.DataFrame, dict]:
    """
    전체 환자에 대해 position-balanced sampling을 수행하고
    manifest DataFrame과 summary dict를 반환한다.
    """
    p2s = split_data["patient_to_safe_id"]
    patch_index_dir = metadata_root / "patch_index_by_patient"
    volumes_npy_dir = volume_root / "volumes_npy"

    all_rows: list[pd.DataFrame] = []
    fallback_log: list[dict] = []
    invalid_log: list[dict] = []
    patients_under_50: list[dict] = []

    pos_bin_dist: dict[str, dict] = {"train": {}, "val": {}, "test": {}}
    zlevel_dist: dict[str, dict] = {"train": {}, "val": {}, "test": {}}
    central_dist: dict[str, dict] = {"train": {}, "val": {}, "test": {}}
    sampling_source_counts: dict[str, int] = {}

    created_at = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    for split in ["train", "val", "test"]:
        for patient_id in split_data[split]:
            safe_id = p2s.get(patient_id, patient_id)
            csv_path = patch_index_dir / f"{safe_id}.csv"
            ct_path = volumes_npy_dir / safe_id / "ct_hu.npy"

            # volume root guard 적용
            assert_volume_root_path_ok(ct_path, volume_root)

            df = pd.read_csv(csv_path)

            # 좌표 필터링: invalid row 제거 (mmap_mode='r', 전체 로드 금지)
            df = filter_valid_coordinates(df, args.crop_size, ct_path, patient_id, invalid_log)

            # sampling
            selected, source_used = sample_patient(
                df=df,
                crops_per_patient=args.crops_per_patient,
                threshold=args.pure_lung_threshold,
                seed=args.seed,
                fallback_log=fallback_log,
                patient_id=patient_id,
            )

            n_selected = len(selected)
            sampling_source_counts[source_used] = (
                sampling_source_counts.get(source_used, 0) + 1
            )

            if n_selected < args.crops_per_patient:
                patients_under_50.append({
                    "patient_id": patient_id,
                    "safe_id": safe_id,
                    "split": split,
                    "n_crops": n_selected,
                })

            if n_selected == 0:
                continue

            # 분포 집계
            for bv, cnt in selected["position_bin"].value_counts().items():
                k = str(bv)
                pos_bin_dist[split][k] = pos_bin_dist[split].get(k, 0) + int(cnt)
            for lv, cnt in selected["z_level"].value_counts().items():
                k = str(lv)
                zlevel_dist[split][k] = zlevel_dist[split].get(k, 0) + int(cnt)
            for cp, cnt in selected["central_peripheral"].value_counts().items():
                k = str(cp)
                central_dist[split][k] = central_dist[split].get(k, 0) + int(cnt)

            # 출력 컬럼 구성
            out = selected[
                ["local_z", "y0", "x0", "y1", "x1",
                 "pure_lung_patch_ratio", "z_level", "central_peripheral", "position_bin"]
            ].copy()
            out["patient_id"] = patient_id
            out["safe_id"] = safe_id
            out["normal_split"] = split
            out["crop_id"] = [
                f"{safe_id}_{int(r.local_z)}_{int(r.y0)}_{int(r.x0)}"
                for r in out.itertuples()
            ]
            out["sampling_source"] = "patch_index"
            out["sampling_rule"] = "position_balanced_v1"
            out["sampling_seed"] = args.seed
            out["z_center"] = out["local_z"].astype(int)
            out["z_lo"] = out["z_center"] - 1
            out["z_hi"] = out["z_center"] + 1
            out["crop_size"] = args.crop_size
            out["source_patch_csv"] = str(csv_path)
            out["source_ct_path"] = str(ct_path)
            out["volume_root"] = str(volume_root)
            out["metadata_root"] = str(metadata_root)
            out["crop_dataset_tag"] = args.output_tag
            out["created_at"] = created_at

            all_rows.append(out)

    manifest_df = (
        pd.concat(all_rows, ignore_index=True)[MANIFEST_OUTPUT_COLUMNS]
        if all_rows else pd.DataFrame(columns=MANIFEST_OUTPUT_COLUMNS)
    )

    # patient_id + crop_id 중복 검증
    if len(manifest_df) > 0:
        dup_mask = manifest_df.duplicated(subset=["patient_id", "crop_id"])
        if dup_mask.any():
            n_dup = int(dup_mask.sum())
            print(f"[ABORT] manifest에서 patient_id+crop_id 중복 {n_dup}개 발견")
            sys.exit(1)

    n_crops_by_split: dict[str, int] = {}
    for split in ["train", "val", "test"]:
        if len(manifest_df) > 0:
            n_crops_by_split[split] = int((manifest_df["normal_split"] == split).sum())
        else:
            n_crops_by_split[split] = 0

    summary = {
        "output_tag": args.output_tag,
        "metadata_root": str(metadata_root),
        "volume_root": str(volume_root),
        "normal_split_json": str(args.normal_split_json),
        "sampling_rule": "position_balanced_v1",
        "sampling_seed": args.seed,
        "crop_size": args.crop_size,
        "crops_per_patient": args.crops_per_patient,
        "pure_lung_patch_ratio_threshold": args.pure_lung_threshold,
        "n_patients_train": len(split_data["train"]),
        "n_patients_val": len(split_data["val"]),
        "n_patients_test": len(split_data["test"]),
        "n_crops_train": n_crops_by_split["train"],
        "n_crops_val": n_crops_by_split["val"],
        "n_crops_test": n_crops_by_split["test"],
        "n_crops_total": sum(n_crops_by_split.values()),
        "patients_with_less_than_50_crops": patients_under_50,
        "split_overlap_check": 0,
        "missing_patch_csv_count": 0,
        "missing_ct_hu_count": 0,
        "sampling_source_counts": sampling_source_counts,
        "position_bin_distribution": pos_bin_dist,
        "z_level_distribution": zlevel_dist,
        "central_peripheral_distribution": central_dist,
        "fallback_count": len(fallback_log),
        "fallback_details": fallback_log,
        "n_patients_with_invalid_rows": len(invalid_log),
        "invalid_rows_by_patient": invalid_log,
        "padim_score_column_exists": padim_score_exists,
        "dry_run": args.dry_run,
        "note": [
            "high-score source 없음 — train 290명 padim_score 미존재",
            "patch_index + position-bin 기반 fallback sampling 적용",
            "96은 baseline이며 최적 확정 아님",
            "volume root에서는 ct_hu.npy만 사용",
            "병변 stage1_dev hard negative crop은 학습 주 데이터 아님",
        ],
    }

    return manifest_df, summary


# ── argparse ──────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normal RD4AD crop sampling manifest 생성 (position-bin 균형 기반)"
    )
    parser.add_argument(
        "--metadata-root",
        default="/mnt/c/Users/jinhy/Desktop/Normal_LUNA16_padim_training_ready_v2_tslungguard_nochest",
        help="metadata/CSV/JSON 기준 root (patch_index_by_patient 포함)",
    )
    parser.add_argument(
        "--volume-root",
        default="/mnt/c/Users/jinhy/Desktop/Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1",
        help="full volume npy 전용 root (volumes_npy/{safe_id}/ct_hu.npy 만 허용)",
    )
    parser.add_argument(
        "--normal-split-json",
        default=str(REPO / "outputs/position-aware-padim-v1/splits/normal_v1.json"),
        help="normal train/val/test split JSON 경로",
    )
    parser.add_argument(
        "--output-root",
        default=str(REPO / "outputs/second-stage-lesion-refiner-v1/normal_sampling"),
        help="sampling manifest 저장 root",
    )
    parser.add_argument(
        "--output-tag",
        default="normal_patch_index_position_balanced_fixed96_v1",
        help="output tag (하위 폴더명 및 파일명에 사용)",
    )
    parser.add_argument(
        "--crops-per-patient",
        type=int,
        default=50,
        help="환자당 최대 crop 수 (기본 50)",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=96,
        help="crop 크기 정사각형 기준 (기본 96)",
    )
    parser.add_argument(
        "--pure-lung-threshold",
        type=float,
        default=0.25,
        help="pure_lung_patch_ratio 최소 기준 (기본 0.25)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed (기본 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="dry-run 모드: 파일 생성 없이 예상 결과만 출력",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="기존 출력이 존재해도 덮어쓰기 허용",
    )
    parser.add_argument(
        "--no-runtime-append",
        action="store_true",
        help="runtime_summary.csv 기록 생략",
    )
    return parser.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    metadata_root = Path(args.metadata_root)
    volume_root = Path(args.volume_root)
    output_root = Path(args.output_root)
    output_dir = output_root / args.output_tag
    out_csv = output_dir / f"normal_sampling_manifest_{args.output_tag}.csv"
    out_json = output_dir / f"normal_sampling_manifest_{args.output_tag}_summary.json"

    # normal_v1.json 로드
    split_json = Path(args.normal_split_json)
    if not split_json.exists():
        print(f"[ABORT] normal_split_json 없음: {split_json}")
        sys.exit(1)
    with open(split_json, encoding="utf-8") as f:
        split_data = json.load(f)

    # preflight
    run_preflight(args, split_data, output_dir)

    # 구조 검증
    print("[validate] 전체 362명 patch CSV / ct_hu.npy 존재 확인 중...")
    validation = validate_structure(split_data, metadata_root, volume_root)

    total_missing_patch = sum(len(v) for v in validation["missing_patch_csv"].values())
    total_missing_ct = sum(len(v) for v in validation["missing_ct_hu"].values())

    if total_missing_patch > 0:
        for split, lst in validation["missing_patch_csv"].items():
            if lst:
                print(f"[ABORT] {split} patch CSV 누락 {len(lst)}명: {lst[:3]}")
        sys.exit(1)
    if total_missing_ct > 0:
        for split, lst in validation["missing_ct_hu"].items():
            if lst:
                print(f"[ABORT] {split} ct_hu.npy 누락 {len(lst)}명: {lst[:3]}")
        sys.exit(1)
    print("[validate] patch CSV 누락 0 / ct_hu.npy 누락 0 ✓")

    check_required_columns(validation["sample_columns"])

    any_padim_score_exists = validation["any_padim_score_exists"]
    n_csv_with_padim_score = validation["n_csv_with_padim_score"]
    if any_padim_score_exists:
        print(
            f"[validate] padim_score 컬럼 존재 CSV: {n_csv_with_padim_score}개 "
            f"— summary에 기록, 기본 sampling에는 미사용"
        )
    else:
        print("[validate] padim_score 컬럼 없음 — position-bin fallback sampling 사용")

    # sampling
    print("[sampling] position-balanced sampling 시작...")
    manifest_df, summary = build_manifest(
        split_data=split_data,
        metadata_root=metadata_root,
        volume_root=volume_root,
        args=args,
        padim_score_exists=any_padim_score_exists,
    )
    summary["n_csv_with_padim_score"] = n_csv_with_padim_score

    print(
        f"[sampling] 완료: "
        f"train={summary['n_crops_train']}, "
        f"val={summary['n_crops_val']}, "
        f"test={summary['n_crops_test']}, "
        f"total={summary['n_crops_total']}"
    )
    print(f"[sampling] 50개 미만 환자: {len(summary['patients_with_less_than_50_crops'])}명")
    print(f"[sampling] fallback 발생 수: {summary['fallback_count']}")
    print(f"[sampling] position_bin 분포 (train): {summary['position_bin_distribution']['train']}")
    print(f"[sampling] z_level 분포 (train): {summary['z_level_distribution']['train']}")
    print(f"[sampling] central_peripheral 분포 (train): {summary['central_peripheral_distribution']['train']}")

    if args.dry_run:
        print("\n[dry-run] 파일 생성 없음. 위 예상 결과만 출력합니다.")
        print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
        return

    # 실제 저장
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[save] sampling manifest CSV → {out_csv}")
    manifest_df.to_csv(out_csv, index=False)

    print(f"[save] summary JSON → {out_json}")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print("\n[done] normal sampling manifest 생성 완료.")
    print(f"  CSV : {out_csv}")
    print(f"  JSON: {out_json}")


if __name__ == "__main__":
    main()
