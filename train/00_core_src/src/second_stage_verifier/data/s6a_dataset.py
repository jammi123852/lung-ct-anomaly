"""
S6-A full crop npz 파일을 PyTorch Dataset으로 로드하는 모듈.
dataset index CSV 생성, patient-level train/val split CSV 생성 함수 포함.

절대 금지:
- npz 수정/재생성 금지
- PNG 생성 금지
- 모델/optimizer/checkpoint 코드 금지
- epoch loop 금지
- stage2_holdout 사용 금지
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# 경로 상수
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

CROPS_S6A_FULL_DIR = _PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/crops_s6a_full"
CROPS_S6A_6CH_FULL_DIR = _PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/crops_s6a_6ch_full"
SUMMARY_CSV_PATH = _PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/reports/crop_s6a_full_summary.csv"
SUMMARY_6CH_CSV_PATH = _PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/reports/crop_s6a_6ch_full_summary.csv"
STAGE_SPLIT_CSV_PATH = _PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1_balanced.csv"
DATASET_INDEX_CSV_PATH = _PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/datasets/s6a_full_dataset_index.csv"
DATASET_INDEX_6CH_CSV_PATH = _PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/datasets/s6a_6ch_full_dataset_index.csv"
TRAIN_VAL_SPLIT_CSV_PATH = _PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/datasets/s6a_stage1_train_val_split.csv"

# ---------------------------------------------------------------------------
# dataset index CSV 컬럼 정의
# ---------------------------------------------------------------------------
_REQUIRED_COLS_FROM_SUMMARY = [
    "patient_id",
    "label_int",          # → label 로 rename
    "sampling_label",
    "local_z",
    "slice_index",
    "slice_index_valid",
    "z_source",
    "sampling_rule",
    "saved_path",         # → npz_path 로 rename
]

_OPTIONAL_COLS_FROM_SUMMARY = [
    "score_original",
    "score_valid950_weighted",
    "score_valid950_soft",
    "composite_rank_v2",
    "lesion_patch_ratio",
    "position_bin",
    "z_level",
    "central_peripheral",
]

# ---------------------------------------------------------------------------
# 함수: build_dataset_index
# ---------------------------------------------------------------------------

def build_dataset_index(
    summary_csv_path: str | Path = SUMMARY_CSV_PATH,
    crops_dir: str | Path = CROPS_S6A_FULL_DIR,
    out_csv_path: str | Path = DATASET_INDEX_CSV_PATH,
) -> tuple[int, int]:
    """summary CSV에서 dataset index CSV를 생성한다.

    Parameters
    ----------
    summary_csv_path:
        full crop summary CSV 경로.
    crops_dir:
        crops_s6a_full 폴더 경로 (존재 확인용).
    out_csv_path:
        출력 dataset index CSV 경로.

    Returns
    -------
    (row_count, patient_count)
    """
    summary_csv_path = Path(summary_csv_path)
    crops_dir = Path(crops_dir)
    out_csv_path = Path(out_csv_path)

    # guard: 출력 파일이 이미 있으면 중단
    if out_csv_path.exists():
        raise FileExistsError(
            f"[build_dataset_index] 출력 파일이 이미 존재합니다. overwrite 방지를 위해 중단합니다.\n"
            f"  경로: {out_csv_path}"
        )

    # 입력 파일 존재 확인
    if not summary_csv_path.exists():
        raise FileNotFoundError(f"summary CSV 없음: {summary_csv_path}")
    if not crops_dir.exists():
        raise FileNotFoundError(f"crops 폴더 없음: {crops_dir}")

    # CSV 로드
    df = pd.read_csv(summary_csv_path, encoding="utf-8-sig")

    # 필수 컬럼 존재 확인
    missing = [c for c in _REQUIRED_COLS_FROM_SUMMARY if c not in df.columns]
    if missing:
        raise KeyError(f"summary CSV에 필수 컬럼 없음: {missing}")

    # 필요한 컬럼만 선택 (추가 컬럼은 있으면 포함)
    optional_present = [c for c in _OPTIONAL_COLS_FROM_SUMMARY if c in df.columns]
    select_cols = _REQUIRED_COLS_FROM_SUMMARY + optional_present
    df = df[select_cols].copy()

    # 컬럼명 변환
    df = df.rename(columns={
        "label_int": "label",
        "saved_path": "npz_path",
    })

    # 컬럼 순서 재정렬: 필수 컬럼 먼저
    required_final = ["npz_path", "patient_id", "label", "sampling_label",
                      "local_z", "slice_index", "slice_index_valid", "z_source", "sampling_rule"]
    optional_final = [c for c in _OPTIONAL_COLS_FROM_SUMMARY if c in df.columns]
    df = df[required_final + optional_final]

    # 출력 폴더 생성
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # 저장
    df.to_csv(out_csv_path, index=False, encoding="utf-8")

    row_count = len(df)
    patient_count = df["patient_id"].nunique()
    print(f"[build_dataset_index] 저장 완료: {out_csv_path}")
    print(f"  rows={row_count}, patients={patient_count}")
    return row_count, patient_count


# ---------------------------------------------------------------------------
# 함수: build_dataset_index_6ch
# ---------------------------------------------------------------------------

def build_dataset_index_6ch() -> tuple[int, int]:
    """S6-A 6ch full dataset index CSV를 생성한다.

    고정 경로:
      summary  : SUMMARY_6CH_CSV_PATH
      crops_dir: CROPS_S6A_6CH_FULL_DIR
      out_csv  : DATASET_INDEX_6CH_CSV_PATH

    기존 3ch index를 절대 덮어쓰지 않는다.
    """
    # guard: 3ch index 경로 오염 방지
    if DATASET_INDEX_6CH_CSV_PATH.resolve() == DATASET_INDEX_CSV_PATH.resolve():
        raise ValueError(
            "[build_dataset_index_6ch] 6ch 출력 경로가 3ch index 경로와 동일합니다. "
            "DATASET_INDEX_6CH_CSV_PATH를 확인하세요."
        )

    # guard: 6ch index가 이미 있으면 중단
    if DATASET_INDEX_6CH_CSV_PATH.exists():
        raise FileExistsError(
            f"[build_dataset_index_6ch] 6ch index 파일이 이미 존재합니다: {DATASET_INDEX_6CH_CSV_PATH}"
        )

    # guard: 6ch crops 폴더 없으면 중단
    if not CROPS_S6A_6CH_FULL_DIR.exists():
        raise FileNotFoundError(
            f"[build_dataset_index_6ch] 6ch crops 폴더 없음: {CROPS_S6A_6CH_FULL_DIR}"
        )

    # guard: 6ch summary CSV 없으면 중단
    if not SUMMARY_6CH_CSV_PATH.exists():
        raise FileNotFoundError(
            f"[build_dataset_index_6ch] 6ch summary CSV 없음: {SUMMARY_6CH_CSV_PATH}"
        )

    row_count, patient_count = build_dataset_index(
        summary_csv_path=SUMMARY_6CH_CSV_PATH,
        crops_dir=CROPS_S6A_6CH_FULL_DIR,
        out_csv_path=DATASET_INDEX_6CH_CSV_PATH,
    )

    # 생성 후 row 수 / patient 수 검증
    if row_count != 130_659:
        raise ValueError(
            f"[build_dataset_index_6ch] row 수 불일치: expected=130659, got={row_count}"
        )
    if patient_count != 154:
        raise ValueError(
            f"[build_dataset_index_6ch] patient 수 불일치: expected=154, got={patient_count}"
        )

    # stage2_holdout guard: 생성된 index에 holdout 포함 여부 확인
    df = pd.read_csv(DATASET_INDEX_6CH_CSV_PATH, encoding="utf-8-sig")
    if "stage" in df.columns:
        holdout_count = (df["stage"] == "stage2_holdout").sum()
        if holdout_count > 0:
            raise ValueError(
                f"[build_dataset_index_6ch] stage2_holdout 환자가 포함되어 있습니다: {holdout_count}명"
            )

    print(f"[build_dataset_index_6ch] 완료: row={row_count}, patient={patient_count}")
    print(f"  출력: {DATASET_INDEX_6CH_CSV_PATH}")
    return row_count, patient_count


# ---------------------------------------------------------------------------
# 함수: build_train_val_split
# ---------------------------------------------------------------------------

def build_train_val_split(
    stage_split_csv_path: str | Path = STAGE_SPLIT_CSV_PATH,
    out_csv_path: str | Path = TRAIN_VAL_SPLIT_CSV_PATH,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> dict:
    """stage1_dev 환자를 patient 단위로 train/val 분할한다.

    Parameters
    ----------
    stage_split_csv_path:
        lesion_stage_split_v1_balanced.csv 경로.
    out_csv_path:
        출력 train/val split CSV 경로.
    train_ratio:
        train 비율 (기본 0.8).
    seed:
        random seed (기본 42).

    Returns
    -------
    split 통계 dict: {
        "train_count": int,
        "val_count": int,
        "train_NSCLC": int,
        "train_MSD_Lung": int,
        "val_NSCLC": int,
        "val_MSD_Lung": int,
    }
    """
    import random as _random

    stage_split_csv_path = Path(stage_split_csv_path)
    out_csv_path = Path(out_csv_path)

    # guard: 출력 파일이 이미 있으면 중단
    if out_csv_path.exists():
        raise FileExistsError(
            f"[build_train_val_split] 출력 파일이 이미 존재합니다. overwrite 방지를 위해 중단합니다.\n"
            f"  경로: {out_csv_path}"
        )

    # 입력 파일 존재 확인
    if not stage_split_csv_path.exists():
        raise FileNotFoundError(f"stage split CSV 없음: {stage_split_csv_path}")

    df = pd.read_csv(stage_split_csv_path, encoding="utf-8-sig")

    # stage2_holdout 포함 여부 확인 (guard)
    holdout_patients = df[df["stage_split"] == "stage2_holdout"]["patient_id"].tolist()
    if holdout_patients:
        # stage1_dev만 필터링 후 진행하면 되지만, holdout 존재 자체는 정상이므로
        # holdout 환자가 실제로 학습 데이터에 포함되는 경우만 차단.
        # 여기서는 stage1_dev 필터링 후 holdout이 섞이는지 검증.
        pass

    # stage1_dev만 사용
    dev_df = df[df["stage_split"] == "stage1_dev"].copy()

    # 재확인: dev_df에 stage2_holdout 환자가 없는지 검증
    overlap = set(dev_df["patient_id"]) & set(holdout_patients)
    if overlap:
        raise ValueError(
            f"[build_train_val_split] stage2_holdout 환자가 stage1_dev에 포함되어 있습니다: {overlap}"
        )

    if len(dev_df) == 0:
        raise ValueError("stage1_dev 환자가 0명입니다. stage split CSV를 확인하세요.")

    # patient 단위 split — group(NSCLC/MSD_Lung)별로 각각 seed 고정 shuffle 후 80% train
    patients = dev_df[["patient_id", "group"]].drop_duplicates("patient_id")

    rng = _random.Random(seed)
    train_ids: set = set()
    val_ids: set = set()

    for group_name, group_df in patients.groupby("group"):
        pids = sorted(group_df["patient_id"].tolist())
        rng.shuffle(pids)
        n_train = round(len(pids) * train_ratio)
        train_ids.update(pids[:n_train])
        val_ids.update(pids[n_train:])

    # guard: train/val overlap 0명
    overlap = train_ids & val_ids
    if overlap:
        raise ValueError(
            f"[build_train_val_split] train/val overlap 발생: {sorted(overlap)}"
        )

    # guard: 합계가 전체 환자 수와 일치
    total_assigned = len(train_ids) + len(val_ids)
    total_patients = len(patients)
    if total_assigned != total_patients:
        raise ValueError(
            f"[build_train_val_split] train/val 합계={total_assigned} != 전체={total_patients}"
        )

    # 최종 CSV: patient_id, group, stage_split, train_val
    result_rows = []
    for _, row in dev_df[["patient_id", "group", "stage_split"]].drop_duplicates("patient_id").iterrows():
        pid = row["patient_id"]
        if pid in train_ids:
            tv = "train"
        elif pid in val_ids:
            tv = "val"
        else:
            raise ValueError(f"patient_id={pid} 가 train/val 어느 쪽에도 포함되지 않았습니다.")
        result_rows.append({
            "patient_id": pid,
            "group": row["group"],
            "stage_split": row["stage_split"],
            "train_val": tv,
        })

    result_df = pd.DataFrame(result_rows)

    # 선택적 컬럼 추가: n_crops, n_positive, n_hard_negative, positive_ratio
    if SUMMARY_CSV_PATH.exists():
        try:
            summary_df = pd.read_csv(SUMMARY_CSV_PATH, encoding="utf-8-sig")
            if "patient_id" in summary_df.columns:
                agg_rows = []
                for pid, g in summary_df.groupby("patient_id"):
                    n_crops = len(g)
                    n_positive = int((g["label_int"] == 1).sum()) if "label_int" in g.columns else 0
                    n_hard_negative = int((g["sampling_label"] == "hard_negative").sum()) if "sampling_label" in g.columns else 0
                    agg_rows.append({
                        "patient_id": pid,
                        "n_crops": n_crops,
                        "n_positive": n_positive,
                        "n_hard_negative": n_hard_negative,
                        "positive_ratio": round(n_positive / max(n_crops, 1), 4),
                    })
                agg_df = pd.DataFrame(agg_rows)
                result_df = result_df.merge(agg_df, on="patient_id", how="left")
        except Exception as e:
            warnings.warn(
                f"[build_train_val_split] 선택적 컬럼 추가 실패: {e}",
                RuntimeWarning,
                stacklevel=2,
            )

    # 출력 폴더 생성
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # 저장
    result_df.to_csv(out_csv_path, index=False, encoding="utf-8")

    # 통계
    stats = {
        "train_count": int((result_df["train_val"] == "train").sum()),
        "val_count": int((result_df["train_val"] == "val").sum()),
        "train_NSCLC": int(((result_df["train_val"] == "train") & (result_df["group"] == "NSCLC")).sum()),
        "train_MSD_Lung": int(((result_df["train_val"] == "train") & (result_df["group"] == "MSD_Lung")).sum()),
        "val_NSCLC": int(((result_df["train_val"] == "val") & (result_df["group"] == "NSCLC")).sum()),
        "val_MSD_Lung": int(((result_df["train_val"] == "val") & (result_df["group"] == "MSD_Lung")).sum()),
    }

    print(f"[build_train_val_split] 저장 완료: {out_csv_path}")
    print(f"  train={stats['train_count']}, val={stats['val_count']}")
    print(f"  train NSCLC={stats['train_NSCLC']}, MSD_Lung={stats['train_MSD_Lung']}")
    print(f"  val   NSCLC={stats['val_NSCLC']}, MSD_Lung={stats['val_MSD_Lung']}")
    return stats


# ---------------------------------------------------------------------------
# Dataset 클래스: S6ADataset
# ---------------------------------------------------------------------------

class S6ADataset(Dataset):
    """S6-A full crop npz 파일을 로드하는 PyTorch Dataset.

    Parameters
    ----------
    index_df:
        build_dataset_index 로 생성한 dataset index DataFrame.
        train_val 컬럼이 있으면 split 필터링에 사용.
    split:
        'all', 'train', 'val' 중 하나.
        'all' 이면 필터링 없이 전체 사용.
        'train' 또는 'val' 이면 index_df의 train_val 컬럼으로 필터링.

    Notes
    -----
    - normalize 미적용: raw HU값 그대로 반환.
    - NaN/Inf 감지 시 경고 출력만 (중단하지 않음).
    - npz 수정/재생성 금지.
    """

    def __init__(
        self,
        index_df: pd.DataFrame,
        split: str = "all",
        image_key: str = "crop",
        expected_channels: int = 3,
    ) -> None:
        if split not in ("all", "train", "val"):
            raise ValueError(f"split은 'all', 'train', 'val' 중 하나여야 합니다. 입력값: {split!r}")

        if split != "all":
            if "train_val" not in index_df.columns:
                raise KeyError(
                    f"split='{split}' 사용 시 index_df에 'train_val' 컬럼이 필요합니다."
                )
            df = index_df[index_df["train_val"] == split].copy()
        else:
            df = index_df.copy()

        self._df = df.reset_index(drop=True)
        self._split = split
        self._image_key = image_key
        self._expected_channels = expected_channels

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> Dict:
        row = self._df.iloc[idx]
        npz_path = str(row["npz_path"])

        # npz read-only 로드
        data = np.load(npz_path, allow_pickle=False, mmap_mode="r")

        # image_key 로 로드: 기본값 "crop" (3ch), 6ch 모드에서는 "image" 지정
        if self._image_key not in data:
            raise KeyError(
                f"[S6ADataset] npz에 key '{self._image_key}' 없음: {npz_path}"
            )
        crop = data[self._image_key].astype(np.float32)

        # shape 검증: channels 불일치 시 명확한 오류 발생
        if crop.shape[0] != self._expected_channels:
            raise ValueError(
                f"[S6ADataset] channel 불일치: expected={self._expected_channels}, "
                f"actual={crop.shape[0]}, path={npz_path}"
            )

        # NaN/Inf 감지 (경고만, 중단하지 않음)
        if not np.isfinite(crop).all():
            nan_count = int(np.sum(np.isnan(crop)))
            inf_count = int(np.sum(np.isinf(crop)))
            warnings.warn(
                f"[S6ADataset] NaN/Inf 감지: idx={idx}, path={npz_path}, "
                f"NaN={nan_count}, Inf={inf_count}",
                RuntimeWarning,
                stacklevel=2,
            )

        image_tensor = torch.tensor(crop, dtype=torch.float32)
        label_tensor = torch.tensor(int(row["label"]), dtype=torch.long)

        return {
            "image": image_tensor,
            "label": label_tensor,
            "patient_id": str(row["patient_id"]),
            "npz_path": npz_path,
            "local_z": int(row["local_z"]),
        }
