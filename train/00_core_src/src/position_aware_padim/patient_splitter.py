from __future__ import annotations

import csv as csv_mod
import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


VALID_SPLITS = {"train", "val", "test"}


@dataclass
class PatientSplit:
    train: List[str] = field(default_factory=list)
    val: List[str] = field(default_factory=list)
    test: List[str] = field(default_factory=list)
    patient_to_safe_id: Dict[str, str] = field(default_factory=dict)
    source: str = ""
    encoding: str = "utf-8-sig"
    seed: str = "original_split"


class PatientSplitter:
    """
    manifests/train_val_test_split.csv 기반으로 train/val/test split을 관리한다.

    우선순위:
      1. load_from_csv: 기존 CSV가 있으면 반드시 이를 읽는다.
      2. create_split: 기존 CSV가 없고 사용자 명시 승인이 있을 때만 사용한다.
    """

    def __init__(
        self,
        repo_root: Optional[str] = None,
        split_csv_path: Optional[str] = None,
    ) -> None:
        if repo_root is None:
            repo_root = str(Path(__file__).resolve().parents[2])
        self.repo_root = Path(repo_root)
        self.reports_dir = (
            self.repo_root / "outputs" / "position-aware-padim-v1" / "reports"
        )
        self.splits_dir = (
            self.repo_root / "outputs" / "position-aware-padim-v1" / "splits"
        )
        self.normal_v1_json = self.splits_dir / "normal_v1.json"

        if split_csv_path is not None:
            self.split_csv_path = Path(split_csv_path)
        else:
            self.split_csv_path = (
                self.repo_root
                / "data"
                / "normal_training_ready"
                / "manifests"
                / "train_val_test_split.csv"
            )

    # ------------------------------------------------------------------
    # 우선순위 1: CSV 로드
    # ------------------------------------------------------------------

    def load_from_csv(self) -> PatientSplit:
        """
        manifests/train_val_test_split.csv를 encoding='utf-8-sig'로 읽어 PatientSplit을 반환한다.

        검증 규칙:
        - 필수 컬럼(patient_id, split, safe_id) 누락 → ValueError
        - 같은 patient_id가 서로 다른 split에 등장 → ValueError
        - split 값이 train/val/test 외 → error.csv 기록 후 ValueError
        """
        if not self.split_csv_path.exists():
            raise FileNotFoundError(
                f"train_val_test_split.csv를 찾을 수 없습니다: {self.split_csv_path}"
            )

        df = pd.read_csv(self.split_csv_path, encoding="utf-8-sig")

        required_cols = {"patient_id", "split", "safe_id"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"필수 컬럼 누락: {missing_cols}")

        # 같은 patient_id가 여러 split에 있는지 확인
        pid_split = (
            df[["patient_id", "split"]]
            .drop_duplicates()
            .groupby("patient_id")["split"]
            .apply(list)
        )
        duplicates = pid_split[pid_split.apply(len) > 1]
        if not duplicates.empty:
            detail = duplicates.to_dict()
            raise ValueError(
                f"같은 patient_id가 여러 split에 존재합니다: {detail}"
            )

        # split 값이 허용 범위 외인 행
        invalid_rows = df[~df["split"].isin(VALID_SPLITS)]
        if not invalid_rows.empty:
            self._record_invalid_splits(invalid_rows)
            raise ValueError(
                f"유효하지 않은 split 값 {len(invalid_rows)}건 발견. "
                f"허용값: {VALID_SPLITS}. "
                f"상세 내용: {self.reports_dir / 'error.csv'}"
            )

        patient_split = PatientSplit(source=str(self.split_csv_path))
        for _, row in df.iterrows():
            pid = str(row["patient_id"]).strip()
            split_val = str(row["split"]).strip()
            safe_id = str(row["safe_id"]).strip()
            patient_split.patient_to_safe_id[pid] = safe_id
            if split_val == "train":
                patient_split.train.append(pid)
            elif split_val == "val":
                patient_split.val.append(pid)
            elif split_val == "test":
                patient_split.test.append(pid)

        return patient_split

    # ------------------------------------------------------------------
    # CSV → JSON 변환
    # ------------------------------------------------------------------

    def convert_csv_to_json(
        self,
        patient_split: PatientSplit,
        overwrite: bool = False,
    ) -> Path:
        """
        PatientSplit을 outputs/position-aware-padim-v1/splits/normal_v1.json으로 저장한다.

        기존 normal_v1.json이 있고 overwrite=False이면 FileExistsError를 발생시킨다.
        JSON 구조: source, encoding, seed, train, val, test, patient_to_safe_id
        """
        self.splits_dir.mkdir(parents=True, exist_ok=True)

        if self.normal_v1_json.exists() and not overwrite:
            raise FileExistsError(
                f"normal_v1.json이 이미 존재합니다: {self.normal_v1_json}\n"
                "덮어쓰려면 overwrite=True를 전달하거나 사용자 승인 후 진행하세요."
            )

        data: Dict[str, Any] = {
            "source": patient_split.source,
            "encoding": patient_split.encoding,
            "seed": patient_split.seed,
            "train": patient_split.train,
            "val": patient_split.val,
            "test": patient_split.test,
            "patient_to_safe_id": patient_split.patient_to_safe_id,
        }
        with open(self.normal_v1_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return self.normal_v1_json

    # ------------------------------------------------------------------
    # 우선순위 2: 새 split 생성 (기존 CSV 없고 사용자 승인 시만)
    # ------------------------------------------------------------------

    def create_split(
        self,
        patient_ids: List[str],
        patient_to_safe_id: Dict[str, str],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
    ) -> PatientSplit:
        """
        랜덤으로 새 split을 생성한다.

        주의: 기존 train_val_test_split.csv가 없고 사용자가 명시 승인한 경우에만 호출한다.
        기존 CSV가 존재하면 RuntimeError를 발생시킨다.
        """
        if self.split_csv_path.exists():
            raise RuntimeError(
                f"train_val_test_split.csv가 이미 존재합니다: {self.split_csv_path}\n"
                "기존 CSV가 있을 때 create_split 사용 금지."
            )
        rng = random.Random(seed)
        ids = sorted(patient_ids)
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        return PatientSplit(
            train=ids[:n_train],
            val=ids[n_train : n_train + n_val],
            test=ids[n_train + n_val :],
            patient_to_safe_id=patient_to_safe_id,
            source="create_split",
            seed=str(seed),
        )

    # ------------------------------------------------------------------
    # 저장 / 로드 / 검증
    # ------------------------------------------------------------------

    def save_split(
        self,
        patient_split: PatientSplit,
        path: Optional[str] = None,
        overwrite: bool = False,
    ) -> Path:
        """PatientSplit을 JSON으로 저장한다. path=None이면 normal_v1.json에 저장한다."""
        if path is None:
            return self.convert_csv_to_json(patient_split, overwrite=overwrite)

        out = Path(path)
        if out.exists() and not overwrite:
            raise FileExistsError(f"파일이 이미 존재합니다: {out}")
        out.parent.mkdir(parents=True, exist_ok=True)

        data: Dict[str, Any] = {
            "source": patient_split.source,
            "encoding": patient_split.encoding,
            "seed": patient_split.seed,
            "train": patient_split.train,
            "val": patient_split.val,
            "test": patient_split.test,
            "patient_to_safe_id": patient_split.patient_to_safe_id,
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return out

    def load_split(self, path: Optional[str] = None) -> PatientSplit:
        """저장된 JSON에서 PatientSplit을 복원한다. path=None이면 normal_v1.json을 읽는다."""
        load_path = Path(path) if path is not None else self.normal_v1_json
        if not load_path.exists():
            raise FileNotFoundError(f"split JSON을 찾을 수 없습니다: {load_path}")

        with open(load_path, encoding="utf-8") as f:
            data = json.load(f)

        return PatientSplit(
            train=data.get("train", []),
            val=data.get("val", []),
            test=data.get("test", []),
            patient_to_safe_id=data.get("patient_to_safe_id", {}),
            source=data.get("source", ""),
            encoding=data.get("encoding", "utf-8-sig"),
            seed=data.get("seed", ""),
        )

    def validate_split(self, patient_split: PatientSplit) -> Dict[str, Any]:
        """split 통계와 중복 여부를 반환한다."""
        all_ids = patient_split.train + patient_split.val + patient_split.test
        unique_ids = set(all_ids)
        duplicates = [pid for pid in unique_ids if all_ids.count(pid) > 1]
        return {
            "n_train": len(patient_split.train),
            "n_val": len(patient_split.val),
            "n_test": len(patient_split.test),
            "n_total": len(all_ids),
            "n_unique": len(unique_ids),
            "duplicates": duplicates,
            "has_duplicates": len(duplicates) > 0,
        }

    # ------------------------------------------------------------------
    # 내부 유틸
    # ------------------------------------------------------------------

    def _record_invalid_splits(self, invalid_rows: pd.DataFrame) -> None:
        """유효하지 않은 split 값을 error.csv에 기록한다.

        error.csv 공통 스키마(4컬럼)를 따른다:
          patient_id, error_type, error_msg, file_logical
        data_validation_summary.csv는 DataValidator 전용이므로 여기서 쓰지 않는다.
        """
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        error_csv = self.reports_dir / "error.csv"
        write_header = not error_csv.exists() or error_csv.stat().st_size == 0
        with open(error_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv_mod.writer(f)
            if write_header:
                writer.writerow(["patient_id", "error_type", "error_msg", "file_logical"])
            for _, row in invalid_rows.iterrows():
                writer.writerow(
                    [
                        row.get("patient_id", ""),
                        "invalid_split_value",
                        f"split 값이 train/val/test 외임: {row.get('split', '')}",
                        "train_val_test_split.csv",
                    ]
                )
