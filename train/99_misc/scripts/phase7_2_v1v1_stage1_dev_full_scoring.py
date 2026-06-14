"""
Phase 7.2 v1/v1 stage1_dev filtered full scoring

목적:
  filtered S6-A manifest (phase6_1b) 의 129,437 crops 전체에 대해
  canonical 27컬럼 reconstruction score CSV를 생성한다.

금지 사항:
  - metric 계산 금지
  - threshold 계산 금지
  - training / backward / optimizer step / checkpoint 생성 금지
  - filtered manifest 수정 금지
  - 원본 s6a_6ch_full_dataset_index.csv 사용 금지
  - crop 파일 수정/삭제/이동 금지
  - score CSV를 output root 밖에 생성 금지
  - 기존 파일 덮어쓰기/삭제/이동 금지
  - stage2_holdout / v2/v2v2 접근 금지
  - pip/conda install 금지
  - 기존 스크립트 수정 금지

참고 파일 (read-only):
  scripts/phase7_1_v1v1_scoring_smoke.py
  scripts/score_rd4ad_2p5d_normal_val_test.py
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ──────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────
MODEL_TAG = "rd4ad_2p5d_normal_mw_fixed96_v1"
MODEL_CLASS = "ConvAutoencoder2p5D"

FULL_MAX_BATCH_SIZE_LIMIT = 32

# manifest safety 검증 기준
EXPECTED_MANIFEST_ROWS = 129437
EXPECTED_UNIQUE_PATIENTS = 152
EXPECTED_STAGE2_HOLDOUT_ROWS = 0
EXCLUDED_PATIENT_IDS = {"LUNG1-295", "LUNG1-415"}

DEFAULT_CHECKPOINT = (
    "outputs/second-stage-lesion-refiner-v1/models/"
    "rd4ad_2p5d_normal_mw_fixed96_v1/checkpoints/best_val_loss.pt"
)
DEFAULT_MANIFEST = (
    "outputs/second-stage-lesion-refiner-v1/review_annotations/"
    "phase6_1b_s6a_stage1_dev_filtered_manifest_v1/"
    "phase6_1b_s6a_stage1_dev_filtered_manifest_v1.csv"
)
DEFAULT_OUTPUT_ROOT = (
    "outputs/second-stage-lesion-refiner-v1/scores/"
    "phase7_2_v1v1_stage1_dev_full_scoring_v1/"
)

# 출력 파일명 (고정)
OUTPUT_CSV_NAME = "phase7_2_v1v1_stage1_dev_full_scoring_v1.csv"
OUTPUT_CSV_TMP_NAME = "phase7_2_v1v1_stage1_dev_full_scoring_v1.csv.tmp"
OUTPUT_JSON_NAME = "phase7_2_v1v1_stage1_dev_full_scoring_summary_v1.json"
OUTPUT_MD_NAME = "phase7_2_v1v1_stage1_dev_full_scoring_report_v1.md"
OUTPUT_ERROR_CSV_NAME = "phase7_2_v1v1_stage1_dev_full_scoring_errors_v1.csv"
OUTPUT_RUNTIME_CSV_NAME = "phase7_2_v1v1_stage1_dev_full_scoring_runtime_summary_v1.csv"
OUTPUT_DONE_JSON_NAME = "phase7_2_v1v1_stage1_dev_full_scoring_DONE.json"

# canonical 27컬럼 (순서 고정)
CANONICAL_COLUMNS = [
    "crop_id",
    "patient_id",
    "npz_path",
    "label",
    "sampling_label",
    "stage_split",
    "model_tag",
    "checkpoint_path",
    "crop_score_l1_mean",
    "crop_score_l1_max",
    "crop_score_mse_mean",
    "channel_0_l1_mean",
    "channel_1_l1_mean",
    "channel_2_l1_mean",
    "channel_3_l1_mean",
    "channel_4_l1_mean",
    "channel_5_l1_mean",
    "lung_channels_l1_mean",
    "mediastinal_channels_l1_mean",
    "input_min",
    "input_max",
    "recon_min",
    "recon_max",
    "error_min",
    "error_max",
    "has_nan",
    "has_inf",
]

# error CSV 컬럼
ERROR_CSV_COLUMNS = [
    "row_id",
    "crop_id",
    "patient_id",
    "npz_path",
    "error_type",
    "error_message",
]

# runtime summary CSV 컬럼
RUNTIME_CSV_COLUMNS = [
    "start_time",
    "end_time",
    "runtime_seconds",
    "batch_size",
    "total_expected_rows",
    "total_processed_rows",
    "total_success_rows",
    "total_error_rows",
    "device",
    "checkpoint_path",
    "output_csv_path",
    "scoring_pass",
]


# ──────────────────────────────────────────────────────────
# Argument Parser
# ──────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Phase 7.2 v1/v1 stage1_dev filtered full scoring: "
            "filtered S6-A manifest의 129,437 crops 전체에 대해 "
            "canonical 27컬럼 reconstruction score CSV를 생성한다. "
            "--run-full 플래그가 없으면 실행되지 않습니다."
        )
    )
    parser.add_argument(
        "--run-full",
        action="store_true",
        help="[필수] full scoring 실행 플래그. 없으면 실행 거부.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help=f"배치 크기 (최대 {FULL_MAX_BATCH_SIZE_LIMIT}, default=32)",
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="checkpoint 경로 (best_val_loss.pt)",
    )
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help="filtered manifest CSV 경로",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="출력 root 디렉토리",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────
# Model (phase7_1 스크립트와 동일한 구조)
# ──────────────────────────────────────────────────────────
class ConvAutoencoder2p5D(nn.Module):
    """
    2.5D reconstruction baseline.
    input_channels=6 (lung z-1/z/z+1, mediastinal z-1/z/z+1)
    Encoder: 96→48→24→12 (3× MaxPool2d)
    Decoder: 12→24→48→96 (3× ConvTranspose2d) + Sigmoid
    phase7_1_v1v1_scoring_smoke.py 와 동일한 구조.
    """

    def __init__(self, input_channels: int = 6, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, c, 3, padding=1),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(c, c * 2, 3, padding=1),
            nn.BatchNorm2d(c * 2), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(c * 2, c * 4, 3, padding=1),
            nn.BatchNorm2d(c * 4), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(c * 4, c * 8, 3, padding=1),
            nn.BatchNorm2d(c * 8), nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2),
            nn.BatchNorm2d(c * 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2),
            nn.BatchNorm2d(c * 2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c * 2, c, 2, stride=2),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, input_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


# ──────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────
class FullScoreDataset(Dataset):
    """
    Phase 7.2 full scoring용 Dataset.
    filtered manifest의 npz_path 컬럼 사용.
    crop_id는 s6a_{row_index:06d} 형식으로 생성 (manifest에 crop_id 없음).
    원본 manifest row index(_original_row_id) 기준.
    """

    def __init__(self, rows: pd.DataFrame):
        # rows는 이미 _original_row_id 컬럼이 있는 상태로 전달됨
        self.df = rows.reset_index(drop=True)
        print(f"[INFO] FullScoreDataset: {len(self.df)} crops")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        npz_path = str(row["npz_path"])
        row_id = int(row["_original_row_id"])

        # crop_id 생성: s6a_{row_index:06d}
        crop_id = f"s6a_{row_id:06d}"

        meta = {
            "row_id": row_id,
            "crop_id": crop_id,
            "patient_id": str(row.get("patient_id", "")),
            "npz_path": npz_path,
            "label": str(row.get("label", "")),
            "sampling_label": str(row.get("sampling_label", "")),
            "stage_split": str(row.get("stage_split", "")),
            "_load_error": None,
            "_shape_error": None,
        }

        # npz 로드 시도
        img = None
        try:
            if not Path(npz_path).exists():
                raise FileNotFoundError(f"npz_path not found: {npz_path}")
            data = np.load(npz_path)
            if "image" not in data:
                raise KeyError(f"'image' key not found in {npz_path}")
            img = data["image"].astype(np.float32)
        except Exception as e:
            meta["_load_error"] = str(e)

        if meta["_load_error"] is not None:
            dummy = torch.zeros(6, 96, 96, dtype=torch.float32)
            return dummy, meta

        # shape 검증
        if img.shape != (6, 96, 96):
            meta["_shape_error"] = f"unexpected shape {img.shape}, expected (6,96,96)"
            dummy = torch.zeros(6, 96, 96, dtype=torch.float32)
            return dummy, meta

        tensor = torch.tensor(img, dtype=torch.float32)
        return tensor, meta


# ──────────────────────────────────────────────────────────
# Collate function
# ──────────────────────────────────────────────────────────
def collate_fn(batch):
    """DataLoader collate: tensor stack + meta dict list"""
    tensors = torch.stack([item[0] for item in batch])
    metas = [item[1] for item in batch]
    return tensors, metas


# ──────────────────────────────────────────────────────────
# Score 계산 (배치 단위, numpy 기반)
# ──────────────────────────────────────────────────────────
def compute_batch_full_scores(
    inputs: torch.Tensor,
    recons: torch.Tensor,
    metas: list,
    checkpoint_path: str,
) -> tuple:
    """
    배치 단위 full score 계산.
    inputs, recons: [B, 6, 96, 96] (CPU 또는 GPU tensor)
    반환: (score_rows list of dict, error_rows list of dict)
    score_rows: CANONICAL_COLUMNS 기준 (에러 crop 포함, 에러 crop은 NaN 값)
    error_rows: ERROR_CSV_COLUMNS 기준
    """
    inputs_np = inputs.detach().cpu().numpy()    # [B, 6, 96, 96]
    recons_np = recons.detach().cpu().numpy()    # [B, 6, 96, 96]

    diff = np.abs(inputs_np - recons_np)         # [B, 6, 96, 96]
    diff_sq = (inputs_np - recons_np) ** 2       # [B, 6, 96, 96]

    score_rows = []
    error_rows = []

    for b in range(inputs_np.shape[0]):
        meta = metas[b]
        inp = inputs_np[b]    # [6, 96, 96]
        rec = recons_np[b]    # [6, 96, 96]
        d = diff[b]           # [6, 96, 96]
        d_sq = diff_sq[b]     # [6, 96, 96]

        row_id = meta["row_id"]
        crop_id = meta["crop_id"]
        patient_id = meta["patient_id"]
        npz_path = meta["npz_path"]
        load_error = meta.get("_load_error")
        shape_error = meta.get("_shape_error")

        # 에러가 있는 crop 처리
        if load_error is not None:
            error_rows.append({
                "row_id": row_id,
                "crop_id": crop_id,
                "patient_id": patient_id,
                "npz_path": npz_path,
                "error_type": "load_error",
                "error_message": load_error,
            })
            # score row는 NaN으로 채워 기록 (has_nan=True)
            score_rows.append(_make_nan_score_row(meta, checkpoint_path))
            continue

        if shape_error is not None:
            error_rows.append({
                "row_id": row_id,
                "crop_id": crop_id,
                "patient_id": patient_id,
                "npz_path": npz_path,
                "error_type": "shape_mismatch",
                "error_message": shape_error,
            })
            score_rows.append(_make_nan_score_row(meta, checkpoint_path))
            continue

        # score 계산 시도
        try:
            crop_score_l1_mean = float(d.mean())
            crop_score_l1_max = float(d.max())
            crop_score_mse_mean = float(d_sq.mean())

            channel_scores = {f"channel_{c}_l1_mean": float(d[c].mean()) for c in range(6)}
            lung_channels_l1_mean = float(d[0:3].mean())
            mediastinal_channels_l1_mean = float(d[3:6].mean())

            input_min = float(inp.min())
            input_max = float(inp.max())
            recon_min = float(rec.min())
            recon_max = float(rec.max())
            error_min = float(d.min())
            error_max = float(d.max())

            # has_nan: input / recon / diff tensor / 모든 scalar score 중 하나라도 NaN이면 True
            scalar_scores = [
                crop_score_l1_mean, crop_score_l1_max, crop_score_mse_mean,
                channel_scores["channel_0_l1_mean"],
                channel_scores["channel_1_l1_mean"],
                channel_scores["channel_2_l1_mean"],
                channel_scores["channel_3_l1_mean"],
                channel_scores["channel_4_l1_mean"],
                channel_scores["channel_5_l1_mean"],
                lung_channels_l1_mean,
                mediastinal_channels_l1_mean,
            ]

            tensor_has_nan = bool(
                np.isnan(inp).any()
                or np.isnan(rec).any()
                or np.isnan(d).any()
            )
            scalar_has_nan = bool(any(math.isnan(v) for v in scalar_scores))
            has_nan = tensor_has_nan or scalar_has_nan

            tensor_has_inf = bool(
                np.isinf(inp).any()
                or np.isinf(rec).any()
                or np.isinf(d).any()
            )
            scalar_has_inf = bool(any(math.isinf(v) for v in scalar_scores))
            has_inf = tensor_has_inf or scalar_has_inf

            # NaN/Inf 에러 기록
            if has_nan:
                error_rows.append({
                    "row_id": row_id,
                    "crop_id": crop_id,
                    "patient_id": patient_id,
                    "npz_path": npz_path,
                    "error_type": "nan_error",
                    "error_message": "NaN detected in input/recon/diff or scalar scores",
                })
            if has_inf:
                error_rows.append({
                    "row_id": row_id,
                    "crop_id": crop_id,
                    "patient_id": patient_id,
                    "npz_path": npz_path,
                    "error_type": "inf_error",
                    "error_message": "Inf detected in input/recon/diff or scalar scores",
                })

            row = {
                "crop_id": crop_id,
                "patient_id": patient_id,
                "npz_path": npz_path,
                "label": meta["label"],
                "sampling_label": meta["sampling_label"],
                "stage_split": meta["stage_split"],
                "model_tag": MODEL_TAG,
                "checkpoint_path": checkpoint_path,
                "crop_score_l1_mean": crop_score_l1_mean,
                "crop_score_l1_max": crop_score_l1_max,
                "crop_score_mse_mean": crop_score_mse_mean,
                "channel_0_l1_mean": channel_scores["channel_0_l1_mean"],
                "channel_1_l1_mean": channel_scores["channel_1_l1_mean"],
                "channel_2_l1_mean": channel_scores["channel_2_l1_mean"],
                "channel_3_l1_mean": channel_scores["channel_3_l1_mean"],
                "channel_4_l1_mean": channel_scores["channel_4_l1_mean"],
                "channel_5_l1_mean": channel_scores["channel_5_l1_mean"],
                "lung_channels_l1_mean": lung_channels_l1_mean,
                "mediastinal_channels_l1_mean": mediastinal_channels_l1_mean,
                "input_min": input_min,
                "input_max": input_max,
                "recon_min": recon_min,
                "recon_max": recon_max,
                "error_min": error_min,
                "error_max": error_max,
                "has_nan": has_nan,
                "has_inf": has_inf,
            }
            score_rows.append(row)

        except Exception as e:
            error_rows.append({
                "row_id": row_id,
                "crop_id": crop_id,
                "patient_id": patient_id,
                "npz_path": npz_path,
                "error_type": "score_calculation_error",
                "error_message": str(e),
            })
            score_rows.append(_make_nan_score_row(meta, checkpoint_path))

    return score_rows, error_rows


def _make_nan_score_row(meta: dict, checkpoint_path: str) -> dict:
    """에러 crop용 NaN score row (has_nan=True)"""
    nan = float("nan")
    return {
        "crop_id": meta["crop_id"],
        "patient_id": meta["patient_id"],
        "npz_path": meta["npz_path"],
        "label": meta["label"],
        "sampling_label": meta["sampling_label"],
        "stage_split": meta["stage_split"],
        "model_tag": MODEL_TAG,
        "checkpoint_path": checkpoint_path,
        "crop_score_l1_mean": nan,
        "crop_score_l1_max": nan,
        "crop_score_mse_mean": nan,
        "channel_0_l1_mean": nan,
        "channel_1_l1_mean": nan,
        "channel_2_l1_mean": nan,
        "channel_3_l1_mean": nan,
        "channel_4_l1_mean": nan,
        "channel_5_l1_mean": nan,
        "lung_channels_l1_mean": nan,
        "mediastinal_channels_l1_mean": nan,
        "input_min": nan,
        "input_max": nan,
        "recon_min": nan,
        "recon_max": nan,
        "error_min": nan,
        "error_max": nan,
        "has_nan": True,
        "has_inf": False,
    }


# ──────────────────────────────────────────────────────────
# Manifest Safety 검증
# ──────────────────────────────────────────────────────────
def verify_manifest_safety(manifest_df: pd.DataFrame) -> list:
    """
    manifest safety 검증 7개 항목.
    실패 항목을 blockers list로 반환. 통과하면 빈 list.
    """
    blockers = []

    # 1. manifest row 수 == 129437
    actual_rows = len(manifest_df)
    if actual_rows != EXPECTED_MANIFEST_ROWS:
        msg = f"manifest row count mismatch: expected={EXPECTED_MANIFEST_ROWS}, actual={actual_rows}"
        blockers.append(msg)
        print(f"[ERROR] {msg}")
    else:
        print(f"[INFO] manifest row count OK: {actual_rows}")

    # 2. unique patient 수 == 152
    actual_patients = manifest_df["patient_id"].nunique()
    if actual_patients != EXPECTED_UNIQUE_PATIENTS:
        msg = f"unique patient count mismatch: expected={EXPECTED_UNIQUE_PATIENTS}, actual={actual_patients}"
        blockers.append(msg)
        print(f"[ERROR] {msg}")
    else:
        print(f"[INFO] unique patient count OK: {actual_patients}")

    # 3. stage2_holdout row 수 == 0
    if "stage_split" in manifest_df.columns:
        actual_holdout = int((manifest_df["stage_split"] == "stage2_holdout").sum())
        if actual_holdout != EXPECTED_STAGE2_HOLDOUT_ROWS:
            msg = f"stage2_holdout rows detected: expected=0, actual={actual_holdout}"
            blockers.append(msg)
            print(f"[ERROR] {msg}")
        else:
            print(f"[INFO] stage2_holdout row count OK: {actual_holdout}")
    else:
        msg = "stage_split column not found in manifest"
        blockers.append(msg)
        print(f"[ERROR] {msg}")

    # 4. LUNG1-295, LUNG1-415 row 수 == 0
    for pid in EXCLUDED_PATIENT_IDS:
        count = int((manifest_df["patient_id"] == pid).sum())
        if count != 0:
            msg = f"excluded patient detected: {pid} has {count} rows (must be 0)"
            blockers.append(msg)
            print(f"[ERROR] {msg}")
        else:
            print(f"[INFO] excluded patient {pid}: 0 rows OK")

    # 5. v2/v2v2 path == 0
    v2_mask = manifest_df["npz_path"].str.contains("v2", na=False)
    v2_count = int(v2_mask.sum())
    if v2_count != 0:
        msg = f"v2/v2v2 path detected: {v2_count} rows (must be 0)"
        blockers.append(msg)
        print(f"[ERROR] {msg}")
    else:
        print(f"[INFO] v2/v2v2 path check OK: 0 rows")

    # 6. training_manifest_status == "not_training_manifest" (모든 row)
    if "training_manifest_status" in manifest_df.columns:
        non_ok = int((manifest_df["training_manifest_status"] != "not_training_manifest").sum())
        if non_ok != 0:
            msg = f"training_manifest_status != 'not_training_manifest' in {non_ok} rows"
            blockers.append(msg)
            print(f"[ERROR] {msg}")
        else:
            print(f"[INFO] training_manifest_status check OK: all 'not_training_manifest'")
    else:
        msg = "training_manifest_status column not found in manifest"
        blockers.append(msg)
        print(f"[ERROR] {msg}")

    # 7. approval_required_before_training == True (모든 row)
    if "approval_required_before_training" in manifest_df.columns:
        col = manifest_df["approval_required_before_training"]
        true_mask = col.apply(lambda x: str(x).strip().lower() == "true")
        non_true = int((~true_mask).sum())
        if non_true != 0:
            msg = f"approval_required_before_training != True in {non_true} rows"
            blockers.append(msg)
            print(f"[ERROR] {msg}")
        else:
            print(f"[INFO] approval_required_before_training check OK: all True")
    else:
        msg = "approval_required_before_training column not found in manifest"
        blockers.append(msg)
        print(f"[ERROR] {msg}")

    return blockers


# ──────────────────────────────────────────────────────────
# Output overwrite guard
# ──────────────────────────────────────────────────────────
def check_output_guard(output_root: Path) -> None:
    """
    output root 및 출력 파일 존재 시 즉시 sys.exit(1).
    기존 파일 덮어쓰기 절대 금지.
    """
    # output root 디렉토리 존재 여부
    if output_root.exists():
        print(f"[ERROR] output root 디렉토리가 이미 존재합니다: {output_root}")
        print("[ERROR] 기존 파일 덮어쓰기 금지. output root를 삭제하거나 다른 경로를 지정하세요.")
        sys.exit(1)

    # 개별 출력 파일 존재 여부
    output_files = [
        OUTPUT_CSV_NAME,
        OUTPUT_CSV_TMP_NAME,
        OUTPUT_JSON_NAME,
        OUTPUT_MD_NAME,
        OUTPUT_ERROR_CSV_NAME,
        OUTPUT_RUNTIME_CSV_NAME,
        OUTPUT_DONE_JSON_NAME,
    ]
    for fname in output_files:
        fpath = output_root / fname
        if fpath.exists():
            print(f"[ERROR] 출력 파일이 이미 존재합니다: {fpath}")
            print("[ERROR] 기존 파일 덮어쓰기 금지.")
            sys.exit(1)


def recheck_output_guard_before_save(output_root: Path, filename: str) -> None:
    """저장 직전 재확인 — 존재하면 즉시 sys.exit(1)"""
    fpath = output_root / filename
    if fpath.exists():
        print(f"[ERROR] 저장 직전 재확인: 출력 파일이 이미 존재합니다: {fpath}")
        print("[ERROR] 기존 파일 덮어쓰기 금지. 즉시 중단합니다.")
        sys.exit(1)


# ──────────────────────────────────────────────────────────
# score 분포 통계
# ──────────────────────────────────────────────────────────
def _scalar_stats(values: list) -> dict:
    """NaN/Inf를 제외한 scalar 값 목록의 min/max/mean/median 계산"""
    valid = [v for v in values if isinstance(v, float) and not math.isnan(v) and not math.isinf(v)]
    if not valid:
        nan = float("nan")
        return {"min": nan, "max": nan, "mean": nan, "median": nan}
    arr = np.array(valid, dtype=np.float64)
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
    }


# ──────────────────────────────────────────────────────────
# MD 보고서 생성
# ──────────────────────────────────────────────────────────
def generate_md_report(
    manifest_path: str,
    checkpoint_path: str,
    batch_size: int,
    scored_count: int,
    score_l1_stats: dict,
    score_l1max_stats: dict,
    score_mse_stats: dict,
    has_nan_count: int,
    has_inf_count: int,
    error_count: int,
    scoring_pass: bool,
    done_marker_created: bool,
    blockers: list,
    timestamp: str,
) -> str:
    """Phase 7.2 full scoring MD 보고서 생성"""

    pass_str = "PASS" if scoring_pass else "FAIL"
    blockers_str = "\n".join(f"- {b}" for b in blockers) if blockers else "없음"

    def fmt_stats(stats: dict) -> str:
        if not stats:
            return "N/A"
        return (
            f"min={stats.get('min', float('nan')):.6f}, "
            f"max={stats.get('max', float('nan')):.6f}, "
            f"mean={stats.get('mean', float('nan')):.6f}, "
            f"median={stats.get('median', float('nan')):.6f}"
        )

    md = f"""# Phase 7.2 v1/v1 stage1_dev Filtered Full Scoring Report

생성 시각: {timestamp}

---

## 1. Phase 7.2 목적

filtered S6-A manifest (phase6_1b) 의 {EXPECTED_MANIFEST_ROWS:,}개 crops 전체에 대해
canonical 27컬럼 reconstruction score CSV를 생성한다.
metric 계산 / threshold 계산 / training은 수행하지 않는다.

---

## 2. 입력 filtered manifest 안전성 확인

- manifest 경로: `{manifest_path}`
- 기대 row 수: {EXPECTED_MANIFEST_ROWS:,}
- 기대 unique patient 수: {EXPECTED_UNIQUE_PATIENTS}
- stage2_holdout rows: 0 (기대값)
- 제외 환자 (LUNG1-295, LUNG1-415): 0 rows (기대값)
- v2/v2v2 path: 0 rows (기대값)
- training_manifest_status: all "not_training_manifest"
- approval_required_before_training: all True
- blockers:
{blockers_str}

---

## 3. 사용한 model / checkpoint

- model_class: `{MODEL_CLASS}`
- model_tag: `{MODEL_TAG}`
- checkpoint: `{checkpoint_path}`

---

## 4. full scoring 범위

- 대상 crops: {EXPECTED_MANIFEST_ROWS:,} crops (전체 filtered manifest)
- batch_size: {batch_size} (상한: {FULL_MAX_BATCH_SIZE_LIMIT})
- 실제 scored crop 수: {scored_count:,}

---

## 5. canonical 27컬럼 schema 확인 (Phase 7.0 기준)

- canonical 컬럼 수: {len(CANONICAL_COLUMNS)}개 (27컬럼 기준)
- canonical 컬럼 목록: {', '.join(CANONICAL_COLUMNS)}

---

## 6. Score 정의

| 컬럼명 | 정의 |
|--------|------|
| crop_score_l1_mean | mean(|input - recon|) over all pixels, 6 channels |
| crop_score_l1_max | max(|input - recon|) over all pixels, 6 channels |
| crop_score_mse_mean | mean((input - recon)^2) over all pixels, 6 channels |
| channel_0_l1_mean ~ channel_5_l1_mean | 채널별 mean(|input - recon|) |
| lung_channels_l1_mean | mean(|input - recon|) over channels 0~2 |
| mediastinal_channels_l1_mean | mean(|input - recon|) over channels 3~5 |

**주의**: lung_channels_l1_mean 및 mediastinal_channels_l1_mean은
channel-group score이며, **anatomical certainty not asserted**.
채널 0~2를 lung window, 채널 3~5를 mediastinal window로 가정하지만
실제 해부학적 의미는 추가 검증 필요.

---

## 7. full score 분포 요약

| 항목 | min | max | mean | median |
|------|-----|-----|------|--------|
| crop_score_l1_mean | {score_l1_stats.get('min', float('nan')):.6f} | {score_l1_stats.get('max', float('nan')):.6f} | {score_l1_stats.get('mean', float('nan')):.6f} | {score_l1_stats.get('median', float('nan')):.6f} |
| crop_score_l1_max | {score_l1max_stats.get('min', float('nan')):.6f} | {score_l1max_stats.get('max', float('nan')):.6f} | {score_l1max_stats.get('mean', float('nan')):.6f} | {score_l1max_stats.get('median', float('nan')):.6f} |
| crop_score_mse_mean | {score_mse_stats.get('min', float('nan')):.6f} | {score_mse_stats.get('max', float('nan')):.6f} | {score_mse_stats.get('mean', float('nan')):.6f} | {score_mse_stats.get('median', float('nan')):.6f} |

---

## 8. NaN/Inf 확인 결과

- has_nan=True인 crop 수: {has_nan_count}
- has_inf=True인 crop 수: {has_inf_count}
- NaN/Inf 확인 통과: {"PASS" if has_nan_count == 0 and has_inf_count == 0 else "FAIL"}

---

## 9. error CSV 결과

- error_count: {error_count}
- error CSV: `{OUTPUT_ERROR_CSV_NAME}`
- error 없음 여부: {"PASS (에러 없음)" if error_count == 0 else f"FAIL (에러 {error_count}건)"}

---

## 10. metric / threshold / training 미수행 확인

- metric_calculation_executed: False
- threshold_calculated: False
- training_executed: False
- backward_executed: False
- optimizer_step_executed: False
- checkpoint_created: False

---

## 11. DONE marker 생성 여부

- DONE marker 생성: {"YES" if done_marker_created else "NO (scoring_pass=False 또는 error 발생)"}
- DONE marker 파일: `{OUTPUT_DONE_JSON_NAME}`

---

## 12. 최종 판정

**scoring_pass: {pass_str}**

{f"blockers: {chr(10).join(f'  - {b}' for b in blockers)}" if blockers else "blocker 없음"}

---

## 13. 다음 단계

{"scoring_pass=PASS 확인됨. Phase 7.3 metric calculation preflight 승인 요청. metric 계산은 별도 승인 후 진행." if scoring_pass else "scoring_pass=FAIL. error CSV 및 FAIL 원인 확인 후 수정 필요. metric 계산 진행 불가."}
"""
    return md


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────
def main():
    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    args = parse_args()

    # --run-full 없으면 실행 거부
    if not args.run_full:
        print("[ERROR] --run-full 플래그가 필요합니다.")
        print(
            "사용법: python scripts/phase7_2_v1v1_stage1_dev_full_scoring.py "
            "--run-full [--batch-size 32]"
        )
        sys.exit(1)

    # batch-size 상한 검증
    if args.batch_size > FULL_MAX_BATCH_SIZE_LIMIT:
        print(
            f"[ERROR] --batch-size={args.batch_size}는 상한 {FULL_MAX_BATCH_SIZE_LIMIT}을 초과합니다. "
            f"최대 {FULL_MAX_BATCH_SIZE_LIMIT} 이하로 설정하세요."
        )
        sys.exit(1)

    print(f"[INFO] Phase 7.2 v1/v1 stage1_dev filtered full scoring 시작")
    print(f"[INFO] batch_size={args.batch_size}, start_time={start_datetime}")

    # ── 경로 기준 설정 (프로젝트 루트 기준 상대 경로 → 절대 경로)
    project_root = Path(__file__).resolve().parent.parent
    checkpoint_path = str(
        Path(args.checkpoint) if Path(args.checkpoint).is_absolute()
        else project_root / args.checkpoint
    )
    manifest_path = str(
        Path(args.manifest) if Path(args.manifest).is_absolute()
        else project_root / args.manifest
    )
    output_root = (
        Path(args.output_root) if Path(args.output_root).is_absolute()
        else project_root / args.output_root
    )

    print(f"[INFO] checkpoint: {checkpoint_path}")
    print(f"[INFO] manifest: {manifest_path}")
    print(f"[INFO] output_root: {output_root}")

    # ── output overwrite guard (실행 초반)
    check_output_guard(output_root)

    # ── checkpoint 존재 확인
    ckpt_p = Path(checkpoint_path)
    if not ckpt_p.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_p}")
        sys.exit(1)

    # ── manifest 존재 확인
    mf_p = Path(manifest_path)
    if not mf_p.exists():
        print(f"[ERROR] Filtered manifest not found: {mf_p}")
        sys.exit(1)

    # ── manifest 로드
    print(f"[INFO] Loading manifest: {mf_p}")
    manifest_df = pd.read_csv(mf_p)
    print(f"[INFO] Manifest loaded: {len(manifest_df)} rows")

    # ── manifest safety 검증 (실패 시 sys.exit(1))
    print("[INFO] Manifest safety 검증 시작...")
    blockers = verify_manifest_safety(manifest_df)
    if blockers:
        print(f"[ERROR] Manifest safety 검증 실패: {len(blockers)}개 blocker")
        for b in blockers:
            print(f"[ERROR]   - {b}")
        sys.exit(1)
    print("[INFO] Manifest safety 검증: 모두 통과")

    # ── manifest에 _original_row_id 추가 (원본 row index 보존, manifest 수정 아님)
    manifest_df = manifest_df.reset_index(drop=False).rename(columns={"index": "_original_row_id"})

    # ── device 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # ── checkpoint 로드 (read-only)
    print(f"[INFO] Loading checkpoint (read-only): {ckpt_p}")
    ckpt = torch.load(str(ckpt_p), map_location=device, weights_only=False)
    print(
        f"[INFO] Checkpoint loaded: epoch={ckpt.get('epoch', 'N/A')}, "
        f"best_val_loss={ckpt.get('best_val_loss', 'N/A')}"
    )

    # ── 모델 구성 및 weight 로드
    model = ConvAutoencoder2p5D(input_channels=6, base_channels=32).to(device)

    if "model_state_dict" not in ckpt:
        print("[ERROR] checkpoint에 'model_state_dict' 키가 없습니다.")
        sys.exit(1)

    model.load_state_dict(ckpt["model_state_dict"])
    print("[INFO] model_state_dict 로드 완료.")
    model.eval()
    print("[INFO] model.eval() 설정 완료.")

    # ── output root 생성 (guard 통과 후)
    output_root.mkdir(parents=True, exist_ok=False)
    print(f"[INFO] output_root 생성: {output_root}")

    # ── error CSV 초기화 (헤더만 먼저 생성)
    error_csv_path = output_root / OUTPUT_ERROR_CSV_NAME
    recheck_output_guard_before_save(output_root, OUTPUT_ERROR_CSV_NAME)
    pd.DataFrame(columns=ERROR_CSV_COLUMNS).to_csv(error_csv_path, index=False)
    print(f"[INFO] error CSV 초기화 완료: {error_csv_path}")

    # ── Dataset / DataLoader 구성
    dataset = FullScoreDataset(rows=manifest_df)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_fn,
    )

    total_batches = len(loader)
    print(f"[INFO] DataLoader: {len(dataset)} crops, {total_batches} batches, batch_size={args.batch_size}")

    # ── tmp CSV 경로
    tmp_csv_path = output_root / OUTPUT_CSV_TMP_NAME
    recheck_output_guard_before_save(output_root, OUTPUT_CSV_TMP_NAME)

    # ── inference (no_grad, no backward, no optimizer)
    all_score_rows = []
    all_error_rows = []
    processed_crops = 0

    inference_start = time.time()

    with torch.no_grad():
        for batch_idx, (batch_tensors, batch_metas) in enumerate(loader):
            # GPU OOM 처리: 자동 축소 금지, 즉시 sys.exit(1)
            try:
                batch_tensors = batch_tensors.to(device)
                recons = model(batch_tensors)
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "cuda out of memory" in str(e).lower():
                    print(f"[ERROR] GPU OOM 발생 (batch_idx={batch_idx}). 자동 batch size 축소 금지.")
                    print(f"[ERROR] OOM 메시지: {e}")
                    print("[ERROR] batch_size를 줄여서 재실행하세요. 즉시 중단합니다.")
                    sys.exit(1)
                else:
                    # forward error 기록
                    for meta in batch_metas:
                        all_error_rows.append({
                            "row_id": meta["row_id"],
                            "crop_id": meta["crop_id"],
                            "patient_id": meta["patient_id"],
                            "npz_path": meta["npz_path"],
                            "error_type": "forward_error",
                            "error_message": str(e),
                        })
                        all_score_rows.append(_make_nan_score_row(meta, checkpoint_path))
                    processed_crops += len(batch_metas)
                    print(f"[WARNING] forward error at batch_idx={batch_idx}: {e}")
                    continue

            # 첫 번째 batch: shape 출력
            if batch_idx == 0:
                print(f"[INFO] first_batch_input_shape={tuple(batch_tensors.shape)}")
                print(f"[INFO] first_batch_recon_shape={tuple(recons.shape)}")

            # score 계산
            score_rows, error_rows = compute_batch_full_scores(
                inputs=batch_tensors,
                recons=recons,
                metas=batch_metas,
                checkpoint_path=checkpoint_path,
            )
            all_score_rows.extend(score_rows)
            all_error_rows.extend(error_rows)
            processed_crops += len(batch_metas)

            # 진행률 출력: 매 10 batch마다
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                elapsed = time.time() - inference_start
                pct = processed_crops / EXPECTED_MANIFEST_ROWS * 100
                print(
                    f"[INFO] batch {batch_idx + 1}/{total_batches} | "
                    f"processed={processed_crops:,}/{EXPECTED_MANIFEST_ROWS:,} "
                    f"({pct:.1f}%) | elapsed={elapsed:.1f}s"
                )

    print(f"[INFO] Inference 완료: {processed_crops:,} crops processed")

    # ── error count 집계
    # error_rows에서 nan/inf 에러는 중복 가능 (동일 crop에 nan + inf)
    # 에러 crop 수 = has_nan 또는 has_inf인 row 수 + load/shape/forward 에러 수
    # 여기서는 에러 CSV 기록 수로 집계
    error_count = len([r for r in all_error_rows if r["error_type"] not in ("nan_error", "inf_error")])
    nan_error_count = len([r for r in all_error_rows if r["error_type"] == "nan_error"])
    inf_error_count = len([r for r in all_error_rows if r["error_type"] == "inf_error"])
    total_error_rows_count = len(all_error_rows)

    print(f"[INFO] error_count (load/shape/forward/score): {error_count}")
    print(f"[INFO] nan_error_count: {nan_error_count}")
    print(f"[INFO] inf_error_count: {inf_error_count}")

    # ── score rows DataFrame 생성
    df_scores = pd.DataFrame(all_score_rows, columns=CANONICAL_COLUMNS)

    # ── temp CSV 저장 (rename 전)
    recheck_output_guard_before_save(output_root, OUTPUT_CSV_TMP_NAME)
    df_scores.to_csv(tmp_csv_path, index=False)
    print(f"[INFO] tmp CSV 저장 완료: {tmp_csv_path} ({len(df_scores)} rows)")

    # ── scoring_pass 조건 검증
    total_rows = len(df_scores)
    has_nan_count = int(df_scores["has_nan"].sum())
    has_inf_count = int(df_scores["has_inf"].sum())

    scoring_pass_conditions = {
        "total_rows_ok": total_rows == EXPECTED_MANIFEST_ROWS,
        "has_nan_zero": has_nan_count == 0,
        "has_inf_zero": has_inf_count == 0,
        "error_count_zero": error_count == 0,
    }

    scoring_pass = all(scoring_pass_conditions.values())

    if not scoring_pass_conditions["total_rows_ok"]:
        print(
            f"[WARNING] total_rows={total_rows} != expected={EXPECTED_MANIFEST_ROWS}. "
            "scoring_pass=False (tmp 상태 유지)"
        )
    if not scoring_pass_conditions["has_nan_zero"]:
        print(
            f"[WARNING] has_nan_count={has_nan_count} != 0. "
            "scoring_pass=False (tmp 상태 유지)"
        )
    if not scoring_pass_conditions["has_inf_zero"]:
        print(
            f"[WARNING] has_inf_count={has_inf_count} != 0. "
            "scoring_pass=False (tmp 상태 유지)"
        )
    if not scoring_pass_conditions["error_count_zero"]:
        print(
            f"[WARNING] error_count={error_count} != 0. "
            "scoring_pass=False (tmp 상태 유지)"
        )

    # ── 조건 통과 시 final CSV로 rename
    final_csv_path = output_root / OUTPUT_CSV_NAME
    if scoring_pass:
        recheck_output_guard_before_save(output_root, OUTPUT_CSV_NAME)
        tmp_csv_path.rename(final_csv_path)
        print(f"[INFO] scoring_pass=True: tmp → final CSV rename 완료: {final_csv_path}")
    else:
        print(f"[WARNING] scoring_pass=False: tmp 상태 유지. final CSV rename 건너뜀.")
        print(f"[WARNING] tmp CSV 경로: {tmp_csv_path}")

    # ── error CSV 업데이트 (에러가 있으면 기록)
    if all_error_rows:
        recheck_output_guard_before_save(output_root, OUTPUT_ERROR_CSV_NAME)
        # 이미 헤더만 있는 파일을 덮어써야 하므로, 헤더 초기화 후 재생성
        # (초기 헤더 파일은 비어 있으므로 새 내용으로 작성)
        # 단, 파일이 이미 있으므로 Path.write 방식으로 처리
        df_errors = pd.DataFrame(all_error_rows, columns=ERROR_CSV_COLUMNS)
        # 헤더 초기화 파일은 이미 생성되어 있으므로, 내용을 교체
        # overwrite guard는 초기화 시점에 이미 통과했으므로 덮어쓰기 허용
        df_errors.to_csv(error_csv_path, index=False)
        print(f"[INFO] error CSV 저장 완료: {error_csv_path} ({len(df_errors)} rows)")
    else:
        print(f"[INFO] error CSV: 에러 없음 (헤더만 존재): {error_csv_path}")

    # ── score 분포 통계 계산
    valid_l1 = [r["crop_score_l1_mean"] for r in all_score_rows]
    valid_l1max = [r["crop_score_l1_max"] for r in all_score_rows]
    valid_mse = [r["crop_score_mse_mean"] for r in all_score_rows]

    score_l1_stats = _scalar_stats(valid_l1)
    score_l1max_stats = _scalar_stats(valid_l1max)
    score_mse_stats = _scalar_stats(valid_mse)

    # ── runtime 계산
    end_time = time.time()
    end_datetime = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    runtime_seconds = end_time - start_time

    total_success_rows = total_rows - has_nan_count
    total_error_rows_summary = has_nan_count  # NaN인 row = 에러 포함 row

    output_csv_path_str = str(final_csv_path) if scoring_pass else str(tmp_csv_path)

    # ── runtime summary CSV 저장
    runtime_row = {
        "start_time": start_datetime,
        "end_time": end_datetime,
        "runtime_seconds": runtime_seconds,
        "batch_size": args.batch_size,
        "total_expected_rows": EXPECTED_MANIFEST_ROWS,
        "total_processed_rows": processed_crops,
        "total_success_rows": total_success_rows,
        "total_error_rows": total_error_rows_summary,
        "device": str(device),
        "checkpoint_path": checkpoint_path,
        "output_csv_path": output_csv_path_str,
        "scoring_pass": scoring_pass,
    }
    recheck_output_guard_before_save(output_root, OUTPUT_RUNTIME_CSV_NAME)
    runtime_csv_path = output_root / OUTPUT_RUNTIME_CSV_NAME
    pd.DataFrame([runtime_row], columns=RUNTIME_CSV_COLUMNS).to_csv(runtime_csv_path, index=False)
    print(f"[INFO] runtime summary CSV 저장 완료: {runtime_csv_path}")

    # ── summary JSON 저장
    timestamp = end_datetime
    summary_json = {
        "input_filtered_manifest_path": manifest_path,
        "output_score_csv_path": output_csv_path_str,
        "checkpoint_path": checkpoint_path,
        "model_class": MODEL_CLASS,
        "batch_size": args.batch_size,
        "scored_crop_count": total_rows,
        "manifest_row_count": EXPECTED_MANIFEST_ROWS,
        "unique_patient_count": EXPECTED_UNIQUE_PATIENTS,
        "stage2_holdout_row_count": 0,
        "excluded_patients_absent": True,
        "v2_path_detected": False,
        "canonical_schema_column_count": len(CANONICAL_COLUMNS),
        "canonical_schema_columns": CANONICAL_COLUMNS,
        "score_l1_min": score_l1_stats["min"],
        "score_l1_max": score_l1_stats["max"],
        "score_l1_mean": score_l1_stats["mean"],
        "score_l1_median": score_l1_stats["median"],
        "score_mse_min": score_mse_stats["min"],
        "score_mse_max": score_mse_stats["max"],
        "score_mse_mean": score_mse_stats["mean"],
        "score_mse_median": score_mse_stats["median"],
        "has_nan_count": has_nan_count,
        "has_inf_count": has_inf_count,
        "failed_crop_count": error_count,
        "metric_calculation_executed": False,
        "threshold_calculated": False,
        "training_executed": False,
        "backward_executed": False,
        "optimizer_step_executed": False,
        "checkpoint_created": False,
        "full_scoring_executed": True,
        "scoring_pass": scoring_pass,
        "blockers": blockers,
        "runtime_seconds": runtime_seconds,
        "next_step_recommendation": (
            "scoring_pass=True. Phase 7.3 metric calculation preflight 승인 요청."
            if scoring_pass else
            "scoring_pass=False. error CSV 및 FAIL 원인 확인 후 수정 필요."
        ),
    }

    recheck_output_guard_before_save(output_root, OUTPUT_JSON_NAME)
    json_path = output_root / OUTPUT_JSON_NAME
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False, default=str)
    print(f"[INFO] summary JSON 저장 완료: {json_path}")

    # ── MD 보고서 저장
    done_marker_created = False  # 아직 생성 전

    md_content = generate_md_report(
        manifest_path=manifest_path,
        checkpoint_path=checkpoint_path,
        batch_size=args.batch_size,
        scored_count=total_rows,
        score_l1_stats=score_l1_stats,
        score_l1max_stats=score_l1max_stats,
        score_mse_stats=score_mse_stats,
        has_nan_count=has_nan_count,
        has_inf_count=has_inf_count,
        error_count=error_count,
        scoring_pass=scoring_pass,
        done_marker_created=done_marker_created,
        blockers=blockers,
        timestamp=timestamp,
    )

    recheck_output_guard_before_save(output_root, OUTPUT_MD_NAME)
    md_path = output_root / OUTPUT_MD_NAME
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[INFO] MD 보고서 저장 완료: {md_path}")

    # ── DONE marker JSON 생성 (scoring_pass=True, error_count=0인 경우에만)
    if scoring_pass and error_count == 0:
        done_marker_created = True
        done_json = {
            "scoring_pass": True,
            "output_csv_row_count": total_rows,
            "error_count": error_count,
            "metric_calculation_executed": False,
            "threshold_calculated": False,
            "training_executed": False,
        }
        recheck_output_guard_before_save(output_root, OUTPUT_DONE_JSON_NAME)
        done_path = output_root / OUTPUT_DONE_JSON_NAME
        with open(done_path, "w", encoding="utf-8") as f:
            json.dump(done_json, f, indent=2, ensure_ascii=False)
        print(f"[INFO] DONE marker JSON 생성 완료: {done_path}")
    else:
        print(
            f"[WARNING] DONE marker 생성 건너뜀 "
            f"(scoring_pass={scoring_pass}, error_count={error_count})"
        )

    # ── 최종 요약 출력
    print("\n[INFO] ===== Phase 7.2 Full Scoring 결과 요약 =====")
    print(f"[INFO] total_rows: {total_rows:,} (expected: {EXPECTED_MANIFEST_ROWS:,})")
    print(f"[INFO] has_nan_count: {has_nan_count}")
    print(f"[INFO] has_inf_count: {has_inf_count}")
    print(f"[INFO] error_count (load/shape/forward/score): {error_count}")
    print(f"[INFO] crop_score_l1_mean: min={score_l1_stats['min']:.6f}, max={score_l1_stats['max']:.6f}, mean={score_l1_stats['mean']:.6f}")
    print(f"[INFO] crop_score_mse_mean: min={score_mse_stats['min']:.6f}, max={score_mse_stats['max']:.6f}, mean={score_mse_stats['mean']:.6f}")
    print(f"[INFO] metric_calculation_executed: False")
    print(f"[INFO] threshold_calculated: False")
    print(f"[INFO] training_executed: False")
    print(f"[INFO] scoring_pass: {scoring_pass}")
    print(f"[INFO] runtime_seconds: {runtime_seconds:.1f}s")
    print(f"\n[INFO] 출력 파일:")
    print(f"[INFO]   CSV:      {final_csv_path if scoring_pass else tmp_csv_path}")
    print(f"[INFO]   JSON:     {json_path}")
    print(f"[INFO]   MD:       {md_path}")
    print(f"[INFO]   error:    {error_csv_path}")
    print(f"[INFO]   runtime:  {runtime_csv_path}")
    if scoring_pass and error_count == 0:
        print(f"[INFO]   DONE:     {output_root / OUTPUT_DONE_JSON_NAME}")
    print("\n[INFO] Phase 7.2 v1/v1 stage1_dev filtered full scoring 완료.")


if __name__ == "__main__":
    main()
