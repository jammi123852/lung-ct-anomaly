"""
Phase 8.4 stage2_holdout RD4AD Full Scoring

목적:
  stage2_holdout 143,735개 crop 전체에 대해 RD4AD canonical 27컬럼
  reconstruction score CSV를 생성한다.

금지 사항:
  - metric 계산 (AUROC/AUPRC/threshold/p95/p99/hit-rate 포함)
  - training / backward / optimizer step / checkpoint 생성
  - 기존 Phase 6/7/8 output 수정·삭제·덮어쓰기
  - v2/v2v2 접근
  - pip/conda install / 외부 다운로드

실행 guard:
  --run-full과 --confirm-run 둘 다 없으면 dry-run 보고만 수행 후 종료.
  실제 full scoring은 두 플래그가 모두 있을 때만 실행.
  batch_size > 32이면 즉시 sys.exit(1).

approval_required_before_scoring 해석 보정:
  manifest의 approval_required_before_scoring=True는
  "scoring 전 사용자 명시 승인 필요" flag이며, scoring 불가를 뜻하지 않는다.
  이중 실행 플래그(--run-full + --confirm-run)와 사용자 승인으로 실행 허용.
  approval_gate_interpretation_corrected=True를 summary JSON에 항상 기록한다.
"""

import argparse
import json
import math
import re
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
EXPECTED_MANIFEST_ROWS = 143735
EXPECTED_UNIQUE_PATIENTS = 154
EXPECTED_POSITIVE_ROWS = 51335
EXPECTED_HARD_NEGATIVE_ROWS = 92400
EXPECTED_STAGE_SPLIT = "stage2_holdout"

DEFAULT_CHECKPOINT = (
    "outputs/second-stage-lesion-refiner-v1/models/"
    "rd4ad_2p5d_normal_mw_fixed96_v1/checkpoints/best_val_loss.pt"
)
DEFAULT_MANIFEST = (
    "outputs/second-stage-lesion-refiner-v1/datasets/"
    "s6a_stage2_holdout_filtered_manifest_v1.csv"
)
DEFAULT_SCORE_OUTPUT_ROOT = (
    "outputs/second-stage-lesion-refiner-v1/scores/"
    "phase8_4_stage2_full_scoring_v1/"
)
DEFAULT_REVIEW_OUTPUT_ROOT = (
    "outputs/second-stage-lesion-refiner-v1/review_annotations/"
    "phase8_4_stage2_full_scoring_v1/"
)

OUTPUT_CSV_NAME = "phase8_4_stage2_full_scoring_v1.csv"
OUTPUT_CSV_TMP_NAME = "phase8_4_stage2_full_scoring_v1.csv.tmp"
OUTPUT_JSON_NAME = "phase8_4_stage2_full_scoring_summary_v1.json"
OUTPUT_MD_NAME = "phase8_4_stage2_full_scoring_report_v1.md"
OUTPUT_ERROR_CSV_NAME = "phase8_4_stage2_full_scoring_errors_v1.csv"
OUTPUT_RUNTIME_CSV_NAME = "phase8_4_stage2_full_scoring_runtime_summary_v1.csv"
OUTPUT_DONE_JSON_NAME = "phase8_4_stage2_full_scoring_DONE.json"

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

# extra 13컬럼
EXTRA_COLUMNS = [
    "row_id",
    "crop_shape",
    "input_nan_count",
    "input_inf_count",
    "recon_nan_count",
    "recon_inf_count",
    "error_nan_count",
    "error_inf_count",
    "score_nan_count",
    "score_inf_count",
    "scoring_status",
    "issue",
    "note",
]

ALL_COLUMNS = CANONICAL_COLUMNS + EXTRA_COLUMNS

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

SCORE_SCALAR_COLUMNS = [
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
]


# ──────────────────────────────────────────────────────────
# Argument Parser
# ──────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Phase 8.4 stage2_holdout 143,735개 crop 전체 full scoring. "
            "--run-full과 --confirm-run 둘 다 없으면 dry-run만 수행합니다."
        )
    )
    parser.add_argument(
        "--run-full",
        action="store_true",
        help="[필수] full scoring 실행 플래그 1. --confirm-run과 함께 사용해야 실행됩니다.",
    )
    parser.add_argument(
        "--confirm-run",
        action="store_true",
        help="[필수] full scoring 실행 플래그 2. --run-full과 함께 사용해야 실행됩니다.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help=f"배치 크기 (상한 {FULL_MAX_BATCH_SIZE_LIMIT}, default=32)",
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="checkpoint 경로 (best_val_loss.pt)",
    )
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help="stage2_holdout filtered manifest CSV 경로",
    )
    parser.add_argument(
        "--score-output-root",
        default=DEFAULT_SCORE_OUTPUT_ROOT,
        help="full scoring score 출력 root 디렉토리",
    )
    parser.add_argument(
        "--review-output-root",
        default=DEFAULT_REVIEW_OUTPUT_ROOT,
        help="full scoring review 출력 root 디렉토리",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────
# Model (phase8_3b와 동일한 구조)
# ──────────────────────────────────────────────────────────
class ConvAutoencoder2p5D(nn.Module):
    """
    2.5D reconstruction baseline.
    input_channels=6 (lung z-1/z/z+1, mediastinal z-1/z/z+1)
    Encoder: 96→48→24→12 (3× MaxPool2d)
    Decoder: 12→24→48→96 (3× ConvTranspose2d) + Sigmoid
    phase8_3b_stage2_scoring_smoke.py와 동일한 구조.
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
    Phase 8.4 full scoring용 Dataset.
    stage2_holdout manifest 전체 (143,735개)를 대상으로 한다.
    crop_id는 s8a_{row_id:06d} 형식으로 생성 (manifest의 row_id 컬럼 기반).
    """

    def __init__(self, rows: pd.DataFrame):
        self.df = rows.reset_index(drop=True)
        print(f"[INFO] FullScoreDataset: {len(self.df)} crops")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        npz_path = str(row["npz_path"])
        row_id = int(row["row_id"])

        crop_id = f"s8a_{row_id:06d}"

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
            "_crop_shape": "",
        }

        img = None
        try:
            if not Path(npz_path).exists():
                raise FileNotFoundError(f"npz_path not found: {npz_path}")
            with np.load(npz_path) as data:
                if "image" not in data:
                    raise KeyError(f"'image' key not found in {npz_path}")
                img = data["image"].astype(np.float32)
            meta["_crop_shape"] = str(tuple(img.shape))
        except Exception as e:
            meta["_load_error"] = str(e)

        if meta["_load_error"] is not None:
            dummy = torch.zeros(6, 96, 96, dtype=torch.float32)
            return dummy, meta

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
    tensors = torch.stack([item[0] for item in batch])
    metas = [item[1] for item in batch]
    return tensors, metas


_V2_PATTERN = re.compile(r'(^|/)v2v2(/|$)|(^|/)v2(/|$)')


def _normalize_label(val) -> str:
    """label 값을 'positive'/'hard_negative'/'unknown' 문자열로 정규화."""
    s = str(val).strip()
    try:
        numeric = float(s)
        if numeric == 1.0:
            return "positive"
        if numeric == 0.0:
            return "hard_negative"
    except (ValueError, TypeError):
        pass
    sl = s.lower()
    if sl == "positive":
        return "positive"
    if sl == "hard_negative":
        return "hard_negative"
    return "unknown"


# ──────────────────────────────────────────────────────────
# Manifest Safety 검증 (10개 항목)
# ──────────────────────────────────────────────────────────
def verify_manifest_safety(manifest_df: pd.DataFrame) -> list:
    """
    manifest safety 검증 10개 항목.
    실패 항목을 blockers list로 반환. 통과하면 빈 list.
    """
    blockers = []

    # 1. row count == 143,735
    actual_rows = len(manifest_df)
    if actual_rows != EXPECTED_MANIFEST_ROWS:
        msg = f"manifest row count mismatch: expected={EXPECTED_MANIFEST_ROWS}, actual={actual_rows}"
        blockers.append(msg)
        print(f"[ERROR] {msg}")
    else:
        print(f"[INFO] manifest row count OK: {actual_rows}")

    # 2. unique patient count == 154
    actual_patients = manifest_df["patient_id"].nunique()
    if actual_patients != EXPECTED_UNIQUE_PATIENTS:
        msg = f"unique patient count mismatch: expected={EXPECTED_UNIQUE_PATIENTS}, actual={actual_patients}"
        blockers.append(msg)
        print(f"[ERROR] {msg}")
    else:
        print(f"[INFO] unique patient count OK: {actual_patients}")

    # 3. stage_split unique == ['stage2_holdout'] (전 행)
    if "stage_split" in manifest_df.columns:
        non_holdout = int((manifest_df["stage_split"] != EXPECTED_STAGE_SPLIT).sum())
        if non_holdout != 0:
            msg = f"stage_split != '{EXPECTED_STAGE_SPLIT}' in {non_holdout} rows"
            blockers.append(msg)
            print(f"[ERROR] {msg}")
        else:
            print(f"[INFO] stage_split check OK: all '{EXPECTED_STAGE_SPLIT}'")
    else:
        msg = "stage_split column not found in manifest"
        blockers.append(msg)
        print(f"[ERROR] {msg}")

    # 4. approval_required_before_scoring == True 전수
    # 해석 보정: True는 사용자 명시 승인 필요 flag이며, scoring 불가가 아님
    if "approval_required_before_scoring" in manifest_df.columns:
        col = manifest_df["approval_required_before_scoring"]
        true_mask = col.apply(lambda x: str(x).strip().lower() == "true")
        non_true = int((~true_mask).sum())
        if non_true != 0:
            msg = f"approval_required_before_scoring != True in {non_true} rows"
            blockers.append(msg)
            print(f"[ERROR] {msg}")
        else:
            print(
                f"[INFO] approval_required_before_scoring check OK: all True "
                f"(해석 보정: True = 사용자 승인 필요 flag, scoring 불가 아님)"
            )
    else:
        msg = "approval_required_before_scoring column not found in manifest"
        blockers.append(msg)
        print(f"[ERROR] {msg}")

    # 5. v2/v2v2 path == 0 (path component 기준 검사)
    v2_mask = manifest_df["npz_path"].apply(
        lambda p: bool(_V2_PATTERN.search(str(p))) if pd.notna(p) else False
    )
    v2_count = int(v2_mask.sum())
    if v2_count != 0:
        msg = f"v2/v2v2 path detected: {v2_count} rows (must be 0)"
        blockers.append(msg)
        print(f"[ERROR] {msg}")
    else:
        print(f"[INFO] v2/v2v2 path check OK: 0 rows")

    # 6. sampling_label=='positive' count == 51,335
    if "sampling_label" in manifest_df.columns:
        actual_positive = int((manifest_df["sampling_label"] == "positive").sum())
        if actual_positive != EXPECTED_POSITIVE_ROWS:
            msg = (
                f"positive row count mismatch: "
                f"expected={EXPECTED_POSITIVE_ROWS}, actual={actual_positive}"
            )
            blockers.append(msg)
            print(f"[ERROR] {msg}")
        else:
            print(f"[INFO] positive row count OK: {actual_positive}")

        # 7. sampling_label=='hard_negative' count == 92,400
        actual_hard_negative = int((manifest_df["sampling_label"] == "hard_negative").sum())
        if actual_hard_negative != EXPECTED_HARD_NEGATIVE_ROWS:
            msg = (
                f"hard_negative row count mismatch: "
                f"expected={EXPECTED_HARD_NEGATIVE_ROWS}, actual={actual_hard_negative}"
            )
            blockers.append(msg)
            print(f"[ERROR] {msg}")
        else:
            print(f"[INFO] hard_negative row count OK: {actual_hard_negative}")
    else:
        msg = "sampling_label column not found in manifest"
        blockers.append(msg)
        print(f"[ERROR] {msg}")

    # 8. label/sampling_label 정합성 (정규화 후 비교)
    if "label" in manifest_df.columns and "sampling_label" in manifest_df.columns:
        normalized_label = manifest_df["label"].apply(_normalize_label)
        normalized_sampling = manifest_df["sampling_label"].apply(
            lambda v: str(v).strip().lower() if pd.notna(v) else "unknown"
        )
        mismatch_count = int((normalized_label != normalized_sampling).sum())
        if mismatch_count > 0:
            msg = (
                f"label/sampling_label mismatch (normalized compare): "
                f"{mismatch_count} rows"
            )
            blockers.append(msg)
            print(f"[ERROR] {msg}")
        else:
            print(f"[INFO] label/sampling_label 정합성 OK (normalized compare)")
    else:
        msg = "label 또는 sampling_label column not found — 정합성 검증 불가"
        blockers.append(msg)
        print(f"[ERROR] {msg}")

    # 9. npz_path empty/null == 0
    null_count = int(manifest_df["npz_path"].isnull().sum())
    empty_count = int((manifest_df["npz_path"].astype(str).str.strip() == "").sum())
    total_null_empty = null_count + empty_count
    if total_null_empty != 0:
        msg = f"npz_path null/empty: {total_null_empty} rows (must be 0)"
        blockers.append(msg)
        print(f"[ERROR] {msg}")
    else:
        print(f"[INFO] npz_path null/empty check OK: 0 rows")

    # 10. npz_path duplicate == 0
    dup_count = int(manifest_df["npz_path"].duplicated().sum())
    if dup_count != 0:
        msg = f"npz_path duplicate: {dup_count} rows (must be 0)"
        blockers.append(msg)
        print(f"[ERROR] {msg}")
    else:
        print(f"[INFO] npz_path duplicate check OK: 0 rows")

    return blockers


# ──────────────────────────────────────────────────────────
# NaN score row 생성
# ──────────────────────────────────────────────────────────
def _make_nan_score_row(
    meta: dict,
    checkpoint_path: str,
    issue: str,
    note: str,
) -> dict:
    """
    load_error 또는 shape_error 시 canonical 27컬럼 + extra를 NaN으로 채운 row 반환.
    has_nan=True, scoring_status='FAIL'로 고정.
    """
    nan = float("nan")
    row = {
        "crop_id": meta.get("crop_id", ""),
        "patient_id": meta.get("patient_id", ""),
        "npz_path": meta.get("npz_path", ""),
        "label": meta.get("label", ""),
        "sampling_label": meta.get("sampling_label", ""),
        "stage_split": meta.get("stage_split", ""),
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
        # extra
        "row_id": meta.get("row_id", ""),
        "crop_shape": meta.get("_crop_shape", ""),
        "input_nan_count": 0,
        "input_inf_count": 0,
        "recon_nan_count": 0,
        "recon_inf_count": 0,
        "error_nan_count": 0,
        "error_inf_count": 0,
        "score_nan_count": 0,
        "score_inf_count": 0,
        "scoring_status": "FAIL",
        "issue": issue,
        "note": note,
    }
    return row


# ──────────────────────────────────────────────────────────
# Score 계산
# ──────────────────────────────────────────────────────────
def compute_batch_scores(
    inputs: torch.Tensor,
    recons: torch.Tensor,
    metas: list,
    checkpoint_path: str,
) -> tuple:
    """
    배치 단위 full score 계산.
    inputs, recons: [B, 6, 96, 96]
    반환: (score_rows list, error_rows list)
    - load_error / shape_error 있으면 error_rows에 추가, score row는 _make_nan_score_row()
    - NaN/Inf score row도 score CSV에 포함 (has_nan=True로 기록), error_rows에도 추가
    """
    inputs_np = inputs.detach().cpu().numpy()
    recons_np = recons.detach().cpu().numpy()

    diff = np.abs(inputs_np - recons_np)
    diff_sq = (inputs_np - recons_np) ** 2

    score_rows = []
    error_rows = []

    for b in range(inputs_np.shape[0]):
        meta = metas[b]
        inp = inputs_np[b]
        rec = recons_np[b]
        d = diff[b]
        d_sq = diff_sq[b]

        load_error = meta.get("_load_error")
        shape_error = meta.get("_shape_error")
        crop_shape_str = meta.get("_crop_shape", "")

        # load_error / shape_error 처리
        if load_error is not None or shape_error is not None:
            issue_parts = []
            if load_error:
                issue_parts.append(f"load_error: {load_error}")
            if shape_error:
                issue_parts.append(f"shape_error: {shape_error}")
            issue_str = "; ".join(issue_parts)

            nan_row = _make_nan_score_row(
                meta=meta,
                checkpoint_path=checkpoint_path,
                issue=issue_str,
                note="",
            )
            score_rows.append(nan_row)

            if load_error:
                error_rows.append({
                    "row_id": meta.get("row_id", ""),
                    "crop_id": meta.get("crop_id", ""),
                    "patient_id": meta.get("patient_id", ""),
                    "npz_path": meta.get("npz_path", ""),
                    "error_type": "load_error",
                    "error_message": load_error,
                })
            if shape_error:
                error_rows.append({
                    "row_id": meta.get("row_id", ""),
                    "crop_id": meta.get("crop_id", ""),
                    "patient_id": meta.get("patient_id", ""),
                    "npz_path": meta.get("npz_path", ""),
                    "error_type": "shape_mismatch",
                    "error_message": shape_error,
                })
            continue

        # 정상 score 계산
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

        input_nan_count = int(np.isnan(inp).sum())
        input_inf_count = int(np.isinf(inp).sum())
        recon_nan_count = int(np.isnan(rec).sum())
        recon_inf_count = int(np.isinf(rec).sum())
        error_nan_count = int(np.isnan(d).sum())
        error_inf_count = int(np.isinf(d).sum())

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
        score_nan_count = int(sum(1 for v in scalar_scores if math.isnan(v)))
        score_inf_count = int(sum(1 for v in scalar_scores if math.isinf(v)))

        has_nan = bool(
            input_nan_count > 0
            or recon_nan_count > 0
            or error_nan_count > 0
            or score_nan_count > 0
        )
        has_inf = bool(
            input_inf_count > 0
            or recon_inf_count > 0
            or error_inf_count > 0
            or score_inf_count > 0
        )

        issue_parts = []
        if has_nan:
            issue_parts.append("has_nan=True")
        if has_inf:
            issue_parts.append("has_inf=True")
        issue_str = "; ".join(issue_parts)

        scoring_status = "PASS" if not (has_nan or has_inf) else "FAIL"

        note_str = "load_ok"

        row = {
            "crop_id": meta["crop_id"],
            "patient_id": meta["patient_id"],
            "npz_path": meta["npz_path"],
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
            # extra
            "row_id": meta["row_id"],
            "crop_shape": crop_shape_str,
            "input_nan_count": input_nan_count,
            "input_inf_count": input_inf_count,
            "recon_nan_count": recon_nan_count,
            "recon_inf_count": recon_inf_count,
            "error_nan_count": error_nan_count,
            "error_inf_count": error_inf_count,
            "score_nan_count": score_nan_count,
            "score_inf_count": score_inf_count,
            "scoring_status": scoring_status,
            "issue": issue_str,
            "note": note_str,
        }
        score_rows.append(row)

        # NaN/Inf score row도 error_rows에 추가
        if has_nan or has_inf:
            error_type = "nan_error" if has_nan else "inf_error"
            error_rows.append({
                "row_id": meta["row_id"],
                "crop_id": meta["crop_id"],
                "patient_id": meta["patient_id"],
                "npz_path": meta["npz_path"],
                "error_type": error_type,
                "error_message": issue_str,
            })

    return score_rows, error_rows


# ──────────────────────────────────────────────────────────
# Output guard
# ──────────────────────────────────────────────────────────
def check_output_guard(score_output_root: Path, review_output_root: Path) -> None:
    """
    output root 및 출력 파일 존재 시 즉시 sys.exit(1).
    기존 파일 덮어쓰기 절대 금지.
    """
    if score_output_root.exists():
        print(f"[ERROR] score output root가 이미 존재합니다: {score_output_root}")
        print("[ERROR] 기존 파일 덮어쓰기 금지. 즉시 중단합니다.")
        sys.exit(1)

    if review_output_root.exists():
        print(f"[ERROR] review output root가 이미 존재합니다: {review_output_root}")
        print("[ERROR] 기존 파일 덮어쓰기 금지. 즉시 중단합니다.")
        sys.exit(1)

    all_output_files = [
        (score_output_root, OUTPUT_CSV_NAME),
        (score_output_root, OUTPUT_CSV_TMP_NAME),
        (review_output_root, OUTPUT_JSON_NAME),
        (review_output_root, OUTPUT_MD_NAME),
        (review_output_root, OUTPUT_ERROR_CSV_NAME),
        (review_output_root, OUTPUT_RUNTIME_CSV_NAME),
        (review_output_root, OUTPUT_DONE_JSON_NAME),
    ]
    for root, fname in all_output_files:
        fpath = root / fname
        if fpath.exists():
            print(f"[ERROR] 출력 파일이 이미 존재합니다: {fpath}")
            print("[ERROR] 기존 파일 덮어쓰기 금지. 즉시 중단합니다.")
            sys.exit(1)


def recheck_before_save(root: Path, filename: str) -> None:
    """저장 직전 재확인: 파일이 이미 존재하면 즉시 sys.exit(1)."""
    fpath = root / filename
    if fpath.exists():
        print(f"[ERROR] 저장 직전 재확인: 출력 파일이 이미 존재합니다: {fpath}")
        print("[ERROR] 즉시 중단합니다.")
        sys.exit(1)


# ──────────────────────────────────────────────────────────
# Score 분포 통계
# ──────────────────────────────────────────────────────────
def _scalar_stats(values: list) -> dict:
    """
    NaN/Inf를 제외한 min/max/mean/median 계산.
    값 목록을 직접 받는다.
    """
    vals = [
        v for v in values
        if isinstance(v, (int, float)) and not math.isnan(float(v)) and not math.isinf(float(v))
    ]
    if not vals:
        nan = float("nan")
        return {"min": nan, "max": nan, "mean": nan, "median": nan}

    sorted_vals = sorted(vals)
    n = len(sorted_vals)
    if n % 2 == 1:
        median = float(sorted_vals[n // 2])
    else:
        median = float((sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2)

    return {
        "min": float(min(vals)),
        "max": float(max(vals)),
        "mean": float(sum(vals) / len(vals)),
        "median": median,
    }


# ──────────────────────────────────────────────────────────
# MD 보고서 생성
# ──────────────────────────────────────────────────────────
def generate_md_report(
    scoring_pass: bool,
    manifest_path: str,
    checkpoint_path: str,
    batch_size: int,
    blockers: list,
    total_processed: int,
    total_error: int,
    score_l1_stats: dict,
    score_l1max_stats: dict,
    score_mse_stats: dict,
    has_nan_count: int,
    has_inf_count: int,
    done_marker_created: bool,
    output_csv_path: str,
    timestamp: str,
) -> str:
    pass_str = "PASS" if scoring_pass else "FAIL"
    blockers_str = "\n".join(f"- {b}" for b in blockers) if blockers else "없음"

    def _fmt(v):
        try:
            if math.isnan(v) or math.isinf(v):
                return str(v)
            return f"{v:.6f}"
        except Exception:
            return str(v)

    md = f"""# Phase 8.4 Stage2 Holdout Full Scoring Report

생성 시각: {timestamp}

---

## 1. 목적

stage2_holdout 143,735개 crop 전체에 대해 RD4AD canonical 27컬럼 reconstruction score CSV를 생성한다.
full scoring 외 metric 계산 / threshold 계산 / training은 수행하지 않는다.

---

## 2. approval_required_before_scoring 해석 보정

- **approval_gate_interpretation_corrected**: True
- manifest의 `approval_required_before_scoring=True`는 "scoring 전 사용자 명시 승인 필요" flag이다.
- scoring 불가로 해석하지 않는다.
- 이중 실행 플래그(`--run-full + --confirm-run`)와 사용자 승인으로 실행 허용.

---

## 3. 입력 manifest 안전성 확인

- manifest 경로: `{manifest_path}`
- 기대 row 수: {EXPECTED_MANIFEST_ROWS:,}
- 기대 unique patient 수: {EXPECTED_UNIQUE_PATIENTS}
- stage_split: all '{EXPECTED_STAGE_SPLIT}'
- approval_required_before_scoring: all True (해석 보정 적용)
- v2/v2v2 path: 0 rows
- positive rows: {EXPECTED_POSITIVE_ROWS:,}
- hard_negative rows: {EXPECTED_HARD_NEGATIVE_ROWS:,}
- label/sampling_label 정합성: 검증 완료
- npz_path null/empty: 0 rows
- npz_path duplicate: 0 rows
- blockers:
{blockers_str}

---

## 4. model / checkpoint

- model_class: `{MODEL_CLASS}`
- model_tag: `{MODEL_TAG}`
- checkpoint: `{checkpoint_path}`
- expected input shape: (6, 96, 96)

---

## 5. Full Scoring 범위

- 대상 crop 수: {EXPECTED_MANIFEST_ROWS:,}개 (stage2_holdout 전체)
- batch_size: {batch_size}
- 처리 완료 rows: {total_processed:,}

---

## 6. canonical 27-column schema

- 컬럼 수: {len(CANONICAL_COLUMNS)}개
- 컬럼 목록: {', '.join(CANONICAL_COLUMNS)}

---

## 7. Score 정의

| 컬럼명 | 정의 |
|--------|------|
| crop_score_l1_mean | mean(|input - recon|) over all pixels, 6 channels |
| crop_score_l1_max | max(|input - recon|) over all pixels, 6 channels |
| crop_score_mse_mean | mean((input - recon)^2) over all pixels, 6 channels |
| channel_0~5_l1_mean | 채널별 mean(|input - recon|) |
| lung_channels_l1_mean | channels 0~2 mean(|input - recon|) |
| mediastinal_channels_l1_mean | channels 3~5 mean(|input - recon|) |

---

## 8. Score 분포 요약

| 항목 | min | max | mean | median |
|------|-----|-----|------|--------|
| crop_score_l1_mean | {_fmt(score_l1_stats.get('min', float('nan')))} | {_fmt(score_l1_stats.get('max', float('nan')))} | {_fmt(score_l1_stats.get('mean', float('nan')))} | {_fmt(score_l1_stats.get('median', float('nan')))} |
| crop_score_l1_max | {_fmt(score_l1max_stats.get('min', float('nan')))} | {_fmt(score_l1max_stats.get('max', float('nan')))} | {_fmt(score_l1max_stats.get('mean', float('nan')))} | {_fmt(score_l1max_stats.get('median', float('nan')))} |
| crop_score_mse_mean | {_fmt(score_mse_stats.get('min', float('nan')))} | {_fmt(score_mse_stats.get('max', float('nan')))} | {_fmt(score_mse_stats.get('mean', float('nan')))} | {_fmt(score_mse_stats.get('median', float('nan')))} |

---

## 9. NaN/Inf 확인

- has_nan=True인 crop 수: {has_nan_count}
- has_inf=True인 crop 수: {has_inf_count}
- NaN/Inf 없음 여부: {"PASS (NaN/Inf 없음)" if has_nan_count == 0 and has_inf_count == 0 else f"FAIL (has_nan={has_nan_count}, has_inf={has_inf_count})"}

---

## 10. error CSV 결과

- error_count: {total_error}
- error 없음 여부: {"PASS (에러 없음)" if total_error == 0 else f"FAIL (에러 {total_error}건)"}

---

## 11. metric / threshold / training 미수행 확인

- metric_calculation_executed: False
- threshold_calculated: False
- training_executed: False
- backward_executed: False
- optimizer_step_executed: False
- checkpoint_created: False

---

## 12. DONE marker 생성 여부

- done_marker_created: {done_marker_created}
- 조건: scoring_pass=True AND error_count=0

---

## 13. 최종 판정

**scoring_pass: {pass_str}**

{("blockers:\n" + chr(10).join(f"  - {b}" for b in blockers)) if blockers else "blocker 없음"}

출력 CSV: `{output_csv_path}`

---

## 14. 다음 단계

{"scoring_pass=True. 다음 단계는 Phase 8.5 metric calculation preflight 승인 요청이다. metric 계산은 별도 승인 후 진행한다." if scoring_pass else "scoring_pass=False. error CSV 및 FAIL 원인 확인 후 수정 필요."}
"""
    return md


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────
def main():
    start_time_float = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    args = parse_args()

    # ── 이중 플래그 guard
    if not (args.run_full and args.confirm_run):
        print("[DRY-RUN] --run-full과 --confirm-run 둘 다 필요합니다.")
        print("[DRY-RUN] 이번 실행은 dry-run 보고만 수행합니다.")
        print("[DRY-RUN] 실제 full scoring은 수행되지 않았습니다.")
        print(
            "[DRY-RUN] 실행 명령: "
            "python scripts/phase8_4_stage2_full_scoring.py "
            "--run-full --confirm-run --batch-size 32"
        )
        sys.exit(0)

    # ── batch_size 범위 검증
    if args.batch_size < 1 or args.batch_size > FULL_MAX_BATCH_SIZE_LIMIT:
        print(
            f"[ERROR] batch_size {args.batch_size} 는 1 이상 {FULL_MAX_BATCH_SIZE_LIMIT} 이하여야 합니다."
        )
        sys.exit(1)

    print(f"[INFO] Phase 8.4 stage2_holdout full scoring 시작")
    print(
        f"[INFO] expected_rows={EXPECTED_MANIFEST_ROWS}, "
        f"batch_size={args.batch_size}, start={start_datetime}"
    )

    # ── 경로 설정
    project_root = Path(__file__).resolve().parent.parent

    ckpt_path = (
        Path(args.checkpoint) if Path(args.checkpoint).is_absolute()
        else project_root / args.checkpoint
    )
    manifest_path = (
        Path(args.manifest) if Path(args.manifest).is_absolute()
        else project_root / args.manifest
    )
    score_output_root = (
        Path(args.score_output_root) if Path(args.score_output_root).is_absolute()
        else project_root / args.score_output_root
    )
    review_output_root = (
        Path(args.review_output_root) if Path(args.review_output_root).is_absolute()
        else project_root / args.review_output_root
    )

    print(f"[INFO] checkpoint: {ckpt_path}")
    print(f"[INFO] manifest: {manifest_path}")
    print(f"[INFO] score_output_root: {score_output_root}")
    print(f"[INFO] review_output_root: {review_output_root}")

    # ── output guard (scoring loop 전 선제 확인)
    check_output_guard(score_output_root, review_output_root)

    # ── checkpoint 존재 확인 (torch.load는 아래 model 구성 시 수행)
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    # ── manifest 존재 확인 및 로드
    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found: {manifest_path}")
        sys.exit(1)

    print(f"[INFO] Loading manifest: {manifest_path}")
    manifest_df = pd.read_csv(str(manifest_path))
    print(f"[INFO] Manifest loaded: {len(manifest_df)} rows")

    # ── manifest safety 검증 (10개 항목)
    print("[INFO] Manifest safety 검증 시작...")
    blockers = verify_manifest_safety(manifest_df)
    if blockers:
        print(f"[ERROR] Manifest safety 검증 실패: {len(blockers)}개 blocker")
        for b in blockers:
            print(f"[ERROR]   - {b}")
        sys.exit(1)
    print("[INFO] Manifest safety 검증: 모두 통과")

    # ── device 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # ── checkpoint 로드 + 모델 구성
    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    print(
        f"[INFO] Checkpoint loaded: epoch={ckpt.get('epoch', 'N/A')}, "
        f"best_val_loss={ckpt.get('best_val_loss', 'N/A')}"
    )

    model = ConvAutoencoder2p5D(input_channels=6, base_channels=32).to(device)

    if "model_state_dict" not in ckpt:
        print("[ERROR] checkpoint에 'model_state_dict' 키가 없습니다.")
        sys.exit(1)

    model.load_state_dict(ckpt["model_state_dict"])
    print("[INFO] model_state_dict 로드 완료.")
    model.eval()
    print("[INFO] model.eval() 설정 완료.")

    # ── DataLoader 구성 (num_workers=0, 안전 기본)
    dataset = FullScoreDataset(manifest_df)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_fn,
    )
    print(f"[INFO] DataLoader 구성 완료: num_workers=0 (안전 기본), batch_size={args.batch_size}")

    # ── scoring loop
    all_score_rows = []
    all_error_rows = []
    total_processed = 0

    print(f"[INFO] Full scoring 시작: {EXPECTED_MANIFEST_ROWS:,}개 crops")

    with torch.no_grad():
        for batch_idx, (inputs, metas) in enumerate(loader):
            inputs = inputs.to(device)
            recons = model(inputs)

            if batch_idx == 0:
                print(f"[INFO] first_batch_input_shape={tuple(inputs.shape)}")
                print(f"[INFO] first_batch_recon_shape={tuple(recons.shape)}")

            score_rows, error_rows = compute_batch_scores(
                inputs=inputs,
                recons=recons,
                metas=metas,
                checkpoint_path=str(ckpt_path),
            )
            all_score_rows.extend(score_rows)
            all_error_rows.extend(error_rows)
            total_processed += len(metas)

            if (batch_idx + 1) % 100 == 0:
                print(
                    f"[INFO] {total_processed:,}/{EXPECTED_MANIFEST_ROWS:,} processed "
                    f"(batch {batch_idx + 1})..."
                )

    print(f"[INFO] Scoring loop 완료: {total_processed:,} crops processed")

    # ── output root 생성 (scoring loop 완료 후 저장 직전, exist_ok=False)
    score_output_root.mkdir(parents=True, exist_ok=False)
    review_output_root.mkdir(parents=True, exist_ok=False)
    print(f"[INFO] score_output_root 생성: {score_output_root}")
    print(f"[INFO] review_output_root 생성: {review_output_root}")

    # ── score CSV 생성 (tmp 경로에 먼저 저장)
    score_df = pd.DataFrame(all_score_rows, columns=ALL_COLUMNS)
    tmp_csv_path = score_output_root / OUTPUT_CSV_TMP_NAME
    final_csv_path = score_output_root / OUTPUT_CSV_NAME

    recheck_before_save(score_output_root, OUTPUT_CSV_TMP_NAME)
    score_df.to_csv(str(tmp_csv_path), index=False)
    print(f"[INFO] tmp CSV 저장 완료: {tmp_csv_path} ({len(score_df)} rows)")

    # ── validation: tmp CSV 재로드 후 row count / NaN / Inf 확인
    loaded_df = pd.read_csv(str(tmp_csv_path))
    has_nan_count = int(loaded_df["has_nan"].sum())
    has_inf_count = int(loaded_df["has_inf"].sum())

    scoring_pass = (
        len(loaded_df) == EXPECTED_MANIFEST_ROWS
        and has_nan_count == 0
        and has_inf_count == 0
        and len(all_error_rows) == 0
    )
    done_marker_expected = scoring_pass
    print(
        f"[INFO] scoring_pass 판정: {scoring_pass} "
        f"(rows={len(loaded_df)}, has_nan={has_nan_count}, "
        f"has_inf={has_inf_count}, error_count={len(all_error_rows)})"
    )
    print(f"[INFO] done_marker_expected: {done_marker_expected}")

    # ── tmp → final rename
    recheck_before_save(score_output_root, OUTPUT_CSV_NAME)
    tmp_csv_path.rename(final_csv_path)
    print(f"[INFO] tmp → final CSV rename 완료: {final_csv_path}")

    # ── error CSV 저장
    error_df = pd.DataFrame(all_error_rows, columns=ERROR_CSV_COLUMNS)
    recheck_before_save(review_output_root, OUTPUT_ERROR_CSV_NAME)
    error_df.to_csv(str(review_output_root / OUTPUT_ERROR_CSV_NAME), index=False)
    print(
        f"[INFO] error CSV 저장 완료: "
        f"{review_output_root / OUTPUT_ERROR_CSV_NAME} ({len(error_df)} rows)"
    )

    # ── runtime summary 저장
    end_time_float = time.time()
    end_datetime = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    runtime_seconds = end_time_float - start_time_float

    runtime_df = pd.DataFrame(
        [{
            "start_time": start_datetime,
            "end_time": end_datetime,
            "runtime_seconds": runtime_seconds,
            "batch_size": args.batch_size,
            "total_expected_rows": EXPECTED_MANIFEST_ROWS,
            "total_processed_rows": total_processed,
            "total_success_rows": total_processed - len(all_error_rows),
            "total_error_rows": len(all_error_rows),
            "device": str(device),
            "checkpoint_path": str(ckpt_path),
            "output_csv_path": str(final_csv_path),
            "scoring_pass": scoring_pass,
        }],
        columns=RUNTIME_CSV_COLUMNS,
    )
    recheck_before_save(review_output_root, OUTPUT_RUNTIME_CSV_NAME)
    runtime_df.to_csv(
        str(review_output_root / OUTPUT_RUNTIME_CSV_NAME), index=False
    )
    print(
        f"[INFO] runtime summary CSV 저장 완료: "
        f"{review_output_root / OUTPUT_RUNTIME_CSV_NAME}"
    )

    # ── score 분포 통계
    l1_stats = _scalar_stats([r["crop_score_l1_mean"] for r in all_score_rows])
    l1max_stats = _scalar_stats([r["crop_score_l1_max"] for r in all_score_rows])
    mse_stats = _scalar_stats([r["crop_score_mse_mean"] for r in all_score_rows])

    has_nan_count_rows = int(sum(1 for r in all_score_rows if r.get("has_nan", False)))
    has_inf_count_rows = int(sum(1 for r in all_score_rows if r.get("has_inf", False)))

    # ── summary JSON 저장
    summary = {
        "approval_gate_interpretation_corrected": True,
        "approval_required_before_scoring_interpretation": (
            "True = 사용자 명시 승인 필요 flag. "
            "scoring 불가로 해석하지 않음. "
            "이중 실행 플래그(--run-full + --confirm-run)로 실행 허용."
        ),
        "stage_split": EXPECTED_STAGE_SPLIT,
        "manifest_path": str(manifest_path),
        "manifest_row_count": len(manifest_df),
        "model_class": MODEL_CLASS,
        "model_tag": MODEL_TAG,
        "checkpoint_path": str(ckpt_path),
        "batch_size": args.batch_size,
        "canonical_schema_column_count": len(CANONICAL_COLUMNS),
        "canonical_schema_columns": CANONICAL_COLUMNS,
        "total_expected_rows": EXPECTED_MANIFEST_ROWS,
        "total_processed_rows": total_processed,
        "total_success_rows": total_processed - len(all_error_rows),
        "total_error_rows": len(all_error_rows),
        "has_nan_count": has_nan_count_rows,
        "has_inf_count": has_inf_count_rows,
        "score_l1_min": l1_stats["min"],
        "score_l1_max": l1_stats["max"],
        "score_l1_mean": l1_stats["mean"],
        "score_l1_median": l1_stats["median"],
        "score_l1max_min": l1max_stats["min"],
        "score_l1max_max": l1max_stats["max"],
        "score_l1max_mean": l1max_stats["mean"],
        "score_l1max_median": l1max_stats["median"],
        "score_mse_min": mse_stats["min"],
        "score_mse_max": mse_stats["max"],
        "score_mse_mean": mse_stats["mean"],
        "score_mse_median": mse_stats["median"],
        "metric_calculation_executed": False,
        "threshold_calculated": False,
        "training_executed": False,
        "backward_executed": False,
        "optimizer_step_executed": False,
        "checkpoint_created": False,
        "scoring_pass": scoring_pass,
        "done_marker_created": done_marker_expected,
        "blockers": blockers,
        "runtime_seconds": runtime_seconds,
        "timestamp": end_datetime,
        "output_score_csv_path": str(final_csv_path),
        "next_step_recommendation": (
            "scoring_pass=True이면 Phase 8.5 metric calculation preflight 승인 요청. "
            "metric 계산은 별도 승인 후 진행."
            if scoring_pass else
            "scoring_pass=False. error CSV 및 FAIL 원인 확인 후 수정 필요."
        ),
    }

    recheck_before_save(review_output_root, OUTPUT_JSON_NAME)
    json_path = review_output_root / OUTPUT_JSON_NAME
    with open(str(json_path), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"[INFO] summary JSON 저장 완료: {json_path}")

    # ── MD 보고서 저장
    md_text = generate_md_report(
        scoring_pass=scoring_pass,
        manifest_path=str(manifest_path),
        checkpoint_path=str(ckpt_path),
        batch_size=args.batch_size,
        blockers=blockers,
        total_processed=total_processed,
        total_error=len(all_error_rows),
        score_l1_stats=l1_stats,
        score_l1max_stats=l1max_stats,
        score_mse_stats=mse_stats,
        has_nan_count=has_nan_count_rows,
        has_inf_count=has_inf_count_rows,
        done_marker_created=done_marker_expected,
        output_csv_path=str(final_csv_path),
        timestamp=end_datetime,
    )
    recheck_before_save(review_output_root, OUTPUT_MD_NAME)
    md_path = review_output_root / OUTPUT_MD_NAME
    with open(str(md_path), "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"[INFO] MD 보고서 저장 완료: {md_path}")

    # ── DONE marker: scoring_pass=True AND error_count=0일 때만 생성
    done_marker_created = False
    if scoring_pass and len(all_error_rows) == 0:
        recheck_before_save(review_output_root, OUTPUT_DONE_JSON_NAME)
        done_path = review_output_root / OUTPUT_DONE_JSON_NAME
        done_summary = dict(summary)
        done_summary["done_marker_created"] = True
        with open(str(done_path), "w", encoding="utf-8") as f:
            json.dump(done_summary, f, ensure_ascii=False, indent=2, default=str)
        done_marker_created = True
        print(f"[INFO] DONE marker 생성: {done_path}")
    else:
        print(
            f"[WARN] DONE marker 생성 안 됨 "
            f"(scoring_pass={scoring_pass}, error_count={len(all_error_rows)})"
        )

    # ── 최종 요약
    print(f"\n[DONE] Phase 8.4 full scoring 완료")
    print(f"  score CSV: {final_csv_path}")
    print(f"  scoring_pass: {scoring_pass}")
    print(f"  total_processed: {total_processed:,}")
    print(f"  error_count: {len(all_error_rows)}")
    print(f"  has_nan_count: {has_nan_count_rows}")
    print(f"  has_inf_count: {has_inf_count_rows}")
    print(f"  runtime_seconds: {runtime_seconds:.1f}s")
    print(f"  DONE marker: {'생성됨' if done_marker_created else '생성 안 됨'}")
    print(f"\n[INFO] 출력 파일:")
    print(f"[INFO]   score CSV:  {final_csv_path}")
    print(f"[INFO]   JSON:       {json_path}")
    print(f"[INFO]   MD:         {md_path}")
    print(f"[INFO]   error:      {review_output_root / OUTPUT_ERROR_CSV_NAME}")
    print(f"[INFO]   runtime:    {review_output_root / OUTPUT_RUNTIME_CSV_NAME}")
    if done_marker_created:
        print(f"[INFO]   DONE:       {review_output_root / OUTPUT_DONE_JSON_NAME}")
    print("\n[INFO] Phase 8.4 stage2_holdout full scoring 완료.")


if __name__ == "__main__":
    main()
