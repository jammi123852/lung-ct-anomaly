"""
Phase 8.3B stage2_holdout v1/v1 RD4AD Scoring Smoke

목적:
  stage2_holdout full crop manifest에서 positive 32 + hard_negative 32 = 64개 crop에 대해
  reconstruction score CSV가 정상 생성되는지 확인한다.

금지 사항:
  - full scoring (max_crops 64 초과 절대 금지)
  - metric 계산 (AUROC/AUPRC/threshold/p95/p99/hit-rate 포함)
  - training / backward / optimizer step / checkpoint 생성
  - stage2_holdout 전체 manifest 로드 후 전체 scoring
  - 기존 Phase 6/7/8 output 수정·삭제·덮어쓰기
  - v2/v2v2 접근
  - pip/conda install / 외부 다운로드

실행 guard:
  --run-smoke와 --confirm-run 둘 다 없으면 dry-run 보고만 수행 후 종료.
  실제 scoring은 두 플래그가 모두 있을 때만 실행.

approval_required_before_scoring 해석 보정:
  manifest의 approval_required_before_scoring=True는
  "scoring 전 사용자 명시 승인 필요" flag이며, scoring 불가를 뜻하지 않는다.
  이중 실행 플래그(--run-smoke + --confirm-run)와 사용자 승인으로 실행 허용.
  approval_gate_interpretation_corrected=True를 summary JSON에 기록한다.

참고 파일 (read-only):
  scripts/phase7_1_v1v1_scoring_smoke.py
  scripts/phase7_2_v1v1_stage1_dev_full_scoring.py
"""

import argparse
import json
import math
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
SMOKE_SCRIPT_PATH = "scripts/phase8_3b_stage2_scoring_smoke.py"

SMOKE_MAX_CROPS = 64
SMOKE_SEED = 42
SMOKE_MAX_BATCH_SIZE = 8

EXPECTED_MANIFEST_ROWS = 143735
EXPECTED_UNIQUE_PATIENTS = 154
EXPECTED_POSITIVE_ROWS = 51335
EXPECTED_HARD_NEGATIVE_ROWS = 92400
EXPECTED_STAGE_SPLIT = "stage2_holdout"

SMOKE_POSITIVE_COUNT = 32
SMOKE_HARD_NEGATIVE_COUNT = 32

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
    "phase8_3_stage2_scoring_smoke_v1/"
)
DEFAULT_REVIEW_OUTPUT_ROOT = (
    "outputs/second-stage-lesion-refiner-v1/review_annotations/"
    "phase8_3_stage2_scoring_smoke_v1/"
)

OUTPUT_CSV_NAME = "phase8_3_stage2_scoring_smoke_v1.csv"
OUTPUT_CSV_TMP_NAME = "phase8_3_stage2_scoring_smoke_v1.csv.tmp"
OUTPUT_JSON_NAME = "phase8_3_stage2_scoring_smoke_summary_v1.json"
OUTPUT_MD_NAME = "phase8_3_stage2_scoring_smoke_report_v1.md"
OUTPUT_ERROR_CSV_NAME = "phase8_3_stage2_scoring_smoke_errors_v1.csv"
OUTPUT_RUNTIME_CSV_NAME = "phase8_3_stage2_scoring_smoke_runtime_summary_v1.csv"
OUTPUT_DONE_JSON_NAME = "phase8_3_stage2_scoring_smoke_DONE.json"

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

# smoke 진단 추가 컬럼
SMOKE_EXTRA_COLUMNS = [
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
    "smoke_status",
    "issue",
    "note",
]

ALL_COLUMNS = CANONICAL_COLUMNS + SMOKE_EXTRA_COLUMNS

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
    "smoke_target_crops",
    "total_processed_rows",
    "total_success_rows",
    "total_error_rows",
    "device",
    "checkpoint_path",
    "output_csv_path",
    "smoke_pass",
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
            "Phase 8.3B stage2_holdout v1/v1 scoring smoke: "
            "positive 32 + hard_negative 32 = 64개 crop에 대해 "
            "reconstruction score CSV 정상 생성 확인. "
            "--run-smoke와 --confirm-run 둘 다 없으면 dry-run만 수행합니다."
        )
    )
    parser.add_argument(
        "--run-smoke",
        action="store_true",
        help="[필수] smoke 실행 플래그 1. --confirm-run과 함께 사용해야 실행됩니다.",
    )
    parser.add_argument(
        "--confirm-run",
        action="store_true",
        help="[필수] smoke 실행 플래그 2. --run-smoke와 함께 사용해야 실행됩니다.",
    )
    parser.add_argument(
        "--max-crops",
        type=int,
        default=SMOKE_MAX_CROPS,
        help=f"최대 처리 crop 수 (상한 {SMOKE_MAX_CROPS}, default={SMOKE_MAX_CROPS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help=f"배치 크기 (상한 {SMOKE_MAX_BATCH_SIZE}, default=8)",
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
        help="smoke score 출력 root 디렉토리",
    )
    parser.add_argument(
        "--review-output-root",
        default=DEFAULT_REVIEW_OUTPUT_ROOT,
        help="smoke review 출력 root 디렉토리",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────
# Model (phase7_1과 동일한 구조)
# ──────────────────────────────────────────────────────────
class ConvAutoencoder2p5D(nn.Module):
    """
    2.5D reconstruction baseline.
    input_channels=6 (lung z-1/z/z+1, mediastinal z-1/z/z+1)
    Encoder: 96→48→24→12 (3× MaxPool2d)
    Decoder: 12→24→48→96 (3× ConvTranspose2d) + Sigmoid
    phase7_1_v1v1_scoring_smoke.py와 동일한 구조.
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
class SmokeScoreDataset(Dataset):
    """
    Phase 8.3B smoke용 Dataset.
    stage2_holdout manifest의 npz_path 컬럼 사용.
    crop_id는 s8a_{row_id:06d} 형식으로 생성 (manifest의 row_id 컬럼 기반).
    """

    def __init__(self, rows: pd.DataFrame):
        self.df = rows.reset_index(drop=True)
        print(f"[INFO] SmokeScoreDataset: {len(self.df)} crops")

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
            data = np.load(npz_path)
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


# ──────────────────────────────────────────────────────────
# Score 계산 (phase7_1과 동일 로직)
# ──────────────────────────────────────────────────────────
def compute_batch_smoke_scores(
    inputs: torch.Tensor,
    recons: torch.Tensor,
    metas: list,
    checkpoint_path: str,
) -> list:
    """
    배치 단위 smoke score 계산.
    inputs, recons: [B, 6, 96, 96]
    반환: list of dict (ALL_COLUMNS 기준)
    """
    inputs_np = inputs.detach().cpu().numpy()
    recons_np = recons.detach().cpu().numpy()

    diff = np.abs(inputs_np - recons_np)
    diff_sq = (inputs_np - recons_np) ** 2

    rows = []
    for b in range(inputs_np.shape[0]):
        meta = metas[b]
        inp = inputs_np[b]
        rec = recons_np[b]
        d = diff[b]
        d_sq = diff_sq[b]

        load_error = meta.get("_load_error")
        shape_error = meta.get("_shape_error")
        crop_shape_str = meta.get("_crop_shape", "")

        issue_parts = []
        if load_error:
            issue_parts.append(f"load_error: {load_error}")
        if shape_error:
            issue_parts.append(f"shape_error: {shape_error}")

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

        smoke_status = "PASS"
        if load_error or shape_error or has_nan or has_inf:
            smoke_status = "FAIL"
            if has_nan:
                issue_parts.append("has_nan=True")
            if has_inf:
                issue_parts.append("has_inf=True")

        issue_str = "; ".join(issue_parts)
        note_str = "load_ok" if (load_error is None and shape_error is None) else ""

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
            "smoke_status": smoke_status,
            "issue": issue_str,
            "note": note_str,
        }
        rows.append(row)

    return rows


# ──────────────────────────────────────────────────────────
# Manifest Safety 검증
# ──────────────────────────────────────────────────────────
def verify_manifest_safety(manifest_df: pd.DataFrame) -> list:
    """
    manifest safety 검증 7개 항목.
    실패 항목을 blockers list로 반환. 통과하면 빈 list.
    """
    blockers = []

    # 1. row count == 143735
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

    # 3. stage_split == 'stage2_holdout' 전수
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

    # 5. v2/v2v2 path == 0
    v2_mask = manifest_df["npz_path"].str.contains("v2", na=False)
    v2_count = int(v2_mask.sum())
    if v2_count != 0:
        msg = f"v2/v2v2 path detected: {v2_count} rows (must be 0)"
        blockers.append(msg)
        print(f"[ERROR] {msg}")
    else:
        print(f"[INFO] v2/v2v2 path check OK: 0 rows")

    # 6. positive row count == 51335
    if "sampling_label" in manifest_df.columns:
        actual_positive = int((manifest_df["sampling_label"] == "positive").sum())
        if actual_positive != EXPECTED_POSITIVE_ROWS:
            msg = f"positive row count mismatch: expected={EXPECTED_POSITIVE_ROWS}, actual={actual_positive}"
            blockers.append(msg)
            print(f"[ERROR] {msg}")
        else:
            print(f"[INFO] positive row count OK: {actual_positive}")

        # 7. hard_negative row count == 92400
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

    return blockers


# ──────────────────────────────────────────────────────────
# Smoke crop 선택
# ──────────────────────────────────────────────────────────
def select_smoke_crops(manifest_df: pd.DataFrame) -> pd.DataFrame:
    """
    positive 32 + hard_negative 32 = 64개 crop 선택.
    seed=42 deterministic. patient_id별 균등 분산 시도.
    """
    positive_df = manifest_df[manifest_df["sampling_label"] == "positive"].copy()
    hard_negative_df = manifest_df[manifest_df["sampling_label"] == "hard_negative"].copy()

    print(f"[INFO] positive pool: {len(positive_df)} crops")
    print(f"[INFO] hard_negative pool: {len(hard_negative_df)} crops")

    # positive 32개 선택 (patient별 분산)
    n_pos = SMOKE_POSITIVE_COUNT
    pos_sample = positive_df.sample(n=n_pos, random_state=SMOKE_SEED).copy()
    pos_unique_patients = pos_sample["patient_id"].nunique()
    print(f"[INFO] positive smoke 선택: {len(pos_sample)} crops, {pos_unique_patients} patients")

    # hard_negative 32개 선택 (patient별 분산)
    n_neg = SMOKE_HARD_NEGATIVE_COUNT
    neg_sample = hard_negative_df.sample(n=n_neg, random_state=SMOKE_SEED).copy()
    neg_unique_patients = neg_sample["patient_id"].nunique()
    print(f"[INFO] hard_negative smoke 선택: {len(neg_sample)} crops, {neg_unique_patients} patients")

    smoke_df = pd.concat([pos_sample, neg_sample], ignore_index=True)
    smoke_df = smoke_df.sample(frac=1, random_state=SMOKE_SEED).reset_index(drop=True)

    print(f"[INFO] 최종 smoke 대상: {len(smoke_df)} crops")
    return smoke_df


# ──────────────────────────────────────────────────────────
# Output guard
# ──────────────────────────────────────────────────────────
def check_output_guard(score_output_root: Path, review_output_root: Path) -> None:
    """
    output root 및 출력 파일 존재 시 즉시 sys.exit(1).
    기존 파일 덮어쓰기 절대 금지.
    """
    if score_output_root.exists():
        print(f"[ERROR] smoke score output root가 이미 존재합니다: {score_output_root}")
        print("[ERROR] 기존 파일 덮어쓰기 금지. 즉시 중단합니다.")
        sys.exit(1)

    if review_output_root.exists():
        print(f"[ERROR] smoke review output root가 이미 존재합니다: {review_output_root}")
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
    fpath = root / filename
    if fpath.exists():
        print(f"[ERROR] 저장 직전 재확인: 출력 파일이 이미 존재합니다: {fpath}")
        print("[ERROR] 즉시 중단합니다.")
        sys.exit(1)


# ──────────────────────────────────────────────────────────
# Score 분포 통계
# ──────────────────────────────────────────────────────────
def _scalar_stats(rows: list, col: str) -> dict:
    vals = [
        r[col] for r in rows
        if isinstance(r.get(col), float) and not math.isnan(r[col]) and not math.isinf(r[col])
    ]
    if not vals:
        nan = float("nan")
        return {"min": nan, "max": nan, "mean": nan}
    return {
        "min": float(min(vals)),
        "max": float(max(vals)),
        "mean": float(sum(vals) / len(vals)),
    }


# ──────────────────────────────────────────────────────────
# MD 보고서 생성
# ──────────────────────────────────────────────────────────
def generate_md_report(
    result_rows: list,
    smoke_pass: bool,
    manifest_path: str,
    checkpoint_path: str,
    max_crops: int,
    batch_size: int,
    blockers: list,
    scored_count: int,
    score_l1_stats: dict,
    score_l1max_stats: dict,
    score_mse_stats: dict,
    nan_inf_pass: bool,
    error_count: int,
    timestamp: str,
) -> str:
    pass_str = "PASS" if smoke_pass else "FAIL"
    blockers_str = "\n".join(f"- {b}" for b in blockers) if blockers else "없음"

    has_nan_count = sum(1 for r in result_rows if r.get("has_nan", False))
    has_inf_count = sum(1 for r in result_rows if r.get("has_inf", False))
    fail_count = sum(1 for r in result_rows if r.get("smoke_status") == "FAIL")

    pos_count = sum(1 for r in result_rows if r.get("sampling_label") == "positive")
    neg_count = sum(1 for r in result_rows if r.get("sampling_label") == "hard_negative")

    md = f"""# Phase 8.3B Stage2 Holdout Scoring Smoke Report

생성 시각: {timestamp}

---

## 1. 목적

stage2_holdout full crop manifest에서 positive {SMOKE_POSITIVE_COUNT}개 + hard_negative {SMOKE_HARD_NEGATIVE_COUNT}개 = {max_crops}개 crop에 대해
reconstruction score CSV가 정상 생성되는지 확인한다.
full scoring / metric 계산 / threshold 계산 / training은 수행하지 않는다.

---

## 2. approval_required_before_scoring 해석 보정

- **approval_gate_interpretation_corrected**: True
- manifest의 `approval_required_before_scoring=True`는 "scoring 전 사용자 명시 승인 필요" flag이다.
- scoring 불가로 해석하지 않는다.
- 이중 실행 플래그(`--run-smoke + --confirm-run`)와 사용자 승인으로 실행 허용.

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
- blockers:
{blockers_str}

---

## 4. model / checkpoint

- model_class: `{MODEL_CLASS}`
- model_tag: `{MODEL_TAG}`
- checkpoint: `{checkpoint_path}`
- expected input shape: (6, 96, 96)

---

## 5. Smoke 대상 선택

- smoke_seed: {SMOKE_SEED} (deterministic)
- positive 선택: {pos_count}개
- hard_negative 선택: {neg_count}개
- 합계: {pos_count + neg_count}개 (max_crops={max_crops})
- batch_size: {batch_size}

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

## 8. Smoke score 분포 요약

| 항목 | min | max | mean |
|------|-----|-----|------|
| crop_score_l1_mean | {score_l1_stats.get('min', float('nan')):.6f} | {score_l1_stats.get('max', float('nan')):.6f} | {score_l1_stats.get('mean', float('nan')):.6f} |
| crop_score_l1_max | {score_l1max_stats.get('min', float('nan')):.6f} | {score_l1max_stats.get('max', float('nan')):.6f} | {score_l1max_stats.get('mean', float('nan')):.6f} |
| crop_score_mse_mean | {score_mse_stats.get('min', float('nan')):.6f} | {score_mse_stats.get('max', float('nan')):.6f} | {score_mse_stats.get('mean', float('nan')):.6f} |

---

## 9. NaN/Inf 확인

- has_nan=True인 crop 수: {has_nan_count}
- has_inf=True인 crop 수: {has_inf_count}
- smoke_status=FAIL인 crop 수: {fail_count}
- NaN/Inf 확인 통과: {"PASS" if nan_inf_pass else "FAIL"}

---

## 10. error CSV 결과

- error_count: {error_count}
- error 없음 여부: {"PASS (에러 없음)" if error_count == 0 else f"FAIL (에러 {error_count}건)"}

---

## 11. metric / threshold / training 미수행 확인

- metric_calculation_executed: False
- threshold_calculated: False
- training_executed: False
- backward_executed: False
- optimizer_step_executed: False
- checkpoint_created: False
- full_scoring_executed: False

---

## 12. 최종 판정

**smoke_pass: {pass_str}**

{("blockers:\n" + chr(10).join(f"  - {b}" for b in blockers)) if blockers else "blocker 없음"}

---

## 13. 다음 단계

{"smoke_pass=True. 다음 단계는 Phase 8.3C smoke score output validation이다. full scoring은 아직 금지하며, smoke output validation 통과 후 별도 Phase 8.4 full scoring preflight에서 승인받는다." if smoke_pass else "smoke_pass=FAIL. blockers 및 FAIL crop 원인 확인 후 수정 필요."}
"""
    return md


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────
def main():
    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    args = parse_args()

    # ── 이중 플래그 guard
    if not args.run_smoke or not args.confirm_run:
        print("[DRY-RUN] --run-smoke와 --confirm-run이 모두 필요합니다.")
        print("[DRY-RUN] 이번 실행은 dry-run 보고만 수행합니다.")
        print("[DRY-RUN] 실제 smoke scoring은 수행되지 않았습니다.")
        print(
            "[DRY-RUN] 실행 명령: source ~/ai_env/bin/activate && "
            "python scripts/phase8_3b_stage2_scoring_smoke.py "
            "--run-smoke --confirm-run --max-crops 64 --batch-size 8"
        )
        sys.exit(0)

    # ── 상한 검증
    if args.max_crops > SMOKE_MAX_CROPS:
        print(
            f"[ERROR] --max-crops={args.max_crops}는 상한 {SMOKE_MAX_CROPS}을 초과합니다."
        )
        sys.exit(1)

    if args.batch_size > SMOKE_MAX_BATCH_SIZE:
        print(
            f"[ERROR] --batch-size={args.batch_size}는 상한 {SMOKE_MAX_BATCH_SIZE}을 초과합니다."
        )
        sys.exit(1)

    print(f"[INFO] Phase 8.3B stage2_holdout scoring smoke 시작")
    print(f"[INFO] max_crops={args.max_crops}, batch_size={args.batch_size}, start={start_datetime}")

    # ── 경로 설정
    project_root = Path(__file__).resolve().parent.parent
    checkpoint_path = str(
        Path(args.checkpoint) if Path(args.checkpoint).is_absolute()
        else project_root / args.checkpoint
    )
    manifest_path = str(
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

    print(f"[INFO] checkpoint: {checkpoint_path}")
    print(f"[INFO] manifest: {manifest_path}")
    print(f"[INFO] score_output_root: {score_output_root}")
    print(f"[INFO] review_output_root: {review_output_root}")

    # ── output guard
    check_output_guard(score_output_root, review_output_root)

    # ── checkpoint 존재 확인
    ckpt_p = Path(checkpoint_path)
    if not ckpt_p.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_p}")
        sys.exit(1)

    # ── manifest 존재 확인 및 로드
    mf_p = Path(manifest_path)
    if not mf_p.exists():
        print(f"[ERROR] Manifest not found: {mf_p}")
        sys.exit(1)

    print(f"[INFO] Loading manifest: {mf_p}")
    manifest_df = pd.read_csv(mf_p)
    print(f"[INFO] Manifest loaded: {len(manifest_df)} rows")

    # ── manifest safety 검증
    print("[INFO] Manifest safety 검증 시작...")
    blockers = verify_manifest_safety(manifest_df)
    if blockers:
        print(f"[ERROR] Manifest safety 검증 실패: {len(blockers)}개 blocker")
        for b in blockers:
            print(f"[ERROR]   - {b}")
        sys.exit(1)
    print("[INFO] Manifest safety 검증: 모두 통과")

    # ── smoke 64개 선택
    smoke_df = select_smoke_crops(manifest_df)

    # ── device 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # ── checkpoint 로드 (read-only, scoring smoke 실행 시에만)
    print(f"[INFO] Loading checkpoint (read-only): {ckpt_p}")
    ckpt = torch.load(str(ckpt_p), map_location=device, weights_only=False)
    print(
        f"[INFO] Checkpoint loaded: epoch={ckpt.get('epoch', 'N/A')}, "
        f"best_val_loss={ckpt.get('best_val_loss', 'N/A')}"
    )

    # ── model 구성 및 weight 로드
    model = ConvAutoencoder2p5D(input_channels=6, base_channels=32).to(device)

    if "model_state_dict" not in ckpt:
        print("[ERROR] checkpoint에 'model_state_dict' 키가 없습니다.")
        sys.exit(1)

    model.load_state_dict(ckpt["model_state_dict"])
    print("[INFO] model_state_dict 로드 완료.")
    model.eval()
    print("[INFO] model.eval() 설정 완료.")

    # ── output root 생성 (guard 통과 후)
    score_output_root.mkdir(parents=True, exist_ok=False)
    review_output_root.mkdir(parents=True, exist_ok=False)
    print(f"[INFO] score_output_root 생성: {score_output_root}")
    print(f"[INFO] review_output_root 생성: {review_output_root}")

    # ── error CSV 초기화 (헤더만)
    error_csv_path = review_output_root / OUTPUT_ERROR_CSV_NAME
    recheck_before_save(review_output_root, OUTPUT_ERROR_CSV_NAME)
    pd.DataFrame(columns=ERROR_CSV_COLUMNS).to_csv(error_csv_path, index=False)
    print(f"[INFO] error CSV 초기화 완료: {error_csv_path}")

    # ── Dataset / DataLoader
    dataset = SmokeScoreDataset(rows=smoke_df)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_fn,
    )

    # ── inference (no_grad, no backward, no optimizer)
    all_rows = []
    all_error_rows = []

    with torch.no_grad():
        for batch_idx, (batch_tensors, batch_metas) in enumerate(loader):
            batch_tensors = batch_tensors.to(device)
            recons = model(batch_tensors)

            if batch_idx == 0:
                print(f"[INFO] first_batch_input_shape={tuple(batch_tensors.shape)}")
                print(f"[INFO] first_batch_recon_shape={tuple(recons.shape)}")

            rows = compute_batch_smoke_scores(
                inputs=batch_tensors,
                recons=recons,
                metas=batch_metas,
                checkpoint_path=checkpoint_path,
            )
            all_rows.extend(rows)

            # error rows 수집
            for r in rows:
                if r.get("smoke_status") == "FAIL":
                    issue = r.get("issue", "")
                    if "load_error" in issue:
                        error_type = "load_error"
                    elif "shape_error" in issue:
                        error_type = "shape_mismatch"
                    elif "has_nan" in issue:
                        error_type = "nan_error"
                    elif "has_inf" in issue:
                        error_type = "inf_error"
                    else:
                        error_type = "unknown"
                    all_error_rows.append({
                        "row_id": r["row_id"],
                        "crop_id": r["crop_id"],
                        "patient_id": r["patient_id"],
                        "npz_path": r["npz_path"],
                        "error_type": error_type,
                        "error_message": issue,
                    })

    print(f"[INFO] Scoring 완료: {len(all_rows)} crops processed")

    scored_count = len(all_rows)
    error_count = len(all_error_rows)

    # ── tmp CSV 저장 (score_output_root)
    tmp_csv_path = score_output_root / OUTPUT_CSV_TMP_NAME
    recheck_before_save(score_output_root, OUTPUT_CSV_TMP_NAME)
    df_out = pd.DataFrame(all_rows, columns=ALL_COLUMNS)
    df_out.to_csv(tmp_csv_path, index=False)
    print(f"[INFO] tmp CSV 저장 완료: {tmp_csv_path} ({len(df_out)} rows)")

    # ── smoke_pass 조건 판정
    actual_pos_count = int((df_out["sampling_label"] == "positive").sum())
    actual_neg_count = int((df_out["sampling_label"] == "hard_negative").sum())
    has_nan_count = int(df_out["has_nan"].sum())
    has_inf_count = int(df_out["has_inf"].sum())
    nan_inf_pass = has_nan_count == 0 and has_inf_count == 0

    smoke_pass_conditions = {
        "row_count_ok": scored_count == SMOKE_MAX_CROPS,
        "positive_count_ok": actual_pos_count == SMOKE_POSITIVE_COUNT,
        "hard_negative_count_ok": actual_neg_count == SMOKE_HARD_NEGATIVE_COUNT,
        "has_nan_zero": has_nan_count == 0,
        "has_inf_zero": has_inf_count == 0,
        "error_count_zero": error_count == 0,
    }
    smoke_pass = all(smoke_pass_conditions.values())
    print(f"[INFO] smoke_pass: {smoke_pass}")
    for k, v in smoke_pass_conditions.items():
        status = "OK" if v else "FAIL"
        print(f"[INFO]   {k}: {status}")

    # ── tmp → final rename (smoke_pass인 경우)
    final_csv_path = score_output_root / OUTPUT_CSV_NAME
    if smoke_pass:
        recheck_before_save(score_output_root, OUTPUT_CSV_NAME)
        tmp_csv_path.rename(final_csv_path)
        print(f"[INFO] smoke_pass=True: tmp → final CSV rename 완료: {final_csv_path}")
    else:
        print(f"[WARNING] smoke_pass=False: tmp 상태 유지. final CSV rename 건너뜀.")
        print(f"[WARNING] tmp CSV 경로: {tmp_csv_path}")

    # ── error CSV 업데이트
    if all_error_rows:
        df_errors = pd.DataFrame(all_error_rows, columns=ERROR_CSV_COLUMNS)
        df_errors.to_csv(error_csv_path, index=False)
        print(f"[INFO] error CSV 저장 완료: {error_csv_path} ({len(df_errors)} rows)")
    else:
        print(f"[INFO] error CSV: 에러 없음 (헤더만 존재): {error_csv_path}")

    # ── score 분포 통계
    score_l1_stats = _scalar_stats(all_rows, "crop_score_l1_mean")
    score_l1max_stats = _scalar_stats(all_rows, "crop_score_l1_max")
    score_mse_stats = _scalar_stats(all_rows, "crop_score_mse_mean")

    # ── runtime 계산
    end_time = time.time()
    end_datetime = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    runtime_seconds = end_time - start_time

    output_csv_path_str = str(final_csv_path) if smoke_pass else str(tmp_csv_path)

    # ── runtime summary CSV 저장 (review_output_root)
    runtime_row = {
        "start_time": start_datetime,
        "end_time": end_datetime,
        "runtime_seconds": runtime_seconds,
        "batch_size": args.batch_size,
        "smoke_target_crops": SMOKE_MAX_CROPS,
        "total_processed_rows": scored_count,
        "total_success_rows": scored_count - has_nan_count,
        "total_error_rows": error_count,
        "device": str(device),
        "checkpoint_path": checkpoint_path,
        "output_csv_path": output_csv_path_str,
        "smoke_pass": smoke_pass,
    }
    recheck_before_save(review_output_root, OUTPUT_RUNTIME_CSV_NAME)
    runtime_csv_path = review_output_root / OUTPUT_RUNTIME_CSV_NAME
    pd.DataFrame([runtime_row], columns=RUNTIME_CSV_COLUMNS).to_csv(runtime_csv_path, index=False)
    print(f"[INFO] runtime summary CSV 저장 완료: {runtime_csv_path}")

    # ── summary JSON 저장 (review_output_root)
    timestamp = end_datetime
    summary_json = {
        "approval_gate_interpretation_corrected": True,
        "approval_required_before_scoring_interpretation": (
            "True = 사용자 명시 승인 필요 flag. "
            "scoring 불가로 해석하지 않음. "
            "이중 실행 플래그(--run-smoke + --confirm-run)로 실행 허용."
        ),
        "smoke_script_path": SMOKE_SCRIPT_PATH,
        "stage_split": EXPECTED_STAGE_SPLIT,
        "manifest_path": manifest_path,
        "manifest_row_count": EXPECTED_MANIFEST_ROWS,
        "manifest_positive_count": EXPECTED_POSITIVE_ROWS,
        "manifest_hard_negative_count": EXPECTED_HARD_NEGATIVE_ROWS,
        "smoke_seed": SMOKE_SEED,
        "smoke_positive_count": actual_pos_count,
        "smoke_hard_negative_count": actual_neg_count,
        "scored_crop_count": scored_count,
        "checkpoint_path": checkpoint_path,
        "model_class": MODEL_CLASS,
        "model_tag": MODEL_TAG,
        "batch_size": args.batch_size,
        "canonical_schema_column_count": len(CANONICAL_COLUMNS),
        "canonical_schema_columns": CANONICAL_COLUMNS,
        "score_l1_min": score_l1_stats["min"],
        "score_l1_max": score_l1_stats["max"],
        "score_l1_mean": score_l1_stats["mean"],
        "score_mse_min": score_mse_stats["min"],
        "score_mse_max": score_mse_stats["max"],
        "score_mse_mean": score_mse_stats["mean"],
        "has_nan_count": has_nan_count,
        "has_inf_count": has_inf_count,
        "error_count": error_count,
        "smoke_pass_conditions": smoke_pass_conditions,
        "metric_calculation_executed": False,
        "threshold_calculated": False,
        "training_executed": False,
        "backward_executed": False,
        "optimizer_step_executed": False,
        "checkpoint_created": False,
        "full_scoring_executed": False,
        "smoke_pass": smoke_pass,
        "blockers": blockers,
        "runtime_seconds": runtime_seconds,
        "timestamp": timestamp,
        "output_score_csv_path": output_csv_path_str,
        "next_step_recommendation": (
            "smoke_pass=True. 다음 단계는 Phase 8.3C smoke score output validation이다. "
            "full scoring은 아직 금지하며, smoke output validation 통과 후 "
            "별도 Phase 8.4 full scoring preflight에서 승인받는다."
            if smoke_pass else
            "smoke_pass=False. error CSV 및 FAIL 원인 확인 후 수정 필요."
        ),
    }

    recheck_before_save(review_output_root, OUTPUT_JSON_NAME)
    json_path = review_output_root / OUTPUT_JSON_NAME
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False, default=str)
    print(f"[INFO] summary JSON 저장 완료: {json_path}")

    # ── MD 보고서 저장 (review_output_root)
    md_content = generate_md_report(
        result_rows=all_rows,
        smoke_pass=smoke_pass,
        manifest_path=manifest_path,
        checkpoint_path=checkpoint_path,
        max_crops=args.max_crops,
        batch_size=args.batch_size,
        blockers=blockers,
        scored_count=scored_count,
        score_l1_stats=score_l1_stats,
        score_l1max_stats=score_l1max_stats,
        score_mse_stats=score_mse_stats,
        nan_inf_pass=nan_inf_pass,
        error_count=error_count,
        timestamp=timestamp,
    )

    recheck_before_save(review_output_root, OUTPUT_MD_NAME)
    md_path = review_output_root / OUTPUT_MD_NAME
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[INFO] MD 보고서 저장 완료: {md_path}")

    # ── DONE marker 생성 (smoke_pass=True, error_count==0인 경우만)
    done_marker_created = False
    if smoke_pass and error_count == 0:
        done_marker_created = True
        done_json = {
            "smoke_pass": True,
            "output_csv_row_count": scored_count,
            "positive_count": actual_pos_count,
            "hard_negative_count": actual_neg_count,
            "error_count": error_count,
            "metric_calculation_executed": False,
            "threshold_calculated": False,
            "training_executed": False,
            "approval_gate_interpretation_corrected": True,
        }
        recheck_before_save(review_output_root, OUTPUT_DONE_JSON_NAME)
        done_path = review_output_root / OUTPUT_DONE_JSON_NAME
        with open(done_path, "w", encoding="utf-8") as f:
            json.dump(done_json, f, indent=2, ensure_ascii=False)
        print(f"[INFO] DONE marker JSON 생성 완료: {done_path}")
    else:
        print(
            f"[WARNING] DONE marker 생성 건너뜀 "
            f"(smoke_pass={smoke_pass}, error_count={error_count})"
        )

    # ── 최종 요약
    print("\n[INFO] ===== Phase 8.3B Smoke 결과 요약 =====")
    print(f"[INFO] scored_crop_count: {scored_count}")
    print(f"[INFO] positive_count: {actual_pos_count} / {SMOKE_POSITIVE_COUNT}")
    print(f"[INFO] hard_negative_count: {actual_neg_count} / {SMOKE_HARD_NEGATIVE_COUNT}")
    print(f"[INFO] has_nan_count: {has_nan_count}")
    print(f"[INFO] has_inf_count: {has_inf_count}")
    print(f"[INFO] error_count: {error_count}")
    print(
        f"[INFO] crop_score_l1_mean: "
        f"min={score_l1_stats['min']:.6f}, "
        f"max={score_l1_stats['max']:.6f}, "
        f"mean={score_l1_stats['mean']:.6f}"
    )
    print(
        f"[INFO] crop_score_mse_mean: "
        f"min={score_mse_stats['min']:.6f}, "
        f"max={score_mse_stats['max']:.6f}, "
        f"mean={score_mse_stats['mean']:.6f}"
    )
    print(f"[INFO] metric_calculation_executed: False")
    print(f"[INFO] threshold_calculated: False")
    print(f"[INFO] training_executed: False")
    print(f"[INFO] smoke_pass: {smoke_pass}")
    print(f"[INFO] runtime_seconds: {runtime_seconds:.1f}s")
    print(f"\n[INFO] 출력 파일:")
    print(f"[INFO]   score CSV: {final_csv_path if smoke_pass else tmp_csv_path}")
    print(f"[INFO]   JSON:      {json_path}")
    print(f"[INFO]   MD:        {md_path}")
    print(f"[INFO]   error:     {error_csv_path}")
    print(f"[INFO]   runtime:   {runtime_csv_path}")
    if done_marker_created:
        print(f"[INFO]   DONE:      {review_output_root / OUTPUT_DONE_JSON_NAME}")
    print("\n[INFO] Phase 8.3B stage2_holdout scoring smoke 완료.")


if __name__ == "__main__":
    main()
