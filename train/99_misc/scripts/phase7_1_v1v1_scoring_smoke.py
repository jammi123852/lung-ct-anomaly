"""
Phase 7.1 v1/v1 scoring smoke script

목적:
  filtered S6-A manifest (phase6_1b) 에서 최대 64개 crop에 대해
  reconstruction score CSV가 정상 생성되는지 확인한다.

금지 사항:
  - full scoring (max_crops 제한 반드시 적용)
  - metric 계산
  - threshold 계산
  - training / backward / optimizer step
  - checkpoint 생성
  - filtered manifest 수정
  - 원본 dataset index 사용/수정
  - crop 파일 수정/삭제/이동
  - score CSV를 smoke output root 밖에 생성
  - stage2_holdout 접근
  - v2/v2v2 접근
  - pip/conda install
  - 외부 다운로드
  - 기존 스크립트 수정

참고 파일 (read-only):
  scripts/score_rd4ad_2p5d_normal_val_test.py
  outputs/.../phase7_0_v1v1_scoring_evaluation_design_preflight_v1.json
"""

import argparse
import json
import math
import sys
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
SCORING_SCRIPT_REFERENCE = "scripts/score_rd4ad_2p5d_normal_val_test.py"
SMOKE_SCRIPT_PATH = "scripts/phase7_1_v1v1_scoring_smoke.py"

SMOKE_MAX_CROPS_LIMIT = 64
SMOKE_MAX_BATCH_SIZE_LIMIT = 8

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
    "outputs/second-stage-lesion-refiner-v1/review_annotations/"
    "phase7_1_v1v1_scoring_smoke_v1/"
)

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

# smoke 진단 컬럼 (canonical 뒤에 추가)
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

# score scalar 컬럼 목록 (NaN/Inf count 계산 대상)
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
            "Phase 7.1 v1/v1 scoring smoke: filtered S6-A manifest에서 "
            "최대 64개 crop에 대해 reconstruction score CSV 정상 생성 확인. "
            "--run-smoke 플래그가 없으면 실행되지 않습니다."
        )
    )
    parser.add_argument(
        "--run-smoke",
        action="store_true",
        help="[필수] smoke 실행 플래그. 없으면 실행 거부.",
    )
    parser.add_argument(
        "--max-crops",
        type=int,
        default=64,
        help=f"최대 처리 crop 수 (최대 {SMOKE_MAX_CROPS_LIMIT}, default=64)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help=f"배치 크기 (최대 {SMOKE_MAX_BATCH_SIZE_LIMIT}, default=8)",
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
        help="smoke 출력 root 디렉토리",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────
# Model (score_rd4ad_2p5d_normal_val_test.py와 동일한 구조)
# ──────────────────────────────────────────────────────────
class ConvAutoencoder2p5D(nn.Module):
    """
    2.5D reconstruction baseline.
    input_channels=6 (lung z-1/z/z+1, mediastinal z-1/z/z+1)
    Encoder: 96→48→24→12 (3× MaxPool2d)
    Decoder: 12→24→48→96 (3× ConvTranspose2d) + Sigmoid
    score_rd4ad_2p5d_normal_val_test.py 와 동일한 구조.
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
    Phase 7.1 smoke용 Dataset.
    filtered manifest의 npz_path 컬럼 사용.
    crop_id는 s6a_{row_index:06d} 형식으로 생성 (manifest에 crop_id 없음).
    """

    def __init__(self, rows: pd.DataFrame):
        self.df = rows.reset_index(drop=True)
        print(f"[INFO] SmokeScoreDataset: {len(self.df)} crops")

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
        }

        # npz 로드 시도
        load_error = None
        img = None
        crop_shape_str = ""

        try:
            if not Path(npz_path).exists():
                raise FileNotFoundError(f"npz_path not found: {npz_path}")
            data = np.load(npz_path)
            if "image" not in data:
                raise KeyError(f"'image' key not found in {npz_path}")
            img = data["image"].astype(np.float32)
            crop_shape_str = str(tuple(img.shape))
        except Exception as e:
            load_error = str(e)

        if load_error is not None:
            # 로드 실패 시 더미 텐서 반환 (배치 처리 유지)
            dummy = torch.zeros(6, 96, 96, dtype=torch.float32)
            meta["_load_error"] = load_error
            meta["_crop_shape"] = ""
            return dummy, meta

        meta["_load_error"] = None
        meta["_crop_shape"] = crop_shape_str

        # shape 검증
        if img.shape != (6, 96, 96):
            meta["_shape_error"] = f"unexpected shape {img.shape}, expected (6,96,96)"
            dummy = torch.zeros(6, 96, 96, dtype=torch.float32)
            return dummy, meta
        else:
            meta["_shape_error"] = None

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
def compute_batch_smoke_scores(
    inputs: torch.Tensor,
    recons: torch.Tensor,
    metas: list,
    checkpoint_path: str,
) -> list:
    """
    배치 단위 smoke score 계산.
    inputs, recons: [B, 6, 96, 96] (CPU 또는 GPU tensor)
    반환: list of dict (ALL_COLUMNS 기준)
    """
    inputs_np = inputs.detach().cpu().numpy()   # [B, 6, 96, 96]
    recons_np = recons.detach().cpu().numpy()   # [B, 6, 96, 96]

    diff = np.abs(inputs_np - recons_np)        # [B, 6, 96, 96]
    diff_sq = (inputs_np - recons_np) ** 2      # [B, 6, 96, 96]

    rows = []
    for b in range(inputs_np.shape[0]):
        meta = metas[b]
        inp = inputs_np[b]    # [6, 96, 96]
        rec = recons_np[b]    # [6, 96, 96]
        d = diff[b]           # [6, 96, 96]
        d_sq = diff_sq[b]     # [6, 96, 96]

        load_error = meta.get("_load_error")
        shape_error = meta.get("_shape_error")
        crop_shape_str = meta.get("_crop_shape", "")

        issue_parts = []
        if load_error:
            issue_parts.append(f"load_error: {load_error}")
        if shape_error:
            issue_parts.append(f"shape_error: {shape_error}")

        # score 계산
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

        # NaN/Inf 개수 계산 (pixel 단위)
        input_nan_count = int(np.isnan(inp).sum())
        input_inf_count = int(np.isinf(inp).sum())
        recon_nan_count = int(np.isnan(rec).sum())
        recon_inf_count = int(np.isinf(rec).sum())
        error_nan_count = int(np.isnan(d).sum())
        error_inf_count = int(np.isinf(d).sum())

        # has_nan/has_inf: input/recon/error tensor 및 scalar score 중 하나라도 해당하면 True
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

        # smoke_status 판정
        smoke_status = "PASS"
        if load_error or shape_error or has_nan or has_inf:
            smoke_status = "FAIL"
            if has_nan:
                issue_parts.append("has_nan=True")
            if has_inf:
                issue_parts.append("has_inf=True")

        issue_str = "; ".join(issue_parts)
        note_str = ""
        if load_error is None and shape_error is None:
            note_str = "load_ok"

        row = {
            # canonical 27컬럼
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
            # smoke 진단 컬럼
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
        # CSV에서 읽을 때 bool 또는 str일 수 있으므로 변환
        col = manifest_df["approval_required_before_training"]
        # True 판단: bool True 또는 str "True"
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
    error_nan_inf_pass: bool,
    shape_pass: bool,
    timestamp: str,
) -> str:
    """Phase 7.1 smoke MD 보고서 생성"""

    pass_str = "PASS" if smoke_pass else "FAIL"
    blockers_str = "\n".join(f"- {b}" for b in blockers) if blockers else "없음"

    # NaN/Inf 통계
    has_nan_count = sum(1 for r in result_rows if r.get("has_nan", False))
    has_inf_count = sum(1 for r in result_rows if r.get("has_inf", False))
    fail_count = sum(1 for r in result_rows if r.get("smoke_status") == "FAIL")

    def fmt_stats(stats: dict) -> str:
        if not stats:
            return "N/A"
        return (
            f"min={stats.get('min', 'N/A'):.6f}, "
            f"max={stats.get('max', 'N/A'):.6f}, "
            f"mean={stats.get('mean', 'N/A'):.6f}"
        )

    md = f"""# Phase 7.1 v1/v1 Scoring Smoke Report

생성 시각: {timestamp}

---

## 1. Phase 7.1 목적

filtered S6-A manifest (phase6_1b) 에서 최대 {max_crops}개 crop에 대해
reconstruction score CSV가 정상 생성되는지 확인한다.
full scoring / metric 계산 / threshold 계산 / training은 수행하지 않는다.

---

## 2. Phase 7.0 canonical schema 반영 여부

- canonical 컬럼 수: {len(CANONICAL_COLUMNS)}개 (27컬럼 기준)
- canonical 컬럼 목록: {', '.join(CANONICAL_COLUMNS)}
- smoke 진단 컬럼 수: {len(SMOKE_EXTRA_COLUMNS)}개
- 총 컬럼 수: {len(ALL_COLUMNS)}개
- Phase 7.0 preflight 정의 기준 반영: 확인됨 (error_min/error_max 포함)

---

## 3. 입력 filtered manifest 안전성 확인

- manifest 경로: `{manifest_path}`
- 기대 row 수: {EXPECTED_MANIFEST_ROWS}
- 기대 unique patient 수: {EXPECTED_UNIQUE_PATIENTS}
- stage2_holdout rows: 0 (기대값)
- 제외 환자 (LUNG1-295, LUNG1-415): 0 rows (기대값)
- v2/v2v2 path: 0 rows (기대값)
- training_manifest_status: all "not_training_manifest"
- approval_required_before_training: all True
- blockers:
{blockers_str}

---

## 4. 사용한 model / checkpoint

- model_class: `{MODEL_CLASS}`
- model_tag: `{MODEL_TAG}`
- checkpoint: `{checkpoint_path}`
- scoring script 참고: `{SCORING_SCRIPT_REFERENCE}`

---

## 5. Scoring smoke 범위

- max_crops: {max_crops} (상한: {SMOKE_MAX_CROPS_LIMIT})
- batch_size: {batch_size} (상한: {SMOKE_MAX_BATCH_SIZE_LIMIT})
- 실제 scored crop 수: {scored_count}

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
channel-group score이며, anatomical certainty not asserted.
채널 0~2를 lung window, 채널 3~5를 mediastinal window로 가정하지만
실제 해부학적 의미는 추가 검증 필요.

---

## 7. score_nan_count / score_inf_count / error_nan_count / error_inf_count 정의

| 컬럼명 | 정의 |
|--------|------|
| score_nan_count | crop_score_l1_mean, crop_score_l1_max, crop_score_mse_mean, channel_0~5_l1_mean, lung_channels_l1_mean, mediastinal_channels_l1_mean 중 NaN인 scalar 값의 개수 |
| score_inf_count | 위 scalar score 컬럼들 중 Inf인 값의 개수 |
| error_nan_count | diff tensor (abs(input - recon))의 NaN pixel 개수 |
| error_inf_count | diff tensor의 Inf pixel 개수 |

has_nan: input / recon / error tensor 또는 scalar score 중 하나라도 NaN이면 True
has_inf: input / recon / error tensor 또는 scalar score 중 하나라도 Inf이면 True

---

## 8. Smoke score 분포 요약

| 항목 | min | max | mean |
|------|-----|-----|------|
| crop_score_l1_mean | {score_l1_stats.get('min', 'N/A'):.6f} | {score_l1_stats.get('max', 'N/A'):.6f} | {score_l1_stats.get('mean', 'N/A'):.6f} |
| crop_score_l1_max | {score_l1max_stats.get('min', 'N/A'):.6f} | {score_l1max_stats.get('max', 'N/A'):.6f} | {score_l1max_stats.get('mean', 'N/A'):.6f} |
| crop_score_mse_mean | {score_mse_stats.get('min', 'N/A'):.6f} | {score_mse_stats.get('max', 'N/A'):.6f} | {score_mse_stats.get('mean', 'N/A'):.6f} |

---

## 9. NaN/Inf 확인 결과

- has_nan=True인 crop 수: {has_nan_count}
- has_inf=True인 crop 수: {has_inf_count}
- smoke_status=FAIL인 crop 수: {fail_count}
- score NaN/Inf 확인 통과: {"PASS" if nan_inf_pass else "FAIL"}
- error NaN/Inf 확인 통과: {"PASS" if error_nan_inf_pass else "FAIL"}
- input/output shape 확인 통과: {"PASS" if shape_pass else "FAIL"}

---

## 10. metric / threshold / training 미수행 확인

- metric_calculation_executed: False
- threshold_calculated: False
- training_executed: False
- backward_executed: False
- optimizer_step_executed: False
- checkpoint_created: False
- full_scoring_executed: False

---

## 11. 최종 판정

**smoke_pass: {pass_str}**

{f"blockers: {chr(10).join(f'  - {b}' for b in blockers)}" if blockers else "blocker 없음"}

---

## 12. 다음 단계

{"smoke_pass=PASS 확인됨. Phase 7.2 stage1_dev filtered full scoring (129437 crops) 진행을 위한 사용자 승인 요청." if smoke_pass else "smoke_pass=FAIL. blockers 및 FAIL crop 목록을 확인한 후 수정 필요."}
"""
    return md


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # --run-smoke 없으면 실행 거부
    if not args.run_smoke:
        print("[ERROR] --run-smoke 플래그가 필요합니다.")
        print(
            "사용법: python scripts/phase7_1_v1v1_scoring_smoke.py "
            "--run-smoke [--max-crops 64] [--batch-size 8]"
        )
        sys.exit(1)

    # max-crops 상한 검증
    if args.max_crops > SMOKE_MAX_CROPS_LIMIT:
        print(
            f"[ERROR] --max-crops={args.max_crops}는 smoke 상한 {SMOKE_MAX_CROPS_LIMIT}을 초과합니다. "
            f"최대 {SMOKE_MAX_CROPS_LIMIT} 이하로 설정하세요."
        )
        sys.exit(1)

    # batch-size 상한 검증
    if args.batch_size > SMOKE_MAX_BATCH_SIZE_LIMIT:
        print(
            f"[ERROR] --batch-size={args.batch_size}는 smoke 상한 {SMOKE_MAX_BATCH_SIZE_LIMIT}을 초과합니다. "
            f"최대 {SMOKE_MAX_BATCH_SIZE_LIMIT} 이하로 설정하세요."
        )
        sys.exit(1)

    print(f"[INFO] Phase 7.1 v1/v1 scoring smoke 시작")
    print(f"[INFO] max_crops={args.max_crops}, batch_size={args.batch_size}")

    # ── 경로 기준 설정 (프로젝트 루트 기준 상대 경로 → 절대 경로)
    project_root = Path(__file__).resolve().parent.parent
    checkpoint_path = str(Path(args.checkpoint) if Path(args.checkpoint).is_absolute()
                          else project_root / args.checkpoint)
    manifest_path = str(Path(args.manifest) if Path(args.manifest).is_absolute()
                         else project_root / args.manifest)
    output_root = Path(args.output_root) if Path(args.output_root).is_absolute() \
        else project_root / args.output_root

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

    # ── manifest safety 검증
    print("[INFO] Manifest safety 검증 시작...")
    blockers = verify_manifest_safety(manifest_df)
    if blockers:
        print(f"[ERROR] Manifest safety 검증 실패: {len(blockers)}개 blocker")
        for b in blockers:
            print(f"[ERROR]   - {b}")
        sys.exit(1)
    print("[INFO] Manifest safety 검증: 모두 통과")

    # ── 상위 max_crops개 선택 (manifest 순서 그대로)
    manifest_df = manifest_df.reset_index(drop=False).rename(columns={"index": "_original_row_id"})
    smoke_df = manifest_df.head(args.max_crops).copy()
    print(f"[INFO] Smoke 대상: {len(smoke_df)} crops (max_crops={args.max_crops})")

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

    # ── Dataset / DataLoader 구성
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

    print(f"[INFO] Scoring 완료: {len(all_rows)} crops processed")

    scored_count = len(all_rows)

    # ── score 분포 통계 계산 (scored rows만, PASS 여부 무관)
    def _scalar_stats(rows, col):
        vals = [r[col] for r in rows if isinstance(r.get(col), float) and not math.isnan(r[col]) and not math.isinf(r[col])]
        if not vals:
            return {"min": float("nan"), "max": float("nan"), "mean": float("nan")}
        return {
            "min": float(min(vals)),
            "max": float(max(vals)),
            "mean": float(sum(vals) / len(vals)),
        }

    score_l1_stats = _scalar_stats(all_rows, "crop_score_l1_mean")
    score_l1max_stats = _scalar_stats(all_rows, "crop_score_l1_max")
    score_mse_stats = _scalar_stats(all_rows, "crop_score_mse_mean")

    # ── 최종 smoke_pass 판정
    fail_rows = [r for r in all_rows if r.get("smoke_status") == "FAIL"]
    nan_inf_pass = all(not r["has_nan"] and not r["has_inf"] for r in all_rows)
    error_nan_inf_pass = all(r["error_nan_count"] == 0 and r["error_inf_count"] == 0 for r in all_rows)
    shape_pass = all(r.get("_shape_error") is None for r in all_rows) if all_rows else True
    # all_rows의 dict에서 _shape_error는 Dataset에서 처리되어 meta에만 있음
    # compute_batch_smoke_scores에서 issue에 포함됨; fail_rows로 판단
    shape_pass = len(fail_rows) == 0 or all(
        "shape_error" not in r.get("issue", "") for r in all_rows
    )

    smoke_pass = (
        len(blockers) == 0
        and len(fail_rows) == 0
        and nan_inf_pass
        and error_nan_inf_pass
    )

    print(f"[INFO] smoke_pass: {smoke_pass}")
    if fail_rows:
        print(f"[WARNING] FAIL crop 수: {len(fail_rows)}")
        for r in fail_rows[:5]:
            print(f"[WARNING]   crop_id={r['crop_id']}, issue={r['issue']}")

    # ── output root 생성
    output_root.mkdir(parents=True, exist_ok=True)

    # ── CSV 저장
    csv_name = "phase7_1_v1v1_scoring_smoke_v1.csv"
    csv_path = output_root / csv_name

    if csv_path.exists():
        print(f"[WARNING] 기존 CSV 존재, 덮어씁니다: {csv_path}")

    df_out = pd.DataFrame(all_rows, columns=ALL_COLUMNS)
    df_out.to_csv(csv_path, index=False)
    print(f"[INFO] CSV 저장 완료: {csv_path} ({len(df_out)} rows, {len(ALL_COLUMNS)} columns)")

    # ── JSON 저장
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    json_name = "phase7_1_v1v1_scoring_smoke_v1.json"
    json_path = output_root / json_name

    # score_nan_inf_check_pass: 모든 crop에서 score_nan_count==0 and score_inf_count==0
    score_nan_inf_pass_flag = all(
        r.get("score_nan_count", 0) == 0 and r.get("score_inf_count", 0) == 0
        for r in all_rows
    )

    result_json = {
        "input_filtered_manifest_path": manifest_path,
        "checkpoint_path": checkpoint_path,
        "model_class": MODEL_CLASS,
        "scoring_script_reference": SCORING_SCRIPT_REFERENCE,
        "smoke_script_path": SMOKE_SCRIPT_PATH,
        "canonical_schema_column_count": len(CANONICAL_COLUMNS),
        "canonical_schema_columns": CANONICAL_COLUMNS,
        "smoke_extra_columns": SMOKE_EXTRA_COLUMNS,
        "max_crops": args.max_crops,
        "batch_size": args.batch_size,
        "scored_crop_count": scored_count,
        "manifest_row_count": EXPECTED_MANIFEST_ROWS,
        "unique_patient_count": EXPECTED_UNIQUE_PATIENTS,
        "stage2_holdout_row_count": 0,
        "excluded_patients_absent": True,
        "v2_path_detected": False,
        "score_l1_min": score_l1_stats["min"],
        "score_l1_max": score_l1_stats["max"],
        "score_l1_mean": score_l1_stats["mean"],
        "score_mse_min": score_mse_stats["min"],
        "score_mse_max": score_mse_stats["max"],
        "score_mse_mean": score_mse_stats["mean"],
        "score_nan_inf_check_pass": score_nan_inf_pass_flag,
        "error_nan_inf_check_pass": error_nan_inf_pass,
        "input_output_shape_check_pass": shape_pass,
        "metric_calculation_executed": False,
        "threshold_calculated": False,
        "training_executed": False,
        "backward_executed": False,
        "optimizer_step_executed": False,
        "checkpoint_created": False,
        "full_scoring_executed": False,
        "smoke_pass": smoke_pass,
        "blockers": blockers,
        "next_step_recommendation": (
            "smoke_pass=True. Phase 7.2 stage1_dev filtered full scoring (129437 crops) 승인 요청."
            if smoke_pass else
            "smoke_pass=False. blockers 및 FAIL crop 원인 확인 후 수정 필요."
        ),
        "timestamp": timestamp,
    }

    if json_path.exists():
        print(f"[WARNING] 기존 JSON 존재, 덮어씁니다: {json_path}")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False, default=str)
    print(f"[INFO] JSON 저장 완료: {json_path}")

    # ── MD 보고서 생성 및 저장
    md_name = "phase7_1_v1v1_scoring_smoke_report_v1.md"
    md_path = output_root / md_name

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
        error_nan_inf_pass=error_nan_inf_pass,
        shape_pass=shape_pass,
        timestamp=timestamp,
    )

    if md_path.exists():
        print(f"[WARNING] 기존 MD 존재, 덮어씁니다: {md_path}")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[INFO] MD 보고서 저장 완료: {md_path}")

    # ── 최종 요약 출력
    print("\n[INFO] ===== Phase 7.1 Smoke 결과 요약 =====")
    print(f"[INFO] scored_crop_count: {scored_count}")
    print(f"[INFO] FAIL crop 수: {len(fail_rows)}")
    print(f"[INFO] crop_score_l1_mean: min={score_l1_stats['min']:.6f}, max={score_l1_stats['max']:.6f}, mean={score_l1_stats['mean']:.6f}")
    print(f"[INFO] crop_score_mse_mean: min={score_mse_stats['min']:.6f}, max={score_mse_stats['max']:.6f}, mean={score_mse_stats['mean']:.6f}")
    print(f"[INFO] metric_calculation_executed: False")
    print(f"[INFO] threshold_calculated: False")
    print(f"[INFO] training_executed: False")
    print(f"[INFO] smoke_pass: {smoke_pass}")
    print(f"\n[INFO] 출력 파일:")
    print(f"[INFO]   CSV: {csv_path}")
    print(f"[INFO]   JSON: {json_path}")
    print(f"[INFO]   MD:   {md_path}")
    print("\n[INFO] Phase 7.1 v1/v1 scoring smoke 완료.")


if __name__ == "__main__":
    main()
