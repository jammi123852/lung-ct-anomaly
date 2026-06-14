"""
RD4AD-style 2.5D reconstruction score script
Normal val/test crop score 계산 (best_val_loss.pt 기준)

Phase 5.30 - score script for normal val/test split
목적: best checkpoint 기준 val/test crop별 reconstruction score 계산
     threshold 후보 산출 (val p90/p95/p99)
     test FP count/rate 확인 (val threshold 기준)

금지 사항:
  - train split 처리 금지
  - normal test를 threshold 튜닝에 사용 금지
  - hard negative / stage2_holdout / v2 경로 접근 금지
  - checkpoint / config / crop 수정 금지
"""

import argparse
import json
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
FORBIDDEN_PATH_PATTERNS = ["stage2_holdout", "v2"]
HARD_NEGATIVE_DIR_MARKER = "rd4ad_train_2p5d_mw_fixed96_thr001"
MODEL_TAG = "rd4ad_2p5d_normal_mw_fixed96_v1"
MODEL_TYPE = "ConvAutoencoder2p5D"
SCORE_FORMULA = "crop_score_l1_mean"

DEFAULT_CONFIG = (
    "outputs/second-stage-lesion-refiner-v1/models/"
    "rd4ad_2p5d_normal_mw_fixed96_v1/configs/"
    "train_config_rd4ad_2p5d_normal_mw_fixed96_v1_executable.yaml"
)
DEFAULT_CHECKPOINT = (
    "outputs/second-stage-lesion-refiner-v1/models/"
    "rd4ad_2p5d_normal_mw_fixed96_v1/checkpoints/best_val_loss.pt"
)
DEFAULT_CROP_MANIFEST = (
    "outputs/second-stage-lesion-refiner-v1/crops_normal/"
    "normal_rd4ad_2p5d_mw_fixed96_v1/manifests/"
    "crop_manifest_normal_rd4ad_2p5d_mw_fixed96_v1.csv"
)
DEFAULT_OUTPUT_ROOT = (
    "outputs/second-stage-lesion-refiner-v1/evaluation/"
    "rd4ad_2p5d_normal_mw_fixed96_v1/normal_val_test_scores_v1/"
)

# CSV 컬럼 순서 (설계 문서 기준)
SCORE_CSV_COLUMNS = [
    "patient_id",
    "safe_id",
    "normal_split",
    "crop_id",
    "crop_path",
    "checkpoint_path",
    "model_tag",
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
    "has_nan",
    "has_inf",
]


# ──────────────────────────────────────────────────────────
# Argument Parser
# ──────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Normal val/test crop reconstruction score 계산 스크립트. "
            "train split 처리 금지. normal test threshold tuning 금지."
        )
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="executable config YAML 경로",
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="checkpoint 경로 (best_val_loss.pt 기준)",
    )
    parser.add_argument(
        "--crop-manifest",
        default=DEFAULT_CROP_MANIFEST,
        help="crop manifest CSV 경로",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="score 저장 root 디렉토리",
    )
    parser.add_argument(
        "--target-splits",
        nargs="+",
        default=["val", "test"],
        choices=["val", "test"],
        help="score 계산 대상 split (val/test만 허용, train 제외)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="배치 크기",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "cuda_if_available"],
        default="cuda_if_available",
        help="실행 디바이스",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="계산만 수행하고 파일 저장 없이 종료",
    )
    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=None,
        help="split별 처리할 최대 crop 수 (None이면 전체)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="기존 output CSV 존재 시 덮어쓰기 허용",
    )
    parser.add_argument(
        "--no-runtime-append",
        action="store_true",
        help="runtime summary 추가 금지",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "config/checkpoint/manifest/output 충돌/target split만 확인하고 "
            "score 계산 없이 종료. 파일 생성 없음. model forward 없음."
        ),
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────
# Config Loader
# ──────────────────────────────────────────────────────────
def load_config(config_path: str) -> dict:
    try:
        import yaml
    except ImportError:
        print("[ERROR] PyYAML is not installed. Cannot load config.")
        print("        Do NOT run pip install. Please check your Python environment.")
        sys.exit(1)

    p = Path(config_path)
    if not p.exists():
        print(f"[ERROR] Config file not found: {p}")
        sys.exit(1)

    with open(p, "r") as f:
        cfg = yaml.safe_load(f)

    print(f"[INFO] Config loaded: {p}")
    return cfg


# ──────────────────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────────────────
def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda_if_available":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_arg == "cuda":
        if not torch.cuda.is_available():
            print("[ERROR] --device cuda requested but CUDA is not available.")
            sys.exit(1)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[INFO] Device: {device}")
    return device


# ──────────────────────────────────────────────────────────
# Safety Guards
# ──────────────────────────────────────────────────────────
def check_path_safety(path: str, label: str):
    """v2 / stage2_holdout 경로 차단"""
    for pattern in FORBIDDEN_PATH_PATTERNS:
        if pattern in str(path):
            print(f"[ERROR] {label} contains forbidden pattern '{pattern}': {path}")
            sys.exit(1)


def check_hard_negative_not_in_path(path: str, label: str):
    """hard negative 경로 차단"""
    if HARD_NEGATIVE_DIR_MARKER in str(path):
        print(f"[ERROR] Hard negative crop dir detected in {label}: {path}")
        print("[ERROR] Hard negative crops must NOT be used in normal val/test scoring.")
        sys.exit(1)


def check_split_not_train(split: str):
    """train split 차단"""
    if split == "train":
        print("[ERROR] train split is not allowed in this score script.")
        print("[ERROR] Only val and test splits are permitted.")
        sys.exit(1)


# ──────────────────────────────────────────────────────────
# Model (train script와 동일한 구조)
# ──────────────────────────────────────────────────────────
class ConvAutoencoder2p5D(nn.Module):
    """
    Minimal reconstruction baseline for 2.5D normal crop verifier.
    NOT a full RD4AD teacher-student implementation.

    input_channels=6  (lung z-1/z/z+1, mediastinal z-1/z/z+1)
    output_channels=6 (reconstruction target == input)
    Encoder: 96→48→24→12 (3× MaxPool2d)
    Decoder: 12→24→48→96 (3× ConvTranspose2d) + Sigmoid
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
class ScoreCropDataset(Dataset):
    """
    val / test split crop score 계산용 Dataset.
    train split은 차단됨.
    """

    def __init__(
        self,
        rows: pd.DataFrame,
        split: str,
        image_key: str = "image",
        limit: int = None,
    ):
        check_split_not_train(split)
        self.split = split
        self.image_key = image_key

        if limit is not None:
            rows = rows.head(limit)
        self.df = rows.reset_index(drop=True)

        # 경로 안전 검증
        for i, row in self.df.iterrows():
            cp = str(row.get("crop_path", ""))
            check_path_safety(cp, f"crop_path row={i}")
            check_hard_negative_not_in_path(cp, f"crop_path row={i}")

        print(f"[INFO] ScoreCropDataset split={split}: {len(self.df)} crops")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        crop_path = str(row["crop_path"])

        if not Path(crop_path).exists():
            raise FileNotFoundError(f"crop_path not found: {crop_path}")

        data = np.load(crop_path)
        if self.image_key not in data:
            raise KeyError(f"image key '{self.image_key}' not found in {crop_path}")

        image = data[self.image_key].astype(np.float32)

        # shape 확인
        if image.shape != (6, 96, 96):
            raise ValueError(
                f"Unexpected shape {image.shape} at {crop_path}, expected (6,96,96)"
            )

        # NaN/Inf 차단
        if not np.isfinite(image).all():
            raise ValueError(f"NaN or Inf detected in input: {crop_path}")

        # intensity 범위 확인 (경고 및 차단)
        if image.min() < -1e-4 or image.max() > 1.0 + 1e-4:
            raise ValueError(
                f"Intensity out of [0,1] in {crop_path}: "
                f"min={image.min():.4f}, max={image.max():.4f}"
            )

        tensor = torch.tensor(image, dtype=torch.float32)

        # metadata 반환
        meta = {
            "patient_id": str(row.get("patient_id", "")),
            "safe_id": str(row.get("safe_id", row.get("patient_id", ""))),
            "normal_split": str(row.get("normal_split", self.split)),
            "crop_id": str(row.get("crop_id", str(idx))),
            "crop_path": crop_path,
        }

        return tensor, meta


# ──────────────────────────────────────────────────────────
# Score Calculation (batch 단위)
# ──────────────────────────────────────────────────────────
def compute_batch_scores(
    inputs: torch.Tensor,
    recons: torch.Tensor,
    checkpoint_path: str,
    metas: list,
) -> list:
    """
    배치 단위 score 계산.
    inputs, recons: [B, 6, 96, 96] (CPU or GPU tensor)
    반환: list of dict (각 crop 1개)
    """
    # recon shape 확인 (input shape와 동일해야 함)
    if recons.shape != inputs.shape:
        raise ValueError(
            f"recon shape {tuple(recons.shape)} != input shape {tuple(inputs.shape)}"
        )

    # CPU로 이동 후 numpy 변환
    inputs_np = inputs.detach().cpu().numpy()   # [B, 6, 96, 96]
    recons_np = recons.detach().cpu().numpy()   # [B, 6, 96, 96]

    diff = np.abs(recons_np - inputs_np)        # L1 diff [B, 6, 96, 96]
    diff_sq = (recons_np - inputs_np) ** 2      # squared diff [B, 6, 96, 96]

    rows = []
    for b in range(inputs_np.shape[0]):
        inp = inputs_np[b]    # [6, 96, 96]
        rec = recons_np[b]    # [6, 96, 96]
        d = diff[b]           # [6, 96, 96]
        d_sq = diff_sq[b]     # [6, 96, 96]

        # 기본 v1 score
        crop_score_l1_mean = float(d.mean())
        crop_score_l1_max = float(d.max())
        crop_score_mse_mean = float(d_sq.mean())

        # 채널별 L1 mean
        channel_scores = {f"channel_{c}_l1_mean": float(d[c].mean()) for c in range(6)}

        # lung / mediastinal window
        lung_channels_l1_mean = float(d[0:3].mean())
        mediastinal_channels_l1_mean = float(d[3:6].mean())

        # input / recon 통계
        input_min = float(inp.min())
        input_max = float(inp.max())
        recon_min = float(rec.min())
        recon_max = float(rec.max())

        # recon NaN / Inf 확인
        has_nan = bool(not np.isfinite(rec).all() and np.isnan(rec).any())
        has_inf = bool(not np.isfinite(rec).all() and np.isinf(rec).any())

        meta = metas[b]

        row = {
            "patient_id": meta["patient_id"],
            "safe_id": meta["safe_id"],
            "normal_split": meta["normal_split"],
            "crop_id": meta["crop_id"],
            "crop_path": meta["crop_path"],
            "checkpoint_path": checkpoint_path,
            "model_tag": MODEL_TAG,
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
            "has_nan": has_nan,
            "has_inf": has_inf,
        }

        rows.append(row)

    return rows


# ──────────────────────────────────────────────────────────
# Collate function (meta dict 처리)
# ──────────────────────────────────────────────────────────
def collate_fn(batch):
    """DataLoader collate: tensor + meta dict list"""
    tensors = torch.stack([item[0] for item in batch])
    metas = [item[1] for item in batch]
    return tensors, metas


# ──────────────────────────────────────────────────────────
# Score Runner (split 단위)
# ──────────────────────────────────────────────────────────
def run_score_for_split(
    split: str,
    manifest_df: pd.DataFrame,
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    checkpoint_path: str,
    image_key: str,
    limit: int = None,
) -> list:
    """
    지정 split에 대해 전체 crop score 계산.
    반환: list of dict (SCORE_CSV_COLUMNS 기준)
    """
    check_split_not_train(split)

    split_df = manifest_df[manifest_df["normal_split"] == split].copy()
    if len(split_df) == 0:
        print(f"[WARNING] No crops found for split='{split}'. Skipping.")
        return []

    dataset = ScoreCropDataset(
        rows=split_df,
        split=split,
        image_key=image_key,
        limit=limit,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_fn,
    )

    all_rows = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (batch_tensors, batch_metas) in enumerate(loader):
            batch_tensors = batch_tensors.to(device)
            recons = model(batch_tensors)

            if batch_idx == 0:
                print(f"  split={split} first_batch_input_shape={tuple(batch_tensors.shape)}")
                print(f"  split={split} first_batch_recon_shape={tuple(recons.shape)}")

            rows = compute_batch_scores(
                inputs=batch_tensors,
                recons=recons,
                checkpoint_path=checkpoint_path,
                metas=batch_metas,
            )
            all_rows.extend(rows)

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"[INFO] split={split} | batch {batch_idx + 1}/{len(loader)} | "
                    f"processed={len(all_rows)} crops"
                )

    print(f"[INFO] split={split}: score 계산 완료 ({len(all_rows)} crops)")
    return all_rows


# ──────────────────────────────────────────────────────────
# Summary JSON 계산
# ──────────────────────────────────────────────────────────
def compute_summary(
    val_rows: list,
    test_rows: list,
    checkpoint_path: str,
    ckpt: dict,
) -> dict:
    """
    val/test score 통계 및 threshold 후보 계산.
    note: normal test는 threshold 튜닝에 사용하지 않음.
    """
    def _stats(rows: list) -> dict:
        if not rows:
            return {
                "n_crops": 0,
                "n_patients": 0,
                "score_mean": None,
                "score_std": None,
                "score_p50": None,
                "score_p90": None,
                "score_p95": None,
                "score_p99": None,
            }
        scores = np.array([r["crop_score_l1_mean"] for r in rows], dtype=np.float64)
        patient_ids = list({r["patient_id"] for r in rows})
        return {
            "n_crops": len(scores),
            "n_patients": len(patient_ids),
            "score_mean": float(scores.mean()),
            "score_std": float(scores.std()),
            "score_p50": float(np.percentile(scores, 50)),
            "score_p90": float(np.percentile(scores, 90)),
            "score_p95": float(np.percentile(scores, 95)),
            "score_p99": float(np.percentile(scores, 99)),
        }

    val_stats = _stats(val_rows)
    test_stats = _stats(test_rows)

    # val threshold 후보 (p90/p95/p99)
    threshold_candidates_from_val = {
        "val_p90": val_stats["score_p90"],
        "val_p95": val_stats["score_p95"],
        "val_p99": val_stats["score_p99"],
    }

    # test FP count/rate (val threshold 기준)
    # normal test crop이 threshold 초과하면 FP (정상인데 이상으로 탐지)
    # 주의: 이 값은 threshold 튜닝에 사용하지 않음, 참고용 FP 확인만
    test_fp_count_at_val_p95 = None
    test_fp_rate_at_val_p95 = None
    test_fp_count_at_val_p99 = None
    test_fp_rate_at_val_p99 = None

    if test_rows and val_stats["score_p95"] is not None:
        test_scores = np.array([r["crop_score_l1_mean"] for r in test_rows], dtype=np.float64)
        thr_p95 = val_stats["score_p95"]
        thr_p99 = val_stats["score_p99"]

        fp_p95 = int((test_scores > thr_p95).sum())
        test_fp_count_at_val_p95 = fp_p95
        test_fp_rate_at_val_p95 = float(fp_p95 / len(test_scores)) if len(test_scores) > 0 else None

        fp_p99 = int((test_scores > thr_p99).sum())
        test_fp_count_at_val_p99 = fp_p99
        test_fp_rate_at_val_p99 = float(fp_p99 / len(test_scores)) if len(test_scores) > 0 else None

    # checkpoint 메타
    ckpt_epoch = ckpt.get("epoch", None)
    ckpt_best_val_loss = ckpt.get("best_val_loss", None)

    summary = {
        "checkpoint_path": checkpoint_path,
        "checkpoint_epoch": ckpt_epoch,
        "checkpoint_best_val_loss": float(ckpt_best_val_loss) if ckpt_best_val_loss is not None else None,
        "model_type": MODEL_TYPE,
        "score_formula": SCORE_FORMULA,
        "n_val_crops": val_stats["n_crops"],
        "n_test_crops": test_stats["n_crops"],
        "n_val_patients": val_stats["n_patients"],
        "n_test_patients": test_stats["n_patients"],
        "val_score_mean": val_stats["score_mean"],
        "val_score_std": val_stats["score_std"],
        "val_score_p50": val_stats["score_p50"],
        "val_score_p90": val_stats["score_p90"],
        "val_score_p95": val_stats["score_p95"],
        "val_score_p99": val_stats["score_p99"],
        "test_score_mean": test_stats["score_mean"],
        "test_score_std": test_stats["score_std"],
        "test_score_p50": test_stats["score_p50"],
        "test_score_p90": test_stats["score_p90"],
        "test_score_p95": test_stats["score_p95"],
        "test_score_p99": test_stats["score_p99"],
        "threshold_candidates_from_val": threshold_candidates_from_val,
        "test_fp_count_at_val_p95": test_fp_count_at_val_p95,
        "test_fp_rate_at_val_p95": test_fp_rate_at_val_p95,
        "test_fp_count_at_val_p99": test_fp_count_at_val_p99,
        "test_fp_rate_at_val_p99": test_fp_rate_at_val_p99,
        "note": (
            "normal test는 threshold 튜닝에 사용하지 않음. "
            "hard negative 평가는 다음 단계. "
            "병변 성능 결론 금지. "
            "이 모델은 full RD4AD teacher-student가 아닌 minimal reconstruction baseline임."
        ),
        "script": "score_rd4ad_2p5d_normal_val_test.py",
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }

    return summary


# ──────────────────────────────────────────────────────────
# Output 충돌 확인
# ──────────────────────────────────────────────────────────
def check_output_conflict(output_root: Path, target_splits: list, force: bool):
    """기존 output CSV가 있으면 --force 없을 때 중단"""
    csv_names = {
        "val": "normal_val_scores_v1.csv",
        "test": "normal_test_scores_v1.csv",
    }
    conflict_found = False
    for split in target_splits:
        csv_path = output_root / csv_names.get(split, f"normal_{split}_scores_v1.csv")
        if csv_path.exists():
            print(f"[ERROR] Output CSV already exists: {csv_path}")
            conflict_found = True

    if conflict_found and not force:
        print("[ERROR] 기존 output CSV가 있습니다. 덮어쓰려면 --force 옵션을 사용하세요.")
        sys.exit(1)

    if conflict_found and force:
        print("[WARNING] --force: 기존 output CSV를 덮어씁니다.")


# ──────────────────────────────────────────────────────────
# Preflight Check
# ──────────────────────────────────────────────────────────
def run_preflight(args) -> None:
    """
    --preflight-only 모드:
    config/checkpoint/manifest/output 충돌/target split만 확인하고
    score 계산 없이 종료.
    파일 생성 없음. model forward 없음. CSV/JSON/MD 저장 없음.
    """
    print("[PREFLIGHT] ===== preflight-only 모드 시작 =====")
    print("[PREFLIGHT] score 계산, model forward, CSV/JSON/MD 저장 없음.")

    # 1. config 로드
    print(f"\n[PREFLIGHT] 1. config 로드: {args.config}")
    cfg = load_config(args.config)
    image_key = cfg.get("data", {}).get("image_key", "image")
    print("[PREFLIGHT] 1. config 로드 OK")

    # 2. checkpoint 존재 확인
    checkpoint_path = args.checkpoint
    print(f"\n[PREFLIGHT] 2. checkpoint 존재 확인: {checkpoint_path}")
    ckpt_p = Path(checkpoint_path)
    if not ckpt_p.exists():
        print(f"[PREFLIGHT][ERROR] Checkpoint not found: {ckpt_p}")
        sys.exit(1)
    print("[PREFLIGHT] 2. checkpoint 존재 확인 OK")

    # 3. checkpoint read-only 로드 가능 확인
    print(f"\n[PREFLIGHT] 3. checkpoint read-only 로드: {ckpt_p}")
    try:
        ckpt = torch.load(str(ckpt_p), map_location="cpu")
    except Exception as e:
        print(f"[PREFLIGHT][ERROR] checkpoint 로드 실패: {e}")
        sys.exit(1)
    print("[PREFLIGHT] 3. checkpoint read-only 로드 OK")

    # 4. checkpoint 내부 key 확인 (epoch, model_state_dict, best_val_loss, config)
    print("\n[PREFLIGHT] 4. checkpoint 내부 key 확인")
    required_keys = ["epoch", "model_state_dict", "best_val_loss", "config"]
    missing_keys = [k for k in required_keys if k not in ckpt]
    if missing_keys:
        print(f"[PREFLIGHT][WARNING] checkpoint에 다음 key가 없습니다: {missing_keys}")
    else:
        print(f"[PREFLIGHT]   필수 key 모두 존재: {required_keys}")
    print(
        f"[PREFLIGHT]   epoch={ckpt.get('epoch', 'N/A')}, "
        f"best_val_loss={ckpt.get('best_val_loss', 'N/A')}"
    )
    print("[PREFLIGHT] 4. checkpoint key 확인 완료")

    # 5. model instantiate 가능 확인
    print("\n[PREFLIGHT] 5. model instantiate 확인")
    try:
        model = ConvAutoencoder2p5D(input_channels=6, base_channels=32)
    except Exception as e:
        print(f"[PREFLIGHT][ERROR] model instantiate 실패: {e}")
        sys.exit(1)
    print("[PREFLIGHT] 5. model instantiate OK")

    # 6. model_state_dict load 가능 확인
    print("\n[PREFLIGHT] 6. model_state_dict load 확인")
    if "model_state_dict" not in ckpt:
        print("[PREFLIGHT][ERROR] checkpoint에 'model_state_dict' 키가 없습니다.")
        sys.exit(1)
    try:
        model.load_state_dict(ckpt["model_state_dict"])
    except Exception as e:
        print(f"[PREFLIGHT][ERROR] model_state_dict load 실패: {e}")
        sys.exit(1)
    print("[PREFLIGHT] 6. model_state_dict load OK")

    # 7. crop manifest 존재 확인
    manifest_path = args.crop_manifest
    print(f"\n[PREFLIGHT] 7. crop manifest 존재 확인: {manifest_path}")
    mf_p = Path(manifest_path)
    if not mf_p.exists():
        print(f"[PREFLIGHT][ERROR] crop manifest not found: {mf_p}")
        sys.exit(1)
    print("[PREFLIGHT] 7. crop manifest 존재 확인 OK")

    # 8. manifest row 수 확인 (total 18,100 / val 1,800 / test 1,800)
    print("\n[PREFLIGHT] 8. manifest row 수 확인")
    manifest_df = pd.read_csv(mf_p)
    EXPECTED_TOTAL = 18100
    EXPECTED_VAL = 1800
    EXPECTED_TEST = 1800
    total_rows = len(manifest_df)
    if total_rows != EXPECTED_TOTAL:
        print(f"[PREFLIGHT][WARNING] total rows={total_rows}, expected={EXPECTED_TOTAL}")
    else:
        print(f"[PREFLIGHT]   total rows OK: {total_rows}")
    if "normal_split" in manifest_df.columns:
        val_count = int((manifest_df["normal_split"] == "val").sum())
        test_count = int((manifest_df["normal_split"] == "test").sum())
        if val_count != EXPECTED_VAL:
            print(f"[PREFLIGHT][WARNING] val rows={val_count}, expected={EXPECTED_VAL}")
        else:
            print(f"[PREFLIGHT]   val rows OK: {val_count}")
        if test_count != EXPECTED_TEST:
            print(f"[PREFLIGHT][WARNING] test rows={test_count}, expected={EXPECTED_TEST}")
        else:
            print(f"[PREFLIGHT]   test rows OK: {test_count}")
    else:
        print("[PREFLIGHT][WARNING] manifest에 'normal_split' 컬럼이 없습니다.")
    print("[PREFLIGHT] 8. manifest row 수 확인 완료")

    # 9. target_splits가 val/test만 포함하는지 확인
    print(f"\n[PREFLIGHT] 9. target_splits 확인: {args.target_splits}")
    if not args.target_splits:
        print("[PREFLIGHT][ERROR] --target-splits가 비어 있습니다.")
        sys.exit(1)
    for sp in args.target_splits:
        if sp not in ("val", "test"):
            print(f"[PREFLIGHT][ERROR] 허용되지 않은 split: {sp}")
            sys.exit(1)
    print("[PREFLIGHT] 9. target_splits val/test 확인 OK")

    # 10. train split이 target에 포함되지 않았는지 확인
    print("\n[PREFLIGHT] 10. train split 차단 확인")
    for sp in args.target_splits:
        check_split_not_train(sp)
    print("[PREFLIGHT] 10. train split 차단 확인 OK")

    # 11. val/test crop_path 일부 존재 확인 (각 split별 최대 3개)
    print("\n[PREFLIGHT] 11. val/test crop_path 일부 존재 확인 (각 split별 최대 3개)")
    for sp in args.target_splits:
        if "normal_split" not in manifest_df.columns:
            print(f"[PREFLIGHT][WARNING] split={sp} crop_path 확인 불가 (normal_split 컬럼 없음)")
            continue
        sp_df = manifest_df[manifest_df["normal_split"] == sp]
        sample_rows = sp_df.head(3)
        for _, row in sample_rows.iterrows():
            cp = str(row.get("crop_path", ""))
            if not Path(cp).exists():
                print(f"[PREFLIGHT][WARNING] crop_path not found (split={sp}): {cp}")
            else:
                print(f"[PREFLIGHT]   split={sp} crop_path OK: {Path(cp).name}")
    print("[PREFLIGHT] 11. val/test crop_path 일부 존재 확인 완료")

    # 12. sample npz shape/dtype/range/NaN 확인 (각 split별 첫 번째 샘플)
    print("\n[PREFLIGHT] 12. sample npz shape/dtype/range/NaN 확인 (각 split별 첫 번째 샘플)")
    for sp in args.target_splits:
        if "normal_split" not in manifest_df.columns:
            print(f"[PREFLIGHT][WARNING] split={sp} npz 확인 불가 (normal_split 컬럼 없음)")
            continue
        sp_df = manifest_df[manifest_df["normal_split"] == sp]
        if len(sp_df) == 0:
            print(f"[PREFLIGHT][WARNING] split={sp}: manifest에 행 없음")
            continue
        first_row = sp_df.iloc[0]
        cp = str(first_row.get("crop_path", ""))
        if not Path(cp).exists():
            print(f"[PREFLIGHT][WARNING] split={sp} 첫 번째 crop_path 없음: {cp}")
            continue
        try:
            data = np.load(cp)
            if image_key not in data:
                print(
                    f"[PREFLIGHT][WARNING] split={sp}: "
                    f"image_key='{image_key}' 없음 in {Path(cp).name}"
                )
                continue
            img = data[image_key].astype(np.float32)
            if img.shape != (6, 96, 96):
                print(
                    f"[PREFLIGHT][WARNING] split={sp}: "
                    f"shape={img.shape}, expected=(6,96,96)"
                )
            else:
                print(f"[PREFLIGHT]   split={sp} shape OK: {img.shape}")
            print(f"[PREFLIGHT]   split={sp} dtype: {img.dtype}")
            vmin, vmax = float(img.min()), float(img.max())
            if vmin < -1e-4 or vmax > 1.0 + 1e-4:
                print(
                    f"[PREFLIGHT][WARNING] split={sp}: intensity out of [0,1] "
                    f"min={vmin:.4f} max={vmax:.4f}"
                )
            else:
                print(f"[PREFLIGHT]   split={sp} range OK: min={vmin:.4f} max={vmax:.4f}")
            if not np.isfinite(img).all():
                print(
                    f"[PREFLIGHT][WARNING] split={sp}: "
                    f"NaN or Inf detected in {Path(cp).name}"
                )
            else:
                print(f"[PREFLIGHT]   split={sp} NaN/Inf OK")
        except Exception as e:
            print(f"[PREFLIGHT][WARNING] split={sp} npz 로드 실패: {e}")
    print("[PREFLIGHT] 12. sample npz 확인 완료")

    # 13. output_root 충돌 확인 (preflight-only: 경고만, sys.exit 하지 않음)
    print(f"\n[PREFLIGHT] 13. output_root 충돌 확인 (경고만): {args.output_root}")
    output_root_pf = Path(args.output_root)
    csv_names_pf = {
        "val": "normal_val_scores_v1.csv",
        "test": "normal_test_scores_v1.csv",
    }
    for sp in args.target_splits:
        csv_path = output_root_pf / csv_names_pf.get(sp, f"normal_{sp}_scores_v1.csv")
        if csv_path.exists():
            print(f"[PREFLIGHT][WARNING] 기존 output CSV 이미 존재: {csv_path}")
            print("[PREFLIGHT]   full run 시 --force 없으면 중단됩니다.")
        else:
            print(f"[PREFLIGHT]   output CSV 없음 (충돌 없음): {csv_path}")
    print("[PREFLIGHT] 13. output_root 충돌 확인 완료")

    # 14. hard negative 경로 차단 확인
    print("\n[PREFLIGHT] 14. hard negative 경로 차단 확인")
    check_hard_negative_not_in_path(checkpoint_path, "--checkpoint (preflight)")
    check_hard_negative_not_in_path(manifest_path, "--crop-manifest (preflight)")
    print("[PREFLIGHT] 14. hard negative 차단 확인 OK")

    # 15. stage2_holdout/v2 차단 확인
    print("\n[PREFLIGHT] 15. stage2_holdout/v2 차단 확인")
    check_path_safety(checkpoint_path, "--checkpoint (preflight)")
    check_path_safety(manifest_path, "--crop-manifest (preflight)")
    check_path_safety(args.output_root, "--output-root (preflight)")
    print("[PREFLIGHT] 15. stage2_holdout/v2 차단 확인 OK")

    # 16. normal test threshold tuning 금지 note
    print("\n[PREFLIGHT] 16. NOTE: normal test는 threshold tuning에 사용하지 않음.")
    print("[PREFLIGHT]   val threshold(p90/p95/p99) 기준으로 test FP 참고 확인만 수행.")
    print("[PREFLIGHT]   threshold 결정에 normal test 수치를 사용하지 않음.")

    # 17. score 계산 없음
    print("\n[PREFLIGHT] 17. score 계산 수행 안 함 (preflight-only)")
    # 18. model forward 없음
    print("[PREFLIGHT] 18. model forward 수행 안 함 (preflight-only)")
    # 19. CSV/JSON/MD 저장 없음
    print("[PREFLIGHT] 19. CSV/JSON/MD 저장 안 함 (preflight-only)")

    print("\n[PREFLIGHT] 모든 확인 완료. score 계산 없이 종료합니다.")
    sys.exit(0)


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────
def main():
    args = parse_args()

    if args.preflight_only:
        run_preflight(args)

    # 1. target_splits 검증 (val/test만 허용, train 제외)
    for sp in args.target_splits:
        check_split_not_train(sp)

    if not args.target_splits:
        print("[ERROR] --target-splits가 비어 있습니다. val 또는 test를 지정하세요.")
        sys.exit(1)

    print(f"[INFO] target_splits: {args.target_splits}")

    # 2. config 로드
    cfg = load_config(args.config)

    # 3. checkpoint 로드 (read-only)
    checkpoint_path = args.checkpoint
    check_path_safety(checkpoint_path, "--checkpoint")
    check_hard_negative_not_in_path(checkpoint_path, "--checkpoint")

    ckpt_p = Path(checkpoint_path)
    if not ckpt_p.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_p}")
        sys.exit(1)

    device = resolve_device(args.device)

    print(f"[INFO] Loading checkpoint (read-only): {ckpt_p}")
    ckpt = torch.load(str(ckpt_p), map_location=device)
    print(
        f"[INFO] Checkpoint loaded: epoch={ckpt.get('epoch', 'N/A')}, "
        f"best_val_loss={ckpt.get('best_val_loss', 'N/A')}"
    )

    # 4. 모델 구조 생성 및 model_state_dict 로드
    model = ConvAutoencoder2p5D(input_channels=6, base_channels=32).to(device)

    if "model_state_dict" not in ckpt:
        print("[ERROR] checkpoint에 'model_state_dict' 키가 없습니다.")
        sys.exit(1)

    model.load_state_dict(ckpt["model_state_dict"])
    print("[INFO] model_state_dict 로드 완료.")

    # 5. model.eval()
    model.eval()
    print("[INFO] model.eval() 설정 완료.")

    # 6. manifest 로드
    manifest_path = args.crop_manifest
    check_path_safety(manifest_path, "--crop-manifest")
    check_hard_negative_not_in_path(manifest_path, "--crop-manifest")

    mf_p = Path(manifest_path)
    if not mf_p.exists():
        print(f"[ERROR] crop manifest not found: {mf_p}")
        sys.exit(1)

    manifest_df = pd.read_csv(mf_p)
    print(f"[INFO] Manifest loaded: {len(manifest_df)} rows from {mf_p}")

    # 7. manifest 내 전체 경로 안전 검증
    print("[INFO] Manifest 경로 안전 검증 중...")
    for i, row in manifest_df.iterrows():
        cp = str(row.get("crop_path", ""))
        check_path_safety(cp, f"manifest crop_path row={i}")
        check_hard_negative_not_in_path(cp, f"manifest crop_path row={i}")

    print("[INFO] Manifest 경로 안전 검증: OK")

    # 8. output root 충돌 확인 (--force 없으면 중단)
    output_root = Path(args.output_root)
    if not args.dry_run:
        check_output_conflict(output_root, args.target_splits, args.force)

    # image_key 설정
    image_key = cfg.get("data", {}).get("image_key", "image")
    print(f"[INFO] image_key: {image_key}")

    # 9. target_splits 순서대로 처리 (val 먼저, test 나중)
    ordered_splits = []
    if "val" in args.target_splits:
        ordered_splits.append("val")
    if "test" in args.target_splits:
        ordered_splits.append("test")

    all_split_rows = {}  # split -> list of dict

    for split in ordered_splits:
        print(f"\n[INFO] ===== split={split} score 계산 시작 =====")
        rows = run_score_for_split(
            split=split,
            manifest_df=manifest_df,
            model=model,
            device=device,
            batch_size=args.batch_size,
            checkpoint_path=checkpoint_path,
            image_key=image_key,
            limit=args.limit_per_split,
        )
        all_split_rows[split] = rows
        print(f"[INFO] split={split}: {len(rows)} rows 수집 완료")

    val_rows = all_split_rows.get("val", [])
    test_rows = all_split_rows.get("test", [])

    # 10. dry-run이면 결과 출력 후 종료 (파일 저장 안 함)
    if args.dry_run:
        print("\n[DRY-RUN] 계산 결과 출력 (파일 저장 없음)")
        for split in ordered_splits:
            rows = all_split_rows.get(split, [])
            if rows:
                scores = [r["crop_score_l1_mean"] for r in rows]
                print(
                    f"  split={split}: n={len(rows)}, "
                    f"score_mean={np.mean(scores):.6f}, "
                    f"score_std={np.std(scores):.6f}, "
                    f"score_p50={np.percentile(scores, 50):.6f}, "
                    f"score_p90={np.percentile(scores, 90):.6f}, "
                    f"score_p95={np.percentile(scores, 95):.6f}, "
                    f"score_p99={np.percentile(scores, 99):.6f}"
                )
                n_has_nan = sum(1 for r in rows if r["has_nan"])
                n_has_inf = sum(1 for r in rows if r["has_inf"])
                print(f"  split={split}: n_has_nan={n_has_nan}, n_has_inf={n_has_inf}")
                _score_cols = [
                    "crop_score_l1_mean", "crop_score_l1_max", "crop_score_mse_mean",
                    "channel_0_l1_mean", "channel_1_l1_mean", "channel_2_l1_mean",
                    "channel_3_l1_mean", "channel_4_l1_mean", "channel_5_l1_mean",
                    "lung_channels_l1_mean", "mediastinal_channels_l1_mean",
                ]
                _missing = [c for c in _score_cols if c not in rows[0]]
                if _missing:
                    print(f"  split={split} [WARN] 누락 score 컬럼: {_missing}")
                else:
                    print(f"  split={split} score 컬럼 모두 존재: OK")
                if rows:
                    print(f"  split={split} 첫 번째 row 예시:")
                    for k, v in list(rows[0].items())[:8]:
                        print(f"    {k}: {v}")
        print("[DRY-RUN] 파일 저장 없이 종료합니다.")
        sys.exit(0)

    # 11. CSV 저장 (split별)
    output_root.mkdir(parents=True, exist_ok=True)

    csv_names = {
        "val": "normal_val_scores_v1.csv",
        "test": "normal_test_scores_v1.csv",
    }

    for split in ordered_splits:
        rows = all_split_rows.get(split, [])
        if not rows:
            print(f"[WARNING] split={split}: rows가 비어 있어 CSV 저장을 건너뜁니다.")
            continue

        csv_path = output_root / csv_names.get(split, f"normal_{split}_scores_v1.csv")
        df_out = pd.DataFrame(rows, columns=SCORE_CSV_COLUMNS)
        df_out.to_csv(csv_path, index=False)
        print(f"[INFO] CSV 저장 완료: {csv_path} ({len(df_out)} rows)")

    # 12. summary JSON 계산 및 저장
    summary = compute_summary(
        val_rows=val_rows,
        test_rows=test_rows,
        checkpoint_path=checkpoint_path,
        ckpt=ckpt,
    )

    summary_path = output_root / "normal_val_test_score_summary_v1.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Summary JSON 저장 완료: {summary_path}")

    # threshold 후보 출력
    print("\n[INFO] ===== Threshold 후보 (val 기준) =====")
    thr = summary.get("threshold_candidates_from_val", {})
    for k, v in thr.items():
        print(f"  {k}: {v:.6f}" if v is not None else f"  {k}: None")

    # test FP 확인 출력
    print("\n[INFO] ===== Test FP 확인 (val threshold 기준, 참고용) =====")
    print(f"  test_fp_count_at_val_p95: {summary.get('test_fp_count_at_val_p95')}")
    print(f"  test_fp_rate_at_val_p95:  {summary.get('test_fp_rate_at_val_p95')}")
    print(f"  test_fp_count_at_val_p99: {summary.get('test_fp_count_at_val_p99')}")
    print(f"  test_fp_rate_at_val_p99:  {summary.get('test_fp_rate_at_val_p99')}")
    print("[INFO] 주의: normal test는 threshold 튜닝에 사용하지 않음 (참고용 FP 확인만)")

    print("\n[INFO] score_rd4ad_2p5d_normal_val_test.py 완료.")


if __name__ == "__main__":
    main()
