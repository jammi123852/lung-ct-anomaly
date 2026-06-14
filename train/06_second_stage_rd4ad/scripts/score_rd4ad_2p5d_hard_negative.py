"""
RD4AD-style 2.5D reconstruction score script
Hard negative crop score 계산 (best_val_loss.pt 기준)

Phase 5.34 - hard negative 7,700 score script
목적: best checkpoint 기준 hard negative crop별 reconstruction score 계산
     normal val threshold 후보 p90/p95/p99를 hard negative score에 적용
     이번 단계에서는 script 작성과 py_compile만 수행 (실행 금지)

금지 사항:
  - stage2_holdout / v2 경로 접근 금지
  - checkpoint / config / crop 수정 금지
  - normal test threshold tuning 금지
  - 병변 성능 결론 금지
  - pip/conda install 코드 포함 금지
  - 외부 다운로드 코드 포함 금지
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
EXPECTED_HARD_NEGATIVE_CROPS = 7700

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
    "outputs/second-stage-lesion-refiner-v1/crops/"
    "rd4ad_train_2p5d_mw_fixed96_thr001_v1/manifests/"
    "crop_manifest_rd4ad_train_2p5d_mw_fixed96_thr001_v1.csv"
)
DEFAULT_NORMAL_SCORE_SUMMARY = (
    "outputs/second-stage-lesion-refiner-v1/evaluation/"
    "rd4ad_2p5d_normal_mw_fixed96_v1/normal_val_test_scores_v1/"
    "normal_val_test_score_summary_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    "outputs/second-stage-lesion-refiner-v1/evaluation/"
    "rd4ad_2p5d_normal_mw_fixed96_v1/hard_negative_scores_v1/"
)

# val threshold fallback (normal score summary 로드 실패 시)
FALLBACK_VAL_P90 = 0.07003
FALLBACK_VAL_P95 = 0.08635
FALLBACK_VAL_P99 = 0.11331

# manifest 실제 컬럼명 → CSV 출력 컬럼명 매핑
OPTIONAL_COL_MANIFEST_TO_CSV = {
    "candidate_id": "source_candidate_id",
    "sample_role": "original_candidate_role",
    "mean_padim_score": "padim_score_mean",
    "max_padim_score": "padim_score_max",
    "bbox_too_large": "large_bbox_flag",
}
# manifest에 없는 컬럼 (→ None 처리)
OPTIONAL_COL_MISSING = ["zero_lc_patient_flag", "weak_case_flag"]
# manifest에서 그대로 사용하는 컬럼
OPTIONAL_COL_PASSTHROUGH = ["rd4ad_label", "binary_label"]

# CSV 기본 컬럼 순서
SCORE_CSV_BASE_COLUMNS = [
    "patient_id",
    "safe_id",
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
    "threshold_exceed_val_p90",
    "threshold_exceed_val_p95",
    "threshold_exceed_val_p99",
]

# CSV optional metadata 컬럼 순서
SCORE_CSV_OPTIONAL_COLUMNS = [
    "source_candidate_id",
    "rd4ad_label",
    "binary_label",
    "original_candidate_role",
    "padim_score_mean",
    "padim_score_max",
    "large_bbox_flag",
    "zero_lc_patient_flag",
    "weak_case_flag",
]


# ──────────────────────────────────────────────────────────
# Argument Parser
# ──────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Hard negative crop reconstruction score 계산 스크립트. "
            "stage2_holdout / v2 경로 접근 금지. "
            "normal test threshold tuning 금지. "
            "병변 성능 결론 금지."
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
        help="hard negative crop manifest CSV 경로 (7,700개 기대)",
    )
    parser.add_argument(
        "--normal-score-summary",
        default=DEFAULT_NORMAL_SCORE_SUMMARY,
        help="normal val/test score summary JSON 경로 (val threshold 로드용)",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="score 저장 root 디렉토리",
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
        "--preflight-only",
        action="store_true",
        help=(
            "config/checkpoint/manifest/output 충돌/threshold 로드만 확인하고 "
            "score 계산 없이 종료. 파일 생성 없음. model forward 없음."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="--limit N개만 처리, model forward 수행, 파일 저장 없이 종료",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="처리할 최대 crop 수 (None이면 전체, dry-run 시 활용)",
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
    """stage2_holdout / v2 경로 차단 (즉시 중단)"""
    for pattern in FORBIDDEN_PATH_PATTERNS:
        if pattern in str(path):
            print(f"[ERROR] {label} contains forbidden pattern '{pattern}': {path}")
            print("[ERROR] stage2_holdout / v2 경로는 이 script에서 사용할 수 없습니다.")
            sys.exit(1)


# ──────────────────────────────────────────────────────────
# Normal Val Threshold Loader
# ──────────────────────────────────────────────────────────
def load_val_thresholds(summary_path: str) -> dict:
    """
    normal score summary JSON에서 val threshold 후보 로드.
    threshold_candidates_from_val.val_p90/p95/p99 사용.
    없으면 fallback 값 사용하고 warning 출력.
    NOTE: normal test는 threshold tuning에 사용하지 않음.
    """
    p = Path(summary_path)
    if not p.exists():
        print(f"[WARNING] normal score summary not found: {p}")
        print("[WARNING] fallback threshold 값을 사용합니다.")
        print(f"[WARNING]   val_p90={FALLBACK_VAL_P90}, val_p95={FALLBACK_VAL_P95}, val_p99={FALLBACK_VAL_P99}")
        return {
            "val_p90": FALLBACK_VAL_P90,
            "val_p95": FALLBACK_VAL_P95,
            "val_p99": FALLBACK_VAL_P99,
            "source": "fallback",
        }

    try:
        with open(p, "r", encoding="utf-8") as f:
            summary = json.load(f)
    except Exception as e:
        print(f"[WARNING] normal score summary 로드 실패: {e}")
        print("[WARNING] fallback threshold 값을 사용합니다.")
        return {
            "val_p90": FALLBACK_VAL_P90,
            "val_p95": FALLBACK_VAL_P95,
            "val_p99": FALLBACK_VAL_P99,
            "source": "fallback",
        }

    thr_candidates = summary.get("threshold_candidates_from_val", {})
    val_p90 = thr_candidates.get("val_p90", None)
    val_p95 = thr_candidates.get("val_p95", None)
    val_p99 = thr_candidates.get("val_p99", None)

    result = {}
    used_fallback = []

    if val_p90 is None:
        print(f"[WARNING] val_p90 not found in summary, using fallback: {FALLBACK_VAL_P90}")
        result["val_p90"] = FALLBACK_VAL_P90
        used_fallback.append("val_p90")
    else:
        result["val_p90"] = float(val_p90)

    if val_p95 is None:
        print(f"[WARNING] val_p95 not found in summary, using fallback: {FALLBACK_VAL_P95}")
        result["val_p95"] = FALLBACK_VAL_P95
        used_fallback.append("val_p95")
    else:
        result["val_p95"] = float(val_p95)

    if val_p99 is None:
        print(f"[WARNING] val_p99 not found in summary, using fallback: {FALLBACK_VAL_P99}")
        result["val_p99"] = FALLBACK_VAL_P99
        used_fallback.append("val_p99")
    else:
        result["val_p99"] = float(val_p99)

    result["source"] = "fallback_partial" if used_fallback else "normal_score_summary"
    print(
        f"[INFO] val threshold 로드 완료: "
        f"val_p90={result['val_p90']:.6f}, "
        f"val_p95={result['val_p95']:.6f}, "
        f"val_p99={result['val_p99']:.6f} "
        f"(source={result['source']})"
    )
    print("[INFO] NOTE: normal test는 threshold tuning에 사용하지 않음. val 기준 threshold만 적용.")
    return result


# ──────────────────────────────────────────────────────────
# Model (참고 script와 동일한 ConvAutoencoder2p5D)
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
class HardNegativeScoreDataset(Dataset):
    """
    Hard negative crop score 계산용 Dataset.
    split 필터링 없음 (manifest에 normal_split 컬럼 없음, 전체 7,700개 처리).
    stage2_holdout / v2 경로 차단.
    hard negative root(rd4ad_train_2p5d_mw_fixed96_thr001) 허용.
    """

    def __init__(
        self,
        rows: pd.DataFrame,
        image_key: str = "image",
        limit: int = None,
    ):
        self.image_key = image_key

        if limit is not None:
            rows = rows.head(limit)
        self.df = rows.reset_index(drop=True)

        # 경로 안전 검증 (stage2_holdout / v2 차단, hard negative root는 허용)
        for i, row in self.df.iterrows():
            cp = str(row.get("crop_path", ""))
            check_path_safety(cp, f"crop_path row={i}")

        print(f"[INFO] HardNegativeScoreDataset: {len(self.df)} crops")

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

        # dtype 확인
        if image.dtype != np.float32:
            raise ValueError(
                f"Unexpected dtype {image.dtype} at {crop_path}, expected float32"
            )

        # NaN/Inf 차단
        if not np.isfinite(image).all():
            raise ValueError(f"NaN or Inf detected in input: {crop_path}")

        # intensity 범위 확인 [0,1]
        if image.min() < -1e-4 or image.max() > 1.0 + 1e-4:
            raise ValueError(
                f"Intensity out of [0,1] in {crop_path}: "
                f"min={image.min():.4f}, max={image.max():.4f}"
            )

        tensor = torch.tensor(image, dtype=torch.float32)

        # metadata 반환
        # patient_id, safe_id, crop_id: 있으면 사용, 없으면 fallback
        patient_id = str(row.get("patient_id", ""))
        safe_id = str(row.get("safe_id", row.get("patient_id", "")))
        crop_id = str(row.get("crop_id", str(idx)))

        # optional metadata (manifest 실제 컬럼명으로 읽기)
        opt_meta = {}
        for manifest_col, csv_col in OPTIONAL_COL_MANIFEST_TO_CSV.items():
            opt_meta[csv_col] = row.get(manifest_col, None)
            if opt_meta[csv_col] is not None:
                opt_meta[csv_col] = str(opt_meta[csv_col]) if not isinstance(opt_meta[csv_col], (int, float, bool)) else opt_meta[csv_col]

        for col in OPTIONAL_COL_MISSING:
            opt_meta[col] = None

        for col in OPTIONAL_COL_PASSTHROUGH:
            val = row.get(col, None)
            opt_meta[col] = val

        meta = {
            "patient_id": patient_id,
            "safe_id": safe_id,
            "crop_id": crop_id,
            "crop_path": crop_path,
            **opt_meta,
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
    val_thresholds: dict,
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

    val_p90 = val_thresholds["val_p90"]
    val_p95 = val_thresholds["val_p95"]
    val_p99 = val_thresholds["val_p99"]

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

        # threshold exceed 컬럼
        threshold_exceed_val_p90 = bool(crop_score_l1_mean > val_p90)
        threshold_exceed_val_p95 = bool(crop_score_l1_mean > val_p95)
        threshold_exceed_val_p99 = bool(crop_score_l1_mean > val_p99)

        meta = metas[b]

        row = {
            # 기본 컬럼
            "patient_id": meta["patient_id"],
            "safe_id": meta["safe_id"],
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
            "threshold_exceed_val_p90": threshold_exceed_val_p90,
            "threshold_exceed_val_p95": threshold_exceed_val_p95,
            "threshold_exceed_val_p99": threshold_exceed_val_p99,
            # optional metadata
            "source_candidate_id": meta.get("source_candidate_id", None),
            "rd4ad_label": meta.get("rd4ad_label", None),
            "binary_label": meta.get("binary_label", None),
            "original_candidate_role": meta.get("original_candidate_role", None),
            "padim_score_mean": meta.get("padim_score_mean", None),
            "padim_score_max": meta.get("padim_score_max", None),
            "large_bbox_flag": meta.get("large_bbox_flag", None),
            "zero_lc_patient_flag": meta.get("zero_lc_patient_flag", None),
            "weak_case_flag": meta.get("weak_case_flag", None),
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
# Score Runner (전체 hard negative)
# ──────────────────────────────────────────────────────────
def run_score_hard_negative(
    manifest_df: pd.DataFrame,
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    checkpoint_path: str,
    image_key: str,
    val_thresholds: dict,
    limit: int = None,
    is_dry_run: bool = False,
) -> list:
    """
    전체 hard negative crop에 대해 score 계산.
    split 필터링 없음 (manifest에 normal_split 컬럼 없음).
    반환: list of dict
    """
    dataset = HardNegativeScoreDataset(
        rows=manifest_df,
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
    first_batch_logged = False

    with torch.no_grad():
        for batch_idx, (batch_tensors, batch_metas) in enumerate(loader):
            batch_tensors = batch_tensors.to(device)
            recons = model(batch_tensors)

            if not first_batch_logged:
                print(f"[INFO] first_batch_input_shape={tuple(batch_tensors.shape)}")
                print(f"[INFO] first_batch_recon_shape={tuple(recons.shape)}")
                first_batch_logged = True

            rows = compute_batch_scores(
                inputs=batch_tensors,
                recons=recons,
                checkpoint_path=checkpoint_path,
                metas=batch_metas,
                val_thresholds=val_thresholds,
            )
            all_rows.extend(rows)

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"[INFO] batch {batch_idx + 1}/{len(loader)} | "
                    f"processed={len(all_rows)} crops"
                )

    print(f"[INFO] hard negative score 계산 완료: {len(all_rows)} crops")
    return all_rows


# ──────────────────────────────────────────────────────────
# Summary JSON 계산
# ──────────────────────────────────────────────────────────
def compute_hard_negative_summary(
    all_rows: list,
    checkpoint_path: str,
    ckpt: dict,
    val_thresholds: dict,
) -> dict:
    """
    hard negative score 통계 및 threshold exceed 요약 계산.
    NOTE:
      - hard negative는 학습 데이터 아님
      - threshold 확정 아님
      - 병변 성능 결론 금지
      - stage2_holdout 미사용
      - v2 미사용
      - full RD4AD teacher-student가 아닌 minimal reconstruction baseline
    """
    n_crops = len(all_rows)
    if n_crops == 0:
        return {
            "checkpoint_path": checkpoint_path,
            "checkpoint_epoch": ckpt.get("epoch", None),
            "checkpoint_best_val_loss": None,
            "score_formula": SCORE_FORMULA,
            "n_hard_negative_crops": 0,
            "n_patients": 0,
            "score_mean": None,
            "score_std": None,
            "score_p50": None,
            "score_p90": None,
            "score_p95": None,
            "score_p99": None,
            "val_threshold_p90": val_thresholds["val_p90"],
            "val_threshold_p95": val_thresholds["val_p95"],
            "val_threshold_p99": val_thresholds["val_p99"],
            "exceed_count_val_p90": None,
            "exceed_rate_val_p90": None,
            "exceed_count_val_p95": None,
            "exceed_rate_val_p95": None,
            "exceed_count_val_p99": None,
            "exceed_rate_val_p99": None,
            "top_score_patients": [],
            "top_score_crops": [],
            "optional_columns_present": [],
            "optional_columns_missing": [],
            "note": {
                "hard_negative": "hard negative는 학습 데이터 아님",
                "threshold": "threshold 확정 아님",
                "lesion_conclusion": "병변 성능 결론 금지",
                "stage2_holdout": "stage2_holdout 미사용",
                "v2": "v2 미사용",
                "model": "full RD4AD teacher-student가 아닌 minimal reconstruction baseline",
            },
        }

    scores = np.array([r["crop_score_l1_mean"] for r in all_rows], dtype=np.float64)
    patient_ids = list({r["patient_id"] for r in all_rows})

    ckpt_best_val_loss = ckpt.get("best_val_loss", None)

    # threshold exceed count/rate
    exceed_p90 = int(sum(1 for r in all_rows if r.get("threshold_exceed_val_p90", False)))
    exceed_p95 = int(sum(1 for r in all_rows if r.get("threshold_exceed_val_p95", False)))
    exceed_p99 = int(sum(1 for r in all_rows if r.get("threshold_exceed_val_p99", False)))

    # top score patients (상위 10개, score 기준 내림차순)
    patient_scores = {}
    for r in all_rows:
        pid = r["patient_id"]
        s = r["crop_score_l1_mean"]
        if pid not in patient_scores or s > patient_scores[pid]:
            patient_scores[pid] = s
    top_score_patients = sorted(patient_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    top_score_patients = [{"patient_id": pid, "max_score": float(s)} for pid, s in top_score_patients]

    # top score crops (상위 10개)
    top_crops = sorted(all_rows, key=lambda r: r["crop_score_l1_mean"], reverse=True)[:10]
    top_score_crops = [
        {
            "crop_id": r["crop_id"],
            "patient_id": r["patient_id"],
            "crop_score_l1_mean": r["crop_score_l1_mean"],
            "crop_path": r["crop_path"],
        }
        for r in top_crops
    ]

    # optional 컬럼 존재/누락 확인
    first_row = all_rows[0]
    optional_present = [c for c in SCORE_CSV_OPTIONAL_COLUMNS if first_row.get(c) is not None]
    optional_missing = [c for c in SCORE_CSV_OPTIONAL_COLUMNS if first_row.get(c) is None]

    summary = {
        "checkpoint_path": checkpoint_path,
        "checkpoint_epoch": ckpt.get("epoch", None),
        "checkpoint_best_val_loss": float(ckpt_best_val_loss) if ckpt_best_val_loss is not None else None,
        "score_formula": SCORE_FORMULA,
        "n_hard_negative_crops": n_crops,
        "n_patients": len(patient_ids),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
        "score_p50": float(np.percentile(scores, 50)),
        "score_p90": float(np.percentile(scores, 90)),
        "score_p95": float(np.percentile(scores, 95)),
        "score_p99": float(np.percentile(scores, 99)),
        "val_threshold_p90": val_thresholds["val_p90"],
        "val_threshold_p95": val_thresholds["val_p95"],
        "val_threshold_p99": val_thresholds["val_p99"],
        "exceed_count_val_p90": exceed_p90,
        "exceed_rate_val_p90": float(exceed_p90 / n_crops) if n_crops > 0 else None,
        "exceed_count_val_p95": exceed_p95,
        "exceed_rate_val_p95": float(exceed_p95 / n_crops) if n_crops > 0 else None,
        "exceed_count_val_p99": exceed_p99,
        "exceed_rate_val_p99": float(exceed_p99 / n_crops) if n_crops > 0 else None,
        "top_score_patients": top_score_patients,
        "top_score_crops": top_score_crops,
        "optional_columns_present": optional_present,
        "optional_columns_missing": optional_missing,
        "note": {
            "hard_negative": "hard negative는 학습 데이터 아님",
            "threshold": "threshold 확정 아님",
            "lesion_conclusion": "병변 성능 결론 금지",
            "stage2_holdout": "stage2_holdout 미사용",
            "v2": "v2 미사용",
            "model": "full RD4AD teacher-student가 아닌 minimal reconstruction baseline",
        },
        "script": "score_rd4ad_2p5d_hard_negative.py",
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }

    return summary


# ──────────────────────────────────────────────────────────
# Output 충돌 확인
# ──────────────────────────────────────────────────────────
def check_output_conflict(output_root: Path, force: bool):
    """기존 output CSV가 있으면 --force 없을 때 중단"""
    csv_path = output_root / "hard_negative_scores_v1.csv"
    if csv_path.exists():
        if not force:
            print(f"[ERROR] Output CSV already exists: {csv_path}")
            print("[ERROR] 기존 output CSV가 있습니다. 덮어쓰려면 --force 옵션을 사용하세요.")
            sys.exit(1)
        else:
            print(f"[WARNING] --force: 기존 output CSV를 덮어씁니다: {csv_path}")


# ──────────────────────────────────────────────────────────
# Preflight Check
# ──────────────────────────────────────────────────────────
def run_preflight(args) -> None:
    """
    --preflight-only 모드:
    config/checkpoint/manifest/output 충돌/threshold 로드만 확인하고
    score 계산 없이 종료.
    파일 생성 없음. model forward 없음. CSV/JSON/MD 저장 없음.
    """
    print("[PREFLIGHT] ===== preflight-only 모드 시작 =====")
    print("[PREFLIGHT] score 계산, model forward, CSV/JSON/MD 저장 없음.")

    # 1. config 로드
    print(f"\n[PREFLIGHT] 1. config 로드: {args.config}")
    cfg = load_config(args.config)
    image_key = cfg.get("data", {}).get("image_key", "image")
    print(f"[PREFLIGHT] 1. config 로드 OK (image_key={image_key})")

    # 2. checkpoint 존재 확인
    checkpoint_path = args.checkpoint
    print(f"\n[PREFLIGHT] 2. checkpoint 존재 확인: {checkpoint_path}")
    check_path_safety(checkpoint_path, "--checkpoint (preflight)")
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
    check_path_safety(manifest_path, "--crop-manifest (preflight)")
    mf_p = Path(manifest_path)
    if not mf_p.exists():
        print(f"[PREFLIGHT][ERROR] crop manifest not found: {mf_p}")
        sys.exit(1)
    print("[PREFLIGHT] 7. crop manifest 존재 확인 OK")

    # 8. manifest row 수 확인 (기대값 7,700)
    print("\n[PREFLIGHT] 8. manifest row 수 확인 (기대값 7,700)")
    manifest_df = pd.read_csv(mf_p)
    total_rows = len(manifest_df)
    if total_rows != EXPECTED_HARD_NEGATIVE_CROPS:
        print(f"[PREFLIGHT][WARNING] total rows={total_rows}, expected={EXPECTED_HARD_NEGATIVE_CROPS}")
    else:
        print(f"[PREFLIGHT]   total rows OK: {total_rows}")
    if "normal_split" in manifest_df.columns:
        print("[PREFLIGHT][INFO] manifest에 normal_split 컬럼 있음 (split 필터 미사용, 전체 처리)")
    else:
        print("[PREFLIGHT]   normal_split 컬럼 없음 → 전체 처리 (기대 동작)")
    print("[PREFLIGHT] 8. manifest row 수 확인 완료")

    # 9. crop_path sample 존재 확인 (최대 3개)
    print("\n[PREFLIGHT] 9. crop_path sample 존재 확인 (최대 3개)")
    sample_rows = manifest_df.head(3)
    for _, row in sample_rows.iterrows():
        cp = str(row.get("crop_path", ""))
        check_path_safety(cp, "crop_path (preflight sample)")
        if not Path(cp).exists():
            print(f"[PREFLIGHT][WARNING] crop_path not found: {cp}")
        else:
            print(f"[PREFLIGHT]   crop_path OK: {Path(cp).name}")
    print("[PREFLIGHT] 9. crop_path sample 확인 완료")

    # 10. sample npz shape/dtype/range/NaN 확인 (첫 번째 샘플)
    print("\n[PREFLIGHT] 10. sample npz shape/dtype/range/NaN 확인 (첫 번째 샘플)")
    first_row = manifest_df.iloc[0]
    cp0 = str(first_row.get("crop_path", ""))
    if Path(cp0).exists():
        try:
            data = np.load(cp0)
            image_key_pf = cfg.get("data", {}).get("image_key", "image")
            if image_key_pf not in data:
                print(f"[PREFLIGHT][WARNING] image key '{image_key_pf}' 없음 in {Path(cp0).name}")
            else:
                img = data[image_key_pf].astype(np.float32)
                if img.shape != (6, 96, 96):
                    print(f"[PREFLIGHT][WARNING] shape={img.shape}, expected=(6,96,96)")
                else:
                    print(f"[PREFLIGHT]   shape OK: {img.shape}")
                print(f"[PREFLIGHT]   dtype: {img.dtype}")
                vmin, vmax = float(img.min()), float(img.max())
                if vmin < -1e-4 or vmax > 1.0 + 1e-4:
                    print(f"[PREFLIGHT][WARNING] intensity out of [0,1]: min={vmin:.4f} max={vmax:.4f}")
                else:
                    print(f"[PREFLIGHT]   range OK: min={vmin:.4f} max={vmax:.4f}")
                if not np.isfinite(img).all():
                    print(f"[PREFLIGHT][WARNING] NaN or Inf detected in {Path(cp0).name}")
                else:
                    print(f"[PREFLIGHT]   NaN/Inf OK")
        except Exception as e:
            print(f"[PREFLIGHT][WARNING] npz 로드 실패: {e}")
    else:
        print(f"[PREFLIGHT][WARNING] 첫 번째 crop_path 없음: {cp0}")
    print("[PREFLIGHT] 10. sample npz 확인 완료")

    # 11. normal score summary 존재 및 threshold 로드 확인
    print(f"\n[PREFLIGHT] 11. normal score summary 및 threshold 로드 확인: {args.normal_score_summary}")
    val_thresholds = load_val_thresholds(args.normal_score_summary)
    print(
        f"[PREFLIGHT]   val_p90={val_thresholds['val_p90']:.6f}, "
        f"val_p95={val_thresholds['val_p95']:.6f}, "
        f"val_p99={val_thresholds['val_p99']:.6f} "
        f"(source={val_thresholds['source']})"
    )
    print("[PREFLIGHT] 11. threshold 로드 확인 완료")

    # 12. output_root 충돌 확인 (preflight-only: 경고만, sys.exit 하지 않음)
    print(f"\n[PREFLIGHT] 12. output_root 충돌 확인 (경고만): {args.output_root}")
    output_root_pf = Path(args.output_root)
    csv_path_pf = output_root_pf / "hard_negative_scores_v1.csv"
    if csv_path_pf.exists():
        print(f"[PREFLIGHT][WARNING] 기존 output CSV 이미 존재: {csv_path_pf}")
        print("[PREFLIGHT]   full run 시 --force 없으면 중단됩니다.")
    else:
        print(f"[PREFLIGHT]   output CSV 없음 (충돌 없음): {csv_path_pf}")
    check_path_safety(args.output_root, "--output-root (preflight)")
    print("[PREFLIGHT] 12. output_root 확인 완료")

    # 13. stage2_holdout/v2 차단 확인
    print("\n[PREFLIGHT] 13. stage2_holdout/v2 차단 확인")
    check_path_safety(checkpoint_path, "--checkpoint (preflight stage2_holdout/v2 check)")
    check_path_safety(manifest_path, "--crop-manifest (preflight stage2_holdout/v2 check)")
    check_path_safety(args.output_root, "--output-root (preflight stage2_holdout/v2 check)")
    print("[PREFLIGHT] 13. stage2_holdout/v2 차단 확인 OK")

    # 14. normal test threshold tuning 금지 note
    print("\n[PREFLIGHT] 14. NOTE: normal test는 threshold tuning에 사용하지 않음.")
    print("[PREFLIGHT]   val threshold(p90/p95/p99) 기준으로 hard negative score에 적용.")
    print("[PREFLIGHT]   threshold 결정에 normal test 수치 사용 금지.")
    print("[PREFLIGHT]   hard negative는 학습 데이터 아님. 병변 성능 결론 금지.")

    # 15. score 계산 없음
    print("\n[PREFLIGHT] 15. score 계산 수행 안 함 (preflight-only)")
    # 16. model forward 없음
    print("[PREFLIGHT] 16. model forward 수행 안 함 (preflight-only)")
    # 17. CSV/JSON/MD 저장 없음
    print("[PREFLIGHT] 17. CSV/JSON/MD 저장 안 함 (preflight-only)")

    print("\n[PREFLIGHT] 모든 확인 완료. score 계산 없이 종료합니다.")
    sys.exit(0)


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────
def main():
    args = parse_args()

    if args.preflight_only:
        run_preflight(args)

    # 1. config 로드
    cfg = load_config(args.config)

    # 2. checkpoint 로드 (read-only)
    checkpoint_path = args.checkpoint
    check_path_safety(checkpoint_path, "--checkpoint")

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

    # checkpoint 필수 key 확인
    required_keys = ["epoch", "model_state_dict", "best_val_loss", "config"]
    missing_keys = [k for k in required_keys if k not in ckpt]
    if missing_keys:
        print(f"[WARNING] checkpoint에 다음 key가 없습니다: {missing_keys}")

    # 3. 모델 구조 생성 및 model_state_dict 로드
    model = ConvAutoencoder2p5D(input_channels=6, base_channels=32).to(device)

    if "model_state_dict" not in ckpt:
        print("[ERROR] checkpoint에 'model_state_dict' 키가 없습니다.")
        sys.exit(1)

    model.load_state_dict(ckpt["model_state_dict"])
    print("[INFO] model_state_dict 로드 완료.")

    # 4. model.eval()
    model.eval()
    print("[INFO] model.eval() 설정 완료.")

    # 5. manifest 로드
    manifest_path = args.crop_manifest
    check_path_safety(manifest_path, "--crop-manifest")

    mf_p = Path(manifest_path)
    if not mf_p.exists():
        print(f"[ERROR] crop manifest not found: {mf_p}")
        sys.exit(1)

    manifest_df = pd.read_csv(mf_p)
    print(f"[INFO] Manifest loaded: {len(manifest_df)} rows from {mf_p}")

    # 7,700개 확인
    if len(manifest_df) != EXPECTED_HARD_NEGATIVE_CROPS:
        print(
            f"[WARNING] manifest rows={len(manifest_df)}, "
            f"expected={EXPECTED_HARD_NEGATIVE_CROPS}"
        )

    # 6. manifest 내 전체 경로 안전 검증 (stage2_holdout / v2 차단)
    print("[INFO] Manifest 경로 안전 검증 중...")
    for i, row in manifest_df.iterrows():
        cp = str(row.get("crop_path", ""))
        check_path_safety(cp, f"manifest crop_path row={i}")
    print("[INFO] Manifest 경로 안전 검증: OK")

    # 7. output root 충돌 확인 (--force 없으면 중단)
    output_root = Path(args.output_root)
    if not args.dry_run:
        check_output_conflict(output_root, args.force)

    # 8. image_key 설정
    image_key = cfg.get("data", {}).get("image_key", "image")
    print(f"[INFO] image_key: {image_key}")

    # 9. val threshold 로드
    check_path_safety(args.normal_score_summary, "--normal-score-summary")
    val_thresholds = load_val_thresholds(args.normal_score_summary)

    # 10. hard negative score 계산
    print("\n[INFO] ===== hard negative score 계산 시작 =====")
    all_rows = run_score_hard_negative(
        manifest_df=manifest_df,
        model=model,
        device=device,
        batch_size=args.batch_size,
        checkpoint_path=checkpoint_path,
        image_key=image_key,
        val_thresholds=val_thresholds,
        limit=args.limit,
        is_dry_run=args.dry_run,
    )

    # 11. dry-run이면 결과 출력 후 종료 (파일 저장 안 함)
    if args.dry_run:
        print("\n[DRY-RUN] 계산 결과 출력 (파일 저장 없음)")
        if all_rows:
            scores = [r["crop_score_l1_mean"] for r in all_rows]
            print(f"  n={len(all_rows)}")
            print(f"  score_mean={np.mean(scores):.6f}")
            print(f"  score_std={np.std(scores):.6f}")
            print(f"  score_p50={np.percentile(scores, 50):.6f}")
            print(f"  score_p90={np.percentile(scores, 90):.6f}")
            print(f"  score_p95={np.percentile(scores, 95):.6f}")
            print(f"  score_p99={np.percentile(scores, 99):.6f}")

            n_has_nan = sum(1 for r in all_rows if r["has_nan"])
            n_has_inf = sum(1 for r in all_rows if r["has_inf"])
            print(f"  n_has_nan={n_has_nan}, n_has_inf={n_has_inf}")

            # score 컬럼 존재 확인
            _score_cols = [
                "crop_score_l1_mean", "crop_score_l1_max", "crop_score_mse_mean",
                "channel_0_l1_mean", "channel_1_l1_mean", "channel_2_l1_mean",
                "channel_3_l1_mean", "channel_4_l1_mean", "channel_5_l1_mean",
                "lung_channels_l1_mean", "mediastinal_channels_l1_mean",
            ]
            _missing = [c for c in _score_cols if c not in all_rows[0]]
            if _missing:
                print(f"  [WARN] 누락 score 컬럼: {_missing}")
            else:
                print(f"  score 컬럼 모두 존재: OK")

            # threshold exceed count/rate 요약
            exceed_p90 = sum(1 for r in all_rows if r.get("threshold_exceed_val_p90", False))
            exceed_p95 = sum(1 for r in all_rows if r.get("threshold_exceed_val_p95", False))
            exceed_p99 = sum(1 for r in all_rows if r.get("threshold_exceed_val_p99", False))
            n = len(all_rows)
            print(f"  threshold_exceed_val_p90: {exceed_p90}/{n} ({exceed_p90/n:.4f})")
            print(f"  threshold_exceed_val_p95: {exceed_p95}/{n} ({exceed_p95/n:.4f})")
            print(f"  threshold_exceed_val_p99: {exceed_p99}/{n} ({exceed_p99/n:.4f})")

            # first_batch_input_shape / first_batch_recon_shape는 run_score_hard_negative에서 출력됨
            print(f"  첫 번째 row 예시:")
            for k, v in list(all_rows[0].items())[:8]:
                print(f"    {k}: {v}")

        print("[DRY-RUN] 파일 저장 없이 종료합니다.")
        sys.exit(0)

    # 12. CSV 저장
    output_root.mkdir(parents=True, exist_ok=True)

    csv_path = output_root / "hard_negative_scores_v1.csv"
    all_csv_columns = SCORE_CSV_BASE_COLUMNS + SCORE_CSV_OPTIONAL_COLUMNS
    df_out = pd.DataFrame(all_rows, columns=all_csv_columns)
    df_out.to_csv(csv_path, index=False)
    print(f"[INFO] CSV 저장 완료: {csv_path} ({len(df_out)} rows)")

    # 13. summary JSON 계산 및 저장
    summary = compute_hard_negative_summary(
        all_rows=all_rows,
        checkpoint_path=checkpoint_path,
        ckpt=ckpt,
        val_thresholds=val_thresholds,
    )

    summary_path = output_root / "hard_negative_score_summary_v1.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Summary JSON 저장 완료: {summary_path}")

    # 14. 결과 출력
    print("\n[INFO] ===== Hard Negative Score 요약 =====")
    print(f"  n_hard_negative_crops: {summary['n_hard_negative_crops']}")
    print(f"  n_patients: {summary['n_patients']}")
    print(f"  score_mean: {summary['score_mean']:.6f}" if summary['score_mean'] is not None else "  score_mean: None")
    print(f"  score_p90: {summary['score_p90']:.6f}" if summary['score_p90'] is not None else "  score_p90: None")
    print(f"  score_p95: {summary['score_p95']:.6f}" if summary['score_p95'] is not None else "  score_p95: None")
    print(f"  score_p99: {summary['score_p99']:.6f}" if summary['score_p99'] is not None else "  score_p99: None")

    print("\n[INFO] ===== Threshold Exceed 요약 (val 기준, 참고용) =====")
    print(f"  val_p90={val_thresholds['val_p90']:.6f}: exceed={summary['exceed_count_val_p90']}, rate={summary['exceed_rate_val_p90']}")
    print(f"  val_p95={val_thresholds['val_p95']:.6f}: exceed={summary['exceed_count_val_p95']}, rate={summary['exceed_rate_val_p95']}")
    print(f"  val_p99={val_thresholds['val_p99']:.6f}: exceed={summary['exceed_count_val_p99']}, rate={summary['exceed_rate_val_p99']}")
    print("[INFO] 주의: threshold 확정 아님. 병변 성능 결론 금지. hard negative는 학습 데이터 아님.")

    print("\n[INFO] score_rd4ad_2p5d_hard_negative.py 완료.")


if __name__ == "__main__":
    main()
