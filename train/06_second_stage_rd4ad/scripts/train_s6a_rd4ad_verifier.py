"""
train_s6a_rd4ad_verifier.py

학습 실행 금지 — 실제 학습은 사용자 승인 후 --preflight-only 플래그 제거

S6-A full crop 기반 2.5D RD4AD Verifier 학습 skeleton.
현재 단계: preflight check 전용 (모델 forward / optimizer / checkpoint / epoch loop 실행 금지)

절대 금지:
- 모델 forward 실행 금지
- optimizer 생성 금지
- checkpoint 저장 금지
- epoch loop 실행 금지
- stage2_holdout 사용 금지
- crop/npz 수정/삭제 금지
- pip/conda install 금지
"""

from __future__ import annotations

import argparse
import sys
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

# ---------------------------------------------------------------------------
# 프로젝트 루트 설정 (scripts/ 기준 상위)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.second_stage_verifier.data.s6a_dataset import S6ADataset  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FORBIDDEN_PATH_PATTERNS = ["stage2_holdout"]
DEFAULT_CONFIG = str(
    _PROJECT_ROOT / "configs/second_stage_verifier/s6a_rd4ad_verifier_config.yaml"
)


# ---------------------------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="S6-A RD4AD Verifier 학습 skeleton (preflight 전용)"
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="config YAML 경로")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        default=True,
        help="preflight check만 수행하고 학습하지 않음 (기본값: True)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="DataLoader 1 batch 확인 후 종료",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="output_dir 기존 존재 시에도 강제 진행 (학습 승인 후만 사용)",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "cuda_if_available"],
        default="cuda_if_available",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config Loader
# ---------------------------------------------------------------------------
def load_config(config_path: str) -> dict:
    try:
        import yaml
    except ImportError:
        print("[ERROR] PyYAML이 설치되어 있지 않습니다. pip install 금지 — 환경을 확인하세요.")
        sys.exit(1)

    p = Path(config_path)
    if not p.exists():
        print(f"[ERROR] config 파일 없음: {p}")
        sys.exit(1)

    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print(f"[CONFIG] 로드 완료: {p}")
    return cfg


# ---------------------------------------------------------------------------
# Guard Functions
# ---------------------------------------------------------------------------
def guard_stage2_holdout(split_df: pd.DataFrame) -> None:
    """stage2_holdout 환자가 포함되어 있으면 즉시 중단."""
    if "stage_split" not in split_df.columns:
        return
    holdout = split_df[split_df["stage_split"] == "stage2_holdout"]
    if len(holdout) > 0:
        print(f"[GUARD FAIL] stage2_holdout 환자 {len(holdout)}명 감지. 즉시 중단.")
        sys.exit(1)
    print("[GUARD PASS] stage2_holdout 0명")


def guard_train_val_overlap(split_df: pd.DataFrame) -> None:
    """train/val patient overlap이 있으면 즉시 중단."""
    if "train_val" not in split_df.columns:
        return
    train_ids = set(split_df[split_df["train_val"] == "train"]["patient_id"])
    val_ids = set(split_df[split_df["train_val"] == "val"]["patient_id"])
    overlap = train_ids & val_ids
    if overlap:
        print(f"[GUARD FAIL] train/val overlap {len(overlap)}명: {sorted(overlap)}")
        sys.exit(1)
    print(f"[GUARD PASS] train/val overlap 0명 (train={len(train_ids)}, val={len(val_ids)})")


def guard_output_dir(output_dir: str, force: bool) -> None:
    """output_dir가 이미 존재하면 --force 없이 중단."""
    p = Path(output_dir)
    if p.exists() and not force:
        print(f"[GUARD FAIL] output_dir 이미 존재: {p}")
        print("  기존 결과 보호를 위해 중단합니다. --force 플래그를 사용하려면 사용자 승인이 필요합니다.")
        sys.exit(1)
    if p.exists() and force:
        print(f"[GUARD WARN] output_dir 이미 존재하나 --force로 계속 진행: {p}")
    else:
        print(f"[GUARD PASS] output_dir 충돌 없음: {p}")


def guard_nan_inf_batch(batch: dict, batch_idx: int) -> None:
    """batch에 NaN/Inf가 있으면 즉시 중단."""
    image = batch["image"]
    if not torch.isfinite(image).all():
        nan_count = int(torch.isnan(image).sum().item())
        inf_count = int(torch.isinf(image).sum().item())
        raise RuntimeError(
            f"[GUARD FAIL] batch {batch_idx}에 NaN/Inf 감지: NaN={nan_count}, Inf={inf_count}"
        )


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
def apply_normalization(image: torch.Tensor, normalization: str,
                        clip_min: float = -1000.0, clip_max: float = 400.0) -> torch.Tensor:
    """config normalization 전략에 따라 image tensor를 변환한다."""
    if normalization == "raw":
        return image
    elif normalization == "clip_scale":
        image = torch.clamp(image, min=clip_min, max=clip_max)
        image = (image - clip_min) / (clip_max - clip_min)
        return image
    elif normalization == "mean_std":
        # TODO: train set mean/std 계산 후 적용 (현재 추정값 사용)
        mean = -350.0
        std = 420.0
        return (image - mean) / (std + 1e-8)
    else:
        raise ValueError(f"알 수 없는 normalization 전략: {normalization!r}")


# ---------------------------------------------------------------------------
# Preflight Check
# ---------------------------------------------------------------------------
def run_preflight(cfg: dict, args: argparse.Namespace) -> bool:
    """
    preflight check를 수행하고 PASS/FAIL 결과를 반환한다.
    모든 항목을 점검하되 학습은 실행하지 않는다.
    """
    print("\n" + "=" * 60)
    print("  S6-A RD4AD Verifier Preflight Check")
    print("=" * 60)

    results: list[tuple[str, str, str]] = []  # (항목, 상태, 상세)

    # 1. config 로드 확인 (이미 완료)
    results.append(("config 로드", "PASS", str(args.config)))

    # 2. dataset index 파일 존재 확인
    idx_path = Path(cfg["paths"]["dataset_index_csv"])
    if idx_path.exists():
        results.append(("dataset_index_csv 존재", "PASS", str(idx_path)))
    else:
        results.append(("dataset_index_csv 존재", "FAIL", f"없음: {idx_path}"))

    # 3. train/val split 파일 존재 확인
    split_path = Path(cfg["paths"]["train_val_split_csv"])
    if split_path.exists():
        results.append(("train_val_split_csv 존재", "PASS", str(split_path)))
    else:
        results.append(("train_val_split_csv 존재", "FAIL", f"없음: {split_path}"))

    # 이후 guard는 파일이 존재해야 진행 가능
    if not idx_path.exists() or not split_path.exists():
        _print_preflight_results(results)
        return False

    # CSV 로드
    index_df = pd.read_csv(idx_path)
    split_df = pd.read_csv(split_path)

    # 4. stage2_holdout 환자 0명 확인
    if "stage_split" in split_df.columns:
        holdout_count = int((split_df["stage_split"] == "stage2_holdout").sum())
        if holdout_count == 0:
            results.append(("stage2_holdout 0명", "PASS", f"holdout={holdout_count}"))
        else:
            results.append(("stage2_holdout 0명", "FAIL", f"holdout={holdout_count}명 감지"))
    else:
        results.append(("stage2_holdout 0명", "WARN", "stage_split 컬럼 없음"))

    # 5. train/val overlap 0명 확인
    if "train_val" in split_df.columns:
        train_ids = set(split_df[split_df["train_val"] == "train"]["patient_id"])
        val_ids = set(split_df[split_df["train_val"] == "val"]["patient_id"])
        overlap = train_ids & val_ids
        if len(overlap) == 0:
            results.append(("train/val overlap 0명", "PASS", f"train={len(train_ids)}, val={len(val_ids)}"))
        else:
            results.append(("train/val overlap 0명", "FAIL", f"overlap={len(overlap)}명: {sorted(overlap)[:5]}"))
    else:
        results.append(("train/val overlap 0명", "WARN", "train_val 컬럼 없음"))

    # 6. train/val crop 수 확인
    if "train_val" in split_df.columns and "label" in index_df.columns:
        # split_df의 patient_id 기준으로 index_df 필터
        train_pids = set(split_df[split_df["train_val"] == "train"]["patient_id"])
        val_pids = set(split_df[split_df["train_val"] == "val"]["patient_id"])
        train_crops = index_df[index_df["patient_id"].isin(train_pids)]
        val_crops = index_df[index_df["patient_id"].isin(val_pids)]
        results.append((
            "train/val crop 수",
            "PASS",
            f"train={len(train_crops)}, val={len(val_crops)}"
        ))

        # 7. class imbalance 비율 출력
        if "label" in train_crops.columns:
            t_pos = int((train_crops["label"] == 1).sum())
            t_neg = int((train_crops["label"] == 0).sum())
            ratio = round(t_neg / max(t_pos, 1), 2)
            results.append((
                "class imbalance (train)",
                "PASS",
                f"positive={t_pos}, hard_negative={t_neg}, ratio=1:{ratio}"
            ))
        else:
            results.append(("class imbalance (train)", "WARN", "label 컬럼 없음"))
    else:
        results.append(("train/val crop 수", "WARN", "컬럼 부족"))
        results.append(("class imbalance (train)", "WARN", "컬럼 부족"))

    # 8. LUNG1-140 split 확인
    if "train_val" in split_df.columns:
        lung140 = split_df[split_df["patient_id"] == "LUNG1-140"]
        if len(lung140) > 0:
            lung140_split = lung140.iloc[0]["train_val"]
            status = "PASS" if lung140_split == "train" else "WARN"
            results.append(("LUNG1-140 split", status, f"split={lung140_split}"))
        else:
            results.append(("LUNG1-140 split", "WARN", "LUNG1-140 없음 (정상일 수 있음)"))

    # 9. output dir 충돌 없음 확인
    out_model_dir = Path(cfg["paths"]["output_model_dir"])
    if out_model_dir.exists() and not args.force:
        results.append(("output_dir 충돌", "FAIL", f"이미 존재: {out_model_dir}"))
    elif out_model_dir.exists() and args.force:
        results.append(("output_dir 충돌", "WARN", f"존재하나 --force 설정됨: {out_model_dir}"))
    else:
        results.append(("output_dir 충돌 없음", "PASS", str(out_model_dir)))

    # 10. GPU 사용 여부 출력
    device_cfg = args.device
    if device_cfg == "cuda_if_available":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_cfg
    gpu_info = f"device={device}"
    if device == "cuda":
        gpu_info += f", GPU={torch.cuda.get_device_name(0)}"
    results.append(("GPU 사용 여부", "PASS", gpu_info))

    # 11. normalization 전략 출력
    norm_strategy = cfg.get("data", {}).get("normalization", "unknown")
    clip_min = cfg.get("data", {}).get("clip_min", -1000)
    clip_max = cfg.get("data", {}).get("clip_max", 400)
    results.append((
        "normalization 전략",
        "PASS",
        f"{norm_strategy} (clip_min={clip_min}, clip_max={clip_max})"
    ))

    _print_preflight_results(results)

    # 12. 전체 PASS/FAIL 판정
    fail_count = sum(1 for _, s, _ in results if s == "FAIL")
    warn_count = sum(1 for _, s, _ in results if s == "WARN")
    if fail_count == 0:
        print(f"\n[PREFLIGHT] 전체 결과: PASS (FAIL=0, WARN={warn_count})")
        return True
    else:
        print(f"\n[PREFLIGHT] 전체 결과: FAIL (FAIL={fail_count}, WARN={warn_count})")
        return False


def _print_preflight_results(results: list[tuple[str, str, str]]) -> None:
    print()
    for item, status, detail in results:
        prefix = "✓" if status == "PASS" else ("!" if status == "WARN" else "✗")
        print(f"  [{status:4s}] {prefix} {item}: {detail}")


# ---------------------------------------------------------------------------
# DataLoader 구성
# ---------------------------------------------------------------------------
def build_dataloaders(cfg: dict, index_df: pd.DataFrame, split_df: pd.DataFrame):
    """
    train/val DataLoader를 구성한다.
    preflight에서는 shape/label 확인만 수행하고 학습 루프는 실행하지 않는다.
    """
    # split 정보를 index_df에 합산
    if "train_val" in split_df.columns:
        split_map = split_df.set_index("patient_id")["train_val"].to_dict()
        index_df = index_df.copy()
        index_df["train_val"] = index_df["patient_id"].map(split_map)
    else:
        raise KeyError("split_df에 train_val 컬럼이 없습니다.")

    train_dataset = S6ADataset(index_df, split="train")
    val_dataset = S6ADataset(index_df, split="val")

    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["training"]["num_workers"]
    pin_memory = cfg["training"]["pin_memory"]

    # class-balanced WeightedRandomSampler 구성
    sampler = None
    if cfg["sampling"].get("use_weighted_sampler", False):
        train_df = index_df[index_df["train_val"] == "train"].reset_index(drop=True)
        labels = train_df["label"].values.astype(int)
        class_counts = np.bincount(labels)  # [n_neg, n_pos]
        class_weights = 1.0 / (class_counts + 1e-8)
        sample_weights = class_weights[labels]
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).float(),
            num_samples=len(sample_weights),
            replacement=True,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print(f"[DATALOADER] train: {len(train_dataset)} samples, val: {len(val_dataset)} samples")
    print(f"[DATALOADER] batch_size={batch_size}, num_workers={num_workers}")
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# pos_weight 계산
# ---------------------------------------------------------------------------
def compute_pos_weight(cfg: dict, index_df: pd.DataFrame, split_df: pd.DataFrame) -> float:
    """
    config의 pos_weight 값을 사용하거나, 자동으로 train set에서 계산한다.
    """
    cfg_pos_weight = cfg.get("loss", {}).get("pos_weight", None)
    if cfg_pos_weight is not None:
        print(f"[POS_WEIGHT] config 값 사용: {cfg_pos_weight}")
        return float(cfg_pos_weight)

    # 자동 계산
    if "train_val" not in split_df.columns or "label" not in index_df.columns:
        print("[POS_WEIGHT] 자동 계산 불가, 기본값 2.0 사용")
        return 2.0

    split_map = split_df.set_index("patient_id")["train_val"].to_dict()
    train_df = index_df[index_df["patient_id"].map(split_map) == "train"]
    n_pos = int((train_df["label"] == 1).sum())
    n_neg = int((train_df["label"] == 0).sum())
    pos_weight = n_neg / max(n_pos, 1)
    print(f"[POS_WEIGHT] 자동 계산: n_neg={n_neg} / n_pos={n_pos} = {pos_weight:.4f}")
    return float(pos_weight)


# ---------------------------------------------------------------------------
# Model skeleton (실제 생성 금지)
# ---------------------------------------------------------------------------
def build_model_skeleton(cfg: dict):
    """
    모델 구조 skeleton — 실제 forward 실행 금지.
    학습 승인 후 이 함수를 완성한다.
    """
    # TODO: 사용자 승인 후 아래 구현 완성
    # backbone = cfg["model"]["backbone"]
    # in_channels = cfg["model"]["in_channels"]
    # num_classes = cfg["model"]["num_classes"]
    # architecture = cfg["model"]["architecture"]
    #
    # if architecture == "encoder_classifier":
    #     # ResNet18 encoder + binary classification head
    #     import torchvision.models as models
    #     model = models.resnet18(pretrained=cfg["model"]["pretrained"])
    #     model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    #     model.fc = nn.Linear(model.fc.in_features, num_classes)
    #     return model
    raise NotImplementedError(
        "[MODEL] 모델 생성은 사용자 승인 후 구현합니다. 현재 preflight-only 모드입니다."
    )


# ---------------------------------------------------------------------------
# Training Loop skeleton (실행 금지)
# ---------------------------------------------------------------------------
def train_one_epoch_skeleton():
    """
    epoch loop skeleton — 실행 금지.
    학습 승인 후 이 함수를 구현한다.
    """
    # TODO: 사용자 승인 후 구현
    # for batch_idx, batch in enumerate(train_loader):
    #     image = batch["image"].to(device)
    #     label = batch["label"].float().to(device)
    #     guard_nan_inf_batch(batch, batch_idx)
    #     apply_normalization(image, normalization, clip_min, clip_max)
    #
    #     # TODO: loss 계산
    #     # logits = model(image)
    #     # loss = criterion(logits.squeeze(1), label)
    #     # loss.backward()
    #     # optimizer.step()
    raise NotImplementedError("train_one_epoch는 사용자 승인 후 구현합니다.")


def evaluate_skeleton():
    """
    validation 평가 skeleton — 실행 금지.
    학습 승인 후 이 함수를 구현한다.
    """
    # TODO: 사용자 승인 후 구현
    # all_logits, all_labels = [], []
    # with torch.no_grad():
    #     for batch in val_loader:
    #         ...
    # TODO: AUROC/AUPRC 계산
    # from sklearn.metrics import roc_auc_score, average_precision_score
    # auroc = roc_auc_score(all_labels, all_scores)
    # auprc = average_precision_score(all_labels, all_scores)
    raise NotImplementedError("evaluate는 사용자 승인 후 구현합니다.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    print(f"[START] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[MODE] preflight_only={args.preflight_only}, dry_run={args.dry_run}")

    # config 로드
    cfg = load_config(args.config)

    # 입력 파일 로드
    idx_path = Path(cfg["paths"]["dataset_index_csv"])
    split_path = Path(cfg["paths"]["train_val_split_csv"])

    # preflight check
    preflight_ok = run_preflight(cfg, args)

    if not preflight_ok:
        print("\n[ABORT] preflight FAIL. 학습을 중단합니다.")
        sys.exit(1)

    if args.preflight_only:
        print("\n[STOP] --preflight-only 모드. 학습을 실행하지 않습니다.")
        print("  실제 학습을 원하면 사용자 승인 후 --preflight-only 플래그를 제거하세요.")
        return

    # --- 아래는 사용자 승인 후 실행 ---

    index_df = pd.read_csv(idx_path)
    split_df = pd.read_csv(split_path)

    # guard
    guard_stage2_holdout(split_df)
    guard_train_val_overlap(split_df)
    guard_output_dir(cfg["paths"]["output_model_dir"], args.force)

    # pos_weight 계산
    pos_weight_val = compute_pos_weight(cfg, index_df, split_df)

    # DataLoader 구성
    train_loader, val_loader = build_dataloaders(cfg, index_df, split_df)

    # dry-run: 1 batch shape 확인 후 종료
    if args.dry_run:
        batch = next(iter(train_loader))
        print(f"[DRY-RUN] train batch: image={batch['image'].shape}, label={batch['label'].shape}")
        guard_nan_inf_batch(batch, 0)
        print("[DRY-RUN] NaN/Inf check PASS")
        print("[DRY-RUN] 완료. 학습하지 않고 종료합니다.")
        return

    # --- 이하 학습 코드: 사용자 승인 후 구현 ---
    # TODO: build_model_skeleton → 실제 모델로 교체
    # TODO: optimizer 생성 (사용자 승인 후)
    # TODO: scheduler 생성 (사용자 승인 후)
    # TODO: criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_val))
    # TODO: epoch loop (사용자 승인 후)
    # TODO: checkpoint 저장 (사용자 승인 후)
    print("\n[ABORT] 학습 루프는 사용자 승인 후 구현합니다. 현재 단계에서는 실행 금지입니다.")
    sys.exit(0)


if __name__ == "__main__":
    main()
