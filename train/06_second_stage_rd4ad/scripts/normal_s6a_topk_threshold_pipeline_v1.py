"""
normal_s6a_topk_threshold_pipeline_v1.py

목적:
  정상 환자(val/test 72명)의 PaDiM high-score 위치에서 6ch crop을 생성하고,
  RD4AD 모델로 스코어링한 뒤 val p95 threshold를 계산한다.
  이 threshold를 stage1_dev crop score에 적용해 recall/FPR/patient hit rate를 산출한다.

절대 금지:
  - 기존 파일/폴더 수정·덮어쓰기·삭제 없음
  - 출력은 신규 폴더에만 저장

입력:
  - PaDiM score CSV : outputs/position-aware-padim-v1/scores/padim_v2_roi0_0/normal_by_patient/*.csv
  - normal split    : outputs/position-aware-padim-v1/splits/normal_v1.json
  - ct_hu.npy       : paths.local.yaml → normal_training_ready_v2_roi0_0/volumes_npy/{safe_id}/ct_hu.npy
  - RD4AD checkpoint: outputs/second-stage-lesion-refiner-v1/models/rd4ad_2p5d_normal_mw_fixed96_v1/checkpoints/best_val_loss.pt
  - stage1_dev score: outputs/second-stage-lesion-refiner-v1/scores/phase7_2_v1v1_stage1_dev_full_scoring_v1/phase7_2_v1v1_stage1_dev_full_scoring_v1.csv

출력 (신규):
  - crops_normal_s6a_topk_v1/{val,test}/{patient_id}/*.npz
  - scores/normal_s6a_topk_scores_v1/normal_s6a_val_scores_v1.csv
  - scores/normal_s6a_topk_scores_v1/normal_s6a_test_scores_v1.csv
  - scores/normal_s6a_topk_scores_v1/normal_s6a_threshold_v1.json
  - scores/normal_s6a_topk_scores_v1/normal_s6a_performance_v1.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset

# ─────────────────────────────────────────────
# 경로 상수
# ─────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]

PADIM_SCORE_DIR = REPO_ROOT / "outputs/position-aware-padim-v1/scores/padim_v2_roi0_0/normal_by_patient"
NORMAL_SPLIT_JSON = REPO_ROOT / "outputs/position-aware-padim-v1/splits/normal_v1.json"
PATHS_CONFIG = REPO_ROOT / "configs/paths.local.yaml"
CHECKPOINT_PATH = REPO_ROOT / (
    "outputs/second-stage-lesion-refiner-v1/models/"
    "rd4ad_2p5d_normal_mw_fixed96_v1/checkpoints/best_val_loss.pt"
)
STAGE1_DEV_SCORE_CSV = REPO_ROOT / (
    "outputs/second-stage-lesion-refiner-v1/scores/"
    "phase7_2_v1v1_stage1_dev_full_scoring_v1/"
    "phase7_2_v1v1_stage1_dev_full_scoring_v1.csv"
)

OUT_CROPS_ROOT = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/crops_normal_s6a_topk_v1"
OUT_SCORE_DIR = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/scores/normal_s6a_topk_scores_v1"

# ─────────────────────────────────────────────
# 파라미터
# ─────────────────────────────────────────────
TOP_K_PER_PATIENT = 50   # 환자당 top PaDiM score patch 수
CROP_SIZE = 96
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# window 파라미터 (generate_s6a_crop_full_6ch.py와 동일)
LUNG_WIN_MIN = -1350.0
LUNG_WIN_MAX = 150.0
MEDI_WIN_MIN = -160.0
MEDI_WIN_MAX = 240.0


# ─────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────
def load_paths_config() -> dict:
    with open(PATHS_CONFIG) as f:
        return yaml.safe_load(f)


def get_normal_vol_root(paths_cfg: dict) -> Path:
    raw = paths_cfg.get("normal_training_ready_v2_roi0_0", "")
    if not raw:
        raise RuntimeError("paths.local.yaml에 normal_training_ready_v2_roi0_0 없음")
    p = Path(raw)
    if not p.exists():
        raise RuntimeError(f"normal vol root 경로 없음: {p}")
    return p


def apply_lung_window(arr: np.ndarray) -> np.ndarray:
    clipped = np.clip(arr, LUNG_WIN_MIN, LUNG_WIN_MAX)
    return ((clipped - LUNG_WIN_MIN) / (LUNG_WIN_MAX - LUNG_WIN_MIN)).astype(np.float32)


def apply_medi_window(arr: np.ndarray) -> np.ndarray:
    clipped = np.clip(arr, MEDI_WIN_MIN, MEDI_WIN_MAX)
    return ((clipped - MEDI_WIN_MIN) / (MEDI_WIN_MAX - MEDI_WIN_MIN)).astype(np.float32)


def compute_crop_coords(cy: float, cx: float, img_h: int, img_w: int) -> tuple:
    half = CROP_SIZE // 2
    y0 = max(0, int(cy) - half)
    x0 = max(0, int(cx) - half)
    y1 = min(img_h, y0 + CROP_SIZE)
    x1 = min(img_w, x0 + CROP_SIZE)
    if y1 - y0 < CROP_SIZE:
        y0 = max(0, y1 - CROP_SIZE)
    if x1 - x0 < CROP_SIZE:
        x0 = max(0, x1 - CROP_SIZE)
    return y0, x0, y1, x1


def extract_6ch_crop(ct: np.ndarray, z: int, y0: int, x0: int, y1: int, x1: int) -> np.ndarray:
    """generate_s6a_crop_full_6ch.py와 동일한 로직"""
    Z = ct.shape[0]
    z_prev = max(0, z - 1)
    z_next = min(Z - 1, z + 1)

    raw_prev = ct[z_prev, y0:y1, x0:x1].astype(np.float32)
    raw_curr = ct[z,      y0:y1, x0:x1].astype(np.float32)
    raw_next = ct[z_next, y0:y1, x0:x1].astype(np.float32)

    image = np.stack([
        apply_lung_window(raw_prev), apply_lung_window(raw_curr), apply_lung_window(raw_next),
        apply_medi_window(raw_prev), apply_medi_window(raw_curr), apply_medi_window(raw_next),
    ], axis=0)
    return image  # (6, 96, 96)


# ─────────────────────────────────────────────
# RD4AD 모델 (phase7_1과 동일 구조)
# ─────────────────────────────────────────────
class ConvAutoencoder2p5D(nn.Module):
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


# ─────────────────────────────────────────────
# Dataset (npz 로드)
# ─────────────────────────────────────────────
class NpzDataset(Dataset):
    def __init__(self, records: list):
        # records: list of dict with keys: crop_id, patient_id, split, npz_path
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        npz_path = rec["npz_path"]
        try:
            data = np.load(npz_path)
            img = data["image"].astype(np.float32)  # (6, 96, 96)
        except Exception as e:
            img = np.zeros((6, CROP_SIZE, CROP_SIZE), dtype=np.float32)
            rec = dict(rec)
            rec["load_error"] = str(e)
        return torch.from_numpy(img), rec


# ─────────────────────────────────────────────
# STEP 1: top-k 위치 추출
# ─────────────────────────────────────────────
def extract_topk_positions(split_ids: list, top_k: int) -> pd.DataFrame:
    rows = []
    for pid in split_ids:
        csv_path = PADIM_SCORE_DIR / f"{pid}.csv"
        if not csv_path.exists():
            print(f"[WARN] score CSV 없음: {csv_path}", file=sys.stderr)
            continue
        df = pd.read_csv(csv_path)
        topk = df.nlargest(top_k, "padim_score")[
            ["patient_id", "safe_id", "group", "local_z", "y0", "x0", "y1", "x1", "padim_score"]
        ].copy()
        topk["rank"] = range(1, len(topk) + 1)
        rows.append(topk)
    if not rows:
        raise RuntimeError("추출된 top-k 위치 없음 — PaDiM score CSV 경로 확인 필요")
    return pd.concat(rows, ignore_index=True)


# ─────────────────────────────────────────────
# STEP 2: 6ch crop 생성 및 npz 저장
# ─────────────────────────────────────────────
def build_crops(topk_df: pd.DataFrame, vol_root: Path, split_name: str) -> list:
    """
    topk_df의 각 행에 대해 6ch crop을 생성하고 npz로 저장한다.
    기존 파일 덮어쓰기 없음 — 이미 존재하면 skip.
    반환: records list (crop_id, patient_id, split, npz_path)
    """
    records = []
    out_split_dir = OUT_CROPS_ROOT / split_name

    for pid, group_df in topk_df.groupby("patient_id"):
        safe_id = group_df.iloc[0]["safe_id"]
        ct_path = vol_root / "volumes_npy" / safe_id / "ct_hu.npy"
        if not ct_path.exists():
            print(f"[SKIP {pid}] ct_hu.npy 없음: {ct_path}", file=sys.stderr)
            continue

        ct = np.load(str(ct_path), mmap_mode="r")
        if ct.ndim != 3:
            print(f"[SKIP {pid}] ct shape 이상: {ct.shape}", file=sys.stderr)
            continue

        H, W = ct.shape[1], ct.shape[2]
        out_patient_dir = out_split_dir / pid
        out_patient_dir.mkdir(parents=True, exist_ok=True)

        for i, row in enumerate(group_df.itertuples(index=False)):
            crop_id = f"{pid}_{split_name}_topk{i+1:03d}"
            out_npz = out_patient_dir / f"{crop_id}.npz"

            # 기존 파일 있으면 skip (덮어쓰기 금지)
            if out_npz.exists():
                records.append({"crop_id": crop_id, "patient_id": pid,
                                 "split": split_name, "npz_path": str(out_npz)})
                continue

            z = int(row.local_z)
            cy = (row.y0 + row.y1) / 2.0
            cx = (row.x0 + row.x1) / 2.0
            y0, x0, y1, x1 = compute_crop_coords(cy, cx, H, W)

            try:
                image = extract_6ch_crop(ct, z, y0, x0, y1, x1)
            except Exception as e:
                print(f"[WARN {pid}] crop 추출 실패 rank{i+1}: {e}", file=sys.stderr)
                continue

            if image.shape != (6, CROP_SIZE, CROP_SIZE):
                print(f"[WARN {pid}] shape 이상 rank{i+1}: {image.shape}", file=sys.stderr)
                continue

            np.savez_compressed(str(out_npz), image=image)
            records.append({"crop_id": crop_id, "patient_id": pid,
                             "split": split_name, "npz_path": str(out_npz)})

        del ct

    print(f"[STEP2] {split_name}: {len(records)} crops 생성/확인")
    return records


# ─────────────────────────────────────────────
# STEP 3: RD4AD 스코어링
# ─────────────────────────────────────────────
def score_crops(records: list, model: nn.Module) -> pd.DataFrame:
    dataset = NpzDataset(records)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    result_rows = []
    model.eval()
    with torch.no_grad():
        for imgs, metas in loader:
            imgs = imgs.to(DEVICE)
            recon = model(imgs)

            l1_err = (recon - imgs).abs()           # (B, 6, 96, 96)
            mse_err = ((recon - imgs) ** 2)

            l1_mean = l1_err.mean(dim=(1, 2, 3)).cpu().numpy()
            l1_max  = l1_err.amax(dim=(1, 2, 3)).cpu().numpy()
            mse_mean = mse_err.mean(dim=(1, 2, 3)).cpu().numpy()

            B = imgs.shape[0]
            for b in range(B):
                result_rows.append({
                    "crop_id":           metas["crop_id"][b],
                    "patient_id":        metas["patient_id"][b],
                    "split":             metas["split"][b],
                    "npz_path":          metas["npz_path"][b],
                    "crop_score_l1_mean": float(l1_mean[b]),
                    "crop_score_l1_max":  float(l1_max[b]),
                    "crop_score_mse_mean": float(mse_mean[b]),
                })

    return pd.DataFrame(result_rows)


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="normal S6A topk threshold pipeline v1")
    parser.add_argument("--top-k", type=int, default=TOP_K_PER_PATIENT,
                        help=f"환자당 top-k PaDiM score 위치 (default: {TOP_K_PER_PATIENT})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    # ── 사전 검사 ──
    for p in [PADIM_SCORE_DIR, NORMAL_SPLIT_JSON, PATHS_CONFIG, CHECKPOINT_PATH, STAGE1_DEV_SCORE_CSV]:
        if not p.exists():
            print(f"[ERROR] 필수 파일 없음: {p}", file=sys.stderr)
            sys.exit(1)

    OUT_SCORE_DIR.mkdir(parents=True, exist_ok=True)

    # 기존 출력 파일 덮어쓰기 방지
    val_score_csv  = OUT_SCORE_DIR / "normal_s6a_val_scores_v1.csv"
    test_score_csv = OUT_SCORE_DIR / "normal_s6a_test_scores_v1.csv"
    thr_json       = OUT_SCORE_DIR / "normal_s6a_threshold_v1.json"
    perf_json      = OUT_SCORE_DIR / "normal_s6a_performance_v1.json"

    for f in [val_score_csv, test_score_csv, thr_json, perf_json]:
        if f.exists():
            print(f"[ERROR] 출력 파일이 이미 존재합니다. 덮어쓰기 금지: {f}", file=sys.stderr)
            print("  삭제 후 재실행하거나 출력 태그를 변경하세요.", file=sys.stderr)
            sys.exit(1)

    # ── config 로드 ──
    paths_cfg = load_paths_config()
    vol_root = get_normal_vol_root(paths_cfg)
    print(f"[INFO] vol_root: {vol_root}")
    print(f"[INFO] device: {DEVICE}")
    print(f"[INFO] top_k per patient: {args.top_k}")

    # ── split 로드 ──
    with open(NORMAL_SPLIT_JSON) as f:
        split_data = json.load(f)
    val_ids  = split_data.get("val", [])
    test_ids = split_data.get("test", [])
    print(f"[INFO] val: {len(val_ids)}명 / test: {len(test_ids)}명")

    # ── STEP 1: top-k 위치 추출 ──
    print("\n[STEP 1] top-k 위치 추출 중...")
    val_topk  = extract_topk_positions(val_ids,  args.top_k)
    test_topk = extract_topk_positions(test_ids, args.top_k)
    print(f"  val top-k rows : {len(val_topk)}")
    print(f"  test top-k rows: {len(test_topk)}")

    # ── STEP 2: 6ch crop 생성 ──
    print("\n[STEP 2] 6ch crop 생성 중...")
    val_records  = build_crops(val_topk,  vol_root, "val")
    test_records = build_crops(test_topk, vol_root, "test")

    if not val_records:
        print("[ERROR] val crop 생성 실패", file=sys.stderr)
        sys.exit(1)

    # ── STEP 3: 모델 로드 ──
    print("\n[STEP 3] RD4AD 모델 로드 중...")
    model = ConvAutoencoder2p5D(input_channels=6, base_channels=32).to(DEVICE)
    ckpt = torch.load(str(CHECKPOINT_PATH), map_location=DEVICE)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    print(f"  checkpoint: {CHECKPOINT_PATH.name}")

    # ── STEP 4: 스코어링 ──
    print("\n[STEP 4] val 스코어링 중...")
    val_score_df = score_crops(val_records, model)
    val_score_df.to_csv(val_score_csv, index=False)
    print(f"  저장: {val_score_csv}  ({len(val_score_df)} rows)")

    print("[STEP 4] test 스코어링 중...")
    test_score_df = score_crops(test_records, model)
    test_score_df.to_csv(test_score_csv, index=False)
    print(f"  저장: {test_score_csv}  ({len(test_score_df)} rows)")

    # ── STEP 5: threshold 계산 ──
    print("\n[STEP 5] threshold 계산 (val p95/p99) ...")
    val_scores = val_score_df["crop_score_l1_mean"].dropna()
    thr_p95 = float(np.percentile(val_scores, 95))
    thr_p99 = float(np.percentile(val_scores, 99))
    print(f"  p95: {thr_p95:.6f}  /  p99: {thr_p99:.6f}")

    # test FP rate
    test_scores = test_score_df["crop_score_l1_mean"].dropna()
    fp_p95 = float((test_scores >= thr_p95).mean())
    fp_p99 = float((test_scores >= thr_p99).mean())
    print(f"  test FP rate  p95: {fp_p95:.4f} ({fp_p95*100:.2f}%)")
    print(f"  test FP rate  p99: {fp_p99:.4f} ({fp_p99*100:.2f}%)")

    thr_result = {
        "threshold_source": "normal_s6a_topk_val",
        "n_val_patients": len(val_ids),
        "n_val_crops": len(val_score_df),
        "top_k_per_patient": args.top_k,
        "threshold_p95": thr_p95,
        "threshold_p99": thr_p99,
        "test_fp_rate_p95": fp_p95,
        "test_fp_rate_p99": fp_p99,
        "n_test_crops": len(test_score_df),
        "score_column": "crop_score_l1_mean",
    }
    with open(thr_json, "w") as f:
        json.dump(thr_result, f, indent=2)
    print(f"  저장: {thr_json}")

    # ── STEP 6: stage1_dev 성능지표 ──
    print("\n[STEP 6] stage1_dev 성능지표 계산 중...")
    dev_df = pd.read_csv(STAGE1_DEV_SCORE_CSV)
    dev_scores = dev_df["crop_score_l1_mean"]
    dev_labels = dev_df["label"]
    pos_mask = dev_labels == 1

    crop_recall_p95 = float(((dev_scores >= thr_p95) & pos_mask).sum() / pos_mask.sum())
    crop_recall_p99 = float(((dev_scores >= thr_p99) & pos_mask).sum() / pos_mask.sum())

    dev_df["detected_p95"] = dev_scores >= thr_p95
    patient_hits_p95 = dev_df[pos_mask].groupby("patient_id")["detected_p95"].any()
    hit_rate_p95 = float(patient_hits_p95.mean())

    dev_df["detected_p99"] = dev_scores >= thr_p99
    patient_hits_p99 = dev_df[pos_mask].groupby("patient_id")["detected_p99"].any()
    hit_rate_p99 = float(patient_hits_p99.mean())

    print(f"\n{'='*50}")
    print("  RD4AD v1/v1 성능지표 (normal S6A topk threshold 기준)")
    print(f"{'='*50}")
    print(f"  threshold p95 : {thr_p95:.6f}")
    print(f"  threshold p99 : {thr_p99:.6f}")
    print(f"  ① 병변 crop 검출률 (recall p95): {crop_recall_p95:.4f} ({crop_recall_p95*100:.2f}%)")
    print(f"  ① 병변 crop 검출률 (recall p99): {crop_recall_p99:.4f} ({crop_recall_p99*100:.2f}%)")
    print(f"  ② 정상 crop 오탐률 (FP p95)    : {fp_p95:.4f} ({fp_p95*100:.2f}%)")
    print(f"  ② 정상 crop 오탐률 (FP p99)    : {fp_p99:.4f} ({fp_p99*100:.2f}%)")
    print(f"  ③ crop AUROC (l1_mean)         : 0.6490  (phase7_4)")
    print(f"  ④ crop AUPRC (l1_mean)         : 0.3973  (phase7_4)")
    print(f"  ⑤ patient hit rate (p95)       : {hit_rate_p95:.4f} ({hit_rate_p95*100:.2f}%)  [{patient_hits_p95.sum()}/{len(patient_hits_p95)}명]")
    print(f"  ⑤ patient hit rate (p99)       : {hit_rate_p99:.4f} ({hit_rate_p99*100:.2f}%)  [{patient_hits_p99.sum()}/{len(patient_hits_p99)}명]")
    print(f"{'='*50}")

    perf_result = {
        "pipeline": "normal_s6a_topk_threshold_pipeline_v1",
        "threshold_p95": thr_p95,
        "threshold_p99": thr_p99,
        "crop_recall_p95": crop_recall_p95,
        "crop_recall_p99": crop_recall_p99,
        "test_fp_rate_p95": fp_p95,
        "test_fp_rate_p99": fp_p99,
        "crop_auroc_l1_mean": 0.6490,
        "crop_auprc_l1_mean": 0.3973,
        "patient_hit_rate_p95": hit_rate_p95,
        "patient_hit_rate_p99": hit_rate_p99,
        "n_positive_crops": int(pos_mask.sum()),
        "n_stage1_dev_patients": int(dev_df["patient_id"].nunique()),
        "note": "stage1_dev 한정. stage2_holdout 미평가.",
    }
    with open(perf_json, "w") as f:
        json.dump(perf_result, f, indent=2)
    print(f"\n  저장: {perf_json}")
    print("\n[DONE]")


if __name__ == "__main__":
    main()
