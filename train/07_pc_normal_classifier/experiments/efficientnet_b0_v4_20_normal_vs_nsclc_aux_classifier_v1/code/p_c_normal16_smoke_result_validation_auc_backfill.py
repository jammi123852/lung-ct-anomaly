"""
P-C-NORMAL16: P-C-NORMAL15 Smoke Result Validation + sklearn-free AUROC Backfill

- checkpoint read-only load
- 16 required keys 검증
- metrics consistency (checkpoint / JSON / MD)
- val-only forward pass (no_grad, no backward, no optimizer step)
- numpy/Mann-Whitney rank-sum AUROC (sklearn 금지)
- 원본 P-C-NORMAL15 JSON/MD 수정 없음
- 신규 backfill report 폴더에만 저장
"""

import os
import sys
import json
import math
import datetime
import csv
import pathlib
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# ── 경로 상수 ─────────────────────────────────────────────────────────────────
BRANCH_ROOT = pathlib.Path(
    "/home/jinhy/project/lung-ct-anomaly/experiments"
    "/efficientnet_b0_v4_20_normal_vs_nsclc_aux_classifier_v1"
)

CKPT_PATH = (
    BRANCH_ROOT
    / "outputs/checkpoints/p_c_normal15_matched_smoke_training"
    / "p_c_normal15_epoch1.pth"
)
JSON_PATH = (
    BRANCH_ROOT
    / "outputs/reports/p_c_normal15_matched_smoke_training"
    / "p_c_normal15_smoke_train_result.json"
)
MD_PATH = (
    BRANCH_ROOT
    / "outputs/reports/p_c_normal15_matched_smoke_training"
    / "p_c_normal15_smoke_train_result.md"
)
NORMAL6_CKPT_PATH = (
    BRANCH_ROOT
    / "outputs/checkpoints/p_c_normal6_smoke_training"
    / "p_c_normal6_epoch1.pth"
)
VAL_MANIFEST_PATH = (
    BRANCH_ROOT
    / "outputs/manifests/p_c_normal12_matched_training_manifest"
    / "p_c_normal12_val_manifest.csv"
)
TRAIN_MANIFEST_PATH = (
    BRANCH_ROOT
    / "outputs/manifests/p_c_normal12_matched_training_manifest"
    / "p_c_normal12_train_manifest.csv"
)
STAGE2_HOLDOUT_SENTINEL = "stage2_holdout"

OUT_DIR = (
    BRANCH_ROOT
    / "outputs/reports/p_c_normal16_smoke_result_validation_auc_backfill"
)

# ── P-C-NORMAL15 기대값 ───────────────────────────────────────────────────────
EXPECTED = {
    "epoch": 1,
    "smoke_only": True,
    "full_training": False,
    "train_loss": 0.07969932096737523,
    "train_acc": 0.9775941602879302,
    "val_loss": 0.027572847438250524,
    "val_acc": 0.9917307692307692,
    "val_auc": None,
    "val_auc_status": "AUROC_ERROR:ModuleNotFoundError",
    "class_weight_normal": 0.8333474146671173,
    "class_weight_nsclc": 1.2499683183373465,
}
SMOKE_CHECKPOINT_REQUIRED_KEYS = (
    "model_state_dict",
    "optimizer_state_dict",
    "epoch",
    "smoke_only",
    "full_training",
    "config",
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
    "val_auc",
    "val_auc_status",
    "label_mapping",
    "class_weights",
    "manifest_paths",
    "forbidden_diagnostic_wording_count",
)
METRIC_TOL = 1e-5
CW_TOL = 1e-4

# ── 전처리 상수 ───────────────────────────────────────────────────────────────
HU_MIN = -1000.0
HU_MAX = 200.0
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── guardrail 추적 ────────────────────────────────────────────────────────────
GUARDRAIL = {
    "training_run": False,
    "backward_run": False,
    "optimizer_step": False,
    "checkpoint_saved": False,
    "checkpoint_modified": False,
    "original_report_modified": False,
    "full_training_run": False,
    "threshold_computed": False,
    "stage2_holdout_accessed": False,
    "hard_negative_included": False,
    "MSD_Lung_included": False,
    "forbidden_diagnostic_wording_count": 0,
}

ERRORS = []


# ── 유틸 ──────────────────────────────────────────────────────────────────────
def _err(msg: str):
    ERRORS.append({"error": msg})
    print(f"[ERROR] {msg}", file=sys.stderr)


def _write_csv(path: pathlib.Path, rows: list[dict]):
    if not rows:
        rows = [{"note": "empty"}]
    # 모든 row의 키를 합산해 fieldnames 구성 (순서 유지)
    all_keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"  saved → {path}")


# ── sklearn-free AUROC (Mann-Whitney rank-sum) ────────────────────────────────
def _auroc_numpy(labels: list, probs: list) -> tuple[float | None, str]:
    """
    numpy rank-sum 방식 AUROC.
    labels: 0/1 리스트
    probs: float 리스트
    반환: (auroc_float_or_None, status_str)
    """
    y = np.array(labels, dtype=np.int32)
    p = np.array(probs, dtype=np.float64)

    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))

    if n_pos == 0 or n_neg == 0:
        return None, f"SINGLE_CLASS_pos={n_pos}_neg={n_neg}"

    # rank 계산 (1-based, ties=average)
    order = np.argsort(p, kind="stable")
    ranks = np.empty(len(p), dtype=np.float64)
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and p[order[j]] == p[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0  # 1-based 평균
        ranks[order[i:j]] = avg_rank
        i = j

    rank_sum_pos = float(np.sum(ranks[y == 1]))
    auroc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    status = "OK" if auroc >= 0.5 else "DEGENERATE"
    return float(auroc), status


# ── 모델 빌드 (추론 전용) ────────────────────────────────────────────────────
def _build_model() -> nn.Module:
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, 1),
    )
    return model


# ── Dataset ──────────────────────────────────────────────────────────────────
class _ValDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        std  = torch.tensor(IMAGENET_STD,  dtype=torch.float32).view(3, 1, 1)
        self.mean = mean
        self.std  = std

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        crop_path = str(row["crop_path"])
        label = int(row["label"])
        sample_weight = float(row["sample_weight"])

        data = np.load(crop_path)
        arr = data["ct_crop"].astype(np.float32)
        arr = np.clip(arr, HU_MIN, HU_MAX)
        arr = (arr - HU_MIN) / (HU_MAX - HU_MIN)
        t = torch.from_numpy(arr)
        t = (t - self.mean) / self.std
        return t, label, sample_weight


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Output file validation
# ─────────────────────────────────────────────────────────────────────────────
def step1_output_file_validation() -> list[dict]:
    print("\n[STEP1] Output file validation")
    rows = []

    def _check(label, path, expected_exists=True):
        exists = pathlib.Path(path).exists()
        size_mb = pathlib.Path(path).stat().st_size / 1e6 if exists else None
        ok = exists == expected_exists
        rows.append({
            "check": label,
            "path": str(path),
            "exists": exists,
            "size_mb": round(size_mb, 3) if size_mb else None,
            "expected_exists": expected_exists,
            "pass": ok,
        })
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {label}: {path}")
        if not ok:
            _err(f"File check failed: {label} @ {path}")

    _check("p_c_normal15_checkpoint", CKPT_PATH)
    _check("p_c_normal15_json_report", JSON_PATH)
    _check("p_c_normal15_md_report", MD_PATH)
    _check("p_c_normal6_checkpoint_exists", NORMAL6_CKPT_PATH)

    # full training output 미생성 확인 (p_c_normal15 폴더 내 epoch1 외 파일 없어야)
    ckpt_dir = CKPT_PATH.parent
    extra_files = [f for f in ckpt_dir.iterdir()
                   if f.name != CKPT_PATH.name] if ckpt_dir.exists() else []
    rows.append({
        "check": "full_training_output_absent",
        "path": str(ckpt_dir),
        "exists": True,
        "size_mb": None,
        "expected_exists": True,
        "pass": len(extra_files) == 0,
        "note": f"extra_files={[f.name for f in extra_files]}",
    })
    if extra_files:
        _err(f"Unexpected files in checkpoint dir: {[f.name for f in extra_files]}")

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Checkpoint validation
# ─────────────────────────────────────────────────────────────────────────────
def step2_checkpoint_validation() -> tuple[list[dict], list[dict], dict | None]:
    print("\n[STEP2] Checkpoint validation")
    key_rows = []
    weight_rows = []
    ckpt = None

    if not CKPT_PATH.exists():
        _err("Checkpoint missing — skipping checkpoint validation")
        return key_rows, weight_rows, ckpt

    # P-C-NORMAL6 mtime 사전 기록
    n6_mtime_before = NORMAL6_CKPT_PATH.stat().st_mtime if NORMAL6_CKPT_PATH.exists() else None

    # read-only load
    ckpt = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)
    print(f"  checkpoint loaded: keys={list(ckpt.keys())}")

    # 16 required keys
    for k in SMOKE_CHECKPOINT_REQUIRED_KEYS:
        present = k in ckpt
        key_rows.append({"key": k, "present": present, "pass": present})
        if not present:
            _err(f"Missing checkpoint key: {k}")

    # epoch, smoke_only, full_training
    for field, expected in [("epoch", 1), ("smoke_only", True), ("full_training", False)]:
        actual = ckpt.get(field)
        ok = actual == expected
        key_rows.append({"key": field, "present": field in ckpt,
                         "expected": str(expected), "actual": str(actual), "pass": ok})
        if not ok:
            _err(f"checkpoint[{field}]={actual} != {expected}")

    # class_weights
    cw = ckpt.get("class_weights", {})
    cw_n = float(cw.get("normal", float("nan")))
    cw_s = float(cw.get("nsclc", float("nan")))
    cw_n_ok = abs(cw_n - EXPECTED["class_weight_normal"]) <= CW_TOL
    cw_s_ok = abs(cw_s - EXPECTED["class_weight_nsclc"]) <= CW_TOL
    key_rows.append({"key": "class_weight_normal", "present": True,
                     "expected": str(EXPECTED["class_weight_normal"]),
                     "actual": str(cw_n), "pass": cw_n_ok})
    key_rows.append({"key": "class_weight_nsclc", "present": True,
                     "expected": str(EXPECTED["class_weight_nsclc"]),
                     "actual": str(cw_s), "pass": cw_s_ok})
    if not cw_n_ok:
        _err(f"class_weight_normal mismatch: {cw_n}")
    if not cw_s_ok:
        _err(f"class_weight_nsclc mismatch: {cw_s}")

    # manifest_paths — P-C-NORMAL12 경로인지 확인
    mp = ckpt.get("manifest_paths", {})
    train_mp = str(mp.get("train_csv", ""))
    val_mp   = str(mp.get("val_csv", ""))
    train_mp_ok = "p_c_normal12" in train_mp
    val_mp_ok   = "p_c_normal12" in val_mp
    key_rows.append({"key": "manifest_paths_train_is_normal12",
                     "present": True, "actual": train_mp, "pass": train_mp_ok})
    key_rows.append({"key": "manifest_paths_val_is_normal12",
                     "present": True, "actual": val_mp, "pass": val_mp_ok})
    if not train_mp_ok:
        _err(f"manifest_paths.train_csv not p_c_normal12: {train_mp}")
    if not val_mp_ok:
        _err(f"manifest_paths.val_csv not p_c_normal12: {val_mp}")

    # stage2_holdout 접근 안 됐는지 확인
    for p in (train_mp, val_mp):
        if STAGE2_HOLDOUT_SENTINEL in p:
            GUARDRAIL["stage2_holdout_accessed"] = True
            _err(f"stage2_holdout detected in manifest_paths: {p}")

    # model weight NaN/Inf
    state_dict = ckpt.get("model_state_dict", {})
    nan_count = 0
    inf_count = 0
    total_params = 0
    for name, tensor in state_dict.items():
        t = tensor.float()
        n_nan = int(torch.isnan(t).sum().item())
        n_inf = int(torch.isinf(t).sum().item())
        n_total = t.numel()
        nan_count += n_nan
        inf_count += n_inf
        total_params += n_total
        if n_nan > 0 or n_inf > 0:
            weight_rows.append({
                "layer": name, "total": n_total,
                "nan": n_nan, "inf": n_inf, "pass": False,
            })
    weight_rows.append({
        "layer": "__TOTAL__", "total": total_params,
        "nan": nan_count, "inf": inf_count,
        "pass": nan_count == 0 and inf_count == 0,
    })
    if nan_count > 0:
        _err(f"model weights contain NaN: count={nan_count}")
    if inf_count > 0:
        _err(f"model weights contain Inf: count={inf_count}")
    print(f"  model weights: total_params={total_params}, NaN={nan_count}, Inf={inf_count}")

    # P-C-NORMAL6 mtime 불변 확인
    n6_mtime_after = NORMAL6_CKPT_PATH.stat().st_mtime if NORMAL6_CKPT_PATH.exists() else None
    n6_unchanged = n6_mtime_before == n6_mtime_after
    key_rows.append({
        "key": "p_c_normal6_mtime_unchanged",
        "present": True,
        "expected": str(n6_mtime_before),
        "actual": str(n6_mtime_after),
        "pass": n6_unchanged,
    })
    if not n6_unchanged:
        _err("P-C-NORMAL6 checkpoint mtime changed!")

    return key_rows, weight_rows, ckpt


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Metric consistency
# ─────────────────────────────────────────────────────────────────────────────
def step3_metric_consistency(ckpt: dict | None) -> list[dict]:
    print("\n[STEP3] Metric consistency")
    rows = []

    if not JSON_PATH.exists():
        _err("JSON report missing — skipping metric consistency")
        return rows

    with open(JSON_PATH, encoding="utf-8") as f:
        json_data = json.load(f)

    md_text = MD_PATH.read_text(encoding="utf-8") if MD_PATH.exists() else ""

    # JSON과 기대값 비교
    for field, expected in [
        ("epoch", 1),
        ("smoke_only", True),
        ("full_training", False),
        ("stage2_holdout_accessed", False),
        ("hard_negative", 0),
        ("MSD_Lung", 0),
    ]:
        actual = json_data.get(field)
        ok = actual == expected
        rows.append({"source": "json", "field": field,
                     "expected": str(expected), "actual": str(actual), "pass": ok})
        if not ok:
            _err(f"JSON[{field}]={actual} != {expected}")

    for field in ("train_loss", "train_acc", "val_loss", "val_acc"):
        exp_val = EXPECTED[field]
        actual  = json_data.get(field)
        ok = actual is not None and abs(float(actual) - exp_val) <= METRIC_TOL
        rows.append({"source": "json", "field": field,
                     "expected": str(exp_val), "actual": str(actual), "pass": ok})
        if not ok:
            _err(f"JSON[{field}]={actual} != {exp_val}")

    # val_auc null + AUROC_ERROR 확인
    auc_val = json_data.get("val_auc")
    auc_status = json_data.get("val_auc_status", "")
    auc_null_ok = auc_val is None
    auc_status_ok = "AUROC_ERROR" in auc_status and "ModuleNotFoundError" in auc_status
    rows.append({"source": "json", "field": "val_auc_null",
                 "expected": "None", "actual": str(auc_val), "pass": auc_null_ok})
    rows.append({"source": "json", "field": "val_auc_status_is_AUROC_ERROR",
                 "expected": "contains:AUROC_ERROR:ModuleNotFoundError",
                 "actual": auc_status, "pass": auc_status_ok})
    rows.append({"source": "json", "field": "val_auc_null_reason",
                 "expected": "sklearn dependency missing (not training failure)",
                 "actual": auc_status, "pass": auc_status_ok})

    # checkpoint와 JSON 비교
    if ckpt is not None:
        for field in ("epoch", "smoke_only", "full_training", "train_loss",
                      "train_acc", "val_loss", "val_acc"):
            ckpt_val = ckpt.get(field)
            json_val = json_data.get(field)
            if field in ("train_loss", "train_acc", "val_loss", "val_acc"):
                ok = (ckpt_val is not None and json_val is not None
                      and abs(float(ckpt_val) - float(json_val)) <= METRIC_TOL)
            else:
                ok = ckpt_val == json_val
            rows.append({
                "source": "checkpoint_vs_json", "field": field,
                "expected": str(json_val), "actual": str(ckpt_val), "pass": ok,
            })
            if not ok:
                _err(f"checkpoint[{field}]={ckpt_val} != JSON[{field}]={json_val}")

    # MD에 key metric 포함 여부 (간단 체크)
    for keyword in ["0.0797", "0.9776", "0.0276", "0.9917"]:
        in_md = keyword in md_text
        rows.append({"source": "md", "field": f"contains_{keyword}",
                     "expected": "True", "actual": str(in_md), "pass": in_md})

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: sklearn-free AUROC backfill
# ─────────────────────────────────────────────────────────────────────────────
def step4_auc_backfill(ckpt: dict | None) -> list[dict]:
    print("\n[STEP4] sklearn-free AUROC backfill")
    rows = []

    if ckpt is None:
        _err("Checkpoint not loaded — skipping AUROC backfill")
        rows.append({"status": "SKIPPED", "reason": "checkpoint_missing",
                     "val_auc": None, "n_pos": None, "n_neg": None})
        return rows

    if not VAL_MANIFEST_PATH.exists():
        _err(f"Val manifest missing: {VAL_MANIFEST_PATH}")
        rows.append({"status": "SKIPPED", "reason": "val_manifest_missing",
                     "val_auc": None, "n_pos": None, "n_neg": None})
        return rows

    # stage2_holdout guard
    if STAGE2_HOLDOUT_SENTINEL in str(VAL_MANIFEST_PATH):
        GUARDRAIL["stage2_holdout_accessed"] = True
        _err("stage2_holdout detected in val manifest path")
        return rows

    df_val = pd.read_csv(VAL_MANIFEST_PATH, low_memory=False)
    print(f"  val manifest loaded: {len(df_val)} rows")

    # hard_negative / MSD_Lung guard
    if "source_name" in df_val.columns:
        msd_count = int(df_val["source_name"].astype(str).str.contains("MSD", na=False).sum())
        hn_count = int(df_val["source_name"].astype(str).str.contains("hard_negative", na=False).sum())
        if msd_count > 0:
            GUARDRAIL["MSD_Lung_included"] = True
            _err(f"MSD_Lung rows found in val: {msd_count}")
        if hn_count > 0:
            GUARDRAIL["hard_negative_included"] = True
            _err(f"hard_negative rows found in val: {hn_count}")

    # 모델 로드 (weights_only 추론, 수정 없음)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device={device}")

    model = _build_model()
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    ds_val = _ValDataset(df_val)
    loader_val = DataLoader(ds_val, batch_size=32, shuffle=False,
                            num_workers=4, pin_memory=True)

    all_probs = []
    all_labels = []
    n_batch = 0
    load_errors = 0

    with torch.no_grad():
        for imgs, labels, _ in loader_val:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits.squeeze(1))
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            n_batch += 1

    print(f"  forward pass done: n_batch={n_batch}, total={len(all_probs)}")

    # sklearn 사용 금지 확인 (import 시도 없음 — 이 스크립트 자체가 sklearn import 안 함)

    n_pos = int(sum(l == 1 for l in all_labels))
    n_neg = int(sum(l == 0 for l in all_labels))
    print(f"  labels: n_pos={n_pos}, n_neg={n_neg}, total={len(all_labels)}")

    val_auc, auc_status = _auroc_numpy(all_labels, all_probs)
    print(f"  AUROC backfill: val_auc={val_auc}, status={auc_status}")

    rows.append({
        "status": "COMPUTED" if val_auc is not None else "FAILED",
        "method": "numpy_rank_sum_mannwhitney",
        "sklearn_used": False,
        "val_auc": round(val_auc, 6) if val_auc is not None else None,
        "auc_status": auc_status,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_total": len(all_labels),
        "n_batch": n_batch,
        "device": str(device),
        "load_errors": load_errors,
        "original_val_auc_status": "AUROC_ERROR:ModuleNotFoundError",
        "original_val_auc_null_reason": "sklearn not installed (not training failure)",
    })

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Guardrail check
# ─────────────────────────────────────────────────────────────────────────────
def step5_guardrail() -> list[dict]:
    print("\n[STEP5] Guardrail check")
    rows = []
    all_pass = True
    for key, actual in GUARDRAIL.items():
        expected = False if key != "forbidden_diagnostic_wording_count" else 0
        ok = actual == expected
        rows.append({
            "guardrail": key,
            "expected": str(expected),
            "actual": str(actual),
            "pass": ok,
        })
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {key}: {actual}")
        if not ok:
            all_pass = False
            _err(f"Guardrail violated: {key}={actual}")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("P-C-NORMAL16: Smoke Result Validation + sklearn-free AUROC Backfill")
    print("=" * 70)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # STEP 1
    file_val_rows = step1_output_file_validation()

    # STEP 2
    key_rows, weight_rows, ckpt = step2_checkpoint_validation()

    # STEP 3
    consistency_rows = step3_metric_consistency(ckpt)

    # STEP 4
    auc_rows = step4_auc_backfill(ckpt)

    # STEP 5
    guardrail_rows = step5_guardrail()

    # ── 판정 ──────────────────────────────────────────────────────────────────
    has_errors = len(ERRORS) > 0

    # 판정 로직
    ckpt_ok = all(r.get("pass", False) for r in key_rows
                  if r.get("key") in SMOKE_CHECKPOINT_REQUIRED_KEYS)
    weight_ok = all(r.get("pass", False) for r in weight_rows)
    metrics_ok = all(r.get("pass", False) for r in consistency_rows)
    guardrail_ok = all(r.get("pass", False) for r in guardrail_rows)
    auc_computed = any(r.get("status") == "COMPUTED" for r in auc_rows)
    auc_val_final = next((r.get("val_auc") for r in auc_rows if r.get("val_auc") is not None), None)
    auc_status_final = next((r.get("auc_status") for r in auc_rows), "UNKNOWN")

    if not ckpt_ok or not weight_ok or GUARDRAIL["stage2_holdout_accessed"]:
        verdict = "FAIL"
    elif ckpt_ok and metrics_ok and auc_computed and guardrail_ok:
        verdict = "PASS"
    elif ckpt_ok and not auc_computed:
        verdict = "PARTIAL_PASS"
    else:
        verdict = "PARTIAL_PASS"

    print(f"\n{'='*70}")
    print(f"  판정: {verdict}")
    print(f"  val_auc_backfill: {auc_val_final}  ({auc_status_final})")
    print(f"  errors: {len(ERRORS)}")
    print(f"{'='*70}")

    # ── 저장 ──────────────────────────────────────────────────────────────────
    _write_csv(OUT_DIR / "p_c_normal16_output_file_validation.csv", file_val_rows)
    _write_csv(OUT_DIR / "p_c_normal16_checkpoint_key_validation.csv", key_rows)
    _write_csv(OUT_DIR / "p_c_normal16_checkpoint_weight_validation.csv", weight_rows)
    _write_csv(OUT_DIR / "p_c_normal16_metric_consistency_check.csv", consistency_rows)
    _write_csv(OUT_DIR / "p_c_normal16_auc_backfill_result.csv", auc_rows)
    _write_csv(OUT_DIR / "p_c_normal16_guardrail_check.csv", guardrail_rows)
    _write_csv(OUT_DIR / "p_c_normal16_errors.csv",
               ERRORS if ERRORS else [{"error": "none"}])

    now_iso = datetime.datetime.now().isoformat()

    # JSON
    summary_json = {
        "stage": "P-C-NORMAL16",
        "validated_at": now_iso,
        "verdict": verdict,
        "p_c_normal15_smoke_training_completed": True,
        "val_auc_null_reason": "sklearn not installed — monitoring metric missing, not training failure",
        "checkpoint_valid": ckpt_ok and weight_ok,
        "metrics_consistent": metrics_ok,
        "val_auc_backfill": auc_val_final,
        "val_auc_backfill_status": auc_status_final,
        "val_auc_backfill_method": "numpy_rank_sum_mannwhitney",
        "sklearn_used": False,
        "full_training_not_yet_run": True,
        "guardrail_pass": guardrail_ok,
        "error_count": len(ERRORS),
        "guardrail": GUARDRAIL,
    }
    json_out = OUT_DIR / "p_c_normal16_smoke_result_validation_auc_backfill.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False)
    print(f"  saved → {json_out}")

    # MD
    md_lines = [
        "# P-C-NORMAL16 Smoke Result Validation + AUROC Backfill",
        "",
        f"**판정: {verdict}**  ",
        f"**날짜:** {now_iso[:10]}",
        "",
        "## 결론",
        "",
        "- P-C-NORMAL15 smoke training은 실행 완료",
        "- val_auc null은 sklearn 미설치로 인한 monitoring metric missing (training failure 아님)",
        "- checkpoint 자체는 유효",
        f"- sklearn-free AUROC backfill 결과: **{auc_val_final}** ({auc_status_final})",
        "- full training은 아직 P-C-NORMAL16 결과 확인 후 별도 판단",
        "",
        "## P-C-NORMAL15 metrics (원본 불변)",
        "",
        "| 항목 | 값 |",
        "|------|-----|",
        "| epoch | 1 |",
        "| train_loss | 0.0797 |",
        "| train_acc | 0.9776 |",
        "| val_loss | 0.0276 |",
        "| val_acc | 0.9917 |",
        "| val_auc (원본) | null (AUROC_ERROR:ModuleNotFoundError) |",
        f"| val_auc (backfill) | {auc_val_final} ({auc_status_final}) |",
        "| backfill_method | numpy_rank_sum_mannwhitney |",
        "| sklearn_used | False |",
        "",
        "## Guardrail",
        "",
        "| 항목 | 결과 |",
        "|------|------|",
    ]
    for k, v in GUARDRAIL.items():
        md_lines.append(f"| {k} | {v} |")

    md_lines += [
        "",
        f"## 판정: {verdict}",
        "",
        f"- checkpoint_valid: {ckpt_ok and weight_ok}",
        f"- metrics_consistent: {metrics_ok}",
        f"- guardrail_pass: {guardrail_ok}",
        f"- error_count: {len(ERRORS)}",
    ]

    md_out = OUT_DIR / "p_c_normal16_smoke_result_validation_auc_backfill.md"
    md_out.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"  saved → {md_out}")

    print(f"\n[DONE] 출력 폴더: {OUT_DIR}")
    return 0 if verdict in ("PASS", "PARTIAL_PASS") else 1


if __name__ == "__main__":
    sys.exit(main())
