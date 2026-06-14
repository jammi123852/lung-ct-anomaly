#!/usr/bin/env python
"""
Phase 6.1: S6-A input asset + crop loader smoke
- dataset index / split join / crop load / DataLoader 구성만 확인
- model forward / training / checkpoint / threshold 금지
"""
import argparse
import json
import csv
import sys
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

DATASET_INDEX  = PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/datasets/s6a_6ch_full_dataset_index.csv"
STAGE_SPLIT    = PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1.csv"
AUX_SPLIT      = PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/datasets/s6a_stage1_train_val_split.csv"
CROP_ROOT      = PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/crops_s6a_6ch_full"
OUTPUT_ROOT    = PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/review_annotations/phase6_1_s6a_input_asset_loader_smoke_v1"

EXPECTED_SHAPE = (6, 96, 96)
EXPECTED_MIN   = 0.0
EXPECTED_MAX   = 1.0
EXPECTED_ROWS  = 130659


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-smoke", action="store_true")
    p.add_argument("--max-crops",   type=int, default=16)
    p.add_argument("--max-batches", type=int, default=2)
    return p.parse_args()


def check_v2_paths(df):
    paths = df["npz_path"].astype(str)
    return int(paths.str.contains("v2", na=False).sum())


def load_crop(npz_path):
    r = {
        "crop_path":  str(npz_path),
        "exists":     False,
        "shape":      None,
        "dtype":      None,
        "min_value":  None,
        "max_value":  None,
        "nan_count":  None,
        "inf_count":  None,
        "status":     "FAIL",
        "issue":      "",
    }
    p = Path(npz_path)
    if not p.exists():
        r["issue"] = "file not found"
        return r
    r["exists"] = True
    try:
        d   = np.load(p)
        arr = d["image"]
        r["shape"]     = list(arr.shape)
        r["dtype"]     = str(arr.dtype)
        r["min_value"] = float(arr.min())
        r["max_value"] = float(arr.max())
        r["nan_count"] = int(np.isnan(arr).sum())
        r["inf_count"] = int(np.isinf(arr).sum())
        issues = []
        if tuple(arr.shape) != EXPECTED_SHAPE:
            issues.append(f"shape {arr.shape} != {EXPECTED_SHAPE}")
        if r["nan_count"] > 0:
            issues.append(f"NaN={r['nan_count']}")
        if r["inf_count"] > 0:
            issues.append(f"Inf={r['inf_count']}")
        if r["min_value"] < EXPECTED_MIN - 1e-6:
            issues.append(f"min {r['min_value']:.4f} < 0")
        if r["max_value"] > EXPECTED_MAX + 1e-6:
            issues.append(f"max {r['max_value']:.4f} > 1")
        r["issue"]  = "; ".join(issues)
        r["status"] = "PASS" if not issues else "FAIL"
    except Exception as e:
        r["issue"] = str(e)
    return r


def sample_rows(df, max_crops):
    groups = {}
    for lbl in ["hard_negative", "positive"]:
        sub = df[df["sampling_label"] == lbl]
        if len(sub):
            groups[lbl] = sub
    normal = df[~df["sampling_label"].isin(["hard_negative", "positive"])]
    if len(normal):
        groups["normal"] = normal

    per = max(1, max_crops // max(len(groups), 1))
    parts = []
    for sub in groups.values():
        parts.append(sub.sample(n=min(per, len(sub)), random_state=42))
    result = pd.concat(parts) if parts else df.head(max_crops)
    return result.head(max_crops)


def run_smoke(args):
    import torch
    from torch.utils.data import Dataset, DataLoader

    blockers = []
    print(f"[1] dataset index: {DATASET_INDEX}")
    df = pd.read_csv(DATASET_INDEX)
    manifest_rows = len(df)
    print(f"    rows: {manifest_rows}  (expected {EXPECTED_ROWS})")

    required_cols = ["npz_path", "patient_id", "label", "sampling_label"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        blockers.append(f"컬럼 없음: {missing}")
        print(f"    [BLOCKER] 컬럼 없음: {missing}")
    path_col_is_npz_path = "npz_path" in df.columns

    print(f"[2] stage split join: {STAGE_SPLIT}")
    stage_df = pd.read_csv(STAGE_SPLIT)
    merged   = df.merge(stage_df[["patient_id", "stage_split"]], on="patient_id", how="left")
    s2_count = int((merged["stage_split"] == "stage2_holdout").sum())
    s1_pats  = int(merged[merged["stage_split"] == "stage1_dev"]["patient_id"].nunique())
    unk      = int(merged["stage_split"].isna().sum())
    print(f"    stage2_holdout rows: {s2_count}  stage1_dev patients: {s1_pats}  unknown: {unk}")
    if s2_count > 0:
        blockers.append(f"stage2_holdout {s2_count}건")

    print("[3] v2/v2v2 경로 검출")
    v2_count = check_v2_paths(df)
    print(f"    v2 경로: {v2_count}건")
    if v2_count > 0:
        blockers.append(f"v2 경로 {v2_count}건")

    print(f"[4] sample crop 로드 (최대 {args.max_crops}개)")
    sdf = sample_rows(df, args.max_crops)
    crop_results = []
    for _, row in sdf.iterrows():
        r = load_crop(row["npz_path"])
        r["patient_id"]    = row["patient_id"]
        r["label_or_group"] = str(row.get("sampling_label", ""))
        crop_results.append(r)

    pass_n  = sum(1 for r in crop_results if r["status"] == "PASS")
    fail_n  = len(crop_results) - pass_n
    print(f"    sampled={len(crop_results)} PASS={pass_n} FAIL={fail_n}")
    if fail_n:
        for r in crop_results:
            if r["status"] != "PASS":
                print(f"    FAIL: {Path(r['crop_path']).name} — {r['issue']}")
        blockers.append(f"crop 로드 실패 {fail_n}건")

    shape_ok = all(r["shape"] == list(EXPECTED_SHAPE) for r in crop_results if r["shape"])
    range_ok = all(
        r["min_value"] is not None
        and r["min_value"] >= EXPECTED_MIN - 1e-6
        and r["max_value"] <= EXPECTED_MAX + 1e-6
        for r in crop_results if r["min_value"] is not None
    )
    nan_total = sum(r["nan_count"] or 0 for r in crop_results)
    inf_total = sum(r["inf_count"] or 0 for r in crop_results)

    print(f"[5] DataLoader smoke (batch_size=3, max_batches={args.max_batches})")

    class CropDataset(Dataset):
        def __init__(self, rows):
            self.rows = rows
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, idx):
            row = self.rows[idx]
            d   = np.load(row["npz_path"])
            img = torch.from_numpy(d["image"].copy())
            # positive/hard_negative는 학습 label로 사용하지 않음 — meta 전달만
            meta = {
                "patient_id":    row["patient_id"],
                "sampling_label": str(row.get("sampling_label", "")),
            }
            return img, meta

    dataset       = CropDataset(sdf.to_dict("records"))
    loader        = DataLoader(dataset, batch_size=3, shuffle=False, num_workers=0)
    batch_shapes  = []
    batch_count   = 0
    for imgs, _ in loader:
        if batch_count >= args.max_batches:
            break
        batch_shapes.append(list(imgs.shape))
        print(f"    batch {batch_count}: shape={list(imgs.shape)} dtype={imgs.dtype}")
        batch_count += 1
    loader_ok = batch_count > 0

    # 보조 train/val split 확인
    aux_used = False
    aux_info = "사용 안 함"
    if AUX_SPLIT.exists():
        aux_df   = pd.read_csv(AUX_SPLIT)
        aux_used = True
        train_n  = int((aux_df["train_val"] == "train").sum())
        val_n    = int((aux_df["train_val"] == "val").sum())
        aux_info = f"rows={len(aux_df)} train={train_n} val={val_n}"
        print(f"[6] 보조 train/val split: {aux_info}")

    # 결과 저장
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    smoke_pass = (
        not blockers
        and shape_ok and range_ok
        and nan_total == 0 and inf_total == 0
        and s2_count == 0 and v2_count == 0
        and loader_ok
    )

    # CSV
    csv_path = OUTPUT_ROOT / "phase6_1_s6a_input_asset_loader_smoke_v1.csv"
    fieldnames = [
        "section","item_id","patient_id","crop_path","split","label_or_group",
        "shape","dtype","min_value","max_value","nan_count","inf_count","status","issue","note",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, r in enumerate(crop_results):
            w.writerow({
                "section":       "crop_smoke",
                "item_id":       i,
                "patient_id":    r.get("patient_id",""),
                "crop_path":     r["crop_path"],
                "split":         "stage1_dev",
                "label_or_group": r.get("label_or_group",""),
                "shape":         str(r["shape"]),
                "dtype":         r["dtype"],
                "min_value":     r["min_value"],
                "max_value":     r["max_value"],
                "nan_count":     r["nan_count"],
                "inf_count":     r["inf_count"],
                "status":        r["status"],
                "issue":         r["issue"],
                "note":          "",
            })

    # JSON
    json_data = {
        "discovered_paths": {
            "dataset_index": str(DATASET_INDEX),
            "stage_split_file": str(STAGE_SPLIT),
            "aux_split_file": str(AUX_SPLIT) if aux_used else None,
            "crop_root": str(CROP_ROOT),
        },
        "manifest_row_count":         manifest_rows,
        "expected_crop_count":        EXPECTED_ROWS,
        "manifest_row_matches_expected": manifest_rows == EXPECTED_ROWS,
        "path_col_is_npz_path":       path_col_is_npz_path,
        "sampled_crop_count":         len(crop_results),
        "batch_count_checked":        batch_count,
        "batch_shapes":               batch_shapes,
        "shape_check_pass":           shape_ok,
        "value_range_check_pass":     range_ok,
        "nan_inf_check_pass":         (nan_total == 0 and inf_total == 0),
        "nan_total":                  nan_total,
        "inf_total":                  inf_total,
        "stage1_dev_only_check":      s2_count == 0,
        "stage2_holdout_count":       s2_count,
        "stage1_dev_patient_count":   s1_pats,
        "v2_path_detected":           v2_count,
        "hard_negative_used_for_training": False,
        "positive_used_for_training":      False,
        "loader_smoke_pass":          loader_ok,
        "aux_split_used":             aux_used,
        "aux_split_info":             aux_info,
        "smoke_pass":                 smoke_pass,
        "blockers":                   blockers,
        "next_step_recommendation": (
            "Phase 6.2 model forward smoke preflight" if smoke_pass else "blockers 해결 후 재실행"
        ),
    }
    json_path = OUTPUT_ROOT / "phase6_1_s6a_input_asset_loader_smoke_v1.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    # MD
    verdict = "전체 통과" if smoke_pass else "미통과"
    md = [
        "# Phase 6.1 S6-A Input Asset + Crop Loader Smoke Report",
        "",
        f"**최종 판정: {verdict}**",
        "",
        "## 1. 목적",
        "S6-A 6ch crop 자산이 2차 파이프라인 입력으로 올바르게 읽히는지 확인한다.",
        "학습·model forward·checkpoint·threshold 없이 data/loader smoke만 수행한다.",
        "",
        "## 2. 발견한 S6-A 경로",
        f"- dataset index: `{DATASET_INDEX}`",
        f"- crop root: `{CROP_ROOT}`",
        f"- stage split: `{STAGE_SPLIT}`",
        f"- aux split: `{AUX_SPLIT}` (사용: {aux_used})",
        "",
        "## 3. manifest row 수",
        f"- 실제: {manifest_rows:,}  /  expected: {EXPECTED_ROWS:,}",
        f"- 일치: {'예' if manifest_rows == EXPECTED_ROWS else '아니오'}",
        "",
        "## 4. sample crop shape 결과",
        f"- sampled: {len(crop_results)}개  PASS: {pass_n}  FAIL: {fail_n}",
        f"- shape check `(6,96,96)`: {'PASS' if shape_ok else 'FAIL'}",
        f"- value range [0,1] check: {'PASS' if range_ok else 'FAIL'}",
        f"- NaN: {nan_total} / Inf: {inf_total}",
        "",
        "## 5. batch loader smoke 결과",
        f"- batch 수: {batch_count}",
        f"- batch shapes: {batch_shapes}",
        f"- loader smoke: {'PASS' if loader_ok else 'FAIL'}",
        "",
        "## 6. stage1_dev-only 확인",
        f"- stage2_holdout row: {s2_count}",
        f"- stage1_dev 환자 수: {s1_pats}",
        f"- 판정: {'PASS' if s2_count == 0 else 'FAIL'}",
        "",
        "## 7. positive/hard_negative 학습 label 미사용 확인",
        "- positive/hard_negative는 shape/read-only smoke 대상으로만 처리",
        "- 학습 label로 사용하지 않음: 확인",
        "",
        "## 8. blockers",
    ]
    md += [f"- {b}" for b in blockers] if blockers else ["- 없음"]
    md += [
        "",
        "## 9. 다음 단계",
        f"- 통과 시: Phase 6.2 model forward smoke preflight",
        f"- 미통과 시: blockers 해결 후 재실행",
    ]
    md_path = OUTPUT_ROOT / "phase6_1_s6a_input_asset_loader_smoke_report_v1.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    print(f"\n=== 결과 ===")
    print(f"smoke_pass:        {smoke_pass}")
    print(f"blockers:          {blockers}")
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")
    print(f"MD:   {md_path}")
    return smoke_pass


def preflight_only():
    print("[preflight] 경로 존재 확인 (smoke 없음)")
    for p, name in [
        (DATASET_INDEX, "dataset index"),
        (STAGE_SPLIT,   "stage split"),
        (CROP_ROOT,     "crop root"),
        (AUX_SPLIT,     "aux split"),
    ]:
        print(f"  {'OK' if Path(p).exists() else 'MISSING'} {name}: {p}")


def main():
    args = parse_args()
    if args.run_smoke:
        ok = run_smoke(args)
        sys.exit(0 if ok else 1)
    else:
        preflight_only()


if __name__ == "__main__":
    main()
