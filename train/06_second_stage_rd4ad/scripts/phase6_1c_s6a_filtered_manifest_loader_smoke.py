#!/usr/bin/env python
"""
Phase 6.1c: S6-A filtered manifest loader smoke
- Phase 6.1b에서 생성된 stage1_dev-only filtered shadow manifest를 입력으로 사용
- 원본 s6a_6ch_full_dataset_index.csv는 smoke 입력으로 사용하지 않음
- model forward / training / checkpoint / threshold 금지
"""
import argparse
import csv
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

FILTERED_MANIFEST = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/review_annotations"
    / "phase6_1b_s6a_stage1_dev_filtered_manifest_v1"
    / "phase6_1b_s6a_stage1_dev_filtered_manifest_v1.csv"
)
PHASE6_1B_SUMMARY = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/review_annotations"
    / "phase6_1b_s6a_stage1_dev_filtered_manifest_v1"
    / "phase6_1b_s6a_stage1_dev_filtered_manifest_summary_v1.json"
)
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/review_annotations"
    / "phase6_1c_s6a_filtered_manifest_loader_smoke_v1"
)

OUT_CSV  = OUTPUT_ROOT / "phase6_1c_s6a_filtered_manifest_loader_smoke_v1.csv"
OUT_JSON = OUTPUT_ROOT / "phase6_1c_s6a_filtered_manifest_loader_smoke_v1.json"
OUT_MD   = OUTPUT_ROOT / "phase6_1c_s6a_filtered_manifest_loader_smoke_report_v1.md"

EXPECTED_ROWS    = 129437
EXPECTED_PATS    = 152
EXCLUDED_PATS    = {"LUNG1-295", "LUNG1-415"}
EXPECTED_SHAPE   = (6, 96, 96)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-smoke",   action="store_true")
    p.add_argument("--max-crops",   type=int, default=16)
    p.add_argument("--max-batches", type=int, default=2)
    return p.parse_args()


def guard_output_exists():
    """output root 또는 결과 파일이 이미 존재하면 즉시 중단."""
    if OUTPUT_ROOT.exists():
        print(f"[ABORT] output root 이미 존재: {OUTPUT_ROOT}")
        sys.exit(1)
    for p in [OUT_CSV, OUT_JSON, OUT_MD]:
        if p.exists():
            print(f"[ABORT] output 파일 이미 존재: {p}")
            sys.exit(1)


def guard_save_exists():
    """저장 직전 재확인."""
    for p in [OUT_CSV, OUT_JSON, OUT_MD]:
        if p.exists():
            print(f"[ABORT] 저장 직전 파일 이미 존재: {p}")
            sys.exit(1)


def check_v2_paths(df):
    return int(df["npz_path"].astype(str).str.contains("v2", na=False).sum())


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
    parts = [sub.sample(n=min(per, len(sub)), random_state=42) for sub in groups.values()]
    result = pd.concat(parts) if parts else df.head(max_crops)
    return result.head(max_crops)


def load_crop(npz_path):
    r = {
        "crop_path": str(npz_path),
        "exists": False,
        "shape": None, "dtype": None,
        "min_value": None, "max_value": None,
        "nan_count": None, "inf_count": None,
        "status": "FAIL", "issue": "",
    }
    if not Path(npz_path).exists():
        r["issue"] = "file not found"
        return r
    r["exists"] = True
    try:
        arr = np.load(npz_path)["image"]
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
        if r["min_value"] < -1e-6:
            issues.append(f"min {r['min_value']:.4f} < 0")
        if r["max_value"] > 1 + 1e-6:
            issues.append(f"max {r['max_value']:.4f} > 1")
        r["issue"]  = "; ".join(issues)
        r["status"] = "PASS" if not issues else "FAIL"
    except Exception as e:
        r["issue"] = str(e)
    return r


def run_smoke(args):
    import torch
    from torch.utils.data import Dataset, DataLoader

    blockers = []

    # ── 1. filtered manifest 로드 ─────────────────────────────────────────
    print(f"[1] filtered manifest 로드: {FILTERED_MANIFEST}")
    if not FILTERED_MANIFEST.exists():
        print("[ABORT] filtered manifest 없음")
        sys.exit(1)
    df = pd.read_csv(FILTERED_MANIFEST)

    # ── 2. 기본 수치 검증 ─────────────────────────────────────────────────
    print(f"[2] 수치 검증")
    filt_rows = len(df)
    filt_pats = df["patient_id"].nunique()
    s2_count  = int((df.get("stage_split", pd.Series(dtype=str)) == "stage2_holdout").sum())
    excl_rows = int(df["patient_id"].isin(EXCLUDED_PATS).sum())
    v2_count  = check_v2_paths(df)

    print(f"    rows={filt_rows} (expected {EXPECTED_ROWS})")
    print(f"    unique patients={filt_pats} (expected {EXPECTED_PATS})")
    print(f"    stage2_holdout rows={s2_count}")
    print(f"    LUNG1-295/LUNG1-415 rows={excl_rows}")
    print(f"    v2/v2v2 경로={v2_count}건")

    if filt_rows != EXPECTED_ROWS:
        blockers.append(f"rows 불일치 expected={EXPECTED_ROWS} actual={filt_rows}")
    if filt_pats != EXPECTED_PATS:
        blockers.append(f"unique patients 불일치 expected={EXPECTED_PATS} actual={filt_pats}")
    if s2_count > 0:
        blockers.append(f"stage2_holdout rows={s2_count}")
    if excl_rows > 0:
        blockers.append(f"LUNG1-295/LUNG1-415 rows={excl_rows} (제거되지 않음)")
    if v2_count > 0:
        blockers.append(f"v2 경로 {v2_count}건")

    # ── 3. npz_path 컬럼 확인 ────────────────────────────────────────────
    if "npz_path" not in df.columns:
        blockers.append("npz_path 컬럼 없음")
        print("[BLOCKER] npz_path 컬럼 없음")

    # ── 4. sample crop 로드 ───────────────────────────────────────────────
    print(f"[3] sample crop 로드 (최대 {args.max_crops}개)")
    sdf = sample_rows(df, args.max_crops)
    crop_results = []
    for _, row in sdf.iterrows():
        r = load_crop(row["npz_path"])
        r["patient_id"]    = row["patient_id"]
        r["label"]         = row.get("label", "")
        r["sampling_label"] = str(row.get("sampling_label", ""))
        r["stage_split"]   = str(row.get("stage_split", ""))
        crop_results.append(r)

    pass_n = sum(1 for r in crop_results if r["status"] == "PASS")
    fail_n = len(crop_results) - pass_n
    print(f"    sampled={len(crop_results)} PASS={pass_n} FAIL={fail_n}")
    if fail_n:
        for r in crop_results:
            if r["status"] != "PASS":
                print(f"    FAIL: {Path(r['crop_path']).name} — {r['issue']}")
        blockers.append(f"crop 로드 실패 {fail_n}건")

    shape_ok  = all(r["shape"] == list(EXPECTED_SHAPE) for r in crop_results if r["shape"])
    range_ok  = all(
        r["min_value"] is not None and r["min_value"] >= -1e-6 and r["max_value"] <= 1 + 1e-6
        for r in crop_results if r["min_value"] is not None
    )
    nan_total = sum(r["nan_count"] or 0 for r in crop_results)
    inf_total = sum(r["inf_count"] or 0 for r in crop_results)
    all_shapes = [r["shape"] for r in crop_results if r["shape"]]

    # ── 5. DataLoader smoke ───────────────────────────────────────────────
    print(f"[4] DataLoader smoke (batch_size=3, max_batches={args.max_batches})")

    class CropDataset(Dataset):
        def __init__(self, rows):
            self.rows = rows
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, idx):
            row = self.rows[idx]
            arr = np.load(row["npz_path"])["image"]
            img = torch.from_numpy(arr.copy())
            # positive/hard_negative는 metadata로만 기록 — 학습 label로 사용하지 않음
            meta = {
                "patient_id":    row["patient_id"],
                "sampling_label": str(row.get("sampling_label", "")),
                "stage_split":   str(row.get("stage_split", "")),
            }
            return img, meta

    dataset      = CropDataset(sdf.to_dict("records"))
    loader       = __import__("torch").utils.data.DataLoader(
        dataset, batch_size=3, shuffle=False, num_workers=0
    )
    batch_shapes = []
    batch_count  = 0
    for imgs, _ in loader:
        if batch_count >= args.max_batches:
            break
        batch_shapes.append(list(imgs.shape))
        print(f"    batch {batch_count}: shape={list(imgs.shape)} dtype={imgs.dtype}")
        batch_count += 1
    loader_ok = batch_count > 0

    # ── 6. 저장 직전 재확인 ───────────────────────────────────────────────
    guard_save_exists()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    smoke_pass = (
        not blockers and shape_ok and range_ok
        and nan_total == 0 and inf_total == 0
        and loader_ok
    )

    # ── 7. CSV ────────────────────────────────────────────────────────────
    fieldnames = [
        "section", "item_id", "patient_id", "npz_path",
        "label", "sampling_label", "stage_split",
        "shape", "dtype", "min_value", "max_value",
        "nan_count", "inf_count", "status", "issue", "note",
    ]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, r in enumerate(crop_results):
            w.writerow({
                "section":       "crop_smoke",
                "item_id":       i,
                "patient_id":    r.get("patient_id", ""),
                "npz_path":      r["crop_path"],
                "label":         r.get("label", ""),
                "sampling_label": r.get("sampling_label", ""),
                "stage_split":   r.get("stage_split", ""),
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

    # ── 8. JSON ───────────────────────────────────────────────────────────
    json_data = {
        "input_filtered_manifest_path":  str(FILTERED_MANIFEST),
        "filtered_manifest_row_count":   filt_rows,
        "filtered_unique_patient_count": filt_pats,
        "stage2_holdout_row_count":      s2_count,
        "excluded_patients_absent":      excl_rows == 0,
        "v2_path_detected":              v2_count,
        "sampled_crop_count":            len(crop_results),
        "all_sample_shapes":             all_shapes,
        "shape_check_pass":              shape_ok,
        "value_range_check_pass":        range_ok,
        "nan_inf_check_pass":            (nan_total == 0 and inf_total == 0),
        "nan_total":                     nan_total,
        "inf_total":                     inf_total,
        "dataloader_batch_count":        batch_count,
        "batch_shapes":                  batch_shapes,
        "positive_used_for_training":    False,
        "hard_negative_used_for_training": False,
        "model_forward_executed":        False,
        "training_executed":             False,
        "checkpoint_created":            False,
        "threshold_calculated":          False,
        "smoke_pass":                    smoke_pass,
        "blockers":                      blockers,
        "next_step_recommendation": (
            "Phase 6.2 model forward smoke preflight" if smoke_pass else "blocker 해결 후 재실행"
        ),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    # ── 9. MD ─────────────────────────────────────────────────────────────
    verdict = "전체 통과" if smoke_pass else "미통과"
    md = [
        "# Phase 6.1c S6-A Filtered Manifest Loader Smoke Report",
        "",
        f"**최종 판정: {verdict}**",
        "",
        "## 1. Phase 6.1c 목적",
        "Phase 6.1b에서 생성된 stage1_dev-only filtered shadow manifest를 입력으로 사용하여",
        "crop 로드·shape/value·DataLoader batch smoke를 재확인한다.",
        "model forward·training·checkpoint·threshold는 이 단계에서 수행하지 않는다.",
        "",
        "## 2. 사용한 filtered manifest 경로",
        f"`{FILTERED_MANIFEST}`",
        "",
        "## 3. row/patient/stage2_holdout 검증",
        f"- rows: {filt_rows:,}  (expected {EXPECTED_ROWS:,}: {'일치' if filt_rows == EXPECTED_ROWS else '불일치'})",
        f"- unique patients: {filt_pats}  (expected {EXPECTED_PATS}: {'일치' if filt_pats == EXPECTED_PATS else '불일치'})",
        f"- stage2_holdout rows: {s2_count}  ({'PASS' if s2_count == 0 else 'FAIL'})",
        "",
        "## 4. excluded patients 부재 확인",
        f"- LUNG1-295, LUNG1-415 rows: {excl_rows}  ({'PASS' if excl_rows == 0 else 'FAIL'})",
        f"- v2/v2v2 경로: {v2_count}건  ({'PASS' if v2_count == 0 else 'FAIL'})",
        "",
        "## 5. sample crop shape/value 결과",
        f"- sampled: {len(crop_results)}개  PASS: {pass_n}  FAIL: {fail_n}",
        f"- shape check `(6,96,96)`: {'PASS' if shape_ok else 'FAIL'}",
        f"- value range [0,1]: {'PASS' if range_ok else 'FAIL'}",
        f"- NaN: {nan_total} / Inf: {inf_total}",
        "",
        "## 6. DataLoader batch 결과",
        f"- batch 수: {batch_count}",
        f"- batch shapes: {batch_shapes}",
        f"- loader smoke: {'PASS' if loader_ok else 'FAIL'}",
        "",
        "## 7. positive/hard_negative 학습 label 미사용 확인",
        "- positive/hard_negative는 metadata로만 기록",
        "- 학습 label로 사용하지 않음: 확인",
        "",
        "## 8. 최종 판정",
        f"**{verdict}**",
        f"blockers: {blockers if blockers else '없음'}",
        "",
        "## 9. 다음 단계",
        "- 통과 시: Phase 6.2 model forward smoke preflight",
        "- 미통과 시: blocker 해결 후 재실행",
    ]
    OUT_MD.write_text("\n".join(md), encoding="utf-8")

    print(f"\n=== Phase 6.1c 결과 ===")
    print(f"smoke_pass: {smoke_pass}")
    print(f"blockers:   {blockers}")
    print(f"CSV:  {OUT_CSV}")
    print(f"JSON: {OUT_JSON}")
    print(f"MD:   {OUT_MD}")
    return smoke_pass


def preflight_only():
    print("[preflight] 경로 존재 확인 (smoke 없음)")
    for p, name in [(FILTERED_MANIFEST, "filtered manifest"), (OUTPUT_ROOT, "output root")]:
        print(f"  {'OK' if Path(p).exists() else 'MISSING'} {name}: {p}")


def main():
    args = parse_args()
    if args.run_smoke:
        guard_output_exists()
        ok = run_smoke(args)
        sys.exit(0 if ok else 1)
    else:
        preflight_only()


if __name__ == "__main__":
    main()
