#!/usr/bin/env python3
"""Phase 6.2b: S6-A filtered batch model forward smoke."""
import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FILTERED_MANIFEST_PATH = PROJECT_ROOT / (
    "outputs/second-stage-lesion-refiner-v1/review_annotations/"
    "phase6_1b_s6a_stage1_dev_filtered_manifest_v1/"
    "phase6_1b_s6a_stage1_dev_filtered_manifest_v1.csv"
)
CHECKPOINT_PATH = PROJECT_ROOT / (
    "outputs/second-stage-lesion-refiner-v1/models/"
    "rd4ad_2p5d_normal_mw_fixed96_v1/checkpoints/best_val_loss.pt"
)
MODEL_SCRIPT = PROJECT_ROOT / "scripts/train_rd4ad_2p5d_normal.py"
OUTPUT_ROOT = PROJECT_ROOT / (
    "outputs/second-stage-lesion-refiner-v1/review_annotations/"
    "phase6_2b_s6a_model_forward_smoke_v1"
)
CSV_NAME = "phase6_2b_s6a_model_forward_smoke_v1.csv"
JSON_NAME = "phase6_2b_s6a_model_forward_smoke_v1.json"
MD_NAME = "phase6_2b_s6a_model_forward_smoke_report_v1.md"

EXPECTED_ROWS = 129_437
EXPECTED_PATIENTS = 152
EXCLUDED_PATIENTS = {"LUNG1-295", "LUNG1-415"}


# ────────────────────────────────────────────────
def check_output_guard():
    if OUTPUT_ROOT.exists():
        print(f"[GUARD] output root already exists: {OUTPUT_ROOT}")
        sys.exit(1)
    for name in [CSV_NAME, JSON_NAME, MD_NAME]:
        p = OUTPUT_ROOT / name
        if p.exists():
            print(f"[GUARD] output file already exists: {p}")
            sys.exit(1)


def recheck_output_guard():
    if OUTPUT_ROOT.exists():
        print(f"[GUARD] output root appeared during run: {OUTPUT_ROOT}")
        sys.exit(1)
    for name in [CSV_NAME, JSON_NAME, MD_NAME]:
        p = OUTPUT_ROOT / name
        if p.exists():
            print(f"[GUARD] output file appeared during run: {p}")
            sys.exit(1)


# ────────────────────────────────────────────────
def validate_manifest(manifest_path: Path):
    df = pd.read_csv(manifest_path)
    row_count = len(df)
    unique_patients = df["patient_id"].nunique()

    if "stage_split" in df.columns:
        holdout_rows = int((df["stage_split"] == "stage2_holdout").sum())
    else:
        holdout_rows = 0

    excluded_counts = {}
    for pid in EXCLUDED_PATIENTS:
        excluded_counts[pid] = int((df["patient_id"] == pid).sum())

    v2_detected = int(df["npz_path"].str.contains("/v2/|/v2v2/", na=False).sum())

    if "training_manifest_status" in df.columns:
        not_training = bool((df["training_manifest_status"] == "not_training_manifest").all())
    else:
        not_training = None

    if "approval_required_before_training" in df.columns:
        approval_required = bool(df["approval_required_before_training"].all())
    else:
        approval_required = None

    issues = []
    if row_count != EXPECTED_ROWS:
        issues.append(f"row_count={row_count}, expected={EXPECTED_ROWS}")
    if unique_patients != EXPECTED_PATIENTS:
        issues.append(f"unique_patients={unique_patients}, expected={EXPECTED_PATIENTS}")
    if holdout_rows != 0:
        issues.append(f"stage2_holdout_rows={holdout_rows}, expected=0")
    for pid, cnt in excluded_counts.items():
        if cnt != 0:
            issues.append(f"{pid} rows={cnt}, expected=0")
    if v2_detected != 0:
        issues.append(f"v2/v2v2 paths detected={v2_detected}, expected=0")
    if not_training is False:
        issues.append("training_manifest_status != not_training_manifest for some rows")
    if approval_required is False:
        issues.append("approval_required_before_training is False for some rows")

    return {
        "df": df,
        "row_count": row_count,
        "unique_patients": unique_patients,
        "holdout_rows": holdout_rows,
        "excluded_counts": excluded_counts,
        "v2_detected": v2_detected,
        "not_training": not_training,
        "approval_required": approval_required,
        "issues": issues,
    }


# ────────────────────────────────────────────────
class CropDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        arr = np.load(row["npz_path"])["image"].astype(np.float32)
        return {
            "crop": torch.from_numpy(arr),
            "patient_id": str(row["patient_id"]),
            "label": str(row.get("label", "")),
            "sampling_label": str(row.get("sampling_label", "")),
        }


# ────────────────────────────────────────────────
def load_model_class(model_script: Path):
    spec = importlib.util.spec_from_file_location("train_rd4ad_2p5d_normal", model_script)
    if spec is None:
        raise ImportError(f"Cannot load spec from {model_script}")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        raise ImportError("Model script called sys.exit() during import — side effect detected (BLOCKED)")
    except Exception as e:
        raise ImportError(f"Model script raised exception during import: {e}")
    if not hasattr(mod, "ConvAutoencoder2p5D"):
        raise ImportError("ConvAutoencoder2p5D not found in model script")
    return mod.ConvAutoencoder2p5D


# ────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-smoke", action="store_true")
    parser.add_argument("--max-crops", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=3)
    args = parser.parse_args()

    if not args.run_smoke:
        print("[INFO] --run-smoke not specified. Dry run only.")
        print(f"[INFO] output root: {OUTPUT_ROOT}")
        return

    # output guard
    check_output_guard()

    print("[STEP 1] Validating filtered manifest...")
    manifest_info = validate_manifest(FILTERED_MANIFEST_PATH)
    df = manifest_info["df"]

    if manifest_info["issues"]:
        print("[MANIFEST FAIL] Issues found:")
        for issue in manifest_info["issues"]:
            print(f"  - {issue}")
        sys.exit(1)

    print(f"[MANIFEST OK] rows={manifest_info['row_count']}, patients={manifest_info['unique_patients']}, "
          f"holdout={manifest_info['holdout_rows']}, v2_detected={manifest_info['v2_detected']}")

    print("[STEP 2] Loading model class via importlib...")
    try:
        ConvAutoencoder2p5D = load_model_class(MODEL_SCRIPT)
    except ImportError as e:
        print(f"[BLOCKED] {e}")
        sys.exit(1)
    print("[MODEL CLASS OK] ConvAutoencoder2p5D loaded")

    print("[STEP 3] Loading checkpoint...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    model = ConvAutoencoder2p5D(input_channels=6)
    raw_ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    if isinstance(raw_ckpt, dict) and "model_state_dict" in raw_ckpt:
        state_dict = raw_ckpt["model_state_dict"]
    elif isinstance(raw_ckpt, dict) and "state_dict" in raw_ckpt:
        state_dict = raw_ckpt["state_dict"]
    else:
        state_dict = raw_ckpt
    load_result = model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    print(f"[CKPT OK] load_state_dict: {load_result}")

    print("[STEP 4] Building DataLoader...")
    sampled_rows = df.head(args.max_crops).to_dict("records")
    dataset = CropDataset(sampled_rows)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    print(f"[LOADER OK] sampled_crops={len(sampled_rows)}, batch_size={args.batch_size}")

    print("[STEP 5] Forward smoke...")
    csv_rows = []
    forward_batch_count = 0
    input_shapes = []
    output_shapes = []
    output_type_global = None
    output_nan_inf_pass = True
    output_shape_pass = True
    blockers = []

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(loader):
            if forward_batch_count >= args.max_batches:
                break

            crop_tensor = batch_data["crop"].to(device)
            patient_ids = list(batch_data["patient_id"])

            in_shape = list(crop_tensor.shape)
            in_dtype = str(crop_tensor.dtype)
            in_min = float(crop_tensor.min())
            in_max = float(crop_tensor.max())
            in_nan = int(torch.isnan(crop_tensor).sum())
            in_inf = int(torch.isinf(crop_tensor).sum())

            shape_ok = (len(in_shape) == 4 and in_shape[1] == 6
                        and in_shape[2] == 96 and in_shape[3] == 96)
            if not shape_ok:
                blockers.append(f"batch {batch_idx}: unexpected input shape {in_shape}")
                output_shape_pass = False

            out = model(crop_tensor)

            if isinstance(out, torch.Tensor):
                output_type_global = "Tensor"
                out_shape = list(out.shape)
                out_dtype = str(out.dtype)
                out_min = float(out.min())
                out_max = float(out.max())
                out_nan = int(torch.isnan(out).sum())
                out_inf = int(torch.isinf(out).sum())
                out_struct_note = ""
                if out_nan > 0 or out_inf > 0:
                    output_nan_inf_pass = False
                    blockers.append(f"batch {batch_idx}: output NaN={out_nan} Inf={out_inf}")
                if out_shape != in_shape:
                    output_shape_pass = False
                    blockers.append(f"batch {batch_idx}: output shape {out_shape} != input {in_shape}")
            elif isinstance(out, (tuple, list)):
                output_type_global = type(out).__name__
                out_shape = [list(t.shape) if isinstance(t, torch.Tensor) else str(t) for t in out]
                out_dtype = str(out[0].dtype) if isinstance(out[0], torch.Tensor) else "unknown"
                out_min = float(out[0].min()) if isinstance(out[0], torch.Tensor) else None
                out_max = float(out[0].max()) if isinstance(out[0], torch.Tensor) else None
                out_nan = sum(int(torch.isnan(t).sum()) for t in out if isinstance(t, torch.Tensor))
                out_inf = sum(int(torch.isinf(t).sum()) for t in out if isinstance(t, torch.Tensor))
                out_struct_note = f"reconstruction candidate: index 0 shape {out_shape[0] if out_shape else 'unknown'}"
                if out_nan > 0 or out_inf > 0:
                    output_nan_inf_pass = False
                    blockers.append(f"batch {batch_idx}: output NaN={out_nan} Inf={out_inf}")
            elif isinstance(out, dict):
                output_type_global = "dict"
                out_shape = {k: list(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)}
                out_dtype = "dict"
                out_min = None
                out_max = None
                out_nan = sum(int(torch.isnan(v).sum()) for v in out.values() if isinstance(v, torch.Tensor))
                out_inf = sum(int(torch.isinf(v).sum()) for v in out.values() if isinstance(v, torch.Tensor))
                out_struct_note = f"dict keys: {list(out.keys())}"
                if out_nan > 0 or out_inf > 0:
                    output_nan_inf_pass = False
                    blockers.append(f"batch {batch_idx}: output NaN={out_nan} Inf={out_inf}")
            else:
                output_type_global = str(type(out))
                out_shape = "unknown"
                out_dtype = "unknown"
                out_min = None
                out_max = None
                out_nan = 0
                out_inf = 0
                out_struct_note = f"unexpected output type: {type(out)}"
                blockers.append(f"batch {batch_idx}: unexpected output type {type(out)}")

            input_shapes.append(in_shape)
            output_shapes.append(out_shape)

            csv_rows.append({
                "section": "forward_smoke",
                "batch_id": batch_idx,
                "patient_ids": "|".join(patient_ids),
                "input_shape": str(in_shape),
                "input_dtype": in_dtype,
                "input_min": round(in_min, 6),
                "input_max": round(in_max, 6),
                "input_nan_count": in_nan,
                "input_inf_count": in_inf,
                "output_type": output_type_global,
                "output_shape": str(out_shape),
                "output_dtype": out_dtype,
                "output_min": round(out_min, 6) if out_min is not None else "",
                "output_max": round(out_max, 6) if out_max is not None else "",
                "output_nan_count": out_nan,
                "output_inf_count": out_inf,
                "status": "FAIL" if blockers else "OK",
                "issue": "; ".join(blockers),
                "note": out_struct_note,
            })

            forward_batch_count += 1
            print(f"[FORWARD OK] batch {batch_idx}: input={in_shape}, output={out_shape}, "
                  f"nan={out_nan}, inf={out_inf}")

    smoke_pass = len(blockers) == 0

    # 저장 직전 output guard 재확인
    recheck_output_guard()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)

    # CSV
    csv_path = OUTPUT_ROOT / CSV_NAME
    if csv_path.exists():
        print(f"[GUARD] CSV already exists: {csv_path}")
        sys.exit(1)
    fieldnames = [
        "section", "batch_id", "patient_ids",
        "input_shape", "input_dtype", "input_min", "input_max",
        "input_nan_count", "input_inf_count",
        "output_type", "output_shape", "output_dtype",
        "output_min", "output_max", "output_nan_count", "output_inf_count",
        "status", "issue", "note",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"[SAVED] CSV: {csv_path}")

    # JSON
    json_path = OUTPUT_ROOT / JSON_NAME
    if json_path.exists():
        print(f"[GUARD] JSON already exists: {json_path}")
        sys.exit(1)
    meta = {
        "input_filtered_manifest_path": str(FILTERED_MANIFEST_PATH),
        "checkpoint_path": str(CHECKPOINT_PATH),
        "model_class": "ConvAutoencoder2p5D",
        "model_script": str(MODEL_SCRIPT),
        "device": str(device),
        "max_crops": args.max_crops,
        "max_batches": args.max_batches,
        "batch_size": args.batch_size,
        "manifest_row_count": manifest_info["row_count"],
        "unique_patient_count": manifest_info["unique_patients"],
        "stage2_holdout_row_count": manifest_info["holdout_rows"],
        "excluded_patients_absent": {k: (v == 0) for k, v in manifest_info["excluded_counts"].items()},
        "v2_path_detected": manifest_info["v2_detected"],
        "checkpoint_loaded": True,
        "load_state_dict_status": str(load_result),
        "model_eval_mode": True,
        "no_grad_used": True,
        "forward_batch_count": forward_batch_count,
        "input_shapes": input_shapes,
        "output_shapes": output_shapes,
        "output_type": output_type_global,
        "output_nan_inf_check_pass": output_nan_inf_pass,
        "output_shape_check_pass": output_shape_pass,
        "training_executed": False,
        "backward_executed": False,
        "optimizer_step_executed": False,
        "checkpoint_created": False,
        "threshold_calculated": False,
        "smoke_pass": smoke_pass,
        "blockers": blockers,
        "next_step_recommendation": (
            "Phase 6.3 smoke result review" if smoke_pass else "블로커 해결 후 재실행"
        ),
    }
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[SAVED] JSON: {json_path}")

    # MD
    md_path = OUTPUT_ROOT / MD_NAME
    if md_path.exists():
        print(f"[GUARD] MD already exists: {md_path}")
        sys.exit(1)
    verdict = "통과" if smoke_pass else "미통과"
    md_lines = [
        "# Phase 6.2b S6-A Model Forward Smoke Report",
        "",
        "## 1. Phase 6.2b 목적",
        "filtered S6-A batch `[B,6,96,96]`를 `ConvAutoencoder2p5D`에 실제로 넣어 forward가 되는지 확인한다.",
        "training, backward, checkpoint 생성은 하지 않는다.",
        "",
        "## 2. 사용한 filtered manifest",
        f"- 경로: `{FILTERED_MANIFEST_PATH}`",
        f"- row 수: {manifest_info['row_count']}",
        f"- unique patient 수: {manifest_info['unique_patients']}",
        f"- stage2_holdout row 수: {manifest_info['holdout_rows']}",
        f"- LUNG1-295 row 수: {manifest_info['excluded_counts'].get('LUNG1-295', 0)}",
        f"- LUNG1-415 row 수: {manifest_info['excluded_counts'].get('LUNG1-415', 0)}",
        f"- v2/v2v2 경로 검출: {manifest_info['v2_detected']}",
        "",
        "## 3. 사용한 model/checkpoint",
        f"- model class: `ConvAutoencoder2p5D`",
        f"- model script: `{MODEL_SCRIPT}`",
        f"- checkpoint: `{CHECKPOINT_PATH}`",
        f"- device: `{device}`",
        "",
        "## 4. manifest safety 확인",
        f"- training_manifest_status == not_training_manifest: {manifest_info['not_training']}",
        f"- approval_required_before_training == True: {manifest_info['approval_required']}",
        f"- manifest issues: {manifest_info['issues'] if manifest_info['issues'] else '없음'}",
        "",
        "## 5. input batch 확인",
        f"- max_crops: {args.max_crops}",
        f"- max_batches: {args.max_batches}",
        f"- batch_size: {args.batch_size}",
        f"- input_shapes: {input_shapes}",
        "",
        "## 6. forward output 확인",
        f"- output_type: {output_type_global}",
        f"- output_shapes: {output_shapes}",
        "",
        "## 7. NaN/Inf 확인",
        f"- output_nan_inf_check_pass: {output_nan_inf_pass}",
        f"- output_shape_check_pass: {output_shape_pass}",
        "",
        "## 8. training/backward/checkpoint/threshold 미수행 확인",
        "- training_executed: False",
        "- backward_executed: False",
        "- optimizer_step_executed: False",
        "- checkpoint_created: False",
        "- threshold_calculated: False",
        "- stage2_holdout 접근: 없음",
        "- v2/v2v2 접근: 없음",
        "",
        "## 9. 최종 판정",
        f"**{verdict}**",
        "",
        f"blockers: {blockers if blockers else '없음'}",
        "",
        "## 10. 다음 단계",
    ]
    if smoke_pass:
        md_lines += [
            "- Phase 6.3 smoke result review",
            "- 이후 Phase 6.4 v1/v1 second-stage design freeze",
            "- full training은 별도 승인 전 금지",
        ]
    else:
        md_lines += [
            "- 블로커 해결 후 재실행",
            "- full training은 별도 승인 전 금지",
        ]
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"[SAVED] MD: {md_path}")

    print(f"\n[RESULT] smoke_pass={smoke_pass}")
    if blockers:
        print("[BLOCKERS]")
        for b in blockers:
            print(f"  - {b}")
    print(f"[DONE] output root: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
