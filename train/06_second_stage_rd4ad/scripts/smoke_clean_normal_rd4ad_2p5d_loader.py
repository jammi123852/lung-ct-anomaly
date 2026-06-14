"""
P1 smoke/preflight script: clean normal crop dataset (rd4ad_2p5d_mw_fixed96_v1)

--preflight : manifest + NPZ sample read-only check (no DataLoader, no model)
--smoke     : --preflight + DataLoader 2-batch load from train/val (no model forward)
              [requires user approval before running]

Absolute forbidden:
  - model forward
  - training / optimizer / epoch loop
  - checkpoint creation
  - scoring
  - file modification / deletion
  - stage2_holdout access
  - pip/conda install
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = (
    BASE_DIR
    / "outputs/second-stage-lesion-refiner-v1/crops_normal"
    / "normal_rd4ad_2p5d_mw_fixed96_v1/manifests"
    / "crop_manifest_normal_rd4ad_2p5d_mw_fixed96_v1.csv"
)
REPORT_DIR = BASE_DIR / "outputs/second-stage-lesion-refiner-v1/reports"

EXPECTED = {
    "total": 18100,
    "train": 14500,
    "val": 1800,
    "test": 1800,
    "patients": 362,
    "shape": (6, 96, 96),
    "dtype": "float32",
    "image_key": "image",
    "split_col": "normal_split",
}

FORBIDDEN_PATTERNS = ["stage2_holdout", "v2"]

SAMPLE_PER_SPLIT = 5


# ─────────────────────────────────────────────
# Guards
# ─────────────────────────────────────────────

def _check_forbidden(path: str):
    for pat in FORBIDDEN_PATTERNS:
        if pat in str(path):
            print(f"[GUARD ERROR] Forbidden pattern '{pat}' in path: {path}")
            sys.exit(1)


# ─────────────────────────────────────────────
# Preflight
# ─────────────────────────────────────────────

def run_preflight() -> dict:
    try:
        import pandas as pd
    except ImportError:
        print("[ERROR] pandas not available. Run with ~/ai_env activated.")
        sys.exit(1)

    results = {
        "timestamp": datetime.now().isoformat(),
        "mode": "preflight",
        "manifest_path": str(MANIFEST_PATH),
        "checks": {},
        "overall": "PASS",
    }
    checks = results["checks"]

    # 1. manifest existence
    if not MANIFEST_PATH.exists():
        checks["manifest_exists"] = {"status": "FAIL", "detail": str(MANIFEST_PATH)}
        results["overall"] = "FAIL"
        _print_result(results)
        return results
    checks["manifest_exists"] = {"status": "PASS", "path": str(MANIFEST_PATH)}
    print(f"[PREFLIGHT OK] manifest exists: {MANIFEST_PATH}")

    # 2. load manifest
    df = pd.read_csv(MANIFEST_PATH)
    print(f"[PREFLIGHT OK] manifest loaded: {len(df)} rows")

    # 3. total count
    total = len(df)
    if total != EXPECTED["total"]:
        checks["total_count"] = {"status": "FAIL", "expected": EXPECTED["total"], "got": total}
        results["overall"] = "FAIL"
    else:
        checks["total_count"] = {"status": "PASS", "count": total}
    print(f"[PREFLIGHT {'OK' if checks['total_count']['status']=='PASS' else 'FAIL'}] total crops: {total}")

    # 4. split column
    if EXPECTED["split_col"] not in df.columns:
        checks["split_col"] = {"status": "FAIL", "detail": f"{EXPECTED['split_col']} not in columns"}
        results["overall"] = "FAIL"
    else:
        checks["split_col"] = {"status": "PASS", "col": EXPECTED["split_col"]}
    print(f"[PREFLIGHT {'OK' if checks['split_col']['status']=='PASS' else 'FAIL'}] split column: {EXPECTED['split_col']}")

    # 5. split distribution
    split_dist = df[EXPECTED["split_col"]].value_counts().to_dict()
    split_ok = True
    for split_name, exp_count in [("train", EXPECTED["train"]), ("val", EXPECTED["val"]), ("test", EXPECTED["test"])]:
        got = split_dist.get(split_name, 0)
        if got != exp_count:
            split_ok = False
            results["overall"] = "FAIL"
        print(f"[PREFLIGHT {'OK' if got==exp_count else 'FAIL'}] {split_name}: {got} (expected {exp_count})")
    checks["split_distribution"] = {"status": "PASS" if split_ok else "FAIL", "distribution": split_dist}

    # 6. patient count
    patient_count = df["patient_id"].nunique()
    if patient_count != EXPECTED["patients"]:
        checks["patient_count"] = {"status": "FAIL", "expected": EXPECTED["patients"], "got": patient_count}
        results["overall"] = "FAIL"
    else:
        checks["patient_count"] = {"status": "PASS", "count": patient_count}
    print(f"[PREFLIGHT {'OK' if checks['patient_count']['status']=='PASS' else 'FAIL'}] patients: {patient_count}")

    # 7. patient overlap between splits
    overlap_ok = True
    split_ids = {
        split: set(df[df[EXPECTED["split_col"]] == split]["patient_id"].unique())
        for split in ["train", "val", "test"]
    }
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = split_ids[a] & split_ids[b]
        if overlap:
            overlap_ok = False
            results["overall"] = "FAIL"
            print(f"[PREFLIGHT FAIL] {a}/{b} patient overlap: {len(overlap)} patients")
        else:
            print(f"[PREFLIGHT OK] {a}/{b} patient overlap: 0")
    checks["patient_overlap"] = {"status": "PASS" if overlap_ok else "FAIL"}

    # 8. crop_path column
    if "crop_path" not in df.columns:
        checks["crop_path_col"] = {"status": "FAIL", "detail": "crop_path column missing"}
        results["overall"] = "FAIL"
        print("[PREFLIGHT FAIL] crop_path column missing")
    else:
        checks["crop_path_col"] = {"status": "PASS"}
        print("[PREFLIGHT OK] crop_path column exists")

    # 9. NPZ sample check per split
    npz_results = {}
    all_npz_ok = True
    for split_name in ["train", "val", "test"]:
        split_df = df[df[EXPECTED["split_col"]] == split_name].head(SAMPLE_PER_SPLIT)
        split_npz_ok = True
        sample_list = []
        for _, row in split_df.iterrows():
            cp = str(row.get("crop_path", ""))
            _check_forbidden(cp)
            cp_path = Path(cp)
            sample = {"path": cp_path.name, "crop_path": cp}
            if not cp_path.exists():
                sample["status"] = "FAIL"
                sample["error"] = "file not found"
                split_npz_ok = False
                all_npz_ok = False
            else:
                try:
                    data = np.load(cp_path)
                    keys = list(data.keys())
                    sample["keys"] = keys
                    if EXPECTED["image_key"] not in data:
                        sample["status"] = "FAIL"
                        sample["error"] = f"key '{EXPECTED['image_key']}' not found"
                        split_npz_ok = False
                        all_npz_ok = False
                    else:
                        arr = data[EXPECTED["image_key"]]
                        shape_ok = tuple(arr.shape) == EXPECTED["shape"]
                        dtype_ok = str(arr.dtype) == EXPECTED["dtype"]
                        range_ok = float(arr.min()) >= -1e-4 and float(arr.max()) <= 1.0 + 1e-4
                        finite_ok = bool(np.isfinite(arr).all())
                        ok = shape_ok and dtype_ok and range_ok and finite_ok
                        sample.update({
                            "status": "PASS" if ok else "FAIL",
                            "shape": list(arr.shape),
                            "dtype": str(arr.dtype),
                            "min": float(arr.min()),
                            "max": float(arr.max()),
                            "has_nan_inf": not finite_ok,
                            "shape_ok": shape_ok,
                            "dtype_ok": dtype_ok,
                            "range_ok": range_ok,
                            "finite_ok": finite_ok,
                        })
                        if not ok:
                            split_npz_ok = False
                            all_npz_ok = False
                except Exception as e:
                    sample["status"] = "FAIL"
                    sample["error"] = str(e)
                    split_npz_ok = False
                    all_npz_ok = False
            sample_list.append(sample)
            status_str = sample.get("status", "FAIL")
            print(f"[PREFLIGHT {status_str}] {split_name} NPZ: {cp_path.name} "
                  f"keys={sample.get('keys', '?')} shape={sample.get('shape', '?')} "
                  f"dtype={sample.get('dtype', '?')} "
                  f"min={sample.get('min', '?'):.4f} max={sample.get('max', '?'):.4f}" if "min" in sample else
                  f"[PREFLIGHT {status_str}] {split_name} NPZ: {cp_path.name} error={sample.get('error', '?')}")
        npz_results[split_name] = {
            "status": "PASS" if split_npz_ok else "FAIL",
            "samples": sample_list,
        }
        if not split_npz_ok:
            results["overall"] = "FAIL"

    checks["npz_samples"] = {"status": "PASS" if all_npz_ok else "FAIL", "per_split": npz_results}

    # 10. confirm no smoke/train/model execution
    checks["no_model_forward"] = {"status": "PASS", "detail": "preflight-only, no DataLoader, no model"}
    checks["no_training"] = {"status": "PASS", "detail": "preflight-only"}
    checks["no_checkpoint"] = {"status": "PASS", "detail": "preflight-only"}

    # summary
    results["summary"] = {
        "total_crops": total,
        "split_distribution": split_dist,
        "patients": patient_count,
        "npz_key_confirmed": EXPECTED["image_key"],
        "shape_confirmed": list(EXPECTED["shape"]),
        "dtype_confirmed": EXPECTED["dtype"],
        "all_preflight_ok": results["overall"] == "PASS",
    }

    _print_result(results)
    return results


# ─────────────────────────────────────────────
# Smoke (user approval required before running)
# ─────────────────────────────────────────────

def run_smoke() -> dict:
    """DataLoader 2-batch smoke. No model forward."""
    try:
        import pandas as pd
        import torch
        from torch.utils.data import Dataset, DataLoader
    except ImportError as e:
        print(f"[ERROR] Required package missing: {e}. Run with ~/ai_env activated.")
        sys.exit(1)

    # run preflight first
    preflight_result = run_preflight()
    if preflight_result["overall"] != "PASS":
        print("[SMOKE ABORT] Preflight failed. Fix preflight issues before smoke.")
        sys.exit(1)

    df = __import__("pandas").read_csv(MANIFEST_PATH)

    class _MinimalNormalDataset(Dataset):
        def __init__(self, df, split, image_key="image"):
            self.df = df[df["normal_split"] == split].reset_index(drop=True)
            self.image_key = image_key

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            cp = Path(str(row["crop_path"]))
            _check_forbidden(str(cp))
            data = np.load(cp)
            arr = data[self.image_key].astype(np.float32)
            return {"image": torch.from_numpy(arr)}

    results = {
        "timestamp": datetime.now().isoformat(),
        "mode": "smoke",
        "checks": {},
        "overall": "PASS",
    }

    for split_name in ["train", "val"]:
        ds = _MinimalNormalDataset(df, split_name)
        loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
        batches_loaded = 0
        for batch in loader:
            img = batch["image"]
            assert img.shape[1:] == torch.Size([6, 96, 96]), f"Unexpected shape: {img.shape}"
            assert img.dtype == torch.float32, f"Unexpected dtype: {img.dtype}"
            batches_loaded += 1
            print(f"[SMOKE OK] {split_name} batch {batches_loaded}: shape={list(img.shape)} dtype={img.dtype} "
                  f"min={img.min():.4f} max={img.max():.4f}")
            if batches_loaded >= 2:
                break
        results["checks"][f"{split_name}_smoke"] = {
            "status": "PASS",
            "batches_loaded": batches_loaded,
            "batch_shape": [list(img.shape)],
        }

    results["checks"]["no_model_forward"] = {"status": "PASS", "detail": "smoke-only, no model"}
    _print_result(results)
    return results


# ─────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────

def _print_result(results: dict):
    overall = results.get("overall", "UNKNOWN")
    print(f"\n{'='*60}")
    print(f"[RESULT] overall: {overall}")
    print(f"{'='*60}\n")


def write_reports(results: dict):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    mode = results.get("mode", "preflight")

    if mode == "smoke":
        json_path = REPORT_DIR / "clean_normal_rd4ad_2p5d_loader_smoke.json"
        md_path = REPORT_DIR / "clean_normal_rd4ad_2p5d_loader_smoke.md"
    else:
        json_path = REPORT_DIR / "clean_normal_rd4ad_2p5d_loader_preflight.json"
        md_path = REPORT_DIR / "clean_normal_rd4ad_2p5d_loader_preflight.md"

    # overwrite guard: 기존 report 보존
    for p in [json_path, md_path]:
        if p.exists():
            print(f"[GUARD ERROR] Report already exists, refusing to overwrite: {p}")
            print("[GUARD] Delete the existing report manually if you want to regenerate.")
            sys.exit(1)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[REPORT] JSON: {json_path}")

    overall = results.get("overall", "UNKNOWN")
    summary = results.get("summary", {})
    checks = results.get("checks", {})

    report_title = (
        "Clean Normal RD4AD 2.5D Loader Smoke Report"
        if mode == "smoke"
        else "Clean Normal RD4AD 2.5D Loader Preflight Report"
    )
    md_lines = [
        f"# {report_title}",
        f"",
        f"- 생성일시: {results.get('timestamp', '')}",
        f"- 모드: `{mode}`",
        f"- 검토 판정: **{overall}**",
        f"",
        f"## 1. Manifest 요약",
        f"",
        f"| 항목 | 값 |",
        f"|------|-----|",
        f"| manifest 경로 | `{results.get('manifest_path', '')}` |",
        f"| total crops | {summary.get('total_crops', '?')} |",
        f"| train | {summary.get('split_distribution', {}).get('train', '?')} |",
        f"| val | {summary.get('split_distribution', {}).get('val', '?')} |",
        f"| test | {summary.get('split_distribution', {}).get('test', '?')} |",
        f"| patients | {summary.get('patients', '?')} |",
        f"",
        f"## 2. NPZ 확인 결과",
        f"",
        f"| 항목 | 값 |",
        f"|------|-----|",
        f"| 실제 사용 key | `{summary.get('npz_key_confirmed', '?')}` |",
        f"| shape | {summary.get('shape_confirmed', '?')} |",
        f"| dtype | {summary.get('dtype_confirmed', '?')} |",
        f"",
        f"## 3. 체크 항목",
        f"",
        f"| 체크 | 결과 |",
        f"|------|------|",
    ]

    for key, val in checks.items():
        status = val.get("status", "?") if isinstance(val, dict) else "?"
        md_lines.append(f"| {key} | {status} |")

    npz_checks = checks.get("npz_samples", {}).get("per_split", {})
    if npz_checks:
        md_lines += ["", "## 4. NPZ 샘플 상세", ""]
        for split_name, split_data in npz_checks.items():
            md_lines.append(f"### {split_name}")
            md_lines.append(f"")
            md_lines.append(f"| 파일 | key | shape | dtype | min | max | NaN/Inf | 결과 |")
            md_lines.append(f"|------|-----|-------|-------|-----|-----|---------|------|")
            for s in split_data.get("samples", []):
                status = s.get("status", "?")
                fname = s.get("path", "?")
                keys = s.get("keys", "?")
                shape = s.get("shape", "?")
                dtype = s.get("dtype", "?")
                mn = f"{s.get('min', 0):.4f}" if "min" in s else "?"
                mx = f"{s.get('max', 0):.4f}" if "max" in s else "?"
                nan = s.get("has_nan_inf", "?")
                md_lines.append(f"| {fname} | {keys} | {shape} | {dtype} | {mn} | {mx} | {nan} | {status} |")
            md_lines.append("")

    md_lines += [
        f"## 5. 안전 확인",
        f"",
        f"- 학습 실행: 미실행 ✓",
        f"- model forward: 미실행 ✓",
        f"- checkpoint 생성: 미실행 ✓",
        f"- scoring 실행: 미실행 ✓",
        f"- NPZ 파일 수정/삭제: 없음 ✓",
        f"- stage2_holdout 접근: 없음 ✓",
        f"",
        f"## 6. 다음 단계",
        f"",
        f"```bash",
        f"# --smoke 실행 (사용자 승인 후)",
        f"source ~/ai_env/bin/activate && \\",
        f"  python scripts/smoke_clean_normal_rd4ad_2p5d_loader.py --smoke",
        f"```",
    ]

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"[REPORT] MD: {md_path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="P1 preflight/smoke for clean normal rd4ad 2p5d crops"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true", help="manifest + NPZ sample check only")
    group.add_argument("--smoke", action="store_true", help="preflight + DataLoader 2-batch (user approval required)")
    parser.add_argument("--write-report", action="store_true", help="write MD/JSON report")
    args = parser.parse_args()

    if args.smoke:
        results = run_smoke()
    else:
        results = run_preflight()

    if args.write_report:
        write_reports(results)

    sys.exit(0 if results.get("overall") == "PASS" else 1)


if __name__ == "__main__":
    main()
