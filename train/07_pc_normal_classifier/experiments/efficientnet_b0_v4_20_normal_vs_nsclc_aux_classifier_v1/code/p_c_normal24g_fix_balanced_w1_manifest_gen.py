"""
p_c_normal24g_fix_balanced_w1_manifest_gen.py

24g-fix usable manifest를 train/val split별로 1:1 balanced downsampling + sample_weight reset.
- balanced downsampling: 다수 클래스 random downsampling (seed=42)
- sample_weight: 1.0으로 reset (원본은 original_sample_weight 컬럼에 보존)
- final_test는 처리하지 않음 (train/val class-balanced ablation 전용)

입력:
  p_c_normal24g_fix_train_feature_manifest_usable.csv  (19,716 rows)
  p_c_normal24g_fix_val_feature_manifest_usable.csv    (5,189 rows)

출력:
  outputs/manifests/p_c_normal24g_fix_balanced_w1_feature_manifest/
    p_c_normal24g_fix_balanced_w1_train_manifest.csv   (15,782: 7891+7891, sw=1.0)
    p_c_normal24g_fix_balanced_w1_val_manifest.csv     (4,160: 2080+2080, sw=1.0)
    p_c_normal24g_fix_balanced_w1_manifest_summary.json
    p_c_normal24g_fix_balanced_w1_sample_weight_check.csv
    DONE.json

절대 생성하지 않음:
  - balanced final_test manifest
  - final_test 관련 CSV/JSON
  - final_test prediction/export/metrics
"""

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SEED         = 42
PROJECT_ROOT = Path(__file__).resolve().parents[3]

SRC_DIR = PROJECT_ROOT / "outputs/manifests/p_c_normal24g_fix_zroi_only_feature_manifest"
OUT_DIR = PROJECT_ROOT / "outputs/manifests/p_c_normal24g_fix_balanced_w1_feature_manifest"

SPLITS = {
    "train": SRC_DIR / "p_c_normal24g_fix_train_feature_manifest_usable.csv",
    "val":   SRC_DIR / "p_c_normal24g_fix_val_feature_manifest_usable.csv",
}

OUT_NAMES = {
    "train": "p_c_normal24g_fix_balanced_w1_train_manifest.csv",
    "val":   "p_c_normal24g_fix_balanced_w1_val_manifest.csv",
}

EXPECTED_COUNTS = {
    "train": 15782,
    "val":   4160,
}


def balance_split(df: pd.DataFrame, split_name: str, rng: np.random.Generator) -> pd.DataFrame:
    n0 = int((df["label"] == 0).sum())
    n1 = int((df["label"] == 1).sum())
    target = min(n0, n1)

    df0 = df[df["label"] == 0]
    df1 = df[df["label"] == 1]

    if n0 > target:
        df0 = df0.sample(n=target, random_state=int(rng.integers(1e6)))
    if n1 > target:
        df1 = df1.sample(n=target, random_state=int(rng.integers(1e6)))

    result = pd.concat([df0, df1], ignore_index=True).sample(
        frac=1, random_state=int(rng.integers(1e6))
    ).reset_index(drop=True)
    print(f"  [{split_name}] before: normal={n0} NSCLC={n1} → after: normal={len(df0)} NSCLC={len(df1)} total={len(result)}")
    return result


def reset_sample_weight(df: pd.DataFrame) -> pd.DataFrame:
    """원본 sample_weight를 original_sample_weight로 보존하고 sample_weight를 1.0으로 reset."""
    df = df.copy()
    if "sample_weight" in df.columns:
        df["original_sample_weight"] = df["sample_weight"]
    else:
        df["original_sample_weight"] = None
    df["sample_weight"] = 1.0
    return df


def compute_sample_weight_stats(df: pd.DataFrame, split_name: str) -> dict:
    if "sample_weight" not in df.columns:
        return {
            "split": split_name, "label0_count": 0, "label1_count": 0,
            "label0_sw_mean": None, "label1_sw_mean": None,
            "label0_sw_sum": None, "label1_sw_sum": None,
            "effective_weight_ratio": None,
            "sw_has_nan": None, "sw_has_inf": None, "sw_has_negative": None,
            "sw_all_ones": None, "orig_sw_preserved": False,
            "verdict": "FAIL_no_sample_weight_col",
        }

    sw       = df["sample_weight"].astype(float)
    has_nan  = bool(sw.isna().any())
    has_inf  = bool(np.isinf(sw.fillna(0)).any())
    has_neg  = bool((sw.dropna() < 0).any())
    all_ones = bool((sw.dropna() == 1.0).all())

    df0    = df[df["label"] == 0]
    df1    = df[df["label"] == 1]
    sw0    = df0["sample_weight"].astype(float)
    sw1    = df1["sample_weight"].astype(float)
    n0_sum = float(sw0.sum())
    n1_sum = float(sw1.sum())
    eff_ratio = n0_sum / n1_sum if n1_sum != 0 else float("nan")

    orig_preserved = (
        "original_sample_weight" in df.columns
        and not df["original_sample_weight"].isna().all()
    )

    verdict = "PASS"
    if has_nan or has_inf or has_neg:
        verdict = "FAIL"
    elif not all_ones:
        verdict = "PARTIAL_PASS"
    elif not np.isnan(eff_ratio) and abs(eff_ratio - 1.0) > 0.05:
        verdict = "PARTIAL_PASS"

    return {
        "split":                  split_name,
        "label0_count":           int(len(df0)),
        "label1_count":           int(len(df1)),
        "label0_sw_mean":         round(float(sw0.mean()), 6),
        "label1_sw_mean":         round(float(sw1.mean()), 6),
        "label0_sw_sum":          round(n0_sum, 6),
        "label1_sw_sum":          round(n1_sum, 6),
        "effective_weight_ratio": round(eff_ratio, 6) if not np.isnan(eff_ratio) else "nan",
        "sw_has_nan":             has_nan,
        "sw_has_inf":             has_inf,
        "sw_has_negative":        has_neg,
        "sw_all_ones":            all_ones,
        "orig_sw_preserved":      orig_preserved,
        "verdict":                verdict,
    }


def main():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        print(f"[ABORT] output dir already exists and is not empty: {OUT_DIR}")
        sys.exit(2)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rng     = np.random.default_rng(SEED)
    summary = {
        "timestamp": ts, "seed": SEED,
        "final_test_processed": False,
        "sample_weight_reset_to_1": True,
        "original_sample_weight_preserved": True,
        "splits": {},
    }
    sw_check_rows = []
    errors        = []

    for split_name, src_path in SPLITS.items():
        if not src_path.exists():
            print(f"[ERROR] not found: {src_path}")
            sys.exit(1)
        df       = pd.read_csv(src_path, low_memory=False)
        balanced = balance_split(df, split_name, rng)
        balanced = reset_sample_weight(balanced)

        expected = EXPECTED_COUNTS[split_name]
        if len(balanced) != expected:
            errors.append(f"{split_name}: expected {expected} rows, got {len(balanced)}")

        n0_out = int((balanced["label"] == 0).sum())
        n1_out = int((balanced["label"] == 1).sum())
        if n0_out != n1_out:
            errors.append(f"{split_name}: label imbalance n0={n0_out} n1={n1_out}")

        if not (balanced["sample_weight"].astype(float) == 1.0).all():
            errors.append(f"{split_name}: sample_weight reset failed — not all 1.0")

        orig_ok = (
            "original_sample_weight" in balanced.columns
            and not balanced["original_sample_weight"].isna().all()
        )
        if not orig_ok:
            errors.append(f"{split_name}: original_sample_weight not preserved")

        out_path = OUT_DIR / OUT_NAMES[split_name]
        balanced.to_csv(out_path, index=False)

        sw_stats = compute_sample_weight_stats(balanced, split_name)
        sw_check_rows.append(sw_stats)

        summary["splits"][split_name] = {
            "src_total":          len(df),
            "src_normal":         int((df["label"] == 0).sum()),
            "src_nsclc":          int((df["label"] == 1).sum()),
            "out_total":          len(balanced),
            "out_normal":         n0_out,
            "out_nsclc":          n1_out,
            "ratio":              round(n0_out / max(n1_out, 1), 3),
            "out_path":           str(out_path),
            "sw_verdict":         sw_stats["verdict"],
            "sw_all_ones":        sw_stats["sw_all_ones"],
            "orig_sw_preserved":  sw_stats["orig_sw_preserved"],
        }

    sw_verdicts     = [r["verdict"] for r in sw_check_rows]
    overall_verdict = "PASS"
    if errors or any("FAIL" in v for v in sw_verdicts):
        overall_verdict = "FAIL"
    elif any("PARTIAL_PASS" in v for v in sw_verdicts):
        overall_verdict = "PARTIAL_PASS"

    if sw_check_rows:
        sw_csv_path = OUT_DIR / "p_c_normal24g_fix_balanced_w1_sample_weight_check.csv"
        with open(sw_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(sw_check_rows[0].keys()))
            writer.writeheader()
            writer.writerows(sw_check_rows)

    summary["overall_verdict"]               = overall_verdict
    summary["errors"]                        = errors
    summary["final_test_processed"]          = False
    with open(OUT_DIR / "p_c_normal24g_fix_balanced_w1_manifest_summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(OUT_DIR / "DONE.json", "w") as f:
        json.dump({
            "step":                             "p_c_normal24g_fix_balanced_w1_manifest_gen",
            "verdict":                          overall_verdict,
            "timestamp":                        ts,
            "train_rows":                       summary["splits"].get("train", {}).get("out_total", 0),
            "val_rows":                         summary["splits"].get("val",   {}).get("out_total", 0),
            "final_test_processed":             False,
            "sample_weight_reset_to_1":         True,
            "original_sample_weight_preserved": True,
            "errors":                           len(errors),
        }, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] → {OUT_DIR}  verdict={overall_verdict}")
    print(json.dumps(summary["splits"], ensure_ascii=False, indent=2))
    if errors:
        print(f"[ERRORS] {errors}")
        sys.exit(1)


if __name__ == "__main__":
    main()
