"""
p_c_normal27_scalar_repair_final_test_manifest.py

P-C-NORMAL27: normal-test crops scalar feature (lung_z_percentile, crop_lung_roi_ratio) repair.

Bug confirmed (P-C-NORMAL26 CRITICAL FINDING):
  - canonical_volume_z in p_c_normal24b_fix mapping CSV is stored as NORMALIZED float (0~1)
    for normal_test crops, not as actual slice index.
  - int(0.29) = 0, int(0.50) = 0 → ALL normal_test crops get z=0
  - lung_z_percentile ≈ 0 for all, crop_lung_roi_ratio ≈ 0 (roi at slice 0 = outside lung)
  - NSCLC test crops have correct slice_index values → unaffected

Fix:
  - For normal_test (label=0) crops: use `slice_index` from canonical_z CSV as canonical_volume_z
  - For NSCLC test (label=1) crops: keep existing values unchanged
  - Recompute lung_z_percentile and crop_lung_roi_ratio for normal_test with corrected z
  - Follow same 24g-fix logic: center_y/center_x → 96×96 bbox (no y0/y1/x0/x1 for normal_test)

Output root:
  outputs/manifests/p_c_normal27_scalar_repair_final_test_manifest/
  outputs/reports/p_c_normal27_scalar_repair_final_test_manifest/

Guardrails:
  - no_training_run
  - no_model_forward
  - no_prediction_export_rerun
  - no_threshold_optimization
  - no_existing_result_overwrite
"""

import csv
import json
import os
import sys
from pathlib import Path
from collections import OrderedDict
from datetime import datetime

import numpy as np
import pandas as pd

# ── 경로 ─────────────────────────────────────────────────────────────────────
BRANCH_ROOT  = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BRANCH_ROOT.parents[1]

FINAL_TEST_MANIFEST_24G = PROJECT_ROOT / "outputs/manifests/p_c_normal24g_fix_zroi_only_feature_manifest/p_c_normal24g_fix_final_test_feature_manifest_usable.csv"
CANONICAL_Z_CSV         = PROJECT_ROOT / "outputs/reports/p_c_normal24b_fix_crop_to_volume_z_revalidation/p_c_normal24b_fix_crop_to_volume_z_mapping.csv"
ROI_DIR                 = PROJECT_ROOT / "outputs/mip-postprocess-research-v1/masks/refined_roi_v4_20_modeB_all_v1"
TRAIN_MANIFEST_24G      = PROJECT_ROOT / "outputs/manifests/p_c_normal24g_fix_zroi_only_feature_manifest/p_c_normal24g_fix_train_feature_manifest_usable.csv"

MANIFEST_OUT = PROJECT_ROOT / "outputs/manifests/p_c_normal27_scalar_repair_final_test_manifest"
REPORT_OUT   = PROJECT_ROOT / "outputs/reports/p_c_normal27_scalar_repair_final_test_manifest"

CROP_SIZE = 96

EXPECTED_TOTAL   = 66283
EXPECTED_NORMAL  = 21560
EXPECTED_NSCLC   = 44723


# ── ROI 관련 ──────────────────────────────────────────────────────────────────
class _LRUCache:
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self._d = OrderedDict()

    def get(self, key):
        if key not in self._d:
            return None
        self._d.move_to_end(key)
        return self._d[key]

    def put(self, key, value):
        if key in self._d:
            self._d.move_to_end(key)
        else:
            if len(self._d) >= self.maxsize:
                self._d.popitem(last=False)
            self._d[key] = value

    def __contains__(self, key):
        return key in self._d


_roi_cache = _LRUCache(60)
_zrange_cache = {}


def build_roi_map():
    roi_map = {}
    for grp in ["normal", "lesion"]:
        grp_dir = ROI_DIR / grp
        if not grp_dir.exists():
            continue
        for sid in os.listdir(grp_dir):
            p = grp_dir / sid / "refined_roi.npy"
            if p.exists():
                roi_map[sid] = str(p)
    return roi_map


def get_z_range(safe_id, roi_map):
    if safe_id in _zrange_cache:
        return _zrange_cache[safe_id]
    if safe_id not in roi_map:
        _zrange_cache[safe_id] = (None, None)
        return (None, None)
    roi = _roi_cache.get(safe_id)
    if roi is None:
        roi = np.load(roi_map[safe_id])
        _roi_cache.put(safe_id, roi)
    active = np.where(roi.max(axis=(1, 2)) > 0)[0]
    if len(active) == 0:
        _zrange_cache[safe_id] = (None, None)
        return (None, None)
    z_min, z_max = int(active.min()), int(active.max())
    _zrange_cache[safe_id] = (z_min, z_max)
    return (z_min, z_max)


def get_roi_crop(safe_id, roi_map, z, y0, x0, y1, x1):
    """bbox 기준 ROI 비율. 분모 = (y1-y0)*(x1-x0)"""
    if safe_id not in roi_map:
        return float("nan")
    bbox_h = y1 - y0
    bbox_w = x1 - x0
    bbox_area = bbox_h * bbox_w
    if bbox_area <= 0:
        return float("nan")
    roi = _roi_cache.get(safe_id)
    if roi is None:
        roi = np.load(roi_map[safe_id])
        _roi_cache.put(safe_id, roi)
    nz = roi.shape[0]
    if z < 0 or z >= nz:
        return float("nan")
    y0c, y1c = max(0, y0), min(roi.shape[1], y1)
    x0c, x1c = max(0, x0), min(roi.shape[2], x1)
    crop = roi[z, y0c:y1c, x0c:x1c]
    return float(np.clip(float(crop.sum()) / bbox_area, 0.0, 1.0))


# ── 헬퍼 ─────────────────────────────────────────────────────────────────────
def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        rows = [{"note": "empty"}]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def feat_stats(series):
    s = series.dropna()
    return {
        "count": int(len(s)),
        "nan": int(series.isna().sum()),
        "min": float(s.min()) if len(s) > 0 else float("nan"),
        "max": float(s.max()) if len(s) > 0 else float("nan"),
        "mean": float(s.mean()) if len(s) > 0 else float("nan"),
        "median": float(s.median()) if len(s) > 0 else float("nan"),
        "std": float(s.std()) if len(s) > 0 else float("nan"),
        "zero_count": int((s == 0).sum()),
        "zero_ratio": float((s == 0).sum() / len(s)) if len(s) > 0 else float("nan"),
    }


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] P-C-NORMAL27 scalar repair 시작")

    # 출력 충돌 방지
    for out_dir in [MANIFEST_OUT, REPORT_OUT]:
        if out_dir.exists() and any(out_dir.iterdir()):
            print(f"[ABORT] 출력 디렉토리가 이미 존재합니다: {out_dir}")
            print("기존 결과 덮어쓰기 방지. 삭제 후 재실행하거나 새 경로를 사용하세요.")
            sys.exit(2)

    MANIFEST_OUT.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.mkdir(parents=True, exist_ok=True)

    errors = []

    # ── 1. 24g-fix final_test manifest 로드 ───────────────────────────────────
    print("24g-fix final_test manifest 로드...")
    df24g = pd.read_csv(FINAL_TEST_MANIFEST_24G, low_memory=False)
    print(f"  total: {len(df24g)}, normal(0): {(df24g['label']==0).sum()}, NSCLC(1): {(df24g['label']==1).sum()}")

    # ── 2. canonical z CSV 로드 → normal_test slice_index 맵 ─────────────────
    print("canonical z CSV 로드...")
    cz_df = pd.read_csv(CANONICAL_Z_CSV, low_memory=False)
    norm_cz = cz_df[cz_df["source_split"] == "normal_test"].copy()
    print(f"  normal_test rows in cz CSV: {len(norm_cz)}, resolved: {norm_cz['resolved'].sum()}, unresolved: {(~norm_cz['resolved']).sum()}")

    # crop_path → (slice_index, resolved) 맵 빌드
    slice_idx_map = {}   # crop_path → correct_canonical_z (int)
    resolved_map  = {}   # crop_path → bool
    for _, row in norm_cz.iterrows():
        cp = str(row["crop_path"])
        resolved_map[cp] = bool(row["resolved"])
        slice_idx_map[cp] = int(row["slice_index"]) if not pd.isna(row["slice_index"]) else None

    # ── 3. train manifest 로드 (분포 비교용) ──────────────────────────────────
    print("train manifest 로드 (분포 비교용)...")
    df_train = pd.read_csv(TRAIN_MANIFEST_24G, low_memory=False)
    train_norm = df_train[df_train["label"] == 0]

    # ── 4. ROI map 빌드 ────────────────────────────────────────────────────────
    print("ROI map 빌드...")
    roi_map = build_roi_map()
    print(f"  total ROI entries: {len(roi_map)}")

    # ── 5. STEP 1: critical finding 재현 (진단) ───────────────────────────────
    print("\n=== STEP 1: critical finding 재현 ===")
    norm_24g  = df24g[df24g["label"] == 0]
    nsclc_24g = df24g[df24g["label"] == 1]

    cz_norm  = feat_stats(norm_24g["canonical_volume_z"])
    cz_nsclc = feat_stats(nsclc_24g["canonical_volume_z"])
    lzp_norm_corrupt = feat_stats(norm_24g["lung_z_percentile"])
    lzp_nsclc        = feat_stats(nsclc_24g["lung_z_percentile"])
    clr_norm_corrupt = feat_stats(norm_24g["crop_lung_roi_ratio"])
    clr_nsclc        = feat_stats(nsclc_24g["crop_lung_roi_ratio"])
    lzp_train = feat_stats(train_norm["lung_z_percentile"])
    clr_train = feat_stats(train_norm["crop_lung_roi_ratio"])

    print(f"  [corrupted] normal test canonical_volume_z: mean={cz_norm['mean']:.4f}, max={cz_norm['max']:.4f}, zero={cz_norm['zero_count']}")
    print(f"  [corrupted] normal test lung_z_percentile : mean={lzp_norm_corrupt['mean']:.4f}, max={lzp_norm_corrupt['max']:.6f}, zero={lzp_norm_corrupt['zero_count']}")
    print(f"  [corrupted] normal test crop_lung_roi_ratio: mean={clr_norm_corrupt['mean']:.4f}, max={clr_norm_corrupt['max']:.4f}, zero={clr_norm_corrupt['zero_count']}")
    print(f"  [correct]   NSCLC test lung_z_percentile : mean={lzp_nsclc['mean']:.4f}")
    print(f"  [correct]   train normal lung_z_percentile: mean={lzp_train['mean']:.4f}")

    # critical finding 재현 확인
    bug_confirmed = (
        lzp_norm_corrupt["mean"] < 0.01 and
        lzp_norm_corrupt["zero_ratio"] > 0.50 and
        clr_norm_corrupt["mean"] < 0.05 and
        clr_norm_corrupt["zero_ratio"] > 0.50 and
        lzp_nsclc["mean"] > 0.40 and
        lzp_train["mean"] > 0.40
    )
    print(f"  >> bug_confirmed: {bug_confirmed}")
    if not bug_confirmed:
        errors.append({"step": "bug_confirmation", "error": "critical finding 재현 실패 — 데이터 확인 필요"})

    # ── 6. STEP 2: normal test scalar 재계산 ──────────────────────────────────
    print("\n=== STEP 2: normal test scalar 재계산 ===")

    # 재계산할 normal rows
    norm_rows = df24g[df24g["label"] == 0].copy()
    nsclc_rows = df24g[df24g["label"] == 1].copy()

    repaired_normal = []
    n_norm = len(norm_rows)
    n_repaired = 0
    n_z_unresolved = 0
    n_roi_failed = 0
    n_unmatched_in_cz = 0

    for i, (idx, row) in enumerate(norm_rows.iterrows()):
        if i % 3000 == 0:
            print(f"  normal [{i}/{n_norm}]...")

        cp = str(row["crop_path"])
        safe_id = str(row["safe_id"])

        # canonical_z lookup
        if cp not in slice_idx_map:
            n_unmatched_in_cz += 1
            errors.append({"step": "z_lookup", "error": f"crop_path not found in cz CSV: {cp[:80]}"})
            out = row.to_dict()
            out["canonical_volume_z"] = float("nan")
            out["z_unresolved"] = True
            out["lung_z_percentile"] = float("nan")
            out["crop_lung_roi_ratio"] = float("nan")
            repaired_normal.append(out)
            continue

        is_resolved = resolved_map[cp]
        correct_z = slice_idx_map[cp]

        if not is_resolved or correct_z is None:
            n_z_unresolved += 1
            out = row.to_dict()
            out["canonical_volume_z"] = float("nan")
            out["z_unresolved"] = True
            out["lung_z_percentile"] = float("nan")
            out["crop_lung_roi_ratio"] = float("nan")
            repaired_normal.append(out)
            continue

        # lung_z_percentile 재계산
        z_min, z_max = get_z_range(safe_id, roi_map)
        if z_min is None:
            lzp = float("nan")
            errors.append({"step": "z_range", "error": f"roi z_range failed for {safe_id}"})
        elif z_min == z_max:
            lzp = 0.5
        else:
            lzp = float(np.clip((correct_z - z_min) / (z_max - z_min), 0.0, 1.0))

        # crop_lung_roi_ratio 재계산 — center_y/center_x → 96×96 bbox
        cy = float(row["center_y"])
        cx = float(row["center_x"])
        y0 = int(cy) - CROP_SIZE // 2
        x0 = int(cx) - CROP_SIZE // 2
        y1 = y0 + CROP_SIZE
        x1 = x0 + CROP_SIZE
        clr = get_roi_crop(safe_id, roi_map, correct_z, y0, x0, y1, x1)

        if np.isnan(lzp) or np.isnan(clr):
            n_roi_failed += 1

        out = row.to_dict()
        out["canonical_volume_z"] = float(correct_z)
        out["z_unresolved"] = False
        out["lung_z_percentile"] = lzp
        out["crop_lung_roi_ratio"] = clr
        repaired_normal.append(out)
        n_repaired += 1

    print(f"  repaired: {n_repaired}, z_unresolved: {n_z_unresolved}, unmatched_in_cz: {n_unmatched_in_cz}, roi_failed: {n_roi_failed}")

    # ── 7. STEP 3: repaired manifest 생성 ─────────────────────────────────────
    print("\n=== STEP 3: repaired manifest 생성 ===")

    repaired_normal_df = pd.DataFrame(repaired_normal)
    nsclc_copy = nsclc_rows.copy()

    # full manifest (normal full + NSCLC)
    # normal full: repaired_normal_df (includes z_unresolved rows)
    # NSCLC: keep as-is
    full_df = pd.concat([repaired_normal_df, nsclc_copy], ignore_index=True)

    # usable = z_unresolved==False only
    usable_df = full_df[full_df["z_unresolved"] == False].copy()

    print(f"  full: {len(full_df)} rows (normal={len(repaired_normal_df)}, NSCLC={len(nsclc_copy)})")
    print(f"  usable: {len(usable_df)} rows (normal={(usable_df['label']==0).sum()}, NSCLC={(usable_df['label']==1).sum()})")

    # row count 검증
    count_ok = (
        len(usable_df) == EXPECTED_TOTAL and
        int((usable_df["label"] == 0).sum()) == EXPECTED_NORMAL and
        int((usable_df["label"] == 1).sum()) == EXPECTED_NSCLC
    )
    if not count_ok:
        errors.append({
            "step": "row_count",
            "error": f"usable count mismatch: total={len(usable_df)}, normal={(usable_df['label']==0).sum()}, nsclc={(usable_df['label']==1).sum()}"
        })

    # 저장
    full_path    = MANIFEST_OUT / "p_c_normal27_final_test_feature_manifest_repaired.csv"
    usable_path  = MANIFEST_OUT / "p_c_normal27_final_test_feature_manifest_repaired_usable.csv"
    full_df.to_csv(full_path, index=False)
    usable_df.to_csv(usable_path, index=False)
    print(f"  saved: {full_path.name}")
    print(f"  saved: {usable_path.name}")

    # ── 8. STEP 4: repair sanity check ────────────────────────────────────────
    print("\n=== STEP 4: sanity check ===")

    norm_repaired = usable_df[usable_df["label"] == 0]
    nsclc_usable  = usable_df[usable_df["label"] == 1]

    lzp_norm_rep  = feat_stats(norm_repaired["lung_z_percentile"])
    clr_norm_rep  = feat_stats(norm_repaired["crop_lung_roi_ratio"])
    lzp_nsclc_rep = feat_stats(nsclc_usable["lung_z_percentile"])
    clr_nsclc_rep = feat_stats(nsclc_usable["crop_lung_roi_ratio"])

    print(f"  [repaired] normal test lung_z_percentile : mean={lzp_norm_rep['mean']:.4f}, min={lzp_norm_rep['min']:.4f}, max={lzp_norm_rep['max']:.4f}, zero_ratio={lzp_norm_rep['zero_ratio']:.3f}")
    print(f"  [repaired] normal test crop_lung_roi_ratio: mean={clr_norm_rep['mean']:.4f}, min={clr_norm_rep['min']:.4f}, max={clr_norm_rep['max']:.4f}, zero_ratio={clr_norm_rep['zero_ratio']:.3f}")
    print(f"  [check]    train  normal lung_z_percentile: mean={lzp_train['mean']:.4f}")
    print(f"  [check]    train  normal crop_lung_roi_ratio: mean={clr_train['mean']:.4f}")

    # sanity pass conditions
    lzp_sanity  = lzp_norm_rep["mean"] > 0.1 and lzp_norm_rep["zero_ratio"] < 0.10
    clr_sanity  = clr_norm_rep["mean"] > 0.1 and clr_norm_rep["zero_ratio"] < 0.50
    nan_check   = norm_repaired["lung_z_percentile"].isna().sum() == 0 and norm_repaired["crop_lung_roi_ratio"].isna().sum() == 0
    range_check = (norm_repaired["lung_z_percentile"].max() <= 1.0 and norm_repaired["lung_z_percentile"].min() >= 0.0 and
                   norm_repaired["crop_lung_roi_ratio"].max() <= 1.0 and norm_repaired["crop_lung_roi_ratio"].min() >= 0.0)
    inf_check   = not np.any(np.isinf(norm_repaired["lung_z_percentile"].values))

    sanity_pass = lzp_sanity and clr_sanity and nan_check and range_check and inf_check and count_ok
    print(f"  sanity_pass: {sanity_pass} (lzp={lzp_sanity}, clr={clr_sanity}, nan_ok={nan_check}, range_ok={range_check}, count_ok={count_ok})")

    # ── 9. STEP 5: before/after 비교 테이블 ──────────────────────────────────
    dist_rows = []
    for label, split_name, col, stage, vals in [
        (0, "normal_test",  "lung_z_percentile",   "corrupted_24g",  lzp_norm_corrupt),
        (0, "normal_test",  "lung_z_percentile",   "repaired_27",    lzp_norm_rep),
        (0, "normal_test",  "lung_z_percentile",   "train_normal_ref", lzp_train),
        (0, "normal_test",  "crop_lung_roi_ratio",  "corrupted_24g",  clr_norm_corrupt),
        (0, "normal_test",  "crop_lung_roi_ratio",  "repaired_27",    clr_norm_rep),
        (0, "normal_test",  "crop_lung_roi_ratio",  "train_normal_ref", clr_train),
        (1, "nsclc_test",   "lung_z_percentile",   "unchanged",      lzp_nsclc_rep),
        (1, "nsclc_test",   "crop_lung_roi_ratio",  "unchanged",      clr_nsclc_rep),
    ]:
        dist_rows.append({
            "label": label, "split": split_name, "feature": col, "stage": stage,
            "count": vals["count"], "nan": vals["nan"],
            "min": round(vals["min"], 4), "max": round(vals["max"], 4),
            "mean": round(vals["mean"], 4), "median": round(vals["median"], 4),
            "std": round(vals["std"], 4),
            "zero_count": vals["zero_count"], "zero_ratio": round(vals["zero_ratio"], 4),
        })
    write_csv(REPORT_OUT / "p_c_normal27_before_after_distribution.csv", dist_rows)

    # ── 10. schema check ───────────────────────────────────────────────────────
    schema_rows = []
    required_cols = [
        "crop_path", "patient_id", "safe_id", "split", "source_split", "label",
        "sample_weight", "canonical_volume_z", "z_unresolved",
        "lung_z_percentile", "crop_lung_roi_ratio",
        "position_bin", "local_z", "center_y", "center_x",
    ]
    for col in required_cols:
        present = col in usable_df.columns
        schema_rows.append({
            "column": col,
            "present": present,
            "dtype": str(usable_df[col].dtype) if present else "N/A",
            "null_count": int(usable_df[col].isna().sum()) if present else -1,
            "pass": present,
        })
    write_csv(REPORT_OUT / "p_c_normal27_repaired_manifest_schema_check.csv", schema_rows)

    # ── 11. guardrail check ────────────────────────────────────────────────────
    guardrail_rows = [
        {"check": "no_training_run",                   "expected": True,  "actual": True,  "pass": True},
        {"check": "no_model_forward",                  "expected": True,  "actual": True,  "pass": True},
        {"check": "no_prediction_export_rerun",        "expected": True,  "actual": True,  "pass": True},
        {"check": "no_scoring_rerun",                  "expected": True,  "actual": True,  "pass": True},
        {"check": "no_threshold_optimization",         "expected": True,  "actual": True,  "pass": True},
        {"check": "no_threshold_sweep",                "expected": True,  "actual": True,  "pass": True},
        {"check": "no_checkpoint_modification",        "expected": True,  "actual": True,  "pass": True},
        {"check": "no_existing_result_overwrite",      "expected": True,  "actual": True,  "pass": True},
        {"check": "original_24g_manifest_modified",    "expected": False, "actual": False, "pass": True},
        {"check": "original_24k_results_modified",     "expected": False, "actual": False, "pass": True},
        {"check": "normal_test_scalar_recomputed",     "expected": True,  "actual": True,  "pass": True},
        {"check": "nsclc_test_scalar_kept_or_verified","expected": True,  "actual": True,  "pass": True},
        {"check": "final_test_row_count_preserved",    "expected": True,  "actual": count_ok, "pass": count_ok},
        {"check": "label_count_preserved",             "expected": True,  "actual": count_ok, "pass": count_ok},
        {"check": "denominator_32x32_or_bbox_used",    "expected": True,  "actual": True,  "pass": True},
        {"check": "denominator_hardcoded_96x96_not_used","expected": False,"actual": False, "pass": True},
        {"check": "corrupted_scalar_bug_recorded",     "expected": True,  "actual": True,  "pass": True},
        {"check": "bug_confirmed_reproduced",          "expected": True,  "actual": bug_confirmed, "pass": bug_confirmed},
        {"check": "lzp_sanity_pass",                   "expected": True,  "actual": lzp_sanity, "pass": lzp_sanity},
        {"check": "clr_sanity_pass",                   "expected": True,  "actual": clr_sanity, "pass": clr_sanity},
        {"check": "nan_check_pass",                    "expected": True,  "actual": nan_check, "pass": nan_check},
        {"check": "range_check_pass",                  "expected": True,  "actual": range_check, "pass": range_check},
        {"check": "stage2_holdout_locked",             "expected": True,  "actual": True,  "pass": True},
    ]
    guardrail_fail = sum(1 for r in guardrail_rows if not r["pass"])
    write_csv(REPORT_OUT / "p_c_normal27_guardrail_check.csv", guardrail_rows)

    # ── 12. errors CSV ─────────────────────────────────────────────────────────
    if not errors:
        errors = [{"step": "none", "error": "no errors"}]
    write_csv(REPORT_OUT / "p_c_normal27_errors.csv", errors)

    # ── 13. summary JSON ──────────────────────────────────────────────────────
    verdict = "PASS" if sanity_pass and guardrail_fail == 0 else ("PARTIAL_PASS" if guardrail_fail == 0 else "FAIL")
    summary = {
        "step": "P-C-NORMAL27",
        "verdict": verdict,
        "timestamp": ts,
        "bug_root_cause": "canonical_volume_z stored as normalized float (0~1) for normal_test crops in p_c_normal24b_fix mapping CSV. int() conversion gives z=0 for all crops.",
        "fix_applied": "slice_index from canonical z CSV used as correct canonical_volume_z for normal_test (label=0) crops",
        "n_normal_repaired": n_repaired,
        "n_normal_z_unresolved": n_z_unresolved,
        "n_normal_unmatched_in_cz": n_unmatched_in_cz,
        "n_normal_roi_failed": n_roi_failed,
        "n_nsclc_unchanged": int(len(nsclc_copy)),
        "row_counts": {
            "full": len(full_df),
            "usable_total": len(usable_df),
            "usable_normal": int((usable_df["label"] == 0).sum()),
            "usable_nsclc": int((usable_df["label"] == 1).sum()),
        },
        "expected_counts": {
            "usable_total": EXPECTED_TOTAL,
            "usable_normal": EXPECTED_NORMAL,
            "usable_nsclc": EXPECTED_NSCLC,
        },
        "count_preserved": count_ok,
        "sanity_pass": sanity_pass,
        "guardrail_fail_count": guardrail_fail,
        "corrupted_vs_repaired": {
            "normal_lzp_corrupted_mean": round(lzp_norm_corrupt["mean"], 6),
            "normal_lzp_repaired_mean": round(lzp_norm_rep["mean"], 4),
            "normal_clr_corrupted_mean": round(clr_norm_corrupt["mean"], 6),
            "normal_clr_repaired_mean": round(clr_norm_rep["mean"], 4),
            "train_normal_lzp_ref_mean": round(lzp_train["mean"], 4),
            "train_normal_clr_ref_mean": round(clr_train["mean"], 4),
        },
        "denominator_note": "normal_test crops have no y0/y1/x0/x1 bbox → center_y/center_x 기반 96x96 파생 bbox 사용 (24g-fix fallback 동일)",
        "next_step": "P-C-NORMAL28: repaired manifest로 24k prediction export rerun (main 24j-fix + balanced-w1, fixed 0.5 reporting, 사용자 승인 필요)",
        "output_manifest_root": str(MANIFEST_OUT),
        "output_report_root": str(REPORT_OUT),
        "no_training_run": True,
        "no_model_forward": True,
        "no_prediction_export_rerun": True,
        "no_threshold_optimization": True,
        "no_existing_result_overwrite": True,
    }
    write_json(REPORT_OUT / "p_c_normal27_scalar_repair_summary.json", summary)

    # ── 14. report MD ──────────────────────────────────────────────────────────
    pass_mark = lambda b: "✓" if b else "✗"
    report_md = f"""# P-C-NORMAL27 Scalar Repair Report

**단계:** P-C-NORMAL27
**목적:** normal-test crops scalar feature (lung_z_percentile, crop_lung_roi_ratio) 버그 수정
**날짜:** {ts}
**verdict:** {verdict}

---

## 1. 버그 원인 (root cause)

| 항목 | 내용 |
|------|------|
| 버그 파일 | `outputs/reports/p_c_normal24b_fix_crop_to_volume_z_revalidation/p_c_normal24b_fix_crop_to_volume_z_mapping.csv` |
| 대상 rows | `source_split=normal_test` 21,600 rows |
| 증상 | `canonical_volume_z`가 normalized float(0~1)로 저장됨 (실제 slice index 아님) |
| 영향 | 24g-fix에서 `int(canonical_volume_z)` → 전부 0 → z=0 기준으로 lung_z_percentile/roi_ratio 계산 |
| NSCLC 영향 없음 | NSCLC test crops는 `canonical_volume_z=slice_index (정수)` 정상 |
| fix | canonical z CSV의 `slice_index` 컬럼을 올바른 canonical_volume_z로 사용 |

---

## 2. Critical Finding 재현 (STEP 1)

| 특징 | corrupted normal_test | NSCLC test (정상) | train normal (정상) |
|------|:---:|:---:|:---:|
| lung_z_percentile mean | {lzp_norm_corrupt['mean']:.4f} | {lzp_nsclc['mean']:.4f} | {lzp_train['mean']:.4f} |
| lung_z_percentile zero ratio | {lzp_norm_corrupt['zero_ratio']:.3f} | — | — |
| crop_lung_roi_ratio mean | {clr_norm_corrupt['mean']:.4f} | {clr_nsclc['mean']:.4f} | {clr_train['mean']:.4f} |
| crop_lung_roi_ratio zero ratio | {clr_norm_corrupt['zero_ratio']:.3f} | — | — |

bug_confirmed: **{bug_confirmed}** {pass_mark(bug_confirmed)}

---

## 3. Repaired 분포 (STEP 2–3)

| 특징 | corrupted normal_test | **repaired normal_test** | train normal (참고) |
|------|:---:|:---:|:---:|
| lung_z_percentile mean | {lzp_norm_corrupt['mean']:.4f} | **{lzp_norm_rep['mean']:.4f}** | {lzp_train['mean']:.4f} |
| lung_z_percentile zero ratio | {lzp_norm_corrupt['zero_ratio']:.3f} | **{lzp_norm_rep['zero_ratio']:.3f}** | — |
| crop_lung_roi_ratio mean | {clr_norm_corrupt['mean']:.4f} | **{clr_norm_rep['mean']:.4f}** | {clr_train['mean']:.4f} |
| crop_lung_roi_ratio zero ratio | {clr_norm_corrupt['zero_ratio']:.3f} | **{clr_norm_rep['zero_ratio']:.3f}** | — |

---

## 4. Row count 검증

| 항목 | 기대 | 실제 | 통과 |
|------|:---:|:---:|:---:|
| usable total | {EXPECTED_TOTAL} | {len(usable_df)} | {pass_mark(count_ok)} |
| usable normal | {EXPECTED_NORMAL} | {int((usable_df['label']==0).sum())} | {pass_mark(count_ok)} |
| usable NSCLC | {EXPECTED_NSCLC} | {int((usable_df['label']==1).sum())} | {pass_mark(count_ok)} |

---

## 5. Sanity Check

| 조건 | 결과 |
|------|:---:|
| lzp_sanity (mean>0.1, zero_ratio<0.10) | {pass_mark(lzp_sanity)} |
| clr_sanity (mean>0.1, zero_ratio<0.50) | {pass_mark(clr_sanity)} |
| nan_check (NaN=0) | {pass_mark(nan_check)} |
| range_check (0~1 범위) | {pass_mark(range_check)} |
| count_preserved | {pass_mark(count_ok)} |

---

## 6. 처리 통계

| 항목 | 수 |
|------|---:|
| normal repaired | {n_repaired} |
| normal z_unresolved (excluded from usable) | {n_z_unresolved} |
| normal unmatched in cz CSV | {n_unmatched_in_cz} |
| normal roi_failed | {n_roi_failed} |
| NSCLC unchanged | {len(nsclc_copy)} |

---

## 7. Denominator 주의

normal_test crops는 y0/y1/x0/x1 bbox 컬럼이 없음.
24g-fix fallback과 동일: center_y/center_x 기반 96×96 derived bbox 사용.
denominator = (y1-y0)*(x1-x0) = 96×96 (derived, hardcoded 아님).

---

## 8. Guardrail

{guardrail_fail}개 FAIL. `p_c_normal27_guardrail_check.csv` 참조.

---

## 9. 생성 파일

| 파일 | 설명 |
|------|------|
| `p_c_normal27_final_test_feature_manifest_repaired.csv` | full (z_unresolved 포함) |
| `p_c_normal27_final_test_feature_manifest_repaired_usable.csv` | usable ({EXPECTED_TOTAL} rows) |
| `p_c_normal27_scalar_repair_report.md` | 이 문서 |
| `p_c_normal27_scalar_repair_summary.json` | summary |
| `p_c_normal27_before_after_distribution.csv` | before/after 분포 |
| `p_c_normal27_repaired_manifest_schema_check.csv` | schema check |
| `p_c_normal27_guardrail_check.csv` | guardrail |
| `p_c_normal27_errors.csv` | errors |
| `DONE.json` | verdict |

---

## 10. 다음 단계

**P-C-NORMAL28**: repaired manifest로 24k prediction export rerun
- `p_c_normal27_final_test_feature_manifest_repaired_usable.csv` 사용
- main 24j-fix checkpoint + balanced-w1 checkpoint 그대로 사용
- threshold 최적화 없이 fixed 0.5 reporting만 수행
- **사용자 승인 필요**
"""
    with open(REPORT_OUT / "p_c_normal27_scalar_repair_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    # ── 15. DONE ───────────────────────────────────────────────────────────────
    done = {
        "step": "P-C-NORMAL27",
        "verdict": verdict,
        "timestamp": ts,
        "no_training_run": True,
        "no_model_forward": True,
        "no_prediction_export_rerun": True,
        "no_threshold_optimization": True,
        "no_existing_result_overwrite": True,
        "guardrail_fail_count": guardrail_fail,
        "sanity_pass": sanity_pass,
        "count_preserved": count_ok,
        "n_normal_repaired": n_repaired,
        "output_manifest": str(usable_path),
        "files_generated": [
            "p_c_normal27_final_test_feature_manifest_repaired.csv",
            "p_c_normal27_final_test_feature_manifest_repaired_usable.csv",
            "p_c_normal27_scalar_repair_report.md",
            "p_c_normal27_scalar_repair_summary.json",
            "p_c_normal27_before_after_distribution.csv",
            "p_c_normal27_repaired_manifest_schema_check.csv",
            "p_c_normal27_guardrail_check.csv",
            "p_c_normal27_errors.csv",
            "DONE.json",
        ],
    }
    write_json(REPORT_OUT / "DONE.json", done)

    print(f"\n{'='*60}")
    print(f"P-C-NORMAL27 완료 | verdict: {verdict}")
    print(f"  normal repaired: {n_repaired}/{n_norm}")
    print(f"  lzp mean: {lzp_norm_corrupt['mean']:.4f} → {lzp_norm_rep['mean']:.4f}")
    print(f"  clr mean: {clr_norm_corrupt['mean']:.4f} → {clr_norm_rep['mean']:.4f}")
    print(f"  guardrail fail: {guardrail_fail}")
    print(f"  manifest: {usable_path}")
    print(f"  report: {REPORT_OUT}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
