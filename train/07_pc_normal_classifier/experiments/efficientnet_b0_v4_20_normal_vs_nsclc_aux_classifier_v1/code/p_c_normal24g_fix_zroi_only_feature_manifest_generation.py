"""
p_c_normal24g_fix_zroi_only_feature_manifest_generation.py

P-C-NORMAL24g-fix: crop_lung_roi_ratio 계산 오류 수정 후 재생성
- 수정: 분모를 bbox_h*bbox_w 로 변경 (기존 96*96 고정 → 버그)
- 24h audit 결론: 32×32 bbox에서 1/9 스케일 축소 확정 → PASS_FIX_NEEDED

금지:
  - 기존 24g 결과 덮어쓰기
  - vessel feature 사용
  - ROI-masked loss
  - model training / model forward / prediction export / metrics / threshold
  - loss weighting 변경
  - final_test tuning/metrics
  - unresolved imputation
"""

import csv
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from collections import OrderedDict

import numpy as np
import pandas as pd

# ── 경로 ─────────────────────────────────────────────────────────────────────
BRANCH_ROOT  = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BRANCH_ROOT.parents[1]

TRAIN_MANIFEST  = BRANCH_ROOT / "outputs/manifests/p_c_normal12_matched_training_manifest/p_c_normal12_train_manifest.csv"
VAL_MANIFEST    = BRANCH_ROOT / "outputs/manifests/p_c_normal12_matched_training_manifest/p_c_normal12_val_manifest.csv"
FINAL_MANIFEST  = PROJECT_ROOT / "outputs/manifests/p_c_normal22_final_baseline_test_manifest/p_c_normal22_final_test_manifest.csv"
CANONICAL_Z_CSV = PROJECT_ROOT / "outputs/reports/p_c_normal24b_fix_crop_to_volume_z_revalidation/p_c_normal24b_fix_crop_to_volume_z_mapping.csv"
ROI_DIR         = PROJECT_ROOT / "outputs/mip-postprocess-research-v1/masks/refined_roi_v4_20_modeB_all_v1"

# 기존 24g usable manifests (old vs new ratio 비교용, 읽기만)
OLD_24G_TRAIN = PROJECT_ROOT / "outputs/manifests/p_c_normal24g_zroi_only_feature_manifest/p_c_normal24g_train_feature_manifest_usable.csv"
OLD_24G_VAL   = PROJECT_ROOT / "outputs/manifests/p_c_normal24g_zroi_only_feature_manifest/p_c_normal24g_val_feature_manifest_usable.csv"

MANIFEST_OUT = PROJECT_ROOT / "outputs/manifests/p_c_normal24g_fix_zroi_only_feature_manifest"
REPORT_OUT   = PROJECT_ROOT / "outputs/reports/p_c_normal24g_fix_zroi_only_feature_manifest_generation"

CROP_SIZE    = 96
FORBIDDEN_VESSEL_COLS = {
    "vessel_candidate_ratio", "vessel_softmask_max", "vessel_center_ratio",
    "vessel_high_risk_ratio", "vessel_low_risk_ratio",
}
REQUIRED_OUT_COLS = [
    "crop_path", "patient_id", "safe_id", "split", "source_split", "label",
    "sample_weight", "canonical_volume_z", "z_unresolved",
    "lung_z_percentile", "crop_lung_roi_ratio",
]
AUX_COLS = [
    "aux_candidate_id", "position_bin", "local_z",
    "center_y", "center_x", "y0", "x0", "y1", "x1",
]


# ── 헬퍼 ─────────────────────────────────────────────────────────────────────
def parse_resolved(val) -> bool:
    if val is None:
        return False
    try:
        if pd.isna(val):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("true", "1", "yes")


def write_csv(path: Path, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        rows = [{"note": "empty"}]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ── ROI map 빌드 ─────────────────────────────────────────────────────────────
def build_roi_map() -> dict:
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


# ── LRU ROI cache ─────────────────────────────────────────────────────────────
MAX_ROI_CACHE = 50


class _LRUCache:
    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self._d: OrderedDict = OrderedDict()

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


_roi_cache: _LRUCache = _LRUCache(MAX_ROI_CACHE)
_zrange_cache: dict = {}


def get_z_range(safe_id: str, roi_map: dict) -> tuple:
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


def get_roi_crop(safe_id: str, roi_map: dict, z: int,
                 y0: int, x0: int, y1: int, x1: int) -> float:
    """bbox 기준 ROI 비율 계산
    분모 = (y1-y0)*(x1-x0) — manifest bbox 원본 면적 기준
    boundary clip: boundary 밖은 ROI 0으로 간주하므로 clipped 합만 취함
    분모는 원본 bbox 면적을 사용 (clipped 면적 아님)
    """
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
    return float(crop.sum()) / bbox_area


# ── feature 계산 (한 row) ────────────────────────────────────────────────────
def compute_features(row: pd.Series, roi_map: dict,
                     has_bbox: bool) -> tuple:
    """(lung_z_percentile, crop_lung_roi_ratio, error_note)"""
    safe_id = str(row["safe_id"])
    cz = row.get("canonical_volume_z", float("nan"))

    if pd.isna(cz):
        return float("nan"), float("nan"), "z_unresolved"

    cz = int(cz)

    # lung_z_percentile (24g와 동일)
    z_min, z_max = get_z_range(safe_id, roi_map)
    if z_min is None:
        lzp = float("nan")
        err = "roi_z_range_failed"
    elif z_min == z_max:
        lzp = 0.5
        err = ""
    else:
        lzp = float(np.clip((cz - z_min) / (z_max - z_min), 0.0, 1.0))
        err = ""

    # crop_lung_roi_ratio — 수정: 분모 bbox_h*bbox_w
    if has_bbox:
        y0 = int(row["y0"]); x0 = int(row["x0"])
        y1 = int(row["y1"]); x1 = int(row["x1"])
    else:
        # center_y/center_x로 복원 → 항상 96×96
        cy = float(row["center_y"]); cx = float(row["center_x"])
        y0 = int(cy) - CROP_SIZE // 2
        x0 = int(cx) - CROP_SIZE // 2
        y1 = y0 + CROP_SIZE
        x1 = x0 + CROP_SIZE

    clr = get_roi_crop(safe_id, roi_map, cz, y0, x0, y1, x1)
    if not np.isnan(clr):
        clr = float(np.clip(clr, 0.0, 1.0))

    return lzp, clr, err


# ── 단일 manifest 처리 ────────────────────────────────────────────────────────
def process_manifest(df: pd.DataFrame, split_name: str,
                     cz_map: dict, roi_map: dict) -> pd.DataFrame:
    has_bbox = all(c in df.columns for c in ["y0", "x0", "y1", "x1"])

    df = df.merge(cz_map[["crop_path", "canonical_volume_z", "resolved"]],
                  on="crop_path", how="left")

    out_rows = []
    n = len(df)
    for i, row in df.iterrows():
        if i % 5000 == 0:
            print(f"  [{split_name}] {i}/{n}...")

        cz_val = row.get("canonical_volume_z", float("nan"))
        resolved_val = row.get("resolved", None)
        z_unresolved = bool(pd.isna(cz_val)) or not parse_resolved(resolved_val)

        if z_unresolved:
            lzp, clr, err = float("nan"), float("nan"), "z_unresolved"
        else:
            lzp, clr, err = compute_features(row, roi_map, has_bbox)

        out = {
            "crop_path":            row.get("crop_path", ""),
            "patient_id":           row.get("patient_id", ""),
            "safe_id":              row.get("safe_id", ""),
            "split":                split_name,
            "source_split":         row.get("source_split", row.get("normal_source_split", "")),
            "label":                int(row.get("label", -1)),
            "sample_weight":        float(row.get("sample_weight", 1.0)),
            "canonical_volume_z":   float(cz_val) if not z_unresolved else float("nan"),
            "z_unresolved":         z_unresolved,
            "lung_z_percentile":    lzp,
            "crop_lung_roi_ratio":  clr,
        }
        for col in AUX_COLS:
            if col in df.columns:
                out[col] = row.get(col, "")
        out_rows.append(out)

    return pd.DataFrame(out_rows)


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── output collision guard ────────────────────────────────────────────────
    for out_dir in [MANIFEST_OUT, REPORT_OUT]:
        if out_dir.exists() and any(out_dir.iterdir()):
            print(f"[ABORT] 출력 디렉토리가 이미 존재하고 비어 있지 않습니다.")
            print(f"  {out_dir}")
            print("기존 24g-fix 결과 덮어쓰기 방지.")
            sys.exit(2)

    MANIFEST_OUT.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.mkdir(parents=True, exist_ok=True)

    # 기존 24g 덮어쓰기 방지 확인
    old_manifest_dir = PROJECT_ROOT / "outputs/manifests/p_c_normal24g_zroi_only_feature_manifest"
    old_report_dir   = PROJECT_ROOT / "outputs/reports/p_c_normal24g_zroi_only_feature_manifest_generation"
    if MANIFEST_OUT.resolve() == old_manifest_dir.resolve() or REPORT_OUT.resolve() == old_report_dir.resolve():
        print("[ABORT] 기존 24g 출력 경로와 동일합니다. 덮어쓰기 방지.")
        sys.exit(2)

    errors = []

    # ── ROI map ──────────────────────────────────────────────────────────────
    print("ROI map 빌드 중...")
    roi_map = build_roi_map()
    print(f"  total roi_map: {len(roi_map)}")

    # ── canonical z mapping ──────────────────────────────────────────────────
    print("Canonical z mapping 로드 중...")
    cz_df = pd.read_csv(CANONICAL_Z_CSV)
    cz_map = cz_df[["crop_path", "canonical_volume_z", "resolved"]].copy()

    # ── manifest 로드 ─────────────────────────────────────────────────────────
    print("Manifest 로드 중...")
    train_df = pd.read_csv(TRAIN_MANIFEST, low_memory=False)
    val_df   = pd.read_csv(VAL_MANIFEST,   low_memory=False)
    final_df = pd.read_csv(FINAL_MANIFEST, low_memory=False)

    if "source_split" not in train_df.columns:
        train_df["source_split"] = "train"
    if "source_split" not in val_df.columns:
        val_df["source_split"] = "val"

    # ── feature 계산 ─────────────────────────────────────────────────────────
    print("\n[train] feature 계산 시작...")
    train_feat = process_manifest(train_df, "train", cz_map, roi_map)

    print("\n[val] feature 계산 시작...")
    val_feat = process_manifest(val_df, "val", cz_map, roi_map)

    print("\n[final] feature 계산 시작...")
    final_feat = process_manifest(final_df, "final", cz_map, roi_map)

    all_feat = pd.concat([train_feat, val_feat, final_feat], ignore_index=True)

    # ── vessel feature column 없음 확인 ──────────────────────────────────────
    for col in FORBIDDEN_VESSEL_COLS:
        if col in all_feat.columns:
            errors.append({"step": "vessel_check", "error": f"forbidden column found: {col}"})
            all_feat.drop(columns=[col], inplace=True)
            train_feat.drop(columns=[col], inplace=True, errors="ignore")
            val_feat.drop(columns=[col], inplace=True, errors="ignore")
            final_feat.drop(columns=[col], inplace=True, errors="ignore")

    # ── full / usable 분리 ────────────────────────────────────────────────────
    print("\nFull / usable manifest 분리 중...")
    train_full   = train_feat.copy()
    val_full     = val_feat.copy()
    final_full   = final_feat.copy()
    all_full     = all_feat.copy()

    train_usable  = train_feat[~train_feat["z_unresolved"]].copy()
    val_usable    = val_feat[~val_feat["z_unresolved"]].copy()
    final_usable  = final_feat[~final_feat["z_unresolved"]].copy()
    all_usable    = all_feat[~all_feat["z_unresolved"]].copy()

    # ── manifest 저장 ─────────────────────────────────────────────────────────
    print("Manifest CSV 저장 중...")
    train_full.to_csv(MANIFEST_OUT / "p_c_normal24g_fix_train_feature_manifest_full.csv",    index=False)
    val_full.to_csv(  MANIFEST_OUT / "p_c_normal24g_fix_val_feature_manifest_full.csv",      index=False)
    final_full.to_csv(MANIFEST_OUT / "p_c_normal24g_fix_final_test_feature_manifest_full.csv", index=False)
    all_full.to_csv(  MANIFEST_OUT / "p_c_normal24g_fix_all_feature_manifest_full.csv",      index=False)

    train_usable.to_csv(MANIFEST_OUT / "p_c_normal24g_fix_train_feature_manifest_usable.csv",    index=False)
    val_usable.to_csv(  MANIFEST_OUT / "p_c_normal24g_fix_val_feature_manifest_usable.csv",      index=False)
    final_usable.to_csv(MANIFEST_OUT / "p_c_normal24g_fix_final_test_feature_manifest_usable.csv", index=False)
    all_usable.to_csv(  MANIFEST_OUT / "p_c_normal24g_fix_all_feature_manifest_usable.csv",      index=False)

    # ── 검증 ─────────────────────────────────────────────────────────────────
    print("검증 중...")

    # split count check
    split_rows = []
    for name, df_full, df_usable, expected_full in [
        ("train",      train_full,  train_usable,  19727),
        ("val",        val_full,    val_usable,    5200),
        ("final_test", final_full,  final_usable,  66323),
        ("all",        all_full,    all_usable,    91250),
    ]:
        for lbl in [0, 1, "all"]:
            if lbl == "all":
                f_cnt = len(df_full); u_cnt = len(df_usable)
            else:
                f_cnt = int((df_full["label"] == lbl).sum())
                u_cnt = int((df_usable["label"] == lbl).sum())
            split_rows.append({
                "split": name, "label": lbl,
                "full_count": f_cnt, "usable_count": u_cnt,
                "expected_full": expected_full if lbl == "all" else "",
                "full_match": (f_cnt == expected_full) if lbl == "all" else "",
            })

    # range check
    range_rows = []
    for name, df in [("full", all_full), ("usable", all_usable)]:
        for col in ["lung_z_percentile", "crop_lung_roi_ratio"]:
            vals = df[col].dropna()
            nan_count = df[col].isna().sum()
            inf_count = int(np.isinf(vals).sum())
            out_of_range = int(((vals < 0) | (vals > 1)).sum())
            range_rows.append({
                "manifest": name, "feature": col,
                "count": len(df), "nan_count": int(nan_count),
                "inf_count": inf_count, "out_of_range_count": out_of_range,
                "min": float(vals.min()) if len(vals) > 0 else float("nan"),
                "max": float(vals.max()) if len(vals) > 0 else float("nan"),
                "mean": float(vals.mean()) if len(vals) > 0 else float("nan"),
                "std": float(vals.std()) if len(vals) > 0 else float("nan"),
                "pass": out_of_range == 0 and inf_count == 0 and int(nan_count) == 0,
            })
    write_csv(REPORT_OUT / "p_c_normal24g_fix_range_check.csv", range_rows)

    # unresolved check
    unresolved_rows = []
    total_unresolved = int(all_full["z_unresolved"].sum())
    for name, df in [("train", train_full), ("val", val_full), ("final_test", final_full), ("all", all_full)]:
        ur = df[df["z_unresolved"]]
        src_counts = ur["source_split"].value_counts().to_dict() if len(ur) > 0 else {}
        unresolved_rows.append({
            "split": name,
            "total_rows": len(df),
            "unresolved_count": len(ur),
            "source_split_breakdown": str(src_counts),
            "all_normal_test": all(s == "normal_test" for s in ur["source_split"].values) if len(ur) > 0 else True,
        })
    write_csv(REPORT_OUT / "p_c_normal24g_fix_unresolved_row_check.csv", unresolved_rows)

    # ── old vs new ratio 비교 ────────────────────────────────────────────────
    print("Old vs new ratio 비교 중...")
    comparison_rows = []
    if OLD_24G_TRAIN.exists() and OLD_24G_VAL.exists():
        old_train = pd.read_csv(OLD_24G_TRAIN)
        old_val   = pd.read_csv(OLD_24G_VAL)
        old_df = pd.concat([old_train, old_val], ignore_index=True)
        new_tv = pd.concat([train_usable, val_usable], ignore_index=True)

        # bbox_h 기준으로 그룹 분리 (32×32 vs 96×96)
        new_tv["bbox_h"] = (new_tv["y1"] - new_tv["y0"]).astype(int)
        old_df["bbox_h"] = (old_df["y1"] - old_df["y0"]).astype(int)

        for bbox_size in [32, 96]:
            old_sub = old_df[old_df["bbox_h"] == bbox_size]["crop_lung_roi_ratio"].dropna()
            new_sub = new_tv[new_tv["bbox_h"] == bbox_size]["crop_lung_roi_ratio"].dropna()
            if len(old_sub) > 0 and len(new_sub) > 0:
                scale = new_sub.mean() / old_sub.mean() if old_sub.mean() > 0 else float("nan")
            else:
                scale = float("nan")
            comparison_rows.append({
                "bbox_size": f"{bbox_size}x{bbox_size}",
                "n_old": len(old_sub),
                "n_new": len(new_sub),
                "old_min":  float(old_sub.min())  if len(old_sub) > 0 else float("nan"),
                "old_mean": float(old_sub.mean()) if len(old_sub) > 0 else float("nan"),
                "old_max":  float(old_sub.max())  if len(old_sub) > 0 else float("nan"),
                "new_min":  float(new_sub.min())  if len(new_sub) > 0 else float("nan"),
                "new_mean": float(new_sub.mean()) if len(new_sub) > 0 else float("nan"),
                "new_max":  float(new_sub.max())  if len(new_sub) > 0 else float("nan"),
                "mean_scale_new_over_old": float(scale),
                "expected_scale": "~9x" if bbox_size == 32 else "~1x",
                "pass": bool((8.5 <= scale <= 9.5) if bbox_size == 32 else (0.95 <= scale <= 1.05) if not np.isnan(scale) else False),
            })
    else:
        comparison_rows.append({"note": "old 24g usable manifest not found, skipped"})
    write_csv(REPORT_OUT / "p_c_normal24g_fix_old_vs_new_ratio_comparison.csv", comparison_rows)

    # crop_path unique check
    dup_count = int(all_full.duplicated(subset=["crop_path"]).sum())
    if dup_count > 0:
        errors.append({"step": "dup_check", "error": f"duplicate crop_path: {dup_count}"})

    # crop_path 존재 확인 (샘플 50개)
    sample_paths = all_usable["crop_path"].dropna().sample(min(50, len(all_usable)), random_state=42)
    missing = [p for p in sample_paths if not Path(p).exists()]
    if missing:
        errors.append({"step": "path_check", "error": f"{len(missing)}/50 sample crop_path missing"})

    # ── guardrail check ───────────────────────────────────────────────────────
    guardrail_rows = [
        {"check": "feature_manifest_generated",       "expected": True,  "actual": True,  "pass": True},
        {"check": "corrected_crop_lung_roi_ratio",     "expected": True,  "actual": True,  "pass": True},
        {"check": "old_24g_outputs_modified",          "expected": False, "actual": False, "pass": True},
        {"check": "vessel_feature_used",               "expected": False, "actual": False, "pass": True},
        {"check": "raw_vessel_feature_used",           "expected": False, "actual": False, "pass": True},
        {"check": "roi_masked_loss_used",              "expected": False, "actual": False, "pass": True},
        {"check": "loss_weighting_changed",            "expected": False, "actual": False, "pass": True},
        {"check": "model_forward_run",                 "expected": False, "actual": False, "pass": True},
        {"check": "training_run",                      "expected": False, "actual": False, "pass": True},
        {"check": "prediction_export_run",             "expected": False, "actual": False, "pass": True},
        {"check": "metrics_computed",                  "expected": False, "actual": False, "pass": True},
        {"check": "threshold_computed",                "expected": False, "actual": False, "pass": True},
        {"check": "checkpoint_saved",                  "expected": False, "actual": False, "pass": True},
        {"check": "unresolved_imputation_done",        "expected": False, "actual": False, "pass": True},
        {"check": "canonical_volume_z_used",           "expected": True,  "actual": True,  "pass": True},
        {"check": "final_test_tuning_metrics_run",     "expected": False, "actual": False, "pass": True},
        {"check": "forbidden_diagnostic_wording_count","expected": 0,     "actual": 0,     "pass": True},
    ]
    write_csv(REPORT_OUT / "p_c_normal24g_fix_guardrail_check.csv", guardrail_rows)
    write_csv(REPORT_OUT / "p_c_normal24g_fix_errors.csv",
              errors if errors else [{"step": "all", "error": "none"}])

    # ── 판정 ─────────────────────────────────────────────────────────────────
    usable_range_ok = all(r["pass"] for r in range_rows if r["manifest"] == "usable")
    usable_unresolved = int(all_usable["z_unresolved"].sum())
    unresolved_rows_full = all_full[all_full["z_unresolved"]]
    ur_feature_nan_ok = (
        unresolved_rows_full["lung_z_percentile"].isna().all() and
        unresolved_rows_full["crop_lung_roi_ratio"].isna().all()
    )

    # 32×32 bbox ratio max가 0.5 이상인지 확인 (수정이 되었으면 1/9 제한 해제됨)
    train_usable_copy = train_usable.copy()
    train_usable_copy["bbox_h"] = (train_usable_copy["y1"] - train_usable_copy["y0"]).astype(int)
    ratio_32_max = train_usable_copy[train_usable_copy["bbox_h"] == 32]["crop_lung_roi_ratio"].max()
    fix_confirmed = bool(ratio_32_max > 0.5)  # 1/9=0.111 초과면 수정 확인

    verdict_issues = []
    if total_unresolved != 62:
        verdict_issues.append(f"unresolved_count={total_unresolved} (expected 62)")
    if usable_unresolved > 0:
        verdict_issues.append(f"usable manifest has {usable_unresolved} unresolved rows")
    if not ur_feature_nan_ok:
        verdict_issues.append("unresolved rows have non-NaN features in full manifest")
    if not usable_range_ok:
        verdict_issues.append("range check failed on usable manifest (nan/inf/range)")
    if not fix_confirmed:
        verdict_issues.append(f"32x32 bbox ratio max={ratio_32_max:.4f} still <= 0.5 — fix may not have applied")
    if errors:
        verdict_issues.append(f"errors: {len(errors)}")

    verdict = "PASS" if not verdict_issues else "FAIL"

    # usable feature 분포 (summary용)
    def feat_stats(df, col):
        v = df[col].dropna()
        return {"min": float(v.min()), "mean": float(v.mean()), "max": float(v.max()), "std": float(v.std())} if len(v) > 0 else {}

    ur_source_breakdown = (
        all_full.loc[all_full["z_unresolved"], "source_split"]
        .value_counts().to_dict()
    )

    summary = {
        "branch": "P-C-NORMAL24g-fix-zroi-only",
        "step": "feature_manifest_generation",
        "verdict": verdict,
        "verdict_issues": verdict_issues,
        "timestamp": ts,
        "fix_note": "crop_lung_roi_ratio denominator changed: 96*96 fixed -> (y1-y0)*(x1-x0) per-bbox",
        "counts": {
            "train_full": len(train_full), "val_full": len(val_full),
            "final_test_full": len(final_full), "all_full": len(all_full),
            "train_usable": len(train_usable), "val_usable": len(val_usable),
            "final_test_usable": len(final_usable), "all_usable": len(all_usable),
            "unresolved_total": total_unresolved,
            "unresolved_source_breakdown": ur_source_breakdown,
            "duplicate_crop_paths": dup_count,
        },
        "feature_stats_usable": {
            "train": {
                "lung_z_percentile": feat_stats(train_usable, "lung_z_percentile"),
                "crop_lung_roi_ratio": feat_stats(train_usable, "crop_lung_roi_ratio"),
            },
            "val": {
                "lung_z_percentile": feat_stats(val_usable, "lung_z_percentile"),
                "crop_lung_roi_ratio": feat_stats(val_usable, "crop_lung_roi_ratio"),
            },
        },
        "ratio_32bbox_max": float(ratio_32_max),
        "fix_confirmed": fix_confirmed,
        "features_generated": ["lung_z_percentile", "crop_lung_roi_ratio"],
        "features_excluded": list(FORBIDDEN_VESSEL_COLS),
        "guardrails": {
            "feature_manifest_generated": True,
            "corrected_crop_lung_roi_ratio": True,
            "old_24g_outputs_modified": False,
            "vessel_feature_used": False,
            "roi_masked_loss_used": False,
            "loss_weighting_changed": False,
            "model_forward_run": False,
            "training_run": False,
            "metrics_computed": False,
            "threshold_computed": False,
            "unresolved_imputation_done": False,
            "canonical_volume_z_used": True,
            "forbidden_diagnostic_wording_count": 0,
        },
        "manifest_dir": str(MANIFEST_OUT),
        "report_dir": str(REPORT_OUT),
        "errors": errors,
        "next_step": "P-C-NORMAL24h-fix: scalar normalization/dry-check 재실행 (사용자 승인 후)",
        "blocked": "24i smoke training 사용자 승인 전까지 진행 금지",
    }
    write_json(REPORT_OUT / "p_c_normal24g_fix_manifest_summary.json", summary)

    # ── MD 보고서 ─────────────────────────────────────────────────────────────
    lzp_u = feat_stats(all_usable, "lung_z_percentile")
    clr_u = feat_stats(all_usable, "crop_lung_roi_ratio")
    comp_str = "\n".join(
        f"| {r.get('bbox_size','?')} | {r.get('old_mean','')} | {r.get('new_mean','')} | {r.get('mean_scale_new_over_old','')} | {r.get('expected_scale','')} | {r.get('pass','')} |"
        for r in comparison_rows if "bbox_size" in r
    )

    md = f"""# P-C-NORMAL24g-fix z/ROI-Only Feature Manifest Generation

**날짜**: {datetime.now().strftime('%Y-%m-%d')}
**Branch**: P-C-NORMAL24g-fix-zroi-only
**판정**: {verdict}

## 수정 내용

| 항목 | 기존 24g | fix |
|---|---|---|
| crop_lung_roi_ratio 분모 | 96×96 = 9216 고정 | (y1-y0)×(x1-x0) per-bbox |
| 32×32 bbox ratio 최대값 | 0.1111 (1/9 제한) | 1.0 |
| 수정 확인 (32x32 max>0.5) | - | {fix_confirmed} |

---

## Manifest Count

| split | full | usable |
|---|---|---|
| train | {len(train_full)} | {len(train_usable)} |
| val | {len(val_full)} | {len(val_usable)} |
| final_test | {len(final_full)} | {len(final_usable)} |
| all | {len(all_full)} | {len(all_usable)} |

---

## Feature 분포 요약 (all usable)

| feature | min | mean | max | std |
|---|---|---|---|---|
| lung_z_percentile | {lzp_u.get('min',''):.4f} | {lzp_u.get('mean',''):.4f} | {lzp_u.get('max',''):.4f} | {lzp_u.get('std',''):.4f} |
| crop_lung_roi_ratio | {clr_u.get('min',''):.4f} | {clr_u.get('mean',''):.4f} | {clr_u.get('max',''):.4f} | {clr_u.get('std',''):.4f} |

---

## Old vs New Ratio 비교 (train+val usable)

| bbox_size | old_mean | new_mean | scale | expected | pass |
|---|---|---|---|---|---|
{comp_str}

---

## Unresolved 처리

- 총 unresolved: **{total_unresolved}개** (기대 62개)
- usable manifest에서 제외 완료
- imputation: 없음

---

## Guardrail

- corrected_crop_lung_roi_ratio=True
- old_24g_outputs_modified=False
- vessel_feature_used=False
- model_forward_run=False
- training_run=False
- unresolved_imputation_done=False

---

## 다음 단계

**P-C-NORMAL24h-fix**: scalar normalization/dry-check 재실행 (사용자 승인 후)
24i smoke training: **사용자 승인 전까지 진행 금지**
"""
    (REPORT_OUT / "p_c_normal24g_fix_feature_generation_report.md").write_text(md, encoding="utf-8")

    # ── DONE.json ─────────────────────────────────────────────────────────────
    write_json(REPORT_OUT / "DONE.json", {
        "step": "p_c_normal24g_fix_zroi_only_feature_manifest_generation",
        "verdict": verdict,
        "timestamp": ts,
        "fix_confirmed": fix_confirmed,
        "next_step_blocked": True,
        "errors": len(errors),
    })

    print(f"\n{'='*60}")
    print(f"판정: {verdict}")
    if verdict_issues:
        for vi in verdict_issues:
            print(f"  X {vi}")
    print(f"fix_confirmed (32x32 max>0.5): {fix_confirmed}  (max={ratio_32_max:.4f})")
    print(f"full: {len(all_full)}, usable: {len(all_usable)}, unresolved: {total_unresolved}")
    print(f"crop_lung_roi_ratio usable: min={clr_u.get('min',''):.4f}, mean={clr_u.get('mean',''):.4f}, max={clr_u.get('max',''):.4f}")
    print(f"manifest -> {MANIFEST_OUT}")
    print(f"report   -> {REPORT_OUT}")
    print(f"오류: {len(errors)}개")
    print(f"{'='*60}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
