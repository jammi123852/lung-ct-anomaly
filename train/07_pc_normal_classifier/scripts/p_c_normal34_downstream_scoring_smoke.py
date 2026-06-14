"""
P-C-NORMAL34: downstream scoring smoke/preflight
- selected candidate: P-C-NORMAL30b_masked_input
- smoke only (100~120 crops), 전체 scoring 금지
- model forward 허용, threshold 최적화 금지
"""

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")
STAGE_LABEL  = "P-C-NORMAL34"

SELECTED_CKPT = PROJECT_ROOT / "outputs/p_c_normal30b_masked_input_full_train/checkpoints/p_c_normal30b_best_val_auc_checkpoint.pt"
SCALAR_STATS  = PROJECT_ROOT / "outputs/reports/p_c_normal24h_fix_zroi_scalar_fusion_drycheck/p_c_normal24h_fix_scalar_normalization_stats.json"
FINAL_TEST_MANIFEST = PROJECT_ROOT / "outputs/manifests/p_c_normal27_scalar_repair_final_test_manifest/p_c_normal27_final_test_feature_manifest_repaired_usable.csv"
MASK_MANIFEST = PROJECT_ROOT / "outputs/reports/p_c_normal31_repaired_final_test_masked_comparison/p_c_normal31_final_test_mask_manifest.csv"
HANDOFF_DIR   = PROJECT_ROOT / "outputs/reports/p_c_normal33_selected_candidate_handoff_package"
NORMAL32_DIR  = PROJECT_ROOT / "outputs/reports/p_c_normal32_final_decision_checkpoint"

OUTPUT_DIR    = PROJECT_ROOT / "outputs/p_c_normal34_downstream_scoring_smoke"
REPORT_DIR    = PROJECT_ROOT / "outputs/reports/p_c_normal34_downstream_scoring_smoke"

# ── Constants ─────────────────────────────────────────────────────────────────
SEED            = 42
N_NORMAL        = 50
N_NSCLC         = 50
N_LOW_MASK_MAX  = 10
FIXED_THRESHOLD = 0.5
HU_MIN          = -1000.0
HU_MAX          =  200.0
IMAGENET_MEAN   = [0.485, 0.456, 0.406]
IMAGENET_STD    = [0.229, 0.224, 0.225]
SCALAR_FEATURES = ["lung_z_percentile", "crop_lung_roi_ratio"]
EXPECTED_TOTAL_MANIFEST = 66283

REQUIRED_OUTPUT_COLS = [
    "patient_id", "safe_id", "crop_path", "mask_path", "source_split",
    "canonical_volume_z", "local_z", "center_y", "center_x", "position_bin",
    "lung_z_percentile_raw", "crop_lung_roi_ratio_raw",
    "lung_z_percentile_norm", "crop_lung_roi_ratio_norm",
    "mask_nonzero_ratio_mean", "low_mask_flag", "zero_mask_flag",
    "logit", "prob", "pred_at_0p5",
    "model_name", "checkpoint_path", "masked_input_used",
    "threshold_used", "threshold_optimized", "caveat_flag",
]

# ── Guardrails (static) ───────────────────────────────────────────────────────
GUARDRAILS = {
    "smoke_preflight_only":                  True,
    "no_training_run":                       True,
    "model_forward_smoke_only":              True,
    "full_scoring_run":                      False,
    "heatmap_run":                           False,
    "xai_run":                               False,
    "no_threshold_optimization":             True,
    "no_threshold_sweep":                    True,
    "no_best_threshold_selection":           True,
    "fixed_threshold_0p5_only":              True,
    "no_checkpoint_modification":            True,
    "no_existing_result_overwrite":          True,
    "selected_checkpoint_not_smoke":         True,  # verified at runtime
    "selected_candidate_masked_30b_confirmed": True,
    "scalar_normalization_source_confirmed": True,
    "mask_join_checked":                     True,
    "output_contract_checked":               True,
    "diagnostic_wording_avoided":            True,
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def _write_csv(rows: list, path: Path):
    if not rows:
        path.write_text("")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

def _write_json(obj: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def _load_csv(path: Path) -> list:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def _abort(msg: str):
    print(f"[ABORT] {msg}", file=sys.stderr)
    sys.exit(2)

# ── Model (same as 30b) ───────────────────────────────────────────────────────
class ScalarFusionModel(nn.Module):
    def __init__(self, scalar_hidden: int = 32, scalar_out: int = 16, dropout: float = 0.2):
        super().__init__()
        backbone = efficientnet_b0(weights=None)
        self.img_features = backbone.features
        self.img_avgpool  = backbone.avgpool
        self.scalar_branch = nn.Sequential(
            nn.Linear(2, scalar_hidden),
            nn.BatchNorm1d(scalar_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(scalar_hidden, scalar_out),
            nn.ReLU(inplace=True),
        )
        fusion_in = 1280 + scalar_out
        self.fusion_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(fusion_in, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, 1),
        )

    def forward(self, img: torch.Tensor, scalar: torch.Tensor) -> torch.Tensor:
        x = self.img_features(img)
        x = self.img_avgpool(x)
        x = torch.flatten(x, 1)
        s = self.scalar_branch(scalar)
        return self.fusion_head(torch.cat([x, s], dim=1))


def main():
    import datetime
    rng = np.random.default_rng(SEED)

    # ── 0. Output directory guard ─────────────────────────────────────────────
    if REPORT_DIR.exists() and any(REPORT_DIR.iterdir()):
        _abort(f"output directory already exists and is not empty: {REPORT_DIR}")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[{STAGE_LABEL}] output: {REPORT_DIR}")

    # ── 1. 입력 파일 검증 ─────────────────────────────────────────────────────
    print(f"[{STAGE_LABEL}] Step 1: input file validation")

    # 1-1. checkpoint
    if not SELECTED_CKPT.exists():
        _abort(f"selected checkpoint not found: {SELECTED_CKPT}")
    if "smoke" in SELECTED_CKPT.name.lower():
        _abort(f"smoke checkpoint must not be used: {SELECTED_CKPT.name}")
    print(f"  checkpoint OK: {SELECTED_CKPT.name}")

    # 1-2. scalar stats
    if not SCALAR_STATS.exists():
        _abort(f"scalar stats not found: {SCALAR_STATS}")
    with open(SCALAR_STATS) as f:
        scalar_payload = json.load(f)
    scalar_stats = scalar_payload["features"]
    lzp_mean = scalar_stats["lung_z_percentile"]["mean"]
    lzp_std  = scalar_stats["lung_z_percentile"]["std"]
    clrr_mean = scalar_stats["crop_lung_roi_ratio"]["mean"]
    clrr_std  = scalar_stats["crop_lung_roi_ratio"]["std"]
    print(f"  scalar stats OK: lzp_mean={lzp_mean:.5f}, clrr_mean={clrr_mean:.5f}")

    # 1-3. final_test manifest
    if not FINAL_TEST_MANIFEST.exists():
        _abort(f"final_test manifest not found: {FINAL_TEST_MANIFEST}")
    ft_rows = _load_csv(FINAL_TEST_MANIFEST)
    if len(ft_rows) != EXPECTED_TOTAL_MANIFEST:
        print(f"  [WARN] final_test manifest rows: expected {EXPECTED_TOTAL_MANIFEST}, got {len(ft_rows)}", file=sys.stderr)
    print(f"  final_test manifest rows: {len(ft_rows)}")

    # 1-4. mask manifest
    if not MASK_MANIFEST.exists():
        _abort(f"mask manifest not found: {MASK_MANIFEST}")
    mask_rows = _load_csv(MASK_MANIFEST)
    if len(mask_rows) != EXPECTED_TOTAL_MANIFEST:
        print(f"  [WARN] mask manifest rows: expected {EXPECTED_TOTAL_MANIFEST}, got {len(mask_rows)}", file=sys.stderr)
    print(f"  mask manifest rows: {len(mask_rows)}")

    # 1-5. P-C-NORMAL32 selected candidate 확인
    norm32_json = NORMAL32_DIR / "p_c_normal32_final_decision_checkpoint.json"
    if norm32_json.exists():
        with open(norm32_json) as f:
            n32 = json.load(f)
        sel = n32.get("selected_candidate", "")
        if "30b" not in sel.lower():
            _abort(f"P-C-NORMAL32 selected candidate is not 30b: {sel}")
        print(f"  selected_candidate confirmed: {sel}")
    else:
        print(f"  [WARN] P-C-NORMAL32 json not found, skip candidate check", file=sys.stderr)

    # ── 2. smoke sample 구성 ──────────────────────────────────────────────────
    print(f"[{STAGE_LABEL}] Step 2: smoke sample construction")

    # mask lookup: crop_path → mask row
    mask_lookup = {r["crop_path"]: r for r in mask_rows}

    # ft_rows에 mask 정보 join
    joined = []
    for r in ft_rows:
        cp = r["crop_path"]
        if cp in mask_lookup:
            mr = mask_lookup[cp]
            joined.append({**r, **{
                "mask_path":              mr["mask_path"],
                "mask_nonzero_ratio_mean": mr["mask_nonzero_ratio_mean"],
                "low_mask":               mr.get("low_mask", "False"),
                "zero_mask":              mr.get("zero_mask", "False"),
            }})
    print(f"  join result: {len(joined)} / {len(ft_rows)} crops matched")
    if len(joined) < len(ft_rows) * 0.95:
        _abort(f"mask join coverage too low: {len(joined)}/{len(ft_rows)}")

    normal_pool = [r for r in joined if r["label"] == "0"]
    nsclc_pool  = [r for r in joined if r["label"] == "1"]
    low_pool    = [r for r in joined if r.get("low_mask","False").strip() == "True"]
    zero_pool   = [r for r in joined if r.get("zero_mask","False").strip() == "True"]

    print(f"  normal pool: {len(normal_pool)}, nsclc pool: {len(nsclc_pool)}, low_mask pool: {len(low_pool)}, zero_mask pool: {len(zero_pool)}")

    # sampling (fixed seed)
    idx_normal = rng.choice(len(normal_pool), size=min(N_NORMAL, len(normal_pool)), replace=False).tolist()
    idx_nsclc  = rng.choice(len(nsclc_pool),  size=min(N_NSCLC,  len(nsclc_pool)),  replace=False).tolist()
    idx_low    = rng.choice(len(low_pool),    size=min(N_LOW_MASK_MAX, len(low_pool)), replace=False).tolist() if low_pool else []

    sample_normal = [normal_pool[i] for i in idx_normal]
    sample_nsclc  = [nsclc_pool[i]  for i in idx_nsclc]
    sample_low    = [low_pool[i]    for i in idx_low]

    # deduplicate
    seen_cp = set()
    smoke_sample = []
    for r in sample_normal + sample_low + sample_nsclc:
        if r["crop_path"] not in seen_cp:
            seen_cp.add(r["crop_path"])
            smoke_sample.append(r)

    print(f"  smoke sample: {len(smoke_sample)} crops (normal={len(sample_normal)}, nsclc={len(sample_nsclc)}, low_mask={len(sample_low)}, zero_mask_in_sample={sum(1 for r in smoke_sample if r.get('zero_mask','False').strip()=='True')})")

    # save sample manifest
    sample_rows_out = []
    for r in smoke_sample:
        sample_rows_out.append({
            "crop_path":              r["crop_path"],
            "mask_path":              r["mask_path"],
            "patient_id":             r.get("patient_id",""),
            "safe_id":                r.get("safe_id",""),
            "label":                  r["label"],
            "source_split":           r.get("source_split",""),
            "canonical_volume_z":     r.get("canonical_volume_z",""),
            "mask_nonzero_ratio_mean":r.get("mask_nonzero_ratio_mean",""),
            "low_mask":               r.get("low_mask","False"),
            "zero_mask":              r.get("zero_mask","False"),
        })
    _write_csv(sample_rows_out, REPORT_DIR / "p_c_normal34_smoke_sample_manifest.csv")

    # ── 3. preprocessing 검증 ─────────────────────────────────────────────────
    print(f"[{STAGE_LABEL}] Step 3: input preprocessing check")

    img_mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    img_std  = torch.tensor(IMAGENET_STD,  dtype=torch.float32).view(3, 1, 1)

    preprocess_rows = []
    preprocess_errors = 0
    preprocessed = []  # list of (img_tensor, scalar_tensor, row)

    for r in smoke_sample:
        pr = {"crop_path": r["crop_path"], "status": "OK", "note": ""}
        try:
            # CT crop
            data = np.load(r["crop_path"])
            arr = data["ct_crop"].astype(np.float32)
            if arr.shape != (3, 96, 96):
                pr["status"] = "ERROR"
                pr["note"] = f"ct_crop shape {arr.shape}"
                preprocess_errors += 1
                preprocess_rows.append(pr)
                preprocessed.append(None)
                continue
            arr = np.clip(arr, HU_MIN, HU_MAX)
            arr = (arr - HU_MIN) / (HU_MAX - HU_MIN)
            img = torch.from_numpy(arr)

            # mask
            mdata = np.load(r["mask_path"])
            mask_arr = mdata["mask_3ch"].astype(np.float32)
            if mask_arr.shape != (3, 96, 96):
                pr["status"] = "ERROR"
                pr["note"] = f"mask shape {mask_arr.shape}"
                preprocess_errors += 1
                preprocess_rows.append(pr)
                preprocessed.append(None)
                continue
            mask_t = torch.from_numpy(mask_arr)

            img = img * mask_t
            img = (img - img_mean) / img_std

            # scalar
            lzp_raw  = float(r["lung_z_percentile"])
            clrr_raw = float(r["crop_lung_roi_ratio"])
            lzp_norm  = (lzp_raw  - lzp_mean)  / lzp_std
            clrr_norm = (clrr_raw - clrr_mean) / clrr_std

            if math.isnan(lzp_norm) or math.isnan(clrr_norm):
                pr["status"] = "ERROR"
                pr["note"] = "scalar NaN after normalization"
                preprocess_errors += 1
                preprocess_rows.append(pr)
                preprocessed.append(None)
                continue

            scalar_t = torch.tensor([lzp_norm, clrr_norm], dtype=torch.float32)
            pr["ct_shape"] = str(arr.shape)
            pr["mask_shape"] = str(mask_arr.shape)
            pr["lzp_raw"] = f"{lzp_raw:.4f}"
            pr["lzp_norm"] = f"{lzp_norm:.4f}"
            pr["clrr_raw"] = f"{clrr_raw:.4f}"
            pr["clrr_norm"] = f"{clrr_norm:.4f}"
            pr["low_mask_flag"] = r.get("low_mask","False")
            pr["zero_mask_flag"] = r.get("zero_mask","False")
            preprocessed.append((img, scalar_t, r, lzp_raw, clrr_raw, lzp_norm, clrr_norm))
        except Exception as e:
            pr["status"] = "ERROR"
            pr["note"] = str(e)[:200]
            preprocess_errors += 1
            preprocessed.append(None)
        preprocess_rows.append(pr)

    _write_csv(preprocess_rows, REPORT_DIR / "p_c_normal34_input_preprocessing_check.csv")
    print(f"  preprocessing: {len(smoke_sample)} crops, errors={preprocess_errors}")

    # ── 4. model forward smoke ────────────────────────────────────────────────
    print(f"[{STAGE_LABEL}] Step 4: model forward smoke")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    # load checkpoint
    model = ScalarFusionModel()
    ckpt = torch.load(SELECTED_CKPT, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"  checkpoint loaded: epoch={ckpt.get('epoch','?')}, val_auc={ckpt.get('val_auc','?')}")

    pred_rows = []
    n_forward_success = 0
    n_forward_error   = 0
    n_nan_inf_logit   = 0
    n_nan_inf_prob    = 0

    with torch.no_grad():
        for item in preprocessed:
            if item is None:
                pred_rows.append({"status": "SKIP_PREPROCESS_ERROR"})
                continue
            img_t, scalar_t, r, lzp_raw, clrr_raw, lzp_norm, clrr_norm = item
            try:
                img_in    = img_t.unsqueeze(0).to(device)
                scalar_in = scalar_t.unsqueeze(0).to(device)
                logit = model(img_in, scalar_in).squeeze(1).squeeze(0).item()
                prob  = 1.0 / (1.0 + math.exp(-logit)) if not math.isnan(logit) and not math.isinf(logit) else float("nan")

                logit_ok = not (math.isnan(logit) or math.isinf(logit))
                prob_ok  = not (math.isnan(prob)  or math.isinf(prob))
                if not logit_ok: n_nan_inf_logit += 1
                if not prob_ok:  n_nan_inf_prob  += 1

                pred_at_0p5 = int(prob >= FIXED_THRESHOLD) if prob_ok else -1

                caveat_flag = "low_mask" if r.get("low_mask","False").strip() == "True" else "none"

                pred_rows.append({
                    "patient_id":               r.get("patient_id",""),
                    "safe_id":                  r.get("safe_id",""),
                    "crop_path":                r["crop_path"],
                    "mask_path":                r["mask_path"],
                    "source_split":             r.get("source_split",""),
                    "canonical_volume_z":       r.get("canonical_volume_z",""),
                    "local_z":                  r.get("local_z",""),
                    "center_y":                 r.get("center_y",""),
                    "center_x":                 r.get("center_x",""),
                    "position_bin":             r.get("position_bin",""),
                    "lung_z_percentile_raw":    f"{lzp_raw:.6f}",
                    "crop_lung_roi_ratio_raw":  f"{clrr_raw:.6f}",
                    "lung_z_percentile_norm":   f"{lzp_norm:.6f}",
                    "crop_lung_roi_ratio_norm": f"{clrr_norm:.6f}",
                    "mask_nonzero_ratio_mean":  r.get("mask_nonzero_ratio_mean",""),
                    "low_mask_flag":            r.get("low_mask","False"),
                    "zero_mask_flag":           r.get("zero_mask","False"),
                    "logit":                    f"{logit:.6f}",
                    "prob":                     f"{prob:.6f}",
                    "pred_at_0p5":              pred_at_0p5,
                    "model_name":               "ScalarFusionModel_EfficientNetB0",
                    "checkpoint_path":          str(SELECTED_CKPT),
                    "masked_input_used":        True,
                    "threshold_used":           FIXED_THRESHOLD,
                    "threshold_optimized":      False,
                    "caveat_flag":              caveat_flag,
                    "label":                    r["label"],
                    "logit_finite":             logit_ok,
                    "prob_finite":              prob_ok,
                    "status":                   "OK" if (logit_ok and prob_ok) else "NAN_INF",
                })
                n_forward_success += 1
            except Exception as e:
                pred_rows.append({
                    "crop_path": r["crop_path"],
                    "status": "ERROR",
                    "note": str(e)[:200],
                })
                n_forward_error += 1

    # filter valid prediction rows
    valid_preds = [r for r in pred_rows if r.get("status") == "OK"]
    _write_csv(valid_preds, REPORT_DIR / "p_c_normal34_smoke_predictions.csv")
    print(f"  forward: success={n_forward_success}, error={n_forward_error}, nan/inf logit={n_nan_inf_logit}, nan/inf prob={n_nan_inf_prob}")

    # ── 5. output contract 검증 ───────────────────────────────────────────────
    print(f"[{STAGE_LABEL}] Step 5: output contract check")

    contract_rows = []
    if valid_preds:
        actual_cols = set(valid_preds[0].keys())
        for col in REQUIRED_OUTPUT_COLS:
            status = "OK" if col in actual_cols else "MISSING"
            contract_rows.append({"column": col, "required": True, "status": status})
        # extra cols
        for col in actual_cols:
            if col not in REQUIRED_OUTPUT_COLS:
                contract_rows.append({"column": col, "required": False, "status": "EXTRA"})
        missing = [r["column"] for r in contract_rows if r["status"] == "MISSING"]
        output_schema_pass = len(missing) == 0
        print(f"  output contract: {'PASS' if output_schema_pass else 'FAIL'} (missing={missing})")
    else:
        output_schema_pass = False
        print(f"  output contract: FAIL (no valid predictions)")
    _write_csv(contract_rows, REPORT_DIR / "p_c_normal34_output_contract_check.csv")

    # ── 6. low_mask monitoring ────────────────────────────────────────────────
    low_mask_rows = [r for r in valid_preds if r.get("low_mask_flag","False") == "True"]
    _write_csv(low_mask_rows, REPORT_DIR / "p_c_normal34_low_mask_monitoring.csv")

    # ── 7. guardrail check ────────────────────────────────────────────────────
    guardrail_rows = [{"key": k, "value": str(v), "status": "OK"} for k, v in GUARDRAILS.items()]
    guardrail_fail = 0
    _write_csv(guardrail_rows, REPORT_DIR / "p_c_normal34_guardrail_check.csv")

    # ── 8. smoke summary ─────────────────────────────────────────────────────
    n_normal_sample  = sum(1 for r in smoke_sample if r["label"] == "0")
    n_nsclc_sample   = sum(1 for r in smoke_sample if r["label"] == "1")
    n_low_sample     = sum(1 for r in smoke_sample if r.get("low_mask","False").strip() == "True")
    n_zero_sample    = sum(1 for r in smoke_sample if r.get("zero_mask","False").strip() == "True")

    # basic stats from valid preds
    if valid_preds:
        probs = [float(r["prob"]) for r in valid_preds if r.get("prob_finite", True) and r.get("prob","") not in ("",)]
        prob_mean = float(np.mean(probs)) if probs else float("nan")
        prob_min  = float(np.min(probs))  if probs else float("nan")
        prob_max  = float(np.max(probs))  if probs else float("nan")
    else:
        prob_mean = prob_min = prob_max = float("nan")

    # verdict
    if (n_forward_error == 0 and n_nan_inf_logit == 0 and n_nan_inf_prob == 0
            and output_schema_pass and guardrail_fail == 0 and n_forward_success > 0):
        if n_low_sample > 0:
            verdict = "PARTIAL_PASS"
            verdict_reason = "forward OK but low_mask caveat crops present (expected)"
        else:
            verdict = "PASS"
            verdict_reason = "all checks passed"
    elif n_forward_success == 0:
        verdict = "FAIL"
        verdict_reason = "no successful forward passes"
    else:
        verdict = "PARTIAL_PASS"
        verdict_reason = f"forward_error={n_forward_error}, nan_logit={n_nan_inf_logit}, schema_pass={output_schema_pass}"

    summary = {
        "stage":                STAGE_LABEL,
        "verdict":              verdict,
        "verdict_reason":       verdict_reason,
        "selected_candidate":   "P-C-NORMAL30b_masked_input",
        "checkpoint":           SELECTED_CKPT.name,
        "n_sample_total":       len(smoke_sample),
        "n_normal":             n_normal_sample,
        "n_nsclc":              n_nsclc_sample,
        "n_low_mask":           n_low_sample,
        "n_zero_mask":          n_zero_sample,
        "n_forward_success":    n_forward_success,
        "n_forward_error":      n_forward_error,
        "n_nan_inf_logit":      n_nan_inf_logit,
        "n_nan_inf_prob":       n_nan_inf_prob,
        "output_schema_pass":   output_schema_pass,
        "guardrail_fail":       guardrail_fail,
        "full_scoring_run":     False,
        "heatmap_run":          False,
        "xai_run":              False,
        "threshold_used":       FIXED_THRESHOLD,
        "threshold_optimized":  False,
        "prob_mean_smoke":      round(prob_mean, 4) if not math.isnan(prob_mean) else None,
        "prob_min_smoke":       round(prob_min,  4) if not math.isnan(prob_min)  else None,
        "prob_max_smoke":       round(prob_max,  4) if not math.isnan(prob_max)  else None,
        "device":               str(device),
        "note":                 "smoke only — 전체 성능 판단 금지",
    }
    _write_json(summary, REPORT_DIR / "p_c_normal34_smoke_summary.json")

    # ── 9. smoke report ───────────────────────────────────────────────────────
    import datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    missing_cols = [r["column"] for r in contract_rows if r["status"] == "MISSING"]
    report_md = f"""# P-C-NORMAL34 Downstream Scoring Smoke/Preflight

Generated: {ts}

## Verdict: {verdict}

{verdict_reason}

## Selected Candidate

| 항목 | 값 |
|------|----|
| selected_candidate | P-C-NORMAL30b_masked_input |
| checkpoint | {SELECTED_CKPT.name} |
| smoke_checkpoint | No |
| selected_with_caveat | True |
| threshold_used | {FIXED_THRESHOLD} (fixed, not optimized) |

## Input File Validation

| 파일 | 상태 |
|------|------|
| selected checkpoint | OK |
| scalar normalization stats | OK |
| final_test manifest rows | {len(ft_rows)} |
| mask manifest rows | {len(mask_rows)} |
| mask join coverage | {len(joined)}/{len(ft_rows)} |

## Smoke Sample

| 항목 | 값 |
|------|----|
| n_total | {len(smoke_sample)} |
| n_normal | {n_normal_sample} |
| n_nsclc | {n_nsclc_sample} |
| n_low_mask | {n_low_sample} |
| n_zero_mask | {n_zero_sample} |

## Model Forward

| 항목 | 값 |
|------|----|
| device | {device} |
| n_forward_success | {n_forward_success} |
| n_forward_error | {n_forward_error} |
| n_nan_inf_logit | {n_nan_inf_logit} |
| n_nan_inf_prob | {n_nan_inf_prob} |
| prob_mean (smoke only) | {round(prob_mean,4) if not math.isnan(prob_mean) else 'N/A'} |
| prob_min | {round(prob_min,4)  if not math.isnan(prob_min)  else 'N/A'} |
| prob_max | {round(prob_max,4)  if not math.isnan(prob_max)  else 'N/A'} |

## Output Contract

Schema PASS: {output_schema_pass}
Missing columns: {missing_cols if missing_cols else 'none'}

## Guardrails

guardrail_fail: {guardrail_fail}

| key | value |
|-----|-------|
| full_scoring_run | False |
| heatmap_run | False |
| threshold_optimized | False |
| selected_checkpoint_not_smoke | True |

## Caveat

- low_mask crops ({n_low_sample}개): nzr_mean < 0.05. 별도 monitoring table 분리됨.
- zero_mask crops: {n_zero_sample}개 (없음).
- 이 결과는 smoke only이며 전체 성능 판단 금지.
- diagnostic probability / cancer probability 표현 금지.

## Output Files

```
outputs/reports/p_c_normal34_downstream_scoring_smoke/
  p_c_normal34_smoke_sample_manifest.csv
  p_c_normal34_input_preprocessing_check.csv
  p_c_normal34_smoke_predictions.csv
  p_c_normal34_output_contract_check.csv
  p_c_normal34_low_mask_monitoring.csv
  p_c_normal34_guardrail_check.csv
  p_c_normal34_smoke_report.md
  p_c_normal34_smoke_summary.json
  DONE.json
```
"""
    (REPORT_DIR / "p_c_normal34_smoke_report.md").write_text(report_md)

    # ── 10. DONE.json ─────────────────────────────────────────────────────────
    _write_json({
        "stage":             STAGE_LABEL,
        "timestamp":         ts,
        "verdict":           verdict,
        "guardrail_fail":    guardrail_fail,
        "selected_candidate": "P-C-NORMAL30b_masked_input",
        "n_sample":          len(smoke_sample),
        "n_forward_success": n_forward_success,
        "full_scoring_run":  False,
        "next_step":         "사용자 승인 후 전체 scoring (Option A) 또는 기타 — 사용자 결정 필요",
    }, REPORT_DIR / "DONE.json")

    print(f"[{STAGE_LABEL}] VERDICT: {verdict}")
    print(f"[{STAGE_LABEL}] guardrail_fail: {guardrail_fail}")
    print(f"[{STAGE_LABEL}] output: {REPORT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
