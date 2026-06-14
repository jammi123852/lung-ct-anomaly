"""
B1-D8g: N-C refiner query scoring
19개 refiner candidate crop을 N-C normal-only distribution(mean/cov_inv)에 query scoring한다.

실행 방법:
  source ~/ai_env/bin/activate && python scripts/b1d8g_nc_refiner_query_scoring.py --confirm-execute

기본 차단:
  --confirm-execute 없으면 실행 불가.

GPU 사용 시:
  --confirm-gpu 추가 필요.

절대 금지:
  stage2_holdout 접근 / threshold 재계산 / score/threshold/ROI 수정
  original_padim_score 수정 / adjusted_score/suppression_weight 생성
  crop 재생성 / N-C distribution artifact 수정 / training

B1-D8g 예상 출력:
  outputs/b1d1_wall_mediastinum_fp_cause_diagnostic_v1/b1d8g_nc_refiner_query_scoring/
    b1d8g_nc_refiner_query_scores.csv
    b1d8g_nc_refiner_query_scoring_summary.json
    b1d8g_nc_refiner_query_scoring_report.md
    b1d8g_nc_refiner_query_score_by_group.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXP_ROOT     = PROJECT_ROOT / "experiments" / "normal_only_second_stage_refiner_v1"
B1D_ROOT     = PROJECT_ROOT / "outputs" / "b1d1_wall_mediastinum_fp_cause_diagnostic_v1"
OUT_DIR      = B1D_ROOT / "b1d8g_nc_refiner_query_scoring"

CROP_DIR     = B1D_ROOT / "b1d8d_refiner_candidate_crops"
LABELS_CSV   = CROP_DIR / "b1d8d_refiner_candidate_crop_labels.csv"
INTEGRITY_CSV= CROP_DIR / "b1d8d_refiner_candidate_crop_integrity.csv"

NC_STATS_NPZ = EXP_ROOT / "outputs" / "models" / "n_c7_full_position_bin_distribution" / "n_c7_position_bin_stats.npz"
NC_META_JSON = EXP_ROOT / "outputs" / "models" / "n_c7_full_position_bin_distribution" / "n_c7_distribution_metadata.json"
NC8_VALID_JSON = EXP_ROOT / "outputs" / "reports" / "n_c8_full_distribution_artifact_validation" / "n_c8_full_distribution_artifact_validation.json"
NC10_THR_JSON  = EXP_ROOT / "outputs" / "evaluation" / "n_c10_normal_val_thresholds" / "n_c10_normal_val_thresholds.json"

EXPECTED_CROP_COUNT  = 19
EXPECTED_FEATURE_DIM = 100
EXPECTED_BINS = {"upper_central","upper_peripheral","middle_central","middle_peripheral","lower_central","lower_peripheral"}

HU_MIN        = -1000.0
HU_MAX        = 200.0
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

OUTPUT_COLUMNS = [
    "refiner_candidate_id",
    "source_stage",
    "patient_id",
    "safe_id",
    "position_bin",
    "proposed_taxonomy_label",
    "rule_b3_flag",
    "gate_p2_flag",
    "train_memory_overlap_risk",
    "original_padim_score",
    "nc_query_score",
    "nc_threshold_p95_if_available",
    "nc_threshold_p99_if_available",
    "nc_p95_exceed",
    "nc_p99_exceed",
    "interpretation_caution",
    "stage2_holdout_flag",
]


_STAGE2_HOLDOUT_PATTERNS = [
    str(PROJECT_ROOT / "outputs" / "position-aware-padim-v1" / "stage2"),
    str(PROJECT_ROOT / "data" / "holdout"),
    "stage2_holdout",
    "holdout",
]


def _abort(msg: str):
    print(f"[BLOCKED] {msg}", file=sys.stderr)
    sys.exit(2)


def _check_stage2_access(path_str: str) -> bool:
    p_lower = str(path_str).lower()
    return any(pat.lower() in p_lower for pat in _STAGE2_HOLDOUT_PATTERNS)


def _safety_preflight(args):
    import numpy as np
    errors = []

    # stage2_holdout 경로 접근 금지 (구체적 경로 체크, rglob 없음)
    stage2_holdout_access = 0
    for pat in _STAGE2_HOLDOUT_PATTERNS:
        p = Path(pat)
        if p.exists():
            print(f"[GUARD] stage2_holdout 경로 감지 (접근 금지): {pat}")
            stage2_holdout_access += 1

    # output collision (final dir)
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        errors.append(f"OUT_DIR already has files: {OUT_DIR}")

    # output tmp collision
    OUT_DIR_TMP = OUT_DIR.parent / (OUT_DIR.name + "_tmp")
    if OUT_DIR_TMP.exists():
        errors.append(f"OUT_DIR_TMP already exists (prior failed run?): {OUT_DIR_TMP}")

    # N-C stats 존재
    if not NC_STATS_NPZ.exists():
        errors.append(f"NC_STATS_NPZ not found: {NC_STATS_NPZ}")
    else:
        stats = np.load(NC_STATS_NPZ, allow_pickle=False)
        if "selected_indices" not in stats.files:
            errors.append("NC_STATS_NPZ missing 'selected_indices' key")
        else:
            idx_shape = stats["selected_indices"].shape
            if list(idx_shape) != [100]:
                errors.append(f"selected_indices shape={idx_shape}, expected=[100]")

    # N-C8 validation pass 확인
    if NC8_VALID_JSON.exists():
        with open(NC8_VALID_JSON) as f:
            nc8 = json.load(f)
        if nc8.get("verdict") not in ("통과", "PASS"):
            errors.append(f"N-C8 verdict not PASS: {nc8.get('verdict')}")
    else:
        errors.append(f"NC8_VALID_JSON not found: {NC8_VALID_JSON}")

    # crop count + NPZ key 전수 검증
    npz_files = sorted(CROP_DIR.glob("RCP_*.npz"))
    if len(npz_files) != EXPECTED_CROP_COUNT:
        errors.append(f"crop_count={len(npz_files)}, expected={EXPECTED_CROP_COUNT}")
    else:
        required_keys = ["crop", "mask_crop", "metadata"]
        for npz_path in npz_files:
            npz = np.load(npz_path, allow_pickle=True)
            for k in required_keys:
                if k not in npz.files:
                    errors.append(f"NPZ key missing: {npz_path.name} missing '{k}'")
            if "crop" in npz.files and npz["crop"].shape != (3, 96, 96):
                errors.append(f"crop shape mismatch: {npz_path.name} shape={npz['crop'].shape}")
            if "mask_crop" in npz.files and npz["mask_crop"].shape != (96, 96):
                errors.append(f"mask_crop shape mismatch: {npz_path.name} shape={npz['mask_crop'].shape}")
            # stage2_holdout 입력 경로 체크
            meta_raw = npz["metadata"]
            if hasattr(meta_raw, "item"):
                meta = json.loads(str(meta_raw.item()))
            else:
                meta = json.loads(str(meta_raw))
            if meta.get("stage2_holdout_flag", 0) != 0:
                errors.append(f"{npz_path.name}: stage2_holdout_flag={meta.get('stage2_holdout_flag')}")

    # threshold artifact
    if not NC10_THR_JSON.exists():
        errors.append(f"NC10_THR_JSON not found: {NC10_THR_JSON}")

    return errors


def _load_labels():
    rows = []
    with open(LABELS_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _preprocess_crop(crop_hu):
    import numpy as np
    clipped = np.clip(crop_hu, HU_MIN, HU_MAX)
    normed  = (clipped - HU_MIN) / (HU_MAX - HU_MIN)
    mean    = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std     = np.array(IMAGENET_STD,  dtype=np.float32).reshape(3, 1, 1)
    return ((normed.astype(np.float32) - mean) / std).astype(np.float32)


def _compute_mahalanobis(feat_100, mean_b, cov_inv_b):
    import numpy as np
    delta  = feat_100.astype(np.float64) - mean_b.astype(np.float64)
    d_sq   = float(delta @ cov_inv_b.astype(np.float64) @ delta)
    return float(np.sqrt(max(0.0, d_sq)))


def _load_feature_extractor():
    """N-C10과 동일한 EfficientNet-B0 feature extractor (f_early, f_mid, f_late 반환)."""
    import torch
    import torchvision.models as tv_models

    class _FeatureExtractorEffNetB0(torch.nn.Module):
        def __init__(self):
            super().__init__()
            base = tv_models.efficientnet_b0(
                weights=tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1
            )
            features = base.features
            self.stem   = features[:2]   # ch=32
            self.stage1 = features[2]    # ch=16
            self.stage2 = features[3]    # ch=24  (early tap)
            self.stage3 = features[4]    # ch=40  (mid tap)
            self.stage4 = features[5]    # ch=80  (late tap)

        def forward(self, x):
            x = self.stem(x)
            x = self.stage1(x)
            f_early = self.stage2(x)
            f_mid   = self.stage3(f_early)
            f_late  = self.stage4(f_mid)
            return f_early, f_mid, f_late

    extractor = _FeatureExtractorEffNetB0()
    extractor.eval()
    return extractor


def _extract_3x3_spatial_features(f_early, f_mid, f_late):
    """N-C10과 동일한 3×3 spatial pooling → (9, 144) numpy array."""
    import torch.nn.functional as F
    import numpy as np

    def _pool_reshape(feat):
        pooled = F.adaptive_avg_pool2d(feat, (3, 3)).squeeze(0)  # (C, 3, 3)
        C = pooled.shape[0]
        arr = pooled.cpu().numpy().transpose(1, 2, 0).reshape(9, C)
        return arr.astype(np.float32)

    a = _pool_reshape(f_early)  # (9, 24)
    b = _pool_reshape(f_mid)    # (9, 40)
    c = _pool_reshape(f_late)   # (9, 80)
    return np.concatenate([a, b, c], axis=1)  # (9, 144)


def run_scoring(args):
    import numpy as np
    import torch

    # OUT_DIR collision guard (final)
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        _abort(f"OUT_DIR already has files: {OUT_DIR}. 덮어쓰기 금지.")

    # OUT_DIR_TMP collision guard
    OUT_DIR_TMP = OUT_DIR.parent / (OUT_DIR.name + "_tmp")
    if OUT_DIR_TMP.exists():
        _abort(f"OUT_DIR_TMP already exists: {OUT_DIR_TMP}. 이전 실패 잔여물 확인 필요.")

    # 모든 출력은 TMP에 먼저 저장
    OUT_DIR_TMP.mkdir(parents=True, exist_ok=True)

    # N-C distribution 로드
    stats = np.load(NC_STATS_NPZ, allow_pickle=False)
    selected_indices = stats["selected_indices"].astype(int)
    assert len(selected_indices) == EXPECTED_FEATURE_DIM, \
        f"selected_indices len={len(selected_indices)}, expected={EXPECTED_FEATURE_DIM}"

    nc_means    = {b: stats[f"mean_{b}"].astype(np.float64)    for b in EXPECTED_BINS}
    nc_cov_invs = {b: stats[f"cov_inv_{b}"].astype(np.float64) for b in EXPECTED_BINS}

    # threshold 로드 (read-only)
    with open(NC10_THR_JSON) as f:
        thr = json.load(f)
    global_p95 = thr["thresholds"]["crop_score_max_p95"]
    global_p99 = thr["thresholds"]["crop_score_max_p99"]
    per_bin_p95 = {
        b: thr["thresholds"].get(f"per_bin_{b}_crop_score_max_p95")
        for b in EXPECTED_BINS
    }

    # feature extractor 초기화 (N-C10과 동일한 FeatureExtractorEffNetB0)
    device = "cuda" if (args.confirm_gpu and torch.cuda.is_available()) else "cpu"
    extractor = _load_feature_extractor()
    extractor = extractor.to(device)
    extractor.eval()

    # labels 로드
    labels = _load_labels()

    output_rows = []

    for row in labels:
        rcp_id       = row["refiner_candidate_id"]
        position_bin = row["position_bin"]
        safe_id      = row.get("safe_id", "")
        patient_id   = row.get("patient_id", "")
        stage2_flag  = int(row.get("stage2_holdout_flag", 0))
        overlap_risk = int(row.get("train_memory_overlap_risk", 0))
        orig_score   = row.get("original_padim_score", "")

        # stage2_holdout 차단
        if stage2_flag != 0:
            _abort(f"{rcp_id}: stage2_holdout_flag={stage2_flag} — 실행 차단")

        # position_bin 확인
        if position_bin not in EXPECTED_BINS:
            _abort(f"{rcp_id}: position_bin={position_bin} not in expected bins")

        # crop 로드 (mask key는 mask_crop)
        npz_path = CROP_DIR / f"{rcp_id}.npz"
        if not npz_path.exists():
            _abort(f"Missing NPZ: {npz_path}")
        npz      = np.load(npz_path, allow_pickle=True)
        crop_hu  = npz["crop"]
        mask     = npz["mask_crop"]

        # shape 확인
        assert crop_hu.shape == (3, 96, 96), f"{rcp_id}: crop shape={crop_hu.shape}"
        assert mask.shape    == (96, 96),     f"{rcp_id}: mask shape={mask.shape}"
        assert not np.isnan(crop_hu).any() and not np.isinf(crop_hu).any(), \
            f"{rcp_id}: NaN/Inf in crop"

        # preprocessing
        crop_tensor = _preprocess_crop(crop_hu)
        crop_t = torch.from_numpy(crop_tensor).unsqueeze(0).to(device)

        # feature extraction (N-C10과 동일: f_early/f_mid/f_late → 3×3 spatial → 9×144 → 9×100)
        with torch.no_grad():
            f_e, f_m, f_l = extractor(crop_t)

        feats_9x144 = _extract_3x3_spatial_features(f_e, f_m, f_l)  # (9, 144)
        feats_9x100 = feats_9x144[:, selected_indices]               # (9, 100)

        assert feats_9x100.shape == (9, EXPECTED_FEATURE_DIM), \
            f"{rcp_id}: feats_9x100 shape={feats_9x100.shape}"

        # 9개 spatial Mahalanobis score → max (N-C10 crop_score_max_p95 기준과 동일)
        spatial_scores = []
        for sp_idx in range(9):
            feat_sel = feats_9x100[sp_idx]
            assert not np.isnan(feat_sel).any() and not np.isinf(feat_sel).any(), \
                f"{rcp_id}[spatial={sp_idx}]: NaN/Inf in feature"
            s = _compute_mahalanobis(feat_sel, nc_means[position_bin], nc_cov_invs[position_bin])
            spatial_scores.append(s)

        nc_query_score = float(max(spatial_scores))  # crop_score_max_p95 기준과 비교 가능

        # threshold 비교 (read-only, 재계산 없음)
        nc_p95_exceed = (nc_query_score > global_p95) if global_p95 is not None else None
        nc_p99_exceed = (nc_query_score > global_p99) if global_p99 is not None else None

        caution = ""
        if overlap_risk:
            caution = "train_memory_overlap_risk=1: 정상-like 결론 강하게 내리지 말 것"

        output_rows.append({
            "refiner_candidate_id":          rcp_id,
            "source_stage":                  row.get("source_stage", ""),
            "patient_id":                    patient_id,
            "safe_id":                       safe_id,
            "position_bin":                  position_bin,
            "proposed_taxonomy_label":       row.get("proposed_taxonomy_label", ""),
            "rule_b3_flag":                  row.get("rule_b3_flag", ""),
            "gate_p2_flag":                  row.get("gate_p2_flag", ""),
            "train_memory_overlap_risk":     overlap_risk,
            "original_padim_score":          orig_score,
            "nc_query_score":                round(nc_query_score, 6),
            "nc_threshold_p95_if_available": round(global_p95, 6) if global_p95 else "",
            "nc_threshold_p99_if_available": round(global_p99, 6) if global_p99 else "",
            "nc_p95_exceed":                 int(nc_p95_exceed) if nc_p95_exceed is not None else "",
            "nc_p99_exceed":                 int(nc_p99_exceed) if nc_p99_exceed is not None else "",
            "interpretation_caution":        caution,
            "stage2_holdout_flag":           stage2_flag,
        })

    # 모든 파일을 TMP에 저장
    out_scores_csv   = OUT_DIR_TMP / "b1d8g_nc_refiner_query_scores.csv"
    out_summary_json = OUT_DIR_TMP / "b1d8g_nc_refiner_query_scoring_summary.json"
    out_report_md    = OUT_DIR_TMP / "b1d8g_nc_refiner_query_scoring_report.md"
    out_group_csv    = OUT_DIR_TMP / "b1d8g_nc_refiner_query_score_by_group.csv"

    # scores CSV 저장
    with open(out_scores_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(output_rows)

    # group CSV (caution group 분리)
    normal_rows  = [r for r in output_rows if not r["train_memory_overlap_risk"]]
    caution_rows = [r for r in output_rows if r["train_memory_overlap_risk"]]

    with open(out_group_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS + ["group"])
        writer.writeheader()
        for r in normal_rows:
            writer.writerow({**r, "group": "standard"})
        for r in caution_rows:
            writer.writerow({**r, "group": "caution_train_overlap"})

    # summary JSON
    summary = {
        "step":              "B1-D8g",
        "crop_count":        len(output_rows),
        "nc_distribution":   str(NC_STATS_NPZ),
        "threshold_source":  str(NC10_THR_JSON),
        "score_formula":     "sqrt_mahalanobis",
        "spatial_feature_count": 9,
        "selected_feature_dim":  EXPECTED_FEATURE_DIM,
        "nc_query_score_formula": "max(9 spatial Mahalanobis distances)",
        "primary_threshold": "crop_score_max_p95",
        "stage2_holdout_accessed": False,
        "feature_extracted": True,
        "scoring_done":      True,
        "threshold_recomputed": False,
        "score_modified":    False,
        "train_overlap_caution_count": len(caution_rows),
        "train_overlap_rcp": [r["refiner_candidate_id"] for r in caution_rows],
        "output_files": {
            "scores_csv":   str(OUT_DIR / "b1d8g_nc_refiner_query_scores.csv"),
            "summary_json": str(OUT_DIR / "b1d8g_nc_refiner_query_scoring_summary.json"),
            "report_md":    str(OUT_DIR / "b1d8g_nc_refiner_query_scoring_report.md"),
            "group_csv":    str(OUT_DIR / "b1d8g_nc_refiner_query_score_by_group.csv"),
        },
        "guardrail_flags": {
            "stage2_holdout_accessed":       False,
            "threshold_recomputed":          False,
            "score_modified":                False,
            "original_padim_score_modified": False,
            "adjusted_score_generated":      False,
            "suppression_weight_generated":  False,
        },
    }
    with open(out_summary_json, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # report MD
    exceed_p95 = [r["refiner_candidate_id"] for r in output_rows if r["nc_p95_exceed"] == 1]
    lines = [
        "# B1-D8g N-C Refiner Query Scoring Report",
        "",
        f"- crop_count: {len(output_rows)}",
        "- score_formula: sqrt_mahalanobis",
        "- spatial_feature_count: 9",
        f"- selected_feature_dim: {EXPECTED_FEATURE_DIM}",
        "- nc_query_score_formula: max(9 spatial Mahalanobis distances)",
        f"- threshold (global p95): {global_p95}",
        f"- threshold (global p99): {global_p99}",
        f"- nc_p95_exceed count: {len(exceed_p95)}",
        f"- nc_p95_exceed RCPs: {exceed_p95}",
        "",
        "## RCP_012 Caution",
        "- train_memory_overlap_risk=1",
        "- score가 낮아도 정상-like 결론을 강하게 내리지 말 것",
        "",
        "## Guardrails",
        "- stage2_holdout_accessed: False",
        "- threshold_recomputed: False",
        "- score_modified: False",
    ]
    with open(out_report_md, "w") as f:
        f.write("\n".join(lines))

    # 검증: 4개 파일 모두 존재
    for p in [out_scores_csv, out_summary_json, out_report_md, out_group_csv]:
        if not p.exists():
            _abort(f"Output file missing after write: {p}")

    # TMP → final rename (atomic)
    OUT_DIR_TMP.rename(OUT_DIR)

    print("[B1-D8g] Scoring 완료.")
    print(f"  scores: {OUT_DIR / 'b1d8g_nc_refiner_query_scores.csv'}")
    print(f"  summary: {OUT_DIR / 'b1d8g_nc_refiner_query_scoring_summary.json'}")


def main():
    parser = argparse.ArgumentParser(description="B1-D8g N-C refiner query scoring")
    parser.add_argument("--confirm-execute", action="store_true",
                        help="실제 실행 승인 (없으면 차단)")
    parser.add_argument("--confirm-gpu", action="store_true",
                        help="GPU 사용 승인")
    parser.add_argument("--dry-run", action="store_true",
                        help="safety preflight만 실행, scoring 없음")
    args = parser.parse_args()

    # dry-run은 --confirm-execute 없이 실행 가능 (scoring 없음)
    # bare-run (flags 없음)은 차단
    if not args.confirm_execute and not args.dry_run:
        _abort(
            "B1-D8g는 기본 차단 상태입니다.\n"
            "  dry-run (preflight only): --dry-run\n"
            "  real scoring: --confirm-execute [--confirm-gpu]\n"
            "예: python scripts/b1d8g_nc_refiner_query_scoring.py --confirm-execute"
        )

    errors = _safety_preflight(args)
    if errors:
        for e in errors:
            print(f"[PREFLIGHT_FAIL] {e}", file=sys.stderr)
        _abort("Preflight 실패. 위 항목을 먼저 해결하세요.")

    if args.dry_run:
        print("[dry-run] Preflight PASS. Scoring은 --confirm-execute 추가 후 실행하세요.")
        return

    run_scoring(args)


if __name__ == "__main__":
    main()
