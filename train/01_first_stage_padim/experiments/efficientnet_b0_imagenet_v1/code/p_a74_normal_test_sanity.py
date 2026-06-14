"""
P-A74: EfficientNet-B0 ImageNet PaDiM — normal test 36명 sanity check.

- P-A73에서 산출한 threshold(p95/p99)를 read-only로 로드.
- normal test 36명 scoring → p95/p99 초과율 확인.
- threshold 재계산 금지. lesion/stage1_dev/stage2_holdout 접근 금지.

실행:
  source ~/ai_env/bin/activate && python experiments/efficientnet_b0_imagenet_v1/code/p_a74_normal_test_sanity.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJ_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT  = Path(__file__).resolve().parents[1]
SRC_DIR   = PROJ_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

BACKBONE            = "efficientnet_b0"
PRETRAIN_SOURCE     = "imagenet"
RAW_FEATURE_DIM     = 144
REDUCED_FEATURE_DIM = 100
MASK_TYPE           = "roi_0_0"
PATHS_CONFIG        = "paths.local.v2_roi0_0.yaml"
SCRIPT_NAME         = "p_a74_normal_test_sanity.py"
RUN_TAG             = "padim_efficientnet_b0_imagenet"
EXPECTED_TEST_N     = 36

# ---- 입력 경로 ----
MODEL_NPZ        = EXP_ROOT / "outputs" / "models" / "distributions" / "position_bin_stats.npz"
SELECTED_INDICES = EXP_ROOT / "outputs" / "models" / "distributions" / "selected_feature_indices.npy"
THRESH_JSON      = EXP_ROOT / "outputs" / "evaluation" / "normal_val_thresholds" / "normal_val_threshold.json"
SPLIT_JSON       = PROJ_ROOT / "outputs" / "position-aware-padim-v1" / "splits" / "normal_v1.json"
P_A73_REPORT_MD  = EXP_ROOT / "outputs" / "reports" / "normal_val" / "p_a73_normal_val_threshold.md"

# ---- 출력 경로 ----
SCORE_DIR   = EXP_ROOT / "outputs" / "scores" / "normal_test_by_patient"
SANITY_DIR  = EXP_ROOT / "outputs" / "evaluation" / "normal_test_sanity"
REPORT_DIR  = EXP_ROOT / "outputs" / "reports" / "normal_test"

SANITY_JSON         = SANITY_DIR / "normal_test_sanity.json"
SANITY_CSV          = SANITY_DIR / "normal_test_sanity.csv"
PER_PATIENT_CSV     = SANITY_DIR / "normal_test_per_patient.csv"
REPORT_MD           = REPORT_DIR / "p_a74_normal_test_sanity.md"
REPORT_JSON         = REPORT_DIR / "p_a74_normal_test_sanity.json"
RUNTIME_CSV         = REPORT_DIR / "p_a74_runtime_summary.csv"

# 고정 threshold (P-A73)
EXPECTED_P95 = 13.240479
EXPECTED_P99 = 15.332286
THRESH_TOLERANCE = 1e-4


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def abort(msg: str, code: int = 2):
    print(f"[P-A74][ABORT] {msg}")
    sys.exit(code)


def main() -> int:
    # ---- G1: P-A73 보고서 통과 확인 ----
    if not P_A73_REPORT_MD.exists():
        abort(f"P-A73 보고서 없음: {P_A73_REPORT_MD}")
    with open(P_A73_REPORT_MD, encoding="utf-8") as f:
        md_text = f.read()
    if "판정: 통과" not in md_text:
        abort(f"P-A73 보고서가 통과 상태가 아님: {P_A73_REPORT_MD}")
    print("[G1] P-A73 보고서 통과 확인 ✅")

    # ---- G2: P-A73 threshold JSON 존재 및 read-only 로드 ----
    if not THRESH_JSON.exists():
        abort(f"P-A73 threshold JSON 없음: {THRESH_JSON}")
    thresh_mtime_before = os.path.getmtime(THRESH_JSON)
    with open(THRESH_JSON, encoding="utf-8") as f:
        thresh_data = json.load(f)
    p95_threshold = float(thresh_data["threshold_p95"])
    p99_threshold = float(thresh_data["threshold_p99"])
    print(f"[G2] threshold JSON 로드: p95={p95_threshold:.6f}, p99={p99_threshold:.6f}")

    # ---- G3: threshold 값 일치 확인 ----
    if abs(p95_threshold - EXPECTED_P95) > THRESH_TOLERANCE:
        abort(f"p95 threshold 불일치: {p95_threshold:.6f} (기대: {EXPECTED_P95})")
    if abs(p99_threshold - EXPECTED_P99) > THRESH_TOLERANCE:
        abort(f"p99 threshold 불일치: {p99_threshold:.6f} (기대: {EXPECTED_P99})")
    print(f"[G3] threshold 값 일치 ✅: p95={p95_threshold:.6f}, p99={p99_threshold:.6f}")

    # ---- G4: distribution npz 존재 확인 ----
    if not MODEL_NPZ.exists():
        abort(f"distribution 파일 없음: {MODEL_NPZ}")
    print(f"[G4] distribution 존재: {MODEL_NPZ}")

    # ---- G5: selected_feature_indices 검증 ----
    if not SELECTED_INDICES.exists():
        abort(f"selected_feature_indices.npy 없음: {SELECTED_INDICES}")
    idx = np.load(SELECTED_INDICES)
    if idx.shape != (REDUCED_FEATURE_DIM,):
        abort(f"selected_index shape 불일치: {idx.shape} (기대: ({REDUCED_FEATURE_DIM},))")
    if len(set(idx.tolist())) != REDUCED_FEATURE_DIM:
        abort(f"selected_index unique 불일치: {len(set(idx.tolist()))}")
    if not ((idx >= 0).all() and (idx < RAW_FEATURE_DIM).all()):
        abort(f"selected_index range 불일치: min={int(idx.min())}, max={int(idx.max())}")
    print(f"[G5] selected_feature_indices OK: shape={idx.shape}, unique={REDUCED_FEATURE_DIM}, range=[{int(idx.min())},{int(idx.max())}]")

    # ---- G6: normal test split 36명 확인 ----
    if not SPLIT_JSON.exists():
        abort(f"split JSON 없음: {SPLIT_JSON}")
    with open(SPLIT_JSON, encoding="utf-8-sig") as f:
        split_data = json.load(f)
    test_patients = list(split_data["test"])
    if len(test_patients) != EXPECTED_TEST_N:
        abort(f"normal test 환자 수가 {EXPECTED_TEST_N}이 아님: {len(test_patients)}")
    print(f"[G6] normal test split: {len(test_patients)}명 확인 ✅")

    # ---- G7: 출력 경로 중복 확인 (덮어쓰기 금지) ----
    if SCORE_DIR.exists() and any(SCORE_DIR.glob("*.csv")):
        abort(f"기존 normal test score 존재 → 덮어쓰기 금지: {SCORE_DIR}")
    if SANITY_JSON.exists():
        abort(f"기존 sanity JSON 존재 → 덮어쓰기 금지: {SANITY_JSON}")
    if REPORT_JSON.exists():
        abort(f"기존 P-A74 report JSON 존재 → 덮어쓰기 금지: {REPORT_JSON}")
    print("[G7] 기존 출력 없음, 충돌 없음 ✅")

    # ---- G8: EfficientNet-B0 weight 존재 확인 (재다운로드 금지) ----
    import torch
    from torchvision.models import EfficientNet_B0_Weights
    wname = EfficientNet_B0_Weights.IMAGENET1K_V1.url.rsplit("/", 1)[-1]
    wpath = Path(torch.hub.get_dir()) / "checkpoints" / wname
    if not wpath.exists():
        abort(f"EfficientNet-B0 weight 없음(재다운로드 금지): {wpath}")
    print(f"[G8] EfficientNet-B0 weight 존재: {wpath}")

    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    SANITY_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    dist_sha = sha256_of(MODEL_NPZ)
    print(f"[P-A74] distribution sha256: {dist_sha}")
    print(f"[P-A74] backbone={BACKBONE}, raw_dim={RAW_FEATURE_DIM}, reduced_dim={REDUCED_FEATURE_DIM}")
    print(f"[P-A74] 사용 threshold: p95={p95_threshold:.6f}, p99={p99_threshold:.6f} (read-only)")

    # ---- 모델 / 추출기 ----
    from position_aware_padim.config_manager import ConfigManager
    from position_aware_padim.data_loader import DataLoader
    from position_aware_padim.path_resolver import PathResolver
    from position_aware_padim.padim_model import PaDiMModel
    from position_aware_padim.feature_extractor_effnet_b0_scaffold import FeatureExtractorEffNetB0

    cfg = ConfigManager(str(PROJ_ROOT))
    cfg.load_config(paths_yaml=PATHS_CONFIG)
    normal_training_ready = cfg.get("paths", "normal_training_ready", "")
    if not normal_training_ready:
        abort("paths config에 normal_training_ready 없음")
    manifest_path = Path(normal_training_ready) / "manifests" / "patient_manifest.csv"
    if not manifest_path.exists():
        abort(f"manifest 없음: {manifest_path}")

    model = PaDiMModel(
        selected_feature_indices_path=str(SELECTED_INDICES),
        feature_dim=REDUCED_FEATURE_DIM,
        eps=1e-5,
    )
    model.load(str(MODEL_NPZ))
    print(f"[P-A74] distribution 로드 완료: position_bin 수={len(model.stats)}")

    feat = FeatureExtractorEffNetB0()
    print(f"[P-A74] device: {feat.device}")

    path_resolver = PathResolver(str(manifest_path), normal_training_ready)
    loader = DataLoader(
        str(manifest_path),
        path_resolver,
        str(REPORT_DIR / "error.csv"),
        use_mmap=True,
    )

    # ---- normal test scoring ----
    all_scores = []
    per_patient_rows = []
    n_csv = 0
    n_failed = 0
    missing = []
    start = time.time()

    for i, pid in enumerate(test_patients, 1):
        data = loader.load_patient_data(pid, mask_type=MASK_TYPE)
        if data is None:
            n_failed += 1
            missing.append(pid)
            print(f"  [FAIL] ({i}/{EXPECTED_TEST_N}) {pid}: 로드 실패")
            per_patient_rows.append({
                "patient_id": pid, "n_patches": 0, "nan": 0, "inf": 0,
                "min": None, "max": None, "mean": None, "std": None, "median": None,
                "n_exceed_p95": 0, "rate_exceed_p95": 0.0,
                "n_exceed_p99": 0, "rate_exceed_p99": 0.0,
                "status": "FAIL",
            })
            continue
        scored = model.score_patient(feat, data)
        out_csv = SCORE_DIR / f"{pid}.csv"
        scored.to_csv(out_csv, index=False, encoding="utf-8-sig")
        n_csv += 1
        s = scored["padim_score"].to_numpy(dtype=np.float64)
        all_scores.append(s)
        s_finite = s[np.isfinite(s)]
        n_exc_p95 = int((s_finite > p95_threshold).sum())
        n_exc_p99 = int((s_finite > p99_threshold).sum())
        n_total_p = len(s_finite) if len(s_finite) > 0 else 1
        per_patient_rows.append({
            "patient_id": pid,
            "n_patches": len(s),
            "nan": int(np.isnan(s).sum()),
            "inf": int(np.isinf(s).sum()),
            "min": float(s_finite.min()) if len(s_finite) > 0 else None,
            "max": float(s_finite.max()) if len(s_finite) > 0 else None,
            "mean": float(s_finite.mean()) if len(s_finite) > 0 else None,
            "std": float(s_finite.std()) if len(s_finite) > 0 else None,
            "median": float(np.median(s_finite)) if len(s_finite) > 0 else None,
            "n_exceed_p95": n_exc_p95,
            "rate_exceed_p95": round(n_exc_p95 / n_total_p, 6),
            "n_exceed_p99": n_exc_p99,
            "rate_exceed_p99": round(n_exc_p99 / n_total_p, 6),
            "status": "OK",
        })
        print(f"  [OK]   ({i}/{EXPECTED_TEST_N}) {pid}: patches={len(s)}, "
              f"nan={int(np.isnan(s).sum())}, exc_p95={n_exc_p95}({n_exc_p95/n_total_p:.2%}), "
              f"exc_p99={n_exc_p99}({n_exc_p99/n_total_p:.2%})")

    elapsed = time.time() - start

    scores = np.concatenate(all_scores) if all_scores else np.array([], dtype=np.float64)
    n_total = int(scores.size)
    n_nan   = int(np.isnan(scores).sum())
    n_inf   = int(np.isinf(scores).sum())
    finite  = scores[np.isfinite(scores)]

    n_exc_p95_total  = int((finite > p95_threshold).sum())
    n_exc_p99_total  = int((finite > p99_threshold).sum())
    rate_exc_p95     = round(n_exc_p95_total / len(finite), 6) if len(finite) > 0 else 0.0
    rate_exc_p99     = round(n_exc_p99_total / len(finite), 6) if len(finite) > 0 else 0.0

    test_stats = {
        "test_n_patches": n_total,
        "test_nan": n_nan,
        "test_inf": n_inf,
        "test_min": float(finite.min()) if len(finite) > 0 else None,
        "test_max": float(finite.max()) if len(finite) > 0 else None,
        "test_mean": float(finite.mean()) if len(finite) > 0 else None,
        "test_std": float(finite.std()) if len(finite) > 0 else None,
        "test_median": float(np.median(finite)) if len(finite) > 0 else None,
        "n_exceed_p95": n_exc_p95_total,
        "rate_exceed_p95": rate_exc_p95,
        "n_exceed_p99": n_exc_p99_total,
        "rate_exceed_p99": rate_exc_p99,
    }

    # ---- G9: threshold JSON mtime 불변 확인 ----
    thresh_mtime_after = os.path.getmtime(THRESH_JSON)
    thresh_unchanged = abs(thresh_mtime_before - thresh_mtime_after) < 1.0
    if not thresh_unchanged:
        abort(f"threshold JSON mtime이 변경됨! before={thresh_mtime_before}, after={thresh_mtime_after}")
    print(f"[G9] threshold JSON mtime 불변 확인 ✅")

    ts = datetime.now().isoformat(timespec="seconds")

    # ---- sanity 결과 저장 ----
    verdict = "통과" if n_failed == 0 and n_nan == 0 and n_inf == 0 else (
        "부분통과" if n_csv > 0 else "실패"
    )

    sanity_result = {
        "stage": "P-A74_normal_test_sanity_efficientnet_b0_imagenet",
        "created": ts,
        "verdict": verdict,
        "backbone": BACKBONE,
        "pretrain_source": PRETRAIN_SOURCE,
        "run_tag": RUN_TAG,
        "n_test_patients": n_csv,
        "n_failed": n_failed,
        "missing": missing,
        "test_stats": test_stats,
        "threshold_used_p95": p95_threshold,
        "threshold_used_p99": p99_threshold,
        "threshold_source_json": str(THRESH_JSON),
        "threshold_json_mtime_unchanged": thresh_unchanged,
        "distribution_npz": str(MODEL_NPZ),
        "distribution_sha256": dist_sha,
        "selected_index": str(SELECTED_INDICES),
        "selected_index_shape": list(idx.shape),
        "selected_index_unique": int(len(set(idx.tolist()))),
        "selected_index_min": int(idx.min()),
        "selected_index_max": int(idx.max()),
        "raw_feature_dim": RAW_FEATURE_DIM,
        "reduced_feature_dim": REDUCED_FEATURE_DIM,
        "weight_file": str(wpath),
        "mask_type": MASK_TYPE,
        "paths_config": PATHS_CONFIG,
        "split_source": str(SPLIT_JSON),
        "score_dir": str(SCORE_DIR),
        "elapsed_sec": round(elapsed, 1),
        "threshold_recalculated": False,
        "lesion_scored": False,
        "stage1_dev_scored": False,
        "stage2_holdout_accessed": False,
        "metrics_calculated": False,
        "auroc_calculated": False,
        "note": "EfficientNet-B0 ImageNet P-A74 normal test sanity. threshold read-only from P-A73.",
    }

    with open(SANITY_JSON, "w", encoding="utf-8") as f:
        json.dump(sanity_result, f, ensure_ascii=False, indent=2)

    # sanity summary CSV
    with open(SANITY_CSV, "w", encoding="utf-8-sig", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["metric", "value"])
        wtr.writerow(["verdict", verdict])
        wtr.writerow(["n_test_patients", n_csv])
        wtr.writerow(["n_failed", n_failed])
        for k, v in test_stats.items():
            wtr.writerow([k, v])
        wtr.writerow(["threshold_p95_used", p95_threshold])
        wtr.writerow(["threshold_p99_used", p99_threshold])
        wtr.writerow(["elapsed_sec", round(elapsed, 1)])

    # per-patient CSV
    pp_fields = ["patient_id", "n_patches", "nan", "inf",
                 "min", "max", "mean", "std", "median",
                 "n_exceed_p95", "rate_exceed_p95",
                 "n_exceed_p99", "rate_exceed_p99", "status"]
    with open(PER_PATIENT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=pp_fields)
        wtr.writeheader()
        wtr.writerows(per_patient_rows)

    # ---- 보고서 MD ----
    md_lines = [
        "# P-A74 EfficientNet-B0 Normal Test Sanity 보고서",
        "",
        f"**판정: {verdict}**",
        "",
        f"- 생성일시: {ts}",
        f"- backbone: {BACKBONE} ({PRETRAIN_SOURCE})",
        "",
        "## 실행 결과",
        "| 항목 | 값 |",
        "|------|-----|",
        f"| 처리 환자 수 | {n_csv}명 / {EXPECTED_TEST_N}명 |",
        f"| 실패 환자 수 | {n_failed}명 |",
        f"| 실패 환자 목록 | {missing if missing else '없음'} |",
        f"| total scored patches | {n_total:,} |",
        f"| NaN count | {n_nan} |",
        f"| Inf count | {n_inf} |",
        f"| score min | {test_stats['test_min']:.6f} |",
        f"| score max | {test_stats['test_max']:.6f} |",
        f"| score mean | {test_stats['test_mean']:.6f} |",
        f"| score std | {test_stats['test_std']:.6f} |",
        f"| score median | {test_stats['test_median']:.6f} |",
        f"| **사용 p95 threshold** | **{p95_threshold:.6f}** |",
        f"| **사용 p99 threshold** | **{p99_threshold:.6f}** |",
        f"| p95 초과 patch 수 | {n_exc_p95_total:,} |",
        f"| p95 초과 비율 | {rate_exc_p95:.4%} |",
        f"| p99 초과 patch 수 | {n_exc_p99_total:,} |",
        f"| p99 초과 비율 | {rate_exc_p99:.4%} |",
        f"| 소요 시간 | {elapsed:.1f}초 |",
        "",
        "## 검증 항목",
        "| 항목 | 확인 |",
        "|------|------|",
        f"| P-A73 보고서 통과 | ✅ |",
        f"| P-A72 distribution 사용 | ✅ sha256={dist_sha[:16]}... |",
        f"| selected_feature_indices | ✅ shape={list(idx.shape)}, unique={REDUCED_FEATURE_DIM}, range=[{int(idx.min())},{int(idx.max())}] |",
        f"| threshold read-only (P-A73) | ✅ p95={p95_threshold:.6f}, p99={p99_threshold:.6f} |",
        f"| threshold JSON mtime 불변 | {'✅' if thresh_unchanged else '❌'} |",
        f"| threshold 재계산 없음 | ✅ |",
        f"| normal val 재실행 없음 | ✅ |",
        f"| lesion 미실행 | ✅ |",
        f"| stage1_dev 미실행 | ✅ |",
        f"| metrics/AUROC/AUPRC 미계산 | ✅ |",
        f"| stage2_holdout 접근 0 | ✅ |",
        f"| 기존 ResNet18/ResNet50/EfficientNet P-A73 결과 무수정 | ✅ |",
        "",
        "## 다음 단계",
        "- P-A75 stage1_dev lesion scoring: 사용자 승인 후 진행 가능",
        "",
        f"생성 시각: {ts}",
    ]
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(sanity_result, f, ensure_ascii=False, indent=2)

    # ---- runtime summary ----
    with open(RUNTIME_CSV, "w", encoding="utf-8-sig", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["stage", "n_patients", "n_failed", "n_patches",
                      "elapsed_sec", "p95_threshold", "p99_threshold",
                      "n_exc_p95", "rate_exc_p95", "n_exc_p99", "rate_exc_p99", "created"])
        wtr.writerow([
            "P-A74", n_csv, n_failed, n_total,
            round(elapsed, 1), round(p95_threshold, 6), round(p99_threshold, 6),
            n_exc_p95_total, rate_exc_p95, n_exc_p99_total, rate_exc_p99, ts,
        ])

    print(f"\n[P-A74] test scoring 완료: {n_csv}명, 실패 {n_failed}명, {elapsed:.1f}초")
    print(f"[P-A74] 전체 score: {n_total:,} | nan {n_nan} | inf {n_inf}")
    print(f"[P-A74] min/max/mean/std: {test_stats['test_min']:.4f}/{test_stats['test_max']:.4f}/"
          f"{test_stats['test_mean']:.4f}/{test_stats['test_std']:.4f}")
    print(f"[P-A74] p95 초과: {n_exc_p95_total:,} ({rate_exc_p95:.4%})")
    print(f"[P-A74] p99 초과: {n_exc_p99_total:,} ({rate_exc_p99:.4%})")
    print(f"[P-A74] sanity JSON : {SANITY_JSON}")
    print(f"[P-A74] report MD   : {REPORT_MD}")
    print(f"\n=== P-A74 완료: {verdict} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
