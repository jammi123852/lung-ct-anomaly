"""
P-A73: EfficientNet-B0 ImageNet PaDiM — normal val 36명 scoring + p95/p99 threshold 계산.

- P-A72.5 검증 완료된 full train distribution(position_bin_stats.npz)으로 normal val 36명만 scoring.
- val 전체 patch의 padim_score에서 p95/p99를 계산한다.
- normal test / lesion / stage2_holdout은 절대 접근하지 않는다 (val만).

실행:
  source ~/ai_env/bin/activate && python experiments/efficientnet_b0_imagenet_v1/code/p_a73_normal_val_threshold.py
"""

from __future__ import annotations

import csv
import hashlib
import json
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
SCRIPT_NAME         = "p_a73_normal_val_threshold.py"
RUN_TAG             = "padim_efficientnet_b0_imagenet"
EXPECTED_VAL_N      = 36

MODEL_NPZ           = EXP_ROOT / "outputs" / "models" / "distributions" / "position_bin_stats.npz"
SELECTED_INDICES    = EXP_ROOT / "outputs" / "models" / "distributions" / "selected_feature_indices.npy"
SCORE_DIR           = EXP_ROOT / "outputs" / "scores" / "normal_val_by_patient"
THRESH_DIR          = EXP_ROOT / "outputs" / "evaluation" / "normal_val_thresholds"
THRESH_JSON         = THRESH_DIR / "normal_val_threshold.json"
THRESH_CSV          = THRESH_DIR / "normal_val_threshold.csv"
REPORT_DIR          = EXP_ROOT / "outputs" / "reports" / "normal_val"
REPORT_MD           = REPORT_DIR / "p_a73_normal_val_threshold.md"
REPORT_JSON         = REPORT_DIR / "p_a73_normal_val_threshold.json"
RUNTIME_CSV         = REPORT_DIR / "p_a73_runtime_summary.csv"

P_A72_5_MD  = EXP_ROOT / "outputs" / "reports" / "full" / "p_a72_5_distribution_validation.md"
SPLIT_JSON  = PROJ_ROOT / "outputs" / "position-aware-padim-v1" / "splits" / "normal_v1.json"


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def abort(msg: str, code: int = 2):
    print(f"[P-A73][ABORT] {msg}")
    sys.exit(code)


def main() -> int:
    # ---- G1: P-A72.5 validation 보고서 통과 확인 ----
    if not P_A72_5_MD.exists():
        abort(f"P-A72.5 validation 보고서 없음: {P_A72_5_MD}")
    with open(P_A72_5_MD, encoding="utf-8") as f:
        md_text = f.read()
    if "판정: 통과" not in md_text:
        abort(f"P-A72.5 보고서가 통과 상태가 아님: {P_A72_5_MD}")
    print("[G1] P-A72.5 validation 보고서 통과 확인 ✅")

    # ---- G2: distribution npz 존재 확인 ----
    if not MODEL_NPZ.exists():
        abort(f"distribution 파일 없음: {MODEL_NPZ}")
    print(f"[G2] distribution 존재: {MODEL_NPZ}")

    # ---- G3: selected_feature_indices 검증 ----
    if not SELECTED_INDICES.exists():
        abort(f"selected_feature_indices.npy 없음: {SELECTED_INDICES}")
    idx = np.load(SELECTED_INDICES)
    if idx.shape != (REDUCED_FEATURE_DIM,):
        abort(f"selected_index shape 불일치: {idx.shape} (기대: ({REDUCED_FEATURE_DIM},))")
    if len(set(idx.tolist())) != REDUCED_FEATURE_DIM:
        abort(f"selected_index unique 불일치: {len(set(idx.tolist()))}")
    if not ((idx >= 0).all() and (idx < RAW_FEATURE_DIM).all()):
        abort(f"selected_index range 불일치: min={int(idx.min())}, max={int(idx.max())}")
    print(f"[G3] selected_feature_indices OK: shape={idx.shape}, unique={REDUCED_FEATURE_DIM}, range=[{int(idx.min())},{int(idx.max())}]")

    # ---- G4: EfficientNet-B0 weight 존재 확인 (재다운로드 금지) ----
    import torch
    from torchvision.models import EfficientNet_B0_Weights
    wname = EfficientNet_B0_Weights.IMAGENET1K_V1.url.rsplit("/", 1)[-1]
    wpath = Path(torch.hub.get_dir()) / "checkpoints" / wname
    if not wpath.exists():
        abort(f"EfficientNet-B0 weight 없음(재다운로드 금지): {wpath}")
    print(f"[G4] EfficientNet-B0 weight 존재: {wpath}")

    # ---- G5: 기존 출력 보호 ----
    if THRESH_JSON.exists():
        abort(f"기존 P-A73 threshold 존재 → 덮어쓰기 금지: {THRESH_JSON}")
    if SCORE_DIR.exists() and any(SCORE_DIR.glob("*.csv")):
        abort(f"기존 P-A73 score CSV 존재 → 덮어쓰기 금지: {SCORE_DIR}")
    if REPORT_JSON.exists():
        abort(f"기존 P-A73 report JSON 존재 → 덮어쓰기 금지: {REPORT_JSON}")
    print("[G5] 기존 출력 없음, 충돌 없음 ✅")

    # ---- split: val 36명 ----
    if not SPLIT_JSON.exists():
        abort(f"split JSON 없음: {SPLIT_JSON}")
    with open(SPLIT_JSON, encoding="utf-8-sig") as f:
        split_data = json.load(f)
    val_patients = list(split_data["val"])
    if len(val_patients) != EXPECTED_VAL_N:
        abort(f"normal val 환자 수가 {EXPECTED_VAL_N}이 아님: {len(val_patients)}")
    print(f"[split] normal val: {len(val_patients)}명 확인 ✅")

    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    THRESH_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    dist_sha = sha256_of(MODEL_NPZ)
    print(f"[P-A73] distribution sha256: {dist_sha}")
    print(f"[P-A73] backbone={BACKBONE}, raw_dim={RAW_FEATURE_DIM}, reduced_dim={REDUCED_FEATURE_DIM}")

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
    print(f"[P-A73] distribution 로드 완료: position_bin 수={len(model.stats)}")

    feat = FeatureExtractorEffNetB0()
    print(f"[P-A73] device: {feat.device}")

    path_resolver = PathResolver(str(manifest_path), normal_training_ready)
    loader = DataLoader(
        str(manifest_path),
        path_resolver,
        str(REPORT_DIR / "error.csv"),
        use_mmap=True,
    )

    # ---- val scoring ----
    all_scores = []
    n_csv = 0
    n_failed = 0
    missing = []
    start = time.time()

    for i, pid in enumerate(val_patients, 1):
        data = loader.load_patient_data(pid, mask_type=MASK_TYPE)
        if data is None:
            n_failed += 1
            missing.append(pid)
            print(f"  [FAIL] ({i}/{EXPECTED_VAL_N}) {pid}: 로드 실패")
            continue
        scored = model.score_patient(feat, data)
        out_csv = SCORE_DIR / f"{pid}.csv"
        scored.to_csv(out_csv, index=False, encoding="utf-8-sig")
        n_csv += 1
        s = scored["padim_score"].to_numpy(dtype=np.float64)
        all_scores.append(s)
        print(f"  [OK]   ({i}/{EXPECTED_VAL_N}) {pid}: patches={len(s)}, nan={int(np.isnan(s).sum())}")

    elapsed = time.time() - start

    scores = np.concatenate(all_scores) if all_scores else np.array([], dtype=np.float64)
    n_total = int(scores.size)
    n_nan = int(np.isnan(scores).sum())
    n_inf = int(np.isinf(scores).sum())
    finite = scores[np.isfinite(scores)]

    p95 = float(np.percentile(finite, 95))
    p99 = float(np.percentile(finite, 99))
    val_stats = {
        "val_n_patches": n_total,
        "val_nan": n_nan,
        "val_inf": n_inf,
        "val_min": float(finite.min()),
        "val_max": float(finite.max()),
        "val_mean": float(finite.mean()),
        "val_std": float(finite.std()),
        "val_median": float(np.median(finite)),
        "val_p95": p95,
        "val_p99": p99,
    }

    # ---- threshold 저장 ----
    ts = datetime.now().isoformat(timespec="seconds")
    result = {
        "stage": "P-A73_normal_val_threshold_efficientnet_b0_imagenet",
        "created": ts,
        "backbone": BACKBONE,
        "pretrain_source": PRETRAIN_SOURCE,
        "run_tag": RUN_TAG,
        "threshold_source": "val_split",
        "n_val_patients": n_csv,
        "n_val_patches": n_total,
        "threshold_p95": p95,
        "threshold_p99": p99,
        "val_stats": val_stats,
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
        "val_missing": missing,
        "n_failed": n_failed,
        "elapsed_sec": round(elapsed, 1),
        "reused_existing_threshold": False,
        "normal_test_scored": False,
        "lesion_scored": False,
        "stage1_dev_scored": False,
        "stage2_holdout_accessed": False,
        "note": "EfficientNet-B0 ImageNet 전용 신규 threshold. normal test/lesion/stage2_holdout 미접근.",
    }

    with open(THRESH_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(THRESH_CSV, "w", encoding="utf-8-sig", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["metric", "value"])
        for k, v in val_stats.items():
            wtr.writerow([k, v])
        wtr.writerow(["threshold_p95", p95])
        wtr.writerow(["threshold_p99", p99])

    # ---- 보고서 ----
    verdict = "통과" if n_failed == 0 and n_nan == 0 and n_inf == 0 else (
        "부분통과" if n_csv > 0 else "실패"
    )
    md_lines = [
        "# P-A73 EfficientNet-B0 Normal Val Threshold 보고서",
        "",
        f"**판정: {verdict}**",
        "",
        f"- 생성일시: {ts}",
        f"- backbone: {BACKBONE} ({PRETRAIN_SOURCE})",
        "",
        "## 실행 결과",
        "| 항목 | 값 |",
        "|------|-----|",
        f"| 처리 환자 수 | {n_csv}명 / {EXPECTED_VAL_N}명 |",
        f"| 실패 환자 수 | {n_failed}명 |",
        f"| 실패 환자 목록 | {missing if missing else '없음'} |",
        f"| total scored patches | {n_total:,} |",
        f"| NaN count | {n_nan} |",
        f"| Inf count | {n_inf} |",
        f"| score min | {val_stats['val_min']:.6f} |",
        f"| score max | {val_stats['val_max']:.6f} |",
        f"| score mean | {val_stats['val_mean']:.6f} |",
        f"| score std | {val_stats['val_std']:.6f} |",
        f"| score median | {val_stats['val_median']:.6f} |",
        f"| **p95 threshold** | **{p95:.6f}** |",
        f"| **p99 threshold** | **{p99:.6f}** |",
        f"| 소요 시간 | {elapsed:.1f}초 |",
        "",
        "## 검증 항목",
        "| 항목 | 확인 |",
        "|------|------|",
        f"| P-A72.5 distribution 사용 | ✅ sha256={dist_sha[:16]}... |",
        f"| selected_feature_indices | ✅ shape={list(idx.shape)}, unique={REDUCED_FEATURE_DIM}, range=[{int(idx.min())},{int(idx.max())}] |",
        f"| backbone | {BACKBONE} ({PRETRAIN_SOURCE}) |",
        f"| raw_feature_dim | {RAW_FEATURE_DIM} |",
        f"| reduced_feature_dim | {REDUCED_FEATURE_DIM} |",
        "| 기존 ResNet18/ResNet50/EfficientNet 결과 무수정 | ✅ |",
        "| normal test 미실행 | ✅ |",
        "| lesion 미실행 | ✅ |",
        "| stage1_dev 미실행 | ✅ |",
        "| metrics/AUROC/AUPRC 미계산 | ✅ |",
        "| stage2_holdout 접근 0 | ✅ |",
        "",
        "## 다음 단계",
        "- P-A74 normal test sanity: 사용자 승인 후 진행 가능",
        "",
        f"생성 시각: {ts}",
    ]
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # ---- runtime summary ----
    with open(RUNTIME_CSV, "w", encoding="utf-8-sig", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["stage", "n_patients", "n_failed", "n_patches", "elapsed_sec", "p95", "p99", "created"])
        wtr.writerow([
            "P-A73", n_csv, n_failed, n_total,
            round(elapsed, 1), round(p95, 6), round(p99, 6), ts,
        ])

    print(f"\n[P-A73] val scoring 완료: {n_csv}명, 실패 {n_failed}명, {elapsed:.1f}초")
    print(f"[P-A73] 전체 score: {n_total:,} | nan {n_nan} | inf {n_inf}")
    print(f"[P-A73] min/max/mean/std: {val_stats['val_min']:.4f}/{val_stats['val_max']:.4f}/"
          f"{val_stats['val_mean']:.4f}/{val_stats['val_std']:.4f}")
    print(f"[P-A73] threshold p95={p95:.6f}  p99={p99:.6f}")
    print(f"[P-A73] threshold json: {THRESH_JSON}")
    print(f"[P-A73] report        : {REPORT_MD}")
    print(f"\n=== P-A73 완료: {verdict} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
