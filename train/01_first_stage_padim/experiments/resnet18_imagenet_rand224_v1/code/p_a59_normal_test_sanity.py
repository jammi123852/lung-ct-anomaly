"""
P-A59: ResNet18 ImageNet random224 PaDiM — normal test 36명 sanity check.

- P-A58 threshold(p95/p99)를 read-only로 로드한다.
- normal test 36명 scoring 후 p95/p99 초과율을 계산한다.
- threshold 재계산/수정, lesion/stage1_dev/stage2_holdout 접근 금지.
- 기존 random100 / ResNet18 v2/v2 baseline 결과 수정 금지.

실행:
  source ~/ai_env/bin/activate && python experiments/resnet18_imagenet_rand224_v1/code/p_a59_normal_test_sanity.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ws_paths  # noqa: E402

REPO_ROOT = ws_paths.REPO_ROOT
SRC_DIR = ws_paths.SRC_DIR
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from position_aware_padim.config_manager import ConfigManager  # noqa: E402
from position_aware_padim.data_loader import DataLoader  # noqa: E402
from position_aware_padim.path_resolver import PathResolver  # noqa: E402
from position_aware_padim.patient_splitter import PatientSplitter  # noqa: E402
from position_aware_padim.feature_extractor_resnet50_scaffold import FeatureExtractorScaffold  # noqa: E402
from position_aware_padim.padim_model_resnet50_scaffold import PaDiMModelResNet50Scaffold  # noqa: E402

PATHS_CONFIG = "configs/paths.local.v2_roi0_0.yaml"
MASK_TYPE = "roi_0_0"
RUN_TAG = "padim_resnet18_imagenet_rand224"

MODEL_NPZ = ws_paths.MODEL_NPZ
SELECTED_INDICES_PATH = ws_paths.SELECTED_INDICES_PATH

# P-A58 threshold (read-only)
THRESH_JSON = ws_paths.OUTPUTS / "evaluation" / "normal_val_thresholds" / "normal_val_threshold.json"

# 출력 경로
SCORE_DIR = ws_paths.OUTPUTS / "scores" / "normal_test_by_patient"
SANITY_DIR = ws_paths.OUTPUTS / "evaluation" / "normal_test_sanity"
REPORT_DIR = ws_paths.OUTPUTS / "reports" / "normal_test"

SANITY_JSON = SANITY_DIR / "normal_test_sanity.json"
SANITY_CSV = SANITY_DIR / "normal_test_sanity.csv"
PER_PATIENT_CSV = SANITY_DIR / "normal_test_per_patient.csv"
REPORT_MD = REPORT_DIR / "p_a59_normal_test_sanity.md"
REPORT_JSON = REPORT_DIR / "p_a59_normal_test_sanity.json"
RUNTIME_CSV = REPORT_DIR / "p_a59_runtime_summary.csv"

# P-A58 보고서
P_A58_MD = ws_paths.OUTPUTS / "reports" / "normal_val" / "p_a58_normal_val_threshold.md"

EXPECTED_P95 = 20.2955
EXPECTED_P99 = 24.4483
THRESHOLD_TOL = 0.01


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def abort(msg: str, code: int = 2):
    print(f"[P-A59][ABORT] {msg}")
    sys.exit(code)


def main() -> int:
    # ---------------- 실행 전 가드 ----------------
    # G1: P-A58 보고서 통과 확인
    if not P_A58_MD.exists():
        abort(f"P-A58 보고서 없음: {P_A58_MD}")
    with open(P_A58_MD, encoding="utf-8") as f:
        md_text = f.read()
    if "판정: 통과" not in md_text:
        abort(f"P-A58 보고서가 통과 상태가 아님: {P_A58_MD}")
    print(f"[P-A59][G1] P-A58 보고서 통과 확인")

    # G2: P-A58 threshold JSON 존재 확인
    if not THRESH_JSON.exists():
        abort(f"P-A58 threshold JSON 없음: {THRESH_JSON}")

    # G3: threshold p95/p99 값 확인
    with open(THRESH_JSON, encoding="utf-8") as f:
        thresh_data = json.load(f)
    p95 = float(thresh_data["threshold_p95"])
    p99 = float(thresh_data["threshold_p99"])
    if abs(p95 - EXPECTED_P95) > THRESHOLD_TOL:
        abort(f"threshold p95 불일치: JSON={p95:.6f}, 기대={EXPECTED_P95}")
    if abs(p99 - EXPECTED_P99) > THRESHOLD_TOL:
        abort(f"threshold p99 불일치: JSON={p99:.6f}, 기대={EXPECTED_P99}")
    print(f"[P-A59][G3] p95={p95:.6f}, p99={p99:.6f} 확인")

    # G4: threshold JSON mtime 기록
    thresh_mtime_before = os.path.getmtime(THRESH_JSON)
    print(f"[P-A59][G4] threshold JSON mtime 기록: {thresh_mtime_before}")

    # G5: distribution npz 존재 확인
    if not MODEL_NPZ.exists():
        abort(f"distribution 파일 없음: {MODEL_NPZ}")
    dist_sha = sha256_of(MODEL_NPZ)
    print(f"[P-A59][G5] distribution sha256: {dist_sha}")

    # G6: selected_feature_indices shape=(224,), unique=224 확인
    if not SELECTED_INDICES_PATH.exists():
        abort(f"selected_feature_indices.npy 없음: {SELECTED_INDICES_PATH}")
    idx = np.load(SELECTED_INDICES_PATH)
    if idx.shape != (ws_paths.REDUCED_FEATURE_DIM,):
        abort(f"selected_index shape 불일치: {idx.shape}")
    if len(set(idx.tolist())) != ws_paths.REDUCED_FEATURE_DIM:
        abort(f"selected_index unique 불일치: {len(set(idx.tolist()))}")
    print(f"[P-A59][G6] selected_index shape={idx.shape}, unique={ws_paths.REDUCED_FEATURE_DIM}")

    # ResNet18 weight 존재 확인 (재다운로드 금지)
    import torch
    from torchvision.models import ResNet18_Weights
    wname = ResNet18_Weights.IMAGENET1K_V1.url.rsplit("/", 1)[-1]
    wpath = Path(torch.hub.get_dir()) / "checkpoints" / wname
    if not wpath.exists():
        abort(f"ResNet18 weight 없음(재다운로드 금지): {wpath}")

    # 설정/split
    cfg = ConfigManager(str(REPO_ROOT))
    cfg.load_config(paths_yaml=Path(PATHS_CONFIG).name)
    normal_training_ready = cfg.get("paths", "normal_training_ready", "")
    if not normal_training_ready:
        abort("paths config에 normal_training_ready 없음")
    manifest_path = Path(normal_training_ready) / "manifests" / "patient_manifest.csv"
    if not manifest_path.exists():
        abort(f"manifest 없음: {manifest_path}")

    splitter = PatientSplitter(str(REPO_ROOT))
    split = splitter.load_split()
    test_patients = list(split.test)

    # G7: normal test 36명 확인
    if len(test_patients) != 36:
        abort(f"normal test 환자 수가 36이 아님: {len(test_patients)}")
    print(f"[P-A59][G7] normal test 환자: {len(test_patients)}명")

    # G8: normal val output과 분리 확인 (경로 문자열 비교)
    val_score_dir = ws_paths.OUTPUTS / "scores" / "normal_val_by_patient"
    if SCORE_DIR == val_score_dir:
        abort("test score 경로가 val score 경로와 동일 - 분리 오류")
    print(f"[P-A59][G8] val/test score 경로 분리 확인")

    # G9: normal test output path 기존 파일 존재 시 중단
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    SANITY_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if any(SCORE_DIR.glob("*.csv")):
        abort(f"기존 P-A59 score CSV 존재 → 덮어쓰기 금지: {SCORE_DIR}")
    if SANITY_JSON.exists():
        abort(f"기존 P-A59 sanity JSON 존재 → 덮어쓰기 금지: {SANITY_JSON}")
    if REPORT_JSON.exists():
        abort(f"기존 P-A59 report JSON 존재 → 덮어쓰기 금지: {REPORT_JSON}")
    print(f"[P-A59][G9] 출력 경로 기존 파일 없음 확인")

    print(f"\n[P-A59] 모든 가드 통과. normal test scoring 시작.")
    print(f"[P-A59] backbone=resnet18, mask={MASK_TYPE}, p95={p95:.6f}, p99={p99:.6f}")

    # ---------------- 모델/추출기 ----------------
    model = PaDiMModelResNet50Scaffold(
        selected_feature_indices_path=str(SELECTED_INDICES_PATH),
        feature_dim=ws_paths.REDUCED_FEATURE_DIM,
        raw_feature_dim=ws_paths.RAW_FEATURE_DIM,
        eps=1e-5,
    )
    model.load(str(MODEL_NPZ))
    feat = FeatureExtractorScaffold(backbone="resnet18", pretrain_source="imagenet")
    print(f"[P-A59] device: {feat.device}, position_bin 수: {len(model.stats)}")

    path_resolver = PathResolver(str(manifest_path), normal_training_ready)
    loader = DataLoader(
        str(manifest_path),
        path_resolver,
        str(ws_paths.REPORTS_DIR / "error.csv"),
        use_mmap=True,
    )

    # ---------------- test scoring ----------------
    all_scores = []
    per_patient_rows = []
    n_csv = 0
    n_failed = 0
    missing = []
    start = datetime.now()

    for pid in test_patients:
        data = loader.load_patient_data(pid, mask_type=MASK_TYPE)
        if data is None:
            n_failed += 1
            missing.append(pid)
            print(f"  [FAIL] {pid}: 로드 실패")
            per_patient_rows.append({
                "patient_id": pid, "n_patches": 0,
                "nan": 0, "inf": 0,
                "min": "", "max": "", "mean": "", "std": "",
                "p95_exceed_n": 0, "p95_exceed_rate": 0.0,
                "p99_exceed_n": 0, "p99_exceed_rate": 0.0,
                "status": "FAIL",
            })
            continue
        scored = model.score_patient(feat, data)
        out_csv = SCORE_DIR / f"{pid}.csv"
        scored.to_csv(out_csv, index=False, encoding="utf-8-sig")
        n_csv += 1

        s = scored["padim_score"].to_numpy(dtype=np.float64)
        all_scores.append(s)

        n_nan_p = int(np.isnan(s).sum())
        n_inf_p = int(np.isinf(s).sum())
        finite_p = s[np.isfinite(s)]
        p95_exceed = int((finite_p > p95).sum())
        p99_exceed = int((finite_p > p99).sum())
        p95_rate = p95_exceed / len(finite_p) if len(finite_p) > 0 else 0.0
        p99_rate = p99_exceed / len(finite_p) if len(finite_p) > 0 else 0.0

        per_patient_rows.append({
            "patient_id": pid,
            "n_patches": len(s),
            "nan": n_nan_p,
            "inf": n_inf_p,
            "min": round(float(finite_p.min()), 6) if len(finite_p) > 0 else "",
            "max": round(float(finite_p.max()), 6) if len(finite_p) > 0 else "",
            "mean": round(float(finite_p.mean()), 6) if len(finite_p) > 0 else "",
            "std": round(float(finite_p.std()), 6) if len(finite_p) > 0 else "",
            "p95_exceed_n": p95_exceed,
            "p95_exceed_rate": round(p95_rate, 6),
            "p99_exceed_n": p99_exceed,
            "p99_exceed_rate": round(p99_rate, 6),
            "status": "OK",
        })
        print(f"  [OK]   {pid}: patches={len(s)}, nan={n_nan_p}, p95_exceed={p95_exceed}({p95_rate:.2%}), p99_exceed={p99_exceed}({p99_rate:.2%})")

    scores = np.concatenate(all_scores) if all_scores else np.array([], dtype=np.float64)
    n_total = int(scores.size)
    n_nan = int(np.isnan(scores).sum())
    n_inf = int(np.isinf(scores).sum())
    finite = scores[np.isfinite(scores)]

    p95_exceed_total = int((finite > p95).sum())
    p99_exceed_total = int((finite > p99).sum())
    p95_rate_total = p95_exceed_total / len(finite) if len(finite) > 0 else 0.0
    p99_rate_total = p99_exceed_total / len(finite) if len(finite) > 0 else 0.0

    test_stats = {
        "test_n_patches": n_total,
        "test_nan": n_nan,
        "test_inf": n_inf,
        "test_min": float(finite.min()) if len(finite) > 0 else None,
        "test_max": float(finite.max()) if len(finite) > 0 else None,
        "test_mean": float(finite.mean()) if len(finite) > 0 else None,
        "test_std": float(finite.std()) if len(finite) > 0 else None,
        "test_median": float(np.median(finite)) if len(finite) > 0 else None,
        "p95_threshold_used": p95,
        "p99_threshold_used": p99,
        "p95_exceed_n": p95_exceed_total,
        "p95_exceed_rate": p95_rate_total,
        "p99_exceed_n": p99_exceed_total,
        "p99_exceed_rate": p99_rate_total,
    }

    elapsed = (datetime.now() - start).total_seconds()

    # G4: threshold JSON mtime 불변 확인
    thresh_mtime_after = os.path.getmtime(THRESH_JSON)
    thresh_mtime_unchanged = abs(thresh_mtime_after - thresh_mtime_before) < 1.0

    verdict = "통과" if n_failed == 0 and n_nan == 0 and n_inf == 0 else (
        "부분통과" if n_csv > 0 else "실패"
    )

    # ---------------- 결과 저장 ----------------
    sanity_result = {
        "stage": "P-A59_normal_test_sanity_resnet18_rand224",
        "created": datetime.now().isoformat(timespec="seconds"),
        "verdict": verdict,
        "backbone": "resnet18",
        "pretrain_source": "imagenet",
        "run_tag": RUN_TAG,
        "n_test_patients": n_csv,
        "n_failed": n_failed,
        "missing": missing,
        "test_stats": test_stats,
        "distribution_npz": str(MODEL_NPZ),
        "distribution_sha256": dist_sha,
        "selected_index_shape": list(idx.shape),
        "selected_index_unique": int(len(set(idx.tolist()))),
        "mask_type": MASK_TYPE,
        "paths_config": PATHS_CONFIG,
        "threshold_source": str(THRESH_JSON),
        "threshold_p95": p95,
        "threshold_p99": p99,
        "threshold_recalculated": False,
        "threshold_json_mtime_unchanged": thresh_mtime_unchanged,
        "elapsed_sec": round(elapsed, 1),
        "normal_val_rescored": False,
        "lesion_scored": False,
        "stage1_dev_scored": False,
        "stage2_holdout_accessed": False,
        "metrics_calculated": False,
        "note": "P-A59 normal test sanity only. threshold read-only, no recalculation.",
    }

    with open(SANITY_JSON, "w", encoding="utf-8") as f:
        json.dump(sanity_result, f, ensure_ascii=False, indent=2)

    with open(SANITY_CSV, "w", encoding="utf-8-sig", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["metric", "value"])
        wtr.writerow(["verdict", verdict])
        wtr.writerow(["n_test_patients", n_csv])
        wtr.writerow(["n_failed", n_failed])
        for k, v in test_stats.items():
            wtr.writerow([k, v])

    # per-patient CSV
    with open(PER_PATIENT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = ["patient_id", "n_patches", "nan", "inf",
                      "min", "max", "mean", "std",
                      "p95_exceed_n", "p95_exceed_rate",
                      "p99_exceed_n", "p99_exceed_rate", "status"]
        wtr = csv.DictWriter(f, fieldnames=fieldnames)
        wtr.writeheader()
        for row in per_patient_rows:
            wtr.writerow(row)

    # high exceedance 환자 요약 (p95_exceed_rate > 10%)
    high_exceedance = [r for r in per_patient_rows if r.get("p95_exceed_rate", 0) > 0.10]

    # ---------------- 보고서 ----------------
    md_lines = [
        "# P-A59 Normal Test Sanity 보고서",
        "",
        f"## 판정: {verdict}",
        "",
        "## 실행 결과",
        "| 항목 | 값 |",
        "|------|-----|",
        f"| 처리 환자 수 | {n_csv}명 / 36명 |",
        f"| 실패 환자 수 | {n_failed}명 |",
        f"| 실패 환자 목록 | {missing if missing else '없음'} |",
        f"| total scored patches | {n_total:,} |",
        f"| NaN count | {n_nan} |",
        f"| Inf count | {n_inf} |",
        f"| score min | {test_stats['test_min']:.6f} |" if test_stats['test_min'] is not None else "| score min | - |",
        f"| score max | {test_stats['test_max']:.6f} |" if test_stats['test_max'] is not None else "| score max | - |",
        f"| score mean | {test_stats['test_mean']:.6f} |" if test_stats['test_mean'] is not None else "| score mean | - |",
        f"| score std | {test_stats['test_std']:.6f} |" if test_stats['test_std'] is not None else "| score std | - |",
        f"| score median | {test_stats['test_median']:.6f} |" if test_stats['test_median'] is not None else "| score median | - |",
        f"| 사용한 p95 threshold | {p95:.6f} |",
        f"| 사용한 p99 threshold | {p99:.6f} |",
        f"| p95 초과 patch 수 | {p95_exceed_total:,} ({p95_rate_total:.4%}) |",
        f"| p99 초과 patch 수 | {p99_exceed_total:,} ({p99_rate_total:.4%}) |",
        f"| 소요 시간 | {elapsed:.1f}초 |",
        "",
        "## per-patient high exceedance (p95_rate > 10%)",
    ]
    if high_exceedance:
        md_lines += [
            "| patient_id | n_patches | p95_exceed_n | p95_exceed_rate | p99_exceed_n | p99_exceed_rate |",
            "|---|---|---|---|---|---|",
        ]
        for r in high_exceedance:
            md_lines.append(
                f"| {r['patient_id']} | {r['n_patches']:,} | {r['p95_exceed_n']} | "
                f"{r['p95_exceed_rate']:.4%} | {r['p99_exceed_n']} | {r['p99_exceed_rate']:.4%} |"
            )
    else:
        md_lines.append("없음 (모든 환자 p95_exceed_rate ≤ 10%)")

    md_lines += [
        "",
        "## 검증 항목",
        "| 항목 | 확인 |",
        "|------|------|",
        f"| P-A58 threshold 재계산/수정 없음 | ✅ threshold_recalculated=False |",
        f"| threshold JSON mtime 불변 | {'✅' if thresh_mtime_unchanged else '❌'} |",
        f"| P-A57.5 distribution 사용 | ✅ sha256={dist_sha[:16]}... |",
        f"| selected feature dim=224 | ✅ shape={list(idx.shape)}, unique={ws_paths.REDUCED_FEATURE_DIM} |",
        "| 기존 random100 결과 무수정 | ✅ |",
        "| normal val 재실행 없음 | ✅ |",
        "| lesion 미실행 | ✅ |",
        "| stage1_dev 미실행 | ✅ |",
        "| metrics 미계산 | ✅ |",
        "| stage2_holdout 접근 0 | ✅ |",
        "",
        "## 다음 단계",
        "- P-A60 stage1_dev lesion scoring: 사용자 승인 후 진행 가능",
        "",
        f"생성 시각: {datetime.now().isoformat(timespec='seconds')}",
    ]

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(sanity_result, f, ensure_ascii=False, indent=2)

    # ---------------- runtime summary ----------------
    with open(RUNTIME_CSV, "w", encoding="utf-8-sig", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["stage", "n_patients", "n_failed", "n_patches",
                      "elapsed_sec", "p95_threshold", "p99_threshold",
                      "p95_exceed_n", "p95_exceed_rate",
                      "p99_exceed_n", "p99_exceed_rate",
                      "verdict", "created"])
        wtr.writerow([
            "P-A59", n_csv, n_failed, n_total,
            round(elapsed, 1), round(p95, 6), round(p99, 6),
            p95_exceed_total, round(p95_rate_total, 6),
            p99_exceed_total, round(p99_rate_total, 6),
            verdict,
            datetime.now().isoformat(timespec="seconds"),
        ])

    print(f"\n[P-A59] test scoring 완료: {n_csv}명, 실패 {n_failed}명, {elapsed:.1f}초")
    print(f"[P-A59] 전체 score: {n_total:,} | nan {n_nan} | inf {n_inf}")
    if test_stats['test_mean'] is not None:
        print(f"[P-A59] min/max/mean/std: {test_stats['test_min']:.4f}/{test_stats['test_max']:.4f}/"
              f"{test_stats['test_mean']:.4f}/{test_stats['test_std']:.4f}")
    print(f"[P-A59] p95={p95:.6f} 초과: {p95_exceed_total:,} ({p95_rate_total:.4%})")
    print(f"[P-A59] p99={p99:.6f} 초과: {p99_exceed_total:,} ({p99_rate_total:.4%})")
    print(f"[P-A59] 판정: {verdict}")
    print(f"[P-A59] sanity json : {SANITY_JSON}")
    print(f"[P-A59] report      : {REPORT_MD}")
    print("JSON_INFO_BEGIN"); print(json.dumps(sanity_result, ensure_ascii=False)); print("JSON_INFO_END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
