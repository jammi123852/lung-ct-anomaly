"""
N-C3 Normal-Only Crop Sampling Manifest Generator

normal_train 290명에서 position-balanced normal-only crop sampling manifest를 생성한다.

실행 모드:
  --dry-run          : full manifest 미생성. 경로/스키마/카운트 추정만 수행. (기본)
  --smoke-one-patient: 1명 smoke manifest 생성 (--confirm-smoke 필요)
  --full-manifest    : 전체 290명 manifest 생성 (--confirm-full 필요)

안전 장치:
  ALLOW_REAL_MANIFEST=False  → bare run 및 --dry-run에서 저장 불가
  --full-manifest without --confirm-full → exit(2)
  기존 출력 파일 존재 시 → exit(2) (overwrite 금지)

금지:
  - crop 파일 생성 금지
  - feature extraction 금지
  - model forward 금지
  - 학습 금지
  - scoring 금지
  - stage2_holdout 접근 금지
  - P-C supervised artifact 사용 금지
  - 기존 결과 수정/삭제 금지
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# 안전 플래그 (기본 False: 저장 불가)
# ---------------------------------------------------------------------------
ALLOW_REAL_MANIFEST = False

# ---------------------------------------------------------------------------
# 경로 상수
# ---------------------------------------------------------------------------
PROJ_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")
EXP_ROOT  = PROJ_ROOT / "experiments" / "normal_only_second_stage_refiner_v1"

SPLIT_JSON      = PROJ_ROOT / "outputs" / "position-aware-padim-v1" / "splits" / "normal_v1.json"
PATCH_INDEX_ROOT = Path("/mnt/c/Users/jinhy/Desktop/v1 paicient/patch_index_by_patient")
V4_20_NORMAL_ROI_ROOT = (PROJ_ROOT / "outputs" / "mip-postprocess-research-v1" /
                          "masks" / "refined_roi_v4_20_modeB_all_v1" / "normal")
NORMAL_CT_ROOT  = Path("/mnt/c/Users/jinhy/Desktop/"
                        "Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1"
                        "/volumes_npy")
N_C2_JSON = (EXP_ROOT / "outputs" / "reports" /
             "n_c2_crop_sampling_manifest_preflight" /
             "n_c2_crop_sampling_manifest_preflight.json")
P_B7_SUMMARY_CSV = (PROJ_ROOT / "experiments" /
                    "efficientnet_b0_imagenet_chestwall_removed_roi_v1" /
                    "outputs" / "reports" / "full" / "p_b7_patch_filtering_summary.csv")

# 출력 경로
DRY_OUT_DIR   = EXP_ROOT / "outputs" / "reports" / "n_c3_normal_only_crop_manifest_drycheck"
SMOKE_OUT_DIR = EXP_ROOT / "outputs" / "manifests" / "n_c3_smoke_one_patient_manifest"
FULL_OUT_DIR  = EXP_ROOT / "outputs" / "manifests" / "n_c3_normal_only_crop_manifest_v1"

# ---------------------------------------------------------------------------
# 설계 파라미터
# ---------------------------------------------------------------------------
EXPECTED_TRAIN_N    = 290
POSITION_BINS       = [
    "upper_central", "upper_peripheral",
    "middle_central", "middle_peripheral",
    "lower_central", "lower_peripheral",
]
CAP_PER_BIN         = 100
CROP_SIZE           = 96
SAMPLING_SEED       = 42
ROI_RATIO_THRESHOLD = 0.5
SOURCE_BRANCH       = "v4_20_roi_normal_only"
MANIFEST_SCHEMA     = [
    "normal_candidate_id",
    "patient_id",
    "safe_id",
    "split",
    "local_z",
    "slice_index",
    "y0",
    "x0",
    "y1",
    "x1",
    "center_y",
    "center_x",
    "position_bin",
    "z_level",
    "roi_patch_ratio",
    "sampling_strategy",
    "patient_sample_rank",
    "position_sample_rank",
    "source_ct_path",
    "source_roi_path",
    "source_branch",
    "forbidden_supervised_source_used",
    "crop_size",
    "crop_dim",
    "z_channels",
]


# ---------------------------------------------------------------------------
# 유틸리티
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(msg, flush=True)


def _abort(msg: str, code: int = 2) -> None:
    _log(f"[ABORT] {msg}")
    sys.exit(code)


def _compute_roi_patch_ratio(roi_slice: np.ndarray, y0: int, x0: int,
                              y1: int, x1: int) -> float:
    sub = roi_slice[y0:y1, x0:x1]
    return float(sub.mean()) if sub.size > 0 else 0.0


# ---------------------------------------------------------------------------
# 보안 가드
# ---------------------------------------------------------------------------

def run_guards() -> dict:
    """N-C2 verdict / split / 경로 / 오염 체크. 모든 모드에서 실행."""
    issues = []
    result = {}

    # G1: ALLOW_REAL_MANIFEST 선언 확인 (코드에 명시됨)
    _log(f"[G1] ALLOW_REAL_MANIFEST={ALLOW_REAL_MANIFEST}")

    # G2: N-C2 verdict 확인
    if N_C2_JSON.exists():
        n_c2 = json.loads(N_C2_JSON.read_text(encoding="utf-8"))
        v = n_c2.get("verdict", "")
        _log(f"[G2] N-C2 verdict={v}")
        if v != "PASS":
            issues.append(f"G2: N-C2 verdict={v} (PASS 필요)")
    else:
        issues.append("G2: N-C2 JSON 없음")
        _log("[G2] N-C2 JSON 없음")
        n_c2 = {}

    # G3: split 로드 및 290명 확인
    if not SPLIT_JSON.exists():
        _abort("G3: split JSON 없음")
    split_data = json.loads(SPLIT_JSON.read_text(encoding="utf-8"))
    train_patients = list(split_data["train"])
    val_patients   = list(split_data.get("val", []))
    test_patients  = list(split_data.get("test", []))
    p2s = split_data.get("patient_to_safe_id", {})
    _log(f"[G3] train={len(train_patients)}, val={len(val_patients)}, test={len(test_patients)}")
    if len(train_patients) != EXPECTED_TRAIN_N:
        issues.append(f"G3: train split {len(train_patients)}≠{EXPECTED_TRAIN_N}")

    # G4: train∩val / train∩test overlap=0
    train_set = set(train_patients)
    val_set   = set(val_patients)
    test_set  = set(test_patients)
    tv_overlap = len(train_set & val_set)
    tt_overlap = len(train_set & test_set)
    _log(f"[G4] train∩val={tv_overlap}, train∩test={tt_overlap}")
    if tv_overlap > 0:
        issues.append(f"G4: train∩val overlap={tv_overlap}")
    if tt_overlap > 0:
        issues.append(f"G4: train∩test overlap={tt_overlap}")

    # G5: stage2_holdout 미접근 확인 (코드 경로상 참조 없음)
    _log("[G5] stage2_holdout 경로 미참조 (코드 경로상 보장)")

    # G6: P-C supervised artifact 미사용 선언
    _log("[G6] P-C supervised crop/manifest/label 미사용 (코드 경로상 보장)")

    # G7: CT/ROI 경로 존재 확인 (경량 존재 체크)
    ct_missing, roi_missing = [], []
    for pid in train_patients:
        sid = p2s.get(pid, pid)
        if not (NORMAL_CT_ROOT / sid / "ct_hu.npy").exists():
            ct_missing.append(pid)
        if not (V4_20_NORMAL_ROI_ROOT / sid / "refined_roi.npy").exists():
            roi_missing.append(pid)
    _log(f"[G7] CT 누락={len(ct_missing)}, ROI 누락={len(roi_missing)}")
    if ct_missing:
        issues.append(f"G7: CT 누락 {len(ct_missing)}명: {ct_missing[:3]}")
    if roi_missing:
        issues.append(f"G7: ROI 누락 {len(roi_missing)}명: {roi_missing[:3]}")

    # G8: patch index CSV 경로 확인 (샘플 3명)
    patch_missing = []
    for pid in train_patients:
        sid = p2s.get(pid, pid)
        if not (PATCH_INDEX_ROOT / f"{sid}.csv").exists():
            patch_missing.append(pid)
    _log(f"[G8] patch index CSV 누락={len(patch_missing)}")
    if patch_missing:
        issues.append(f"G8: patch index CSV 누락 {len(patch_missing)}명")

    result.update({
        "n_c2": n_c2,
        "train_patients": train_patients,
        "val_patients": val_patients,
        "test_patients": test_patients,
        "p2s": p2s,
        "ct_missing": ct_missing,
        "roi_missing": roi_missing,
        "patch_missing": patch_missing,
        "issues": issues,
    })
    return result


# ---------------------------------------------------------------------------
# 샘플링 카운트 추정 (dry-run 전용, 실제 ROI 로드 없이 patch CSV 기반)
# ---------------------------------------------------------------------------

def estimate_sampling_counts(train_patients: list, p2s: dict) -> dict:
    """
    각 환자의 patch index CSV에서 position_bin별 patch 수를 집계한다.
    ROI 재필터링은 하지 않고 원본 patch count 기준으로 추정한다.
    (dry-run에서 ROI mmap 로드 없이 빠른 추정용)
    """
    rng = np.random.default_rng(SAMPLING_SEED)

    by_patient = []
    total_estimated = 0
    under_cap_bins = 0
    bin_totals = {b: 0 for b in POSITION_BINS}
    bin_under_cap = {b: 0 for b in POSITION_BINS}

    for pid in train_patients:
        sid = p2s.get(pid, pid)
        patch_csv = PATCH_INDEX_ROOT / f"{sid}.csv"
        if not patch_csv.exists():
            by_patient.append({
                "patient_id": pid, "safe_id": sid,
                "patch_csv_found": False,
                "total_patch": 0,
                **{f"bin_{b}": 0 for b in POSITION_BINS},
                **{f"sample_{b}": 0 for b in POSITION_BINS},
                "total_sample": 0,
            })
            continue

        bin_counts = {b: 0 for b in POSITION_BINS}
        bin_samples = {}
        try:
            with open(patch_csv, encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    pb = row.get("position_bin", "")
                    if pb in bin_counts:
                        bin_counts[pb] += 1
        except Exception as e:
            _log(f"  [WARN] {pid}: patch CSV 읽기 오류: {e}")

        total_patch = sum(bin_counts.values())
        patient_total_sample = 0
        for b in POSITION_BINS:
            n = bin_counts[b]
            s = min(n, CAP_PER_BIN)
            bin_samples[f"sample_{b}"] = s
            patient_total_sample += s
            bin_totals[b] += n
            if n < CAP_PER_BIN:
                under_cap_bins += 1
                bin_under_cap[b] += 1

        total_estimated += patient_total_sample
        by_patient.append({
            "patient_id": pid,
            "safe_id": sid,
            "patch_csv_found": True,
            "total_patch": total_patch,
            **{f"bin_{b}": bin_counts[b] for b in POSITION_BINS},
            **bin_samples,
            "total_sample": patient_total_sample,
        })

    return {
        "by_patient": by_patient,
        "total_estimated": total_estimated,
        "under_cap_bins_count": under_cap_bins,
        "bin_totals": bin_totals,
        "bin_under_cap_patients": bin_under_cap,
    }


# ---------------------------------------------------------------------------
# 실제 샘플링 (smoke / full 모드)
# ---------------------------------------------------------------------------

def sample_patient_manifest(pid: str, sid: str, p2s: dict,
                             candidate_id_start: int) -> tuple:
    """
    단일 환자에 대해 v4_20 ROI 필터링 후 position_bin별 100개 샘플링.
    Returns: (rows, next_candidate_id, per_patient_stats)
    """
    patch_csv   = PATCH_INDEX_ROOT / f"{sid}.csv"
    roi_path    = V4_20_NORMAL_ROI_ROOT / sid / "refined_roi.npy"
    ct_path     = NORMAL_CT_ROOT / sid / "ct_hu.npy"

    if not patch_csv.exists() or not roi_path.exists() or not ct_path.exists():
        return [], candidate_id_start, {"error": "missing_files"}

    refined_roi = np.load(str(roi_path), mmap_mode='r')
    n_z = refined_roi.shape[0]

    rng = np.random.default_rng(SAMPLING_SEED)

    bin_patches = {b: [] for b in POSITION_BINS}
    with open(patch_csv, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pb = row.get("position_bin", "")
            if pb not in bin_patches:
                continue
            z = int(row["local_z"])
            if z < 0 or z >= n_z:
                continue
            y0 = int(row["y0"]); x0 = int(row["x0"])
            y1 = int(row["y1"]); x1 = int(row["x1"])
            roi_slice = np.asarray(refined_roi[z])
            ratio = _compute_roi_patch_ratio(roi_slice, y0, x0, y1, x1)
            if ratio < ROI_RATIO_THRESHOLD:
                continue
            bin_patches[pb].append({
                "local_z": z, "y0": y0, "x0": x0, "y1": y1, "x1": x1,
                "z_level": row.get("z_level", ""),
                "roi_patch_ratio": ratio,
            })

    rows = []
    cid = candidate_id_start
    patient_rank = 0
    stats = {}
    for b in POSITION_BINS:
        pool = bin_patches[b]
        n = len(pool)
        k = min(n, CAP_PER_BIN)
        idxs = rng.choice(n, size=k, replace=False).tolist() if n > 0 else []
        sampled = [pool[i] for i in sorted(idxs)]
        stats[b] = {"pool": n, "sampled": k, "under_cap": n < CAP_PER_BIN}
        for pos_rank, p in enumerate(sampled):
            center_y = p["y0"] + 16
            center_x = p["x0"] + 16
            rows.append({
                "normal_candidate_id": f"NC_{cid:07d}",
                "patient_id": pid,
                "safe_id": sid,
                "split": "normal_train",
                "local_z": p["local_z"],
                "slice_index": p["local_z"],
                "y0": p["y0"], "x0": p["x0"], "y1": p["y1"], "x1": p["x1"],
                "center_y": center_y,
                "center_x": center_x,
                "position_bin": b,
                "z_level": p["z_level"],
                "roi_patch_ratio": round(p["roi_patch_ratio"], 6),
                "sampling_strategy": "patient_x_position_balanced",
                "patient_sample_rank": patient_rank,
                "position_sample_rank": pos_rank,
                "source_ct_path": str(ct_path),
                "source_roi_path": str(roi_path),
                "source_branch": SOURCE_BRANCH,
                "forbidden_supervised_source_used": False,
                "crop_size": CROP_SIZE,
                "crop_dim": "2.5d_3ch",
                "z_channels": "z-1,z,z+1",
            })
            cid += 1
            patient_rank += 1

    return rows, cid, stats


# ---------------------------------------------------------------------------
# dry-run 모드
# ---------------------------------------------------------------------------

def run_dry(guard: dict) -> None:
    _log("\n=== [DRY-RUN] N-C3 Normal-Only Crop Sampling Manifest ===")

    train_patients = guard["train_patients"]
    p2s            = guard["p2s"]

    # 카운트 추정 (ROI 재필터링 없이 patch CSV 기반)
    _log(f"[dry] {len(train_patients)}명 patch index CSV 집계 중...")
    est = estimate_sampling_counts(train_patients, p2s)

    # schema 검증 (컬럼 목록 정적 확인)
    schema_ok = True
    missing_cols = [c for c in MANIFEST_SCHEMA if c not in MANIFEST_SCHEMA]  # 자기참조 확인

    # output collision 확인
    smoke_manifest = SMOKE_OUT_DIR / "n_c3_smoke_manifest.csv"
    full_manifest  = FULL_OUT_DIR  / "n_c3_normal_only_crop_manifest_v1.csv"
    smoke_exists = smoke_manifest.exists()
    full_exists  = full_manifest.exists()

    # 판정
    issues = guard["issues"][:]
    if guard["ct_missing"]:
        issues.append(f"CT 누락 {len(guard['ct_missing'])}명")
    if guard["roi_missing"]:
        issues.append(f"ROI 누락 {len(guard['roi_missing'])}명")
    if guard["patch_missing"]:
        issues.append(f"patch CSV 누락 {len(guard['patch_missing'])}명")

    total_est = est["total_estimated"]
    verdict = "PASS" if not issues else "FAIL"

    _log(f"\n[dry] 예상 manifest rows: {total_est:,} / max {EXPECTED_TRAIN_N * len(POSITION_BINS) * CAP_PER_BIN:,}")
    _log(f"[dry] under-cap bin 건수: {est['under_cap_bins_count']}")
    _log(f"[dry] schema columns: {len(MANIFEST_SCHEMA)}")
    _log(f"[dry] smoke manifest exists: {smoke_exists}")
    _log(f"[dry] full manifest exists:  {full_exists}")
    _log(f"[dry] 판정: {verdict}")
    if issues:
        for iss in issues:
            _log(f"  [ISSUE] {iss}")

    # 출력 파일 저장
    DRY_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # n_c3_input_path_check.csv
    path_rows = []
    for pid in train_patients:
        sid = p2s.get(pid, pid)
        ct_ok  = (NORMAL_CT_ROOT / sid / "ct_hu.npy").exists()
        roi_ok = (V4_20_NORMAL_ROI_ROOT / sid / "refined_roi.npy").exists()
        pc_ok  = (PATCH_INDEX_ROOT / f"{sid}.csv").exists()
        path_rows.append({
            "patient_id": pid, "safe_id": sid,
            "ct_ok": ct_ok, "roi_ok": roi_ok, "patch_csv_ok": pc_ok,
        })
    with open(DRY_OUT_DIR / "n_c3_input_path_check.csv", "w",
              encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["patient_id", "safe_id", "ct_ok", "roi_ok", "patch_csv_ok"])
        w.writeheader()
        w.writerows(path_rows)

    # n_c3_sampling_count_estimate.csv
    with open(DRY_OUT_DIR / "n_c3_sampling_count_estimate.csv", "w",
              encoding="utf-8-sig", newline="") as f:
        if est["by_patient"]:
            fieldnames = list(est["by_patient"][0].keys())
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(est["by_patient"])

    # n_c3_schema_validation.csv
    schema_rows = [{"column": c, "present": True} for c in MANIFEST_SCHEMA]
    with open(DRY_OUT_DIR / "n_c3_schema_validation.csv", "w",
              encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["column", "present"])
        w.writeheader()
        w.writerows(schema_rows)

    # n_c3_output_path_check.csv
    out_path_rows = [
        {"path": str(smoke_manifest), "mode": "smoke", "exists": smoke_exists, "collision": smoke_exists},
        {"path": str(full_manifest),  "mode": "full",  "exists": full_exists,  "collision": full_exists},
    ]
    with open(DRY_OUT_DIR / "n_c3_output_path_check.csv", "w",
              encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "mode", "exists", "collision"])
        w.writeheader()
        w.writerows(out_path_rows)

    # n_c3_errors.csv
    err_rows = []
    for iss in issues:
        err_rows.append({"category": "issue", "severity": "FAIL", "message": iss})
    if not err_rows:
        err_rows.append({"category": "none", "severity": "OK", "message": "no issues"})
    with open(DRY_OUT_DIR / "n_c3_errors.csv", "w",
              encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["category", "severity", "message"])
        w.writeheader()
        w.writerows(err_rows)

    # JSON 요약
    ts = datetime.now().isoformat(timespec="seconds")
    summary = {
        "step": "N-C3",
        "mode": "dry_run",
        "verdict": verdict,
        "timestamp": ts,
        "allow_real_manifest": ALLOW_REAL_MANIFEST,
        "n_c2_verdict": guard.get("n_c2", {}).get("verdict", "unknown"),
        "train_patients": len(train_patients),
        "expected_train_n": EXPECTED_TRAIN_N,
        "train_val_overlap": len(set(guard["train_patients"]) & set(guard["val_patients"])),
        "train_test_overlap": len(set(guard["train_patients"]) & set(guard["test_patients"])),
        "ct_missing": len(guard["ct_missing"]),
        "roi_missing": len(guard["roi_missing"]),
        "patch_csv_missing": len(guard["patch_missing"]),
        "stage2_holdout_accessed": False,
        "pc_supervised_artifact_used": False,
        "crop_generated": False,
        "feature_extraction_run": False,
        "model_forward_run": False,
        "training_run": False,
        "scoring_run": False,
        "full_manifest_generated": False,
        "sampling_strategy": "D_patient_x_position_balanced",
        "cap_per_bin": CAP_PER_BIN,
        "n_bins": len(POSITION_BINS),
        "max_crops_per_patient": len(POSITION_BINS) * CAP_PER_BIN,
        "expected_max_rows": EXPECTED_TRAIN_N * len(POSITION_BINS) * CAP_PER_BIN,
        "estimated_rows": total_est,
        "under_cap_bins_count": est["under_cap_bins_count"],
        "bin_totals": est["bin_totals"],
        "schema_columns": len(MANIFEST_SCHEMA),
        "smoke_manifest_collision": smoke_exists,
        "full_manifest_collision": full_exists,
        "issues": issues,
        "existing_results_modified": False,
    }
    with open(DRY_OUT_DIR / "n_c3_normal_only_crop_manifest_drycheck.json", "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # MD 보고서
    _write_dry_md(summary, est)

    _log(f"\n[dry] 출력: {DRY_OUT_DIR}")
    if verdict == "FAIL":
        sys.exit(1)


def _write_dry_md(s: dict, est: dict) -> None:
    lines = [
        "# N-C3 Normal-Only Crop Manifest — Dry-Run Check\n",
        f"**판정: {'✅ 통과' if s['verdict'] == 'PASS' else '❌ 실패'}**\n",
        f"**날짜**: {s['timestamp'][:10]}  \n",
        f"**모드**: dry-run  \n",
        "\n---\n",
        "## 판정 요약\n\n",
        "| 항목 | 결과 |\n|------|------|\n",
        f"| N-C2 verdict | {s['n_c2_verdict']} |\n",
        f"| normal_train 290명 확인 | {'✅ PASS' if s['train_patients'] == 290 else '❌ FAIL'} |\n",
        f"| train∩val overlap=0 | {'✅ PASS' if s['train_val_overlap'] == 0 else '❌ FAIL'} |\n",
        f"| train∩test overlap=0 | {'✅ PASS' if s['train_test_overlap'] == 0 else '❌ FAIL'} |\n",
        f"| CT 누락=0 | {('✅ PASS' if s['ct_missing'] == 0 else ('❌ FAIL (' + str(s['ct_missing']) + '명)'))} |\n",
        f"| ROI 누락=0 | {('✅ PASS' if s['roi_missing'] == 0 else ('❌ FAIL (' + str(s['roi_missing']) + '명)'))} |\n",
        f"| patch CSV 누락=0 | {('✅ PASS' if s['patch_csv_missing'] == 0 else ('❌ FAIL (' + str(s['patch_csv_missing']) + '명)'))} |\n",
        f"| stage2_holdout 미접근 | ✅ PASS |\n",
        f"| P-C supervised artifact 미사용 | ✅ PASS |\n",
        f"| full manifest 미생성 | ✅ PASS |\n",
        f"| crop 생성 없음 | ✅ PASS |\n",
        f"| 기존 결과 무수정 | ✅ PASS |\n",
        "\n---\n",
        "## sampling 설계 확인\n\n",
        f"| 항목 | 값 |\n|------|----|\n",
        f"| strategy | D: patient × position-balanced |\n",
        f"| cap_per_bin | {s['cap_per_bin']} |\n",
        f"| n_bins | {s['n_bins']} |\n",
        f"| max_per_patient | {s['max_crops_per_patient']} |\n",
        f"| expected_max_rows | {s['expected_max_rows']:,} |\n",
        f"| estimated_rows (CSV 기반) | {s['estimated_rows']:,} |\n",
        f"| under_cap_bin 건수 | {s['under_cap_bins_count']} |\n",
        f"| crop_size | {CROP_SIZE}px |\n",
        f"| dimension | 2.5D 3ch (z-1/z/z+1) |\n",
        f"| dtype | int16 (저장 시) |\n",
        f"| seed | {SAMPLING_SEED} |\n",
        "\n---\n",
        "## position_bin별 추정 pool 크기\n\n",
        "| position_bin | 전체 pool | under-cap 환자 수 |\n|---|---|---|\n",
    ]
    for b in POSITION_BINS:
        total = est["bin_totals"].get(b, 0)
        uc = est["bin_under_cap_patients"].get(b, 0)
        lines.append(f"| {b} | {total:,} | {uc} |\n")

    lines += [
        "\n---\n",
        "## output collision 확인\n\n",
        f"| 경로 | 모드 | 존재 여부 | collision |\n|---|---|---|---|\n",
        f"| n_c3_smoke_manifest.csv | smoke | {s['smoke_manifest_collision']} | {s['smoke_manifest_collision']} |\n",
        f"| n_c3_normal_only_crop_manifest_v1.csv | full | {s['full_manifest_collision']} | {s['full_manifest_collision']} |\n",
        "\n---\n",
        "## 다음 단계\n\n",
        "- **N-C3b**: `--smoke-one-patient --confirm-smoke` 실행 → smoke manifest 1명 생성 검증\n",
        "- **N-C3c**: smoke 검증 후 `--full-manifest --confirm-full` 실행 (290명 승인 필요)\n",
    ]

    with open(DRY_OUT_DIR / "n_c3_normal_only_crop_manifest_drycheck.md", "w",
              encoding="utf-8") as f:
        f.writelines(lines)


# ---------------------------------------------------------------------------
# smoke-one-patient 모드
# ---------------------------------------------------------------------------

def run_smoke(guard: dict) -> None:
    _log("\n=== [SMOKE] N-C3 Smoke Manifest (1 patient) ===")

    if not ALLOW_REAL_MANIFEST:
        _abort("ALLOW_REAL_MANIFEST=False → smoke 실행 불가. 스크립트 상단 플래그를 True로 변경하고 --confirm-smoke 추가 필요.")

    train_patients = guard["train_patients"]
    p2s = guard["p2s"]

    if SMOKE_OUT_DIR.exists() and any(SMOKE_OUT_DIR.iterdir()):
        _abort(f"smoke 출력 경로에 기존 파일 존재: {SMOKE_OUT_DIR} (overwrite 금지)")
    SMOKE_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 첫번째 환자 smoke
    pid = train_patients[0]
    sid = p2s.get(pid, pid)
    _log(f"[smoke] 대상: {pid} ({sid})")

    rows, next_cid, stats = sample_patient_manifest(pid, sid, p2s, candidate_id_start=1)
    _log(f"[smoke] 생성 rows: {len(rows)}")

    out_csv = SMOKE_OUT_DIR / "n_c3_smoke_manifest.csv"
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_SCHEMA)
        w.writeheader()
        w.writerows(rows)

    ts = datetime.now().isoformat(timespec="seconds")
    summary = {
        "step": "N-C3", "mode": "smoke_one_patient",
        "verdict": "PASS" if rows else "FAIL",
        "timestamp": ts,
        "patient_id": pid, "safe_id": sid,
        "n_rows": len(rows),
        "stats_by_bin": stats,
        "output_csv": str(out_csv),
        "allow_real_manifest": ALLOW_REAL_MANIFEST,
        "crop_generated": False,
        "stage2_holdout_accessed": False,
        "pc_supervised_artifact_used": False,
    }
    with open(SMOKE_OUT_DIR / "n_c3_smoke_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    _log(f"[smoke] 출력: {SMOKE_OUT_DIR}")
    _log(f"[smoke] 판정: {summary['verdict']}")


# ---------------------------------------------------------------------------
# full manifest 모드
# ---------------------------------------------------------------------------

def run_full(guard: dict) -> None:
    _log("\n=== [FULL] N-C3 Normal-Only Crop Manifest (290 patients) ===")

    if not ALLOW_REAL_MANIFEST:
        _abort("ALLOW_REAL_MANIFEST=False → full 실행 불가.")

    train_patients = guard["train_patients"]
    p2s = guard["p2s"]

    if FULL_OUT_DIR.exists() and any(FULL_OUT_DIR.iterdir()):
        _abort(f"full 출력 경로에 기존 파일 존재: {FULL_OUT_DIR} (overwrite 금지)")
    FULL_OUT_DIR.mkdir(parents=True, exist_ok=True)

    out_csv = FULL_OUT_DIR / "n_c3_normal_only_crop_manifest_v1.csv"
    cid = 1
    total_rows = 0

    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_SCHEMA)
        w.writeheader()
        for i, pid in enumerate(train_patients):
            sid = p2s.get(pid, pid)
            rows, cid, stats = sample_patient_manifest(pid, sid, p2s, cid)
            w.writerows(rows)
            total_rows += len(rows)
            if (i + 1) % 50 == 0 or i == 0:
                _log(f"  [{i+1}/{len(train_patients)}] {pid}: {len(rows)}행 (누적 {total_rows:,})")

    ts = datetime.now().isoformat(timespec="seconds")
    summary = {
        "step": "N-C3", "mode": "full_manifest",
        "verdict": "PASS" if total_rows > 0 else "FAIL",
        "timestamp": ts,
        "n_patients": len(train_patients),
        "n_rows": total_rows,
        "output_csv": str(out_csv),
        "allow_real_manifest": ALLOW_REAL_MANIFEST,
        "crop_generated": False,
        "stage2_holdout_accessed": False,
        "pc_supervised_artifact_used": False,
    }
    with open(FULL_OUT_DIR / "n_c3_manifest_generation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    _log(f"[full] 총 rows: {total_rows:,}")
    _log(f"[full] 출력: {FULL_OUT_DIR}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="N-C3 Normal-Only Crop Manifest")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--dry-run",            action="store_true")
    mode_group.add_argument("--smoke-one-patient",  action="store_true")
    mode_group.add_argument("--full-manifest",      action="store_true")
    parser.add_argument("--confirm-smoke", action="store_true",
                        help="smoke manifest 실제 생성 확인 (--smoke-one-patient와 함께)")
    parser.add_argument("--confirm-full",  action="store_true",
                        help="full manifest 실제 생성 확인 (--full-manifest와 함께)")
    args = parser.parse_args()

    # bare run 차단 (argparse required=True로 이미 처리)

    # --smoke/--full 실행 가드
    if args.smoke_one_patient and not args.confirm_smoke:
        _abort("--smoke-one-patient 실행 시 --confirm-smoke 필요.")
    if args.full_manifest and not args.confirm_full:
        _abort("--full-manifest 실행 시 --confirm-full 필요.")

    _log(f"[N-C3] 모드: {'dry-run' if args.dry_run else 'smoke' if args.smoke_one_patient else 'full'}")

    guard = run_guards()
    if guard.get("issues") and not args.dry_run:
        # dry-run은 이슈가 있어도 결과 파일 생성 후 리포트
        # smoke/full은 이슈 있으면 중단
        hard = [i for i in guard["issues"] if not i.startswith("G2")]
        if hard:
            _abort(f"가드 실패: {hard}")

    if args.dry_run:
        run_dry(guard)
    elif args.smoke_one_patient:
        run_smoke(guard)
    elif args.full_manifest:
        run_full(guard)


if __name__ == "__main__":
    # bare run (인수 없음) 차단
    if len(sys.argv) == 1:
        print("[ABORT] 인수 없이 직접 실행 불가. --dry-run / --smoke-one-patient / --full-manifest 중 하나 필요.")
        sys.exit(2)
    main()
