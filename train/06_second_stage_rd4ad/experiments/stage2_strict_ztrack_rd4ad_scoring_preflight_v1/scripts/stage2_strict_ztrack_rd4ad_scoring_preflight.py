"""
stage2_strict_ztrack_rd4ad_scoring_preflight.py

목적:
  stage2_holdout 데이터에 대해 strict z-track RD4AD scoring을 진행하기 전
  scoring manifest 생성 및 사전 조건 검증.

핵심 수정 사항 (stage1 대비):
  - stage2 manifest의 y0/x0/y1/x1는 32×32 position 좌표 (96×96 아님)
  - z-track grouping key는 32×32 position 좌표 유지
  - RD4AD crop은 center±48으로 96×96 새로 생성
  - pos_y0/pos_x0/pos_y1/pos_x1 (32×32) vs crop_y0/crop_x0/crop_y1/crop_x1 (96×96) 분리 저장

가드레일:
  - model forward / checkpoint load / crop 실제 생성 / RD4AD scoring 금지
  - stage2 결과를 보고 method 변경 금지 (eval-only)
  - min_run=1 예외 추가 금지
  - score_original rescue 금지

실행:
  bare run           → sys.exit(2) (금지)
  dry-run            → --dry-run
  actual preflight   → --run-preflight --confirm-readonly --confirm-stage2-holdout-eval-only
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT    = Path("/home/jinhy/project/lung-ct-anomaly")
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments/stage2_strict_ztrack_rd4ad_scoring_preflight_v1"

SURVIVAL_CSV  = (
    PROJECT_ROOT
    / "experiments/stage2_strict_ztrack_schema_survival_preflight_v1"
    / "manifests/stage2_ztrack_candidate_survival_minrun2.csv"
)
ZTRACK_MANIFEST = (
    PROJECT_ROOT
    / "experiments/stage2_strict_ztrack_schema_survival_preflight_v1"
    / "manifests/stage2_ztrack_manifest_minrun2.csv"
)
CKPT_PATH = (
    PROJECT_ROOT
    / "outputs/models/rd_d1s_true_rd4ad_resnet18_medi3ch_shard_v1"
    / "checkpoints/best_train_loss.pth"
)
RESNET_WEIGHT = Path("/home/jinhy/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth")
CT_ROOT = Path(
    "/mnt/c/Users/jinhy/Desktop"
    "/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1"
    "/volumes_npy"
)
ROI_MASK_ROOT = (
    PROJECT_ROOT
    / "outputs/mip-postprocess-research-v1/masks"
    / "refined_roi_v4_20_modeB_all_v1"
)

# output paths
OUT_MANIFEST    = EXPERIMENT_ROOT / "manifests/stage2_rd4ad_scoring_manifest_minrun2.csv"
OUT_SHARD_PLAN  = EXPERIMENT_ROOT / "manifests/stage2_rd4ad_scoring_shard_plan.csv"
OUT_ERRORS_CSV  = EXPERIMENT_ROOT / "logs/errors.csv"
OUT_REPORT      = EXPERIMENT_ROOT / "reports/stage2_strict_ztrack_rd4ad_scoring_preflight_report.md"
OUT_SUMMARY     = EXPERIMENT_ROOT / "reports/stage2_strict_ztrack_rd4ad_scoring_preflight_summary.json"
OUT_DONE        = EXPERIMENT_ROOT / "DONE.json"

# ── constants ──────────────────────────────────────────────────────────────────

GUARDRAILS = {
    "stage2_holdout_used_for_method_tuning":  False,
    "model_forward_executed":                 False,
    "checkpoint_loaded":                      False,
    "crop_generation_executed":               False,
    "training_executed":                      False,
    "scoring_executed":                       False,
    "existing_artifact_modified":             False,
    "existing_script_modified":               False,
    "output_overwrite":                       False,
    "label_used_for_evaluation_only":         True,
    "label_used_as_selector":                 False,
    "min_run_exception_added":                False,
    "score_original_rescue_used":             False,
    "roi_hard_filter_applied":                False,
    "vessel_mask_applied":                    False,
    "all_survived_track_candidates_preserved": True,
    "primary_min_run_len":                    2,
    # eval-only 방식 확정 (stage1_dev에서 결정)
    "primary_candidate_score":                "P1_times_roi",
    "primary_track_score":                    "P1_track_top3_mean",
    "auxiliary_track_score":                  "P1_track_top2_mean",
}

SHARD_COUNT          = 8
CROP_SIZE            = 96
HALF_CROP            = CROP_SIZE // 2      # = 48
POS_SIZE             = 32                  # stage2 manifest position 좌표 크기
EXPECTED_SURVIVED    = 128_827
EXPECTED_TRACKS      = 20_910
EXPECTED_POS_PAT_COV = 153                 # 154명 중 153명 (LUNG1-415 complete miss)
COMPLETE_MISS_KNOWN  = {"LUNG1-415"}       # z-track min_run=2 구조적 탈락, 변경 금지

# stage1 scoring runtime ref (n_forward, elapsed_sec)
# → sec_per_forward 추정 기준
RUNTIME_REF = [(4913, 84.2), (4037, 85.7), (4843, 90.1), (6423, 106.2)]

# ── helpers ────────────────────────────────────────────────────────────────────

errors = []

def _add_error(step: str, code: str, msg: str, row: dict = None):
    entry = {"step": step, "code": code, "message": msg}
    if row:
        entry["candidate_id"] = row.get("candidate_id", "")
        entry["patient_id"]   = row.get("patient_id", "")
    errors.append(entry)


def _patient_shard(patient_id: str, n_shards: int = SHARD_COUNT) -> int:
    return int(hashlib.md5(patient_id.encode()).hexdigest(), 16) % n_shards


def _sec_per_forward() -> float:
    if not RUNTIME_REF:
        return 0.018
    weighted = sum(s / n for n, s in RUNTIME_REF) / len(RUNTIME_REF)
    return round(weighted, 6)


def _make_crop_coords(y0: int, x0: int, y1: int, x1: int):
    """32×32 position 좌표 → center±48 으로 96×96 crop 생성."""
    cy = (y0 + y1) / 2.0
    cx = (x0 + x1) / 2.0
    crop_y0 = round(cy - HALF_CROP)
    crop_x0 = round(cx - HALF_CROP)
    crop_y1 = crop_y0 + CROP_SIZE
    crop_x1 = crop_x0 + CROP_SIZE
    return crop_y0, crop_x0, crop_y1, crop_x1


# ── step functions ─────────────────────────────────────────────────────────────

def _check_input_files() -> dict:
    """필수 입력 파일 존재 확인."""
    checks = {
        "survival_csv":    SURVIVAL_CSV.exists(),
        "ztrack_manifest": ZTRACK_MANIFEST.exists(),
        "ckpt_path":       CKPT_PATH.exists(),
        "resnet_weight":   RESNET_WEIGHT.exists(),
        "ct_root":         CT_ROOT.exists(),
        "roi_mask_root":   ROI_MASK_ROOT.exists(),
    }
    for name, ok in checks.items():
        mark = "OK" if ok else "MISSING"
        print(f"      [{mark}] {name}")
    return checks


def _load_survival_csv() -> list:
    """survived=True 후보만 로딩."""
    rows = []
    with open(SURVIVAL_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("survived", "").strip().lower() == "true":
                rows.append(row)
    return rows


def _load_ztrack_manifest() -> dict:
    """track_id → {track_len, has_positive, n_positive, z_start, z_end}."""
    d = {}
    with open(ZTRACK_MANIFEST, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            d[row["track_id"]] = {
                "track_len":   int(row["track_len"]),
                "has_positive": row.get("has_positive", "False").strip().lower() == "true",
                "n_positive":   int(row.get("n_positive", 0)),
                "z_start":      int(row.get("z_start", 0)),
                "z_end":        int(row.get("z_end", 0)),
            }
    return d


def _validate_position_coords(survived_rows: list) -> dict:
    """pos(y0,x0,y1,x1) 32×32 크기 확인."""
    bad_h, bad_w, sample_bad = [], [], []
    for row in survived_rows:
        py0, px0, py1, px1 = (
            int(row["y0"]), int(row["x0"]),
            int(row["y1"]), int(row["x1"]),
        )
        h, w = py1 - py0, px1 - px0
        if h != POS_SIZE:
            bad_h.append(row["candidate_id"])
        if w != POS_SIZE:
            bad_w.append(row["candidate_id"])
        if (h != POS_SIZE or w != POS_SIZE) and len(sample_bad) < 5:
            sample_bad.append({
                "candidate_id": row["candidate_id"],
                "h": h, "w": w,
                "y0": py0, "x0": px0, "y1": py1, "x1": px1,
            })
    return {
        "total":            len(survived_rows),
        "bad_height_count": len(bad_h),
        "bad_width_count":  len(bad_w),
        "all_32x32":        len(bad_h) == 0 and len(bad_w) == 0,
        "sample_bad":       sample_bad,
    }


def _validate_crop_coords(scoring_manifest: list) -> dict:
    """center±48 변환 결과 crop 96×96 크기 확인."""
    bad, sample_bad = [], []
    for row in scoring_manifest:
        h = int(row["crop_y1"]) - int(row["crop_y0"])
        w = int(row["crop_x1"]) - int(row["crop_x0"])
        if h != CROP_SIZE or w != CROP_SIZE:
            bad.append(row["candidate_id"])
            if len(sample_bad) < 5:
                sample_bad.append({
                    "candidate_id": row["candidate_id"],
                    "h": h, "w": w,
                })
    return {
        "total_checked":   len(scoring_manifest),
        "non_96x96_count": len(bad),
        "crop_coord_ok":   len(bad) == 0,
        "sample_bad":      sample_bad,
    }


def _check_ct_readiness(scoring_manifest: list) -> dict:
    """샘플 환자에 대해 CT ct_hu.npy 존재 확인 (model forward 없음)."""
    safe_id_by_patient = {}
    label_by_patient   = {}
    for row in scoring_manifest:
        pid = row["patient_id"]
        if pid not in safe_id_by_patient:
            safe_id_by_patient[pid] = row["safe_id"]
        if row.get("label", "").strip() == "1":
            label_by_patient[pid] = "positive"

    all_pids = list(safe_id_by_patient.keys())
    import random
    random.seed(42)
    sample_pids = random.sample(all_pids, min(20, len(all_pids)))
    # positive 환자 우선 포함
    pos_pids = [p for p in all_pids if label_by_patient.get(p) == "positive"]
    sample_pids = list(set(sample_pids + pos_pids[:10]))[:30]

    ct_root_exists = CT_ROOT.exists()
    results = []
    missing_ct = []
    for pid in sorted(sample_pids):
        sid      = safe_id_by_patient[pid]
        npy_path = CT_ROOT / sid / "ct_hu.npy"
        ok       = npy_path.exists()
        results.append({"patient_id": pid, "safe_id": sid, "ct_exists": ok})
        if not ok:
            missing_ct.append(pid)
            _add_error("ct_readiness", "CT_MISSING", f"{sid}/ct_hu.npy not found")

    return {
        "ct_root":          str(CT_ROOT),
        "ct_root_exists":   ct_root_exists,
        "unique_patients":  len(safe_id_by_patient),
        "sampled":          len(sample_pids),
        "ct_missing_count": len(missing_ct),
        "ct_ok":            ct_root_exists and len(missing_ct) == 0,
        "sample_results":   results,
        "missing_pids":     missing_ct,
    }


def _check_roi_mask_readiness(scoring_manifest: list) -> dict:
    """샘플 환자에 대해 ROI mask (refined_roi.npy) 존재 확인."""
    safe_id_by_patient = {}
    for row in scoring_manifest:
        pid = row["patient_id"]
        if pid not in safe_id_by_patient:
            safe_id_by_patient[pid] = row["safe_id"]

    all_pids = list(safe_id_by_patient.keys())
    import random
    random.seed(42)
    sample_pids = random.sample(all_pids, min(20, len(all_pids)))

    missing_mask = []
    results      = []
    for pid in sorted(sample_pids):
        sid  = safe_id_by_patient[pid]
        # lesion 또는 normal 폴더 중 하나에 있음
        found = False
        for subset in ("lesion", "normal"):
            p = ROI_MASK_ROOT / subset / sid / "refined_roi.npy"
            if p.exists():
                found = True
                break
        results.append({"patient_id": pid, "safe_id": sid, "mask_exists": found})
        if not found:
            missing_mask.append(pid)
            _add_error("roi_mask_readiness", "MASK_MISSING",
                       f"refined_roi.npy not found for {sid}")

    return {
        "roi_mask_root":     str(ROI_MASK_ROOT),
        "roi_root_exists":   ROI_MASK_ROOT.exists(),
        "sampled":           len(sample_pids),
        "mask_missing_count": len(missing_mask),
        "mask_ok":           ROI_MASK_ROOT.exists() and len(missing_mask) == 0,
        "sample_results":    results,
        "missing_pids":      missing_mask,
    }


def _check_complete_miss(scoring_manifest: list, ztrack_dict: dict) -> dict:
    """survived 후보에 positive가 0인 환자 집계."""
    patients_in_manifest = defaultdict(lambda: {"has_positive": False, "count": 0})
    for row in scoring_manifest:
        pid = row["patient_id"]
        patients_in_manifest[pid]["count"] += 1
        if row.get("label", "").strip() in ("1", "positive", "True", "true"):
            patients_in_manifest[pid]["has_positive"] = True

    all_positive_pids = {
        row["patient_id"]
        for row in _load_survival_csv_positive_only()
    }

    complete_miss = []
    for pid in all_positive_pids:
        if pid not in patients_in_manifest or not patients_in_manifest[pid]["has_positive"]:
            complete_miss.append(pid)

    # LUNG1-415 확인
    known_ok = all(p in complete_miss for p in COMPLETE_MISS_KNOWN)

    return {
        "n_positive_patients_total": len(all_positive_pids),
        "complete_miss_count":       len(complete_miss),
        "complete_miss_pids":        sorted(complete_miss),
        "known_complete_miss_confirmed": known_ok,
        "positive_patient_coverage": round(
            (len(all_positive_pids) - len(complete_miss)) / max(len(all_positive_pids), 1), 4
        ),
    }


def _load_survival_csv_positive_only() -> list:
    """positive label 후보만 로딩 (complete_miss 계산용)."""
    rows = []
    with open(SURVIVAL_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("label", "").strip() in ("1", "positive", "True", "true"):
                rows.append(row)
    return rows


def _build_scoring_manifest(survived_rows: list, ztrack_dict: dict) -> list:
    """scoring manifest 생성: pos/crop 좌표 분리, track_len join."""
    scoring_manifest = []
    missing_track = []

    for row in survived_rows:
        cid = row["candidate_id"]
        pid = row["patient_id"]
        tid = row.get("track_id", "")

        py0, px0, py1, px1 = (
            int(row["y0"]), int(row["x0"]),
            int(row["y1"]), int(row["x1"]),
        )
        # 96×96 crop 생성 (center±48)
        crop_y0, crop_x0, crop_y1, crop_x1 = _make_crop_coords(py0, px0, py1, px1)

        # track_len join
        track_info = ztrack_dict.get(tid, {})
        track_len  = track_info.get("track_len", 0)
        if not tid or tid not in ztrack_dict:
            missing_track.append(cid)
            _add_error("build_manifest", "MISSING_TRACK",
                       f"track_id not in ztrack_manifest: {tid}", row)

        row_out = {
            # ID
            "candidate_id":    cid,
            "patient_id":      pid,
            "safe_id":         row.get("safe_id", ""),
            "local_z":         row.get("local_z", ""),
            "label":           row.get("label", ""),
            # 32×32 position (z-track grouping 기준)
            "pos_y0":          py0,
            "pos_x0":          px0,
            "pos_y1":          py1,
            "pos_x1":          px1,
            # 96×96 RD4AD crop (center±48)
            "crop_y0":         crop_y0,
            "crop_x0":         crop_x0,
            "crop_y1":         crop_y1,
            "crop_x1":         crop_x1,
            # track info
            "track_id":        tid,
            "track_len":       track_len,
            "ztrack_min_run_len": row.get("ztrack_min_run_len", "2"),
            # original scoring fields
            "score_original":  row.get("score_original", ""),
            # position source 보존
            "position_source": row.get("position_source", "y0_x0_y1_x1_as_crop_coords"),
        }
        scoring_manifest.append(row_out)

    if missing_track:
        print(f"      [WARN] track_id 미매칭: {len(missing_track)}건")

    return scoring_manifest


def _build_shard_plan(scoring_manifest: list, spf: float) -> list:
    """patient stable hash 기반 shard plan."""
    shard_cands   = defaultdict(list)
    shard_pats    = defaultdict(set)
    shard_pos_pats= defaultdict(set)

    for row in scoring_manifest:
        pid = row["patient_id"]
        sid = _patient_shard(pid)
        shard_cands[sid].append(row)
        shard_pats[sid].add(pid)
        if row.get("label", "").strip() in ("1", "positive", "True", "true"):
            shard_pos_pats[sid].add(pid)

    shard_rows = []
    for sid in range(SHARD_COUNT):
        cands    = shard_cands.get(sid, [])
        n_cand   = len(cands)
        est_sec  = round(n_cand * spf, 1)
        shard_rows.append({
            "shard_id":              sid,
            "candidate_count":       n_cand,
            "patient_count":         len(shard_pats.get(sid, set())),
            "positive_patient_count": len(shard_pos_pats.get(sid, set())),
            "estimated_runtime_sec": est_sec,
            "estimated_runtime_min": round(est_sec / 60, 1),
        })
    return shard_rows


def _compute_patient_coverage(scoring_manifest: list) -> dict:
    """positive 환자 커버리지 계산."""
    pos_patients    = set()
    covered_pos     = set()
    all_patients    = set()

    for row in scoring_manifest:
        pid = row["patient_id"]
        all_patients.add(pid)
        if row.get("label", "").strip() in ("1", "positive", "True", "true"):
            pos_patients.add(pid)
            covered_pos.add(pid)

    return {
        "total_patients":      len(all_patients),
        "positive_patients":   len(pos_patients),
        "covered_positive":    len(covered_pos),
        "pos_patient_coverage": round(
            len(covered_pos) / max(len(pos_patients), 1), 4
        ),
    }


def _write_scoring_manifest(scoring_manifest: list):
    """scoring manifest CSV 저장."""
    if not scoring_manifest:
        return
    fieldnames = list(scoring_manifest[0].keys())
    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MANIFEST, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scoring_manifest)


def _write_shard_plan(shard_rows: list):
    if not shard_rows:
        return
    fieldnames = list(shard_rows[0].keys())
    OUT_SHARD_PLAN.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_SHARD_PLAN, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(shard_rows)


def _write_errors_csv():
    OUT_ERRORS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_ERRORS_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["step", "code", "message", "candidate_id", "patient_id"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(errors)


def _write_report(summary: dict):
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage2 Strict Z-Track RD4AD Scoring Preflight v1\n\n",
        "## 목적\n",
        "stage2_holdout strict z-track RD4AD actual scoring 전 사전 조건 검증.\n\n",
        "## 핵심 수정 (stage1 대비)\n",
        "- y0/x0/y1/x1 = 32×32 position 좌표 (96×96 아님)\n",
        "- RD4AD crop = center±48으로 96×96 새로 생성\n",
        "- pos_y0/pos_x0/pos_y1/pos_x1 vs crop_y0/crop_x0/crop_y1/crop_x1 분리 저장\n\n",
        "## 검증 결과\n\n",
        "| 항목 | 값 |\n",
        "|------|----|\n",
        f"| verdict | {summary['verdict']} |\n",
        f"| survived candidates | {summary['n_survived']:,} |\n",
        f"| expected survived | {EXPECTED_SURVIVED:,} |\n",
        f"| tracks | {summary['n_tracks']:,} |\n",
        f"| positive patient coverage | {summary['pos_patient_coverage']:.4f} |\n",
        f"| complete miss | {', '.join(summary['complete_miss_pids']) or 'none'} |\n",
        f"| position 32×32 ok | {summary['pos_coord_ok']} |\n",
        f"| crop 96×96 ok | {summary['crop_coord_ok']} |\n",
        f"| CT readiness ok | {summary['ct_ok']} |\n",
        f"| ROI mask ok | {summary['roi_mask_ok']} |\n",
        f"| checkpoint exists | {summary['ckpt_exists']} |\n",
        f"| errors | {summary['n_errors']} |\n\n",
        "## Shard Plan 요약\n\n",
        "| shard_id | candidates | patients | positive_patients | est_min |\n",
        "|----------|-----------|---------|-----------------|--------|\n",
    ]
    for s in summary.get("shard_plan", []):
        lines.append(
            f"| {s['shard_id']} | {s['candidate_count']:,} | "
            f"{s['patient_count']} | {s['positive_patient_count']} | "
            f"{s['estimated_runtime_min']} |\n"
        )

    total_min = sum(s["estimated_runtime_min"] for s in summary.get("shard_plan", []))
    spf       = summary.get("sec_per_forward", 0)
    n_surv    = summary.get("n_survived", 0)
    lines.append(f"\n총 예상 시간 (직렬): {total_min:.1f}분 (spf={spf:.4f}s, {n_surv:,}개)\n\n")

    lines += [
        "## actual scoring 예정 출력 feature 컬럼\n\n",
        "```\n",
        "rd4ad_score_raw, score_layer1, score_layer2, score_layer3\n",
        "crop_hu_mean, crop_hu_std, crop_hu_p10, crop_hu_p50, crop_hu_p90\n",
        "roi_0_0_patch_ratio\n",
        "P1_times_roi (= rd4ad_score_raw × roi_0_0_patch_ratio)\n",
        "P2_times_sqrt_roi (= rd4ad_score_raw × sqrt(roi_0_0_patch_ratio))\n",
        "track_id, track_len\n",
        "```\n\n",
        "## 가드레일\n\n",
        "- stage2 결과를 보고 method 변경 금지 (eval-only)\n",
        f"- primary_candidate_score: {GUARDRAILS['primary_candidate_score']}\n",
        f"- primary_track_score: {GUARDRAILS['primary_track_score']}\n",
        f"- auxiliary_track_score: {GUARDRAILS['auxiliary_track_score']}\n",
        "- vessel mask 미적용\n",
        "- ROI hard filter 금지\n",
        f"- complete miss {list(COMPLETE_MISS_KNOWN)}: method 변경 금지\n",
    ]

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.writelines(lines)


def _write_summary(summary: dict):
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def _write_done(summary: dict):
    done = {
        "verdict":    summary["verdict"],
        "guardrails": GUARDRAILS,
        "n_survived": summary["n_survived"],
        "n_tracks":   summary["n_tracks"],
    }
    with open(OUT_DONE, "w", encoding="utf-8") as f:
        json.dump(done, f, indent=2, ensure_ascii=False)


# ── dry-run ────────────────────────────────────────────────────────────────────

def run_dry():
    print("=" * 70)
    print("[DRY-RUN] stage2 strict z-track RD4AD scoring preflight v1")
    print("=" * 70)

    print("\n[1] 입력 파일 존재 확인...")
    checks = _check_input_files()
    all_ok = all(v for k, v in checks.items()
                 if k not in ("resnet_weight",))  # resnet_weight는 WARN만
    print(f"\n  → 필수 파일 {'전부 OK' if all_ok else '일부 MISSING'}")

    print(f"\n[2] 출력 경로")
    print(f"  manifest:   {OUT_MANIFEST}")
    print(f"  shard plan: {OUT_SHARD_PLAN}")
    print(f"  report:     {OUT_REPORT}")
    print(f"  summary:    {OUT_SUMMARY}")
    print(f"  errors:     {OUT_ERRORS_CSV}")
    print(f"  DONE:       {OUT_DONE}")

    print(f"\n[3] 가드레일 확인")
    for k, v in GUARDRAILS.items():
        print(f"  {k}: {v}")

    print(f"\n[4] crop 변환 예시")
    print(f"  position(y0=128, x0=368, y1=160, x1=400)")
    cy0, cx0, cy1, cx1 = _make_crop_coords(128, 368, 160, 400)
    print(f"  → crop_y0={cy0}, crop_x0={cx0}, crop_y1={cy1}, crop_x1={cx1}")
    h, w = cy1 - cy0, cx1 - cx0
    print(f"  → crop 크기: {h}×{w} ({'OK' if h == CROP_SIZE and w == CROP_SIZE else 'ERROR'})")

    print(f"\n[DRY-RUN 완료] 실제 preflight: --run-preflight --confirm-readonly --confirm-stage2-holdout-eval-only")


# ── actual preflight ───────────────────────────────────────────────────────────

def run_preflight():
    print("=" * 70)
    print("[PREFLIGHT] stage2 strict z-track RD4AD scoring preflight v1")
    print("=" * 70)

    # ── [1] 입력 파일 확인 ──────────────────────────────────────────────────────
    print("\n[1/9] 입력 파일 존재 확인...")
    file_checks = _check_input_files()
    ckpt_exists = file_checks["ckpt_path"]
    ct_root_ok  = file_checks["ct_root"]
    if not ckpt_exists:
        _add_error("file_check", "CKPT_MISSING", str(CKPT_PATH))
    if not ct_root_ok:
        _add_error("file_check", "CT_ROOT_MISSING", str(CT_ROOT))

    # ── [2] survival CSV 로딩 ────────────────────────────────────────────────
    print("\n[2/9] survival CSV 로딩 (survived=True)...")
    survived_rows = _load_survival_csv()
    n_survived = len(survived_rows)
    print(f"      survived=True: {n_survived:,}  (expected: {EXPECTED_SURVIVED:,})")
    survived_match = n_survived == EXPECTED_SURVIVED
    if not survived_match:
        _add_error("survival_check", "SURVIVED_COUNT_MISMATCH",
                   f"expected {EXPECTED_SURVIVED}, got {n_survived}")

    # ── [3] ztrack manifest 로딩 ─────────────────────────────────────────────
    print("\n[3/9] ztrack manifest 로딩...")
    ztrack_dict = _load_ztrack_manifest()
    n_tracks = len(ztrack_dict)
    print(f"      tracks: {n_tracks:,}  (expected: {EXPECTED_TRACKS:,})")
    tracks_match = n_tracks == EXPECTED_TRACKS
    if not tracks_match:
        _add_error("ztrack_check", "TRACK_COUNT_MISMATCH",
                   f"expected {EXPECTED_TRACKS}, got {n_tracks}")

    # ── [4] position 좌표 32×32 검증 ─────────────────────────────────────────
    print("\n[4/9] position 좌표 32×32 검증...")
    pos_check = _validate_position_coords(survived_rows)
    mark = "OK" if pos_check["all_32x32"] else "WARN"
    print(f"      [{mark}] 32×32 아닌 position: "
          f"{pos_check['bad_height_count']} (h) / {pos_check['bad_width_count']} (w)")
    if pos_check["sample_bad"]:
        for s in pos_check["sample_bad"]:
            print(f"        sample: {s}")
        _add_error("pos_coord_check", "NON_32x32_POSITION",
                   f"bad_h={pos_check['bad_height_count']} bad_w={pos_check['bad_width_count']}")

    # ── [5] scoring manifest 생성 ─────────────────────────────────────────────
    print("\n[5/9] scoring manifest 생성 (center±48 crop 변환)...")
    scoring_manifest = _build_scoring_manifest(survived_rows, ztrack_dict)
    print(f"      scoring manifest rows: {len(scoring_manifest):,}")

    # ── [6] crop 96×96 검증 ───────────────────────────────────────────────────
    print("\n[6/9] crop 좌표 96×96 검증...")
    crop_check = _validate_crop_coords(scoring_manifest)
    mark = "OK" if crop_check["crop_coord_ok"] else "ERROR"
    print(f"      [{mark}] 96×96 아닌 crop: "
          f"{crop_check['non_96x96_count']:,} / {crop_check['total_checked']:,}")
    if not crop_check["crop_coord_ok"]:
        _add_error("crop_coord_check", "NON_96x96_CROP",
                   f"non_96x96_count={crop_check['non_96x96_count']}")
        for s in crop_check["sample_bad"]:
            print(f"        sample: {s}")

    # ── [7] CT readiness 확인 ─────────────────────────────────────────────────
    print("\n[7/9] CT readiness 확인 (샘플, model forward 없음)...")
    ct_check = _check_ct_readiness(scoring_manifest)
    ct_ok = ct_check["ct_ok"]
    mark  = "OK" if ct_ok else "WARN"
    print(f"      [{mark}] CT ct_hu.npy 확인: "
          f"{ct_check['sampled'] - ct_check['ct_missing_count']}/{ct_check['sampled']} OK")
    print(f"      unique patients: {ct_check['unique_patients']:,}")
    if not ct_root_ok:
        print(f"      [WARN] CT_ROOT 접근 불가: {CT_ROOT} → Windows 마운트 확인 필요")

    # ── ROI mask readiness ─────────────────────────────────────────────────────
    roi_check = _check_roi_mask_readiness(scoring_manifest)
    mask_ok   = roi_check["mask_ok"]
    mark      = "OK" if mask_ok else "WARN"
    print(f"      [{mark}] ROI mask: "
          f"{roi_check['sampled'] - roi_check['mask_missing_count']}/{roi_check['sampled']} OK")

    # ── [8] complete miss 확인 ────────────────────────────────────────────────
    print("\n[8/9] complete miss 확인...")
    pat_cov = _compute_patient_coverage(scoring_manifest)

    # positive manifest는 survived=True 기준으로 확인
    # LUNG1-415는 survived=False이므로 scoring_manifest에 없음
    complete_miss_in_scoring = []
    # positive 환자 중 scoring_manifest에 positive 후보가 없는 환자
    pos_patients_in_scoring  = set()
    all_pos_patients         = set()

    for row in scoring_manifest:
        if row.get("label", "").strip() in ("1", "positive", "True", "true"):
            pos_patients_in_scoring.add(row["patient_id"])

    # survival CSV 전체에서 positive 환자 목록
    with open(SURVIVAL_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("label", "").strip() in ("1", "positive", "True", "true"):
                all_pos_patients.add(row["patient_id"])

    complete_miss_in_scoring = sorted(all_pos_patients - pos_patients_in_scoring)

    n_pos_total   = len(all_pos_patients)
    n_covered     = n_pos_total - len(complete_miss_in_scoring)
    pos_coverage  = round(n_covered / max(n_pos_total, 1), 4)

    known_miss_ok = all(p in complete_miss_in_scoring for p in COMPLETE_MISS_KNOWN)

    print(f"      positive patients (total): {n_pos_total:,}")
    print(f"      covered in scoring:        {n_covered:,}  (coverage={pos_coverage:.4f})")
    print(f"      complete miss:             {len(complete_miss_in_scoring)} → {complete_miss_in_scoring}")
    print(f"      LUNG1-415 confirmed miss:  {'YES (방법 변경 없음)' if 'LUNG1-415' in complete_miss_in_scoring else 'NOT IN MISS'}")

    # ── [9] shard plan 생성 ───────────────────────────────────────────────────
    print("\n[9/9] shard plan 생성...")
    spf        = _sec_per_forward()
    shard_rows = _build_shard_plan(scoring_manifest, spf)
    total_sec  = sum(s["estimated_runtime_sec"] for s in shard_rows)
    total_min  = total_sec / 60
    print(f"      sec_per_forward (추정): {spf:.4f}s")
    print(f"      총 candidates:          {len(scoring_manifest):,}")
    print(f"      예상 직렬 시간:          {total_min:.1f}분")
    print(f"      shard count:            {SHARD_COUNT}")
    for s in shard_rows:
        print(f"        shard {s['shard_id']}: {s['candidate_count']:,}개  "
              f"{s['patient_count']}명  ~{s['estimated_runtime_min']}분")

    # ── verdict ───────────────────────────────────────────────────────────────
    fatal_errors = [e for e in errors if e["code"] in (
        "CKPT_MISSING",
        "SURVIVED_COUNT_MISMATCH",
        "NON_96x96_CROP",
    )]
    warn_errors  = [e for e in errors if e["code"] in (
        "CT_ROOT_MISSING", "CT_MISSING",
        "MASK_MISSING",
        "MISSING_TRACK",
        "TRACK_COUNT_MISMATCH",
    )]

    if fatal_errors:
        verdict = "FAIL"
    elif warn_errors or not survived_match:
        verdict = "PARTIAL_PASS"
    else:
        verdict = "PASS"

    print(f"\n{'=' * 70}")
    print(f"[VERDICT] {verdict}")
    print(f"  survived match:       {survived_match} ({n_survived:,} / {EXPECTED_SURVIVED:,})")
    print(f"  pos_coord_ok:         {pos_check['all_32x32']}")
    print(f"  crop_coord_ok:        {crop_check['crop_coord_ok']}")
    print(f"  ckpt_exists:          {ckpt_exists}")
    print(f"  ct_ok:                {ct_ok}")
    print(f"  roi_mask_ok:          {mask_ok}")
    print(f"  complete_miss:        {complete_miss_in_scoring}")
    print(f"  errors:               {len(errors)} (fatal={len(fatal_errors)}, warn={len(warn_errors)})")
    print(f"{'=' * 70}")

    # ── 저장 ──────────────────────────────────────────────────────────────────
    print("\n[저장]...")
    _write_scoring_manifest(scoring_manifest)
    print(f"  → {OUT_MANIFEST}")

    _write_shard_plan(shard_rows)
    print(f"  → {OUT_SHARD_PLAN}")

    _write_errors_csv()
    print(f"  → {OUT_ERRORS_CSV}")

    summary = {
        "verdict":              verdict,
        "n_survived":           n_survived,
        "n_survived_expected":  EXPECTED_SURVIVED,
        "n_tracks":             n_tracks,
        "n_tracks_expected":    EXPECTED_TRACKS,
        "pos_patient_coverage": pos_coverage,
        "n_positive_patients":  n_pos_total,
        "n_covered_positive":   n_covered,
        "complete_miss_pids":   complete_miss_in_scoring,
        "known_complete_miss":  sorted(COMPLETE_MISS_KNOWN),
        "pos_coord_ok":         pos_check["all_32x32"],
        "crop_coord_ok":        crop_check["crop_coord_ok"],
        "ckpt_exists":          ckpt_exists,
        "ct_ok":                ct_ok,
        "roi_mask_ok":          mask_ok,
        "n_errors":             len(errors),
        "n_fatal":              len(fatal_errors),
        "n_warn":               len(warn_errors),
        "sec_per_forward":      spf,
        "estimated_total_min":  round(total_min, 1),
        "shard_plan":           shard_rows,
        "pos_coord_check":      pos_check,
        "crop_coord_check":     crop_check,
        "ct_check":             ct_check,
        "roi_mask_check":       roi_check,
        "guardrails":           GUARDRAILS,
        "output_feature_columns_planned": [
            "rd4ad_score_raw", "score_layer1", "score_layer2", "score_layer3",
            "crop_hu_mean", "crop_hu_std", "crop_hu_p10", "crop_hu_p50", "crop_hu_p90",
            "roi_0_0_patch_ratio",
            "P1_times_roi", "P2_times_sqrt_roi",
            "track_id", "track_len",
        ],
    }

    _write_report(summary)
    print(f"  → {OUT_REPORT}")

    _write_summary(summary)
    print(f"  → {OUT_SUMMARY}")

    if verdict in ("PASS", "PARTIAL_PASS"):
        _write_done(summary)
        print(f"  → {OUT_DONE}")

    print("\n[완료]")
    return verdict


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("[ERROR] bare run 금지.", file=sys.stderr)
        print("  dry-run:   --dry-run", file=sys.stderr)
        print("  preflight: --run-preflight --confirm-readonly --confirm-stage2-holdout-eval-only",
              file=sys.stderr)
        sys.exit(2)

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dry-run",                           action="store_true")
    parser.add_argument("--run-preflight",                     action="store_true")
    parser.add_argument("--confirm-readonly",                  action="store_true")
    parser.add_argument("--confirm-stage2-holdout-eval-only",  action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        run_dry()
        return

    if args.run_preflight:
        if not (args.confirm_readonly and args.confirm_stage2_holdout_eval_only):
            print("[ERROR] --run-preflight 실행 시 --confirm-readonly와 "
                  "--confirm-stage2-holdout-eval-only 필요.", file=sys.stderr)
            sys.exit(2)

        EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
        (EXPERIMENT_ROOT / "manifests").mkdir(exist_ok=True)
        (EXPERIMENT_ROOT / "reports").mkdir(exist_ok=True)
        (EXPERIMENT_ROOT / "logs").mkdir(exist_ok=True)

        run_preflight()
        return

    print("[ERROR] 알 수 없는 인수. --dry-run 또는 --run-preflight 사용.", file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
