#!/usr/bin/env python3
"""
validate_s6a_crop_full.py
=========================
crops_s6a_full 130,659개 npz 전수 검증 스크립트.

목적:
- S6-A crop full 결과가 실제 학습 입력으로 안전한지 확인
- read-only 검증만 수행 (npz 수정/재생성/PNG 생성 금지)

검증 항목 (25개):
  1.  npz 총 개수 130,659개인지
  2.  환자 폴더 수 154개인지
  3.  summary CSV 총 행 수와 npz 실제 개수 일치 여부
  4.  summary JSON total_crops와 npz 실제 개수 일치 여부
  5.  manifest row 수와 npz 실제 개수 일치 여부
  6.  positive 43,553개 / hard_negative 87,106개 일치 여부
  7.  환자별 npz 개수와 summary/manifest 환자별 개수 일치 여부
  8.  LUNG1-140 npz 수 2,232개인지
  9.  필수 key 존재 여부 (24개 key 전부)
  10. crop shape 전부 (3,96,96)인지
  11. crop dtype 전부 float32인지
  12. crop NaN/Inf 0개인지
  13. label 값이 0 또는 1 정수인지
  14. sampling_label 값이 'positive' 또는 'hard_negative'인지
  15. label과 sampling_label 일치 여부 (positive→1, hard_negative→0)
  16. z_source가 전부 "local_z"인지
  17. local_z 값이 존재하고 음수가 아닌지
  18. slice_index_valid key가 전부 존재하는지 (9번과 별도 체크)
  19. crop_coords 길이가 4인지
  20. orig_bbox 길이가 4인지
  21. crop_coords가 96×96 크기인지 (y1-y0==96, x1-x0==96)
  22. sampling_rule이 전부 'S6-A_positive_all_hn_ratio2'인지
  23. patient_id 폴더명과 npz 내부 patient_id가 일치하는지
  24. npz 파일명이 000000.npz 형식으로 환자별 0부터 연속되는지
  25. 기존 score/candidate/evaluation/crop 파일 미수정 확인

절대 금지:
- npz 수정/재생성 금지
- PNG 생성 금지
- 기존 score/candidate/evaluation/crop 파일 수정 금지

syntax check (실행 아님):
  python -m py_compile scripts/validate_s6a_crop_full.py

실행:
  source ~/ai_env/bin/activate && \\
  python scripts/validate_s6a_crop_full.py [--dry-run]
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# 경로 상수
# ─────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent
CROPS_DIR     = BASE_DIR / "outputs/second-stage-lesion-refiner-v1/crops_s6a_full"
SUMMARY_CSV   = BASE_DIR / "outputs/second-stage-lesion-refiner-v1/reports/crop_s6a_full_summary.csv"
SUMMARY_JSON  = BASE_DIR / "outputs/second-stage-lesion-refiner-v1/reports/crop_s6a_full_summary.json"
MANIFEST_CSV  = BASE_DIR / "outputs/second-stage-lesion-refiner-v1/candidates/rule_s6a_gs2_selected_candidate_manifest_dryrun.csv"
OUT_RPT_DIR   = BASE_DIR / "outputs/second-stage-lesion-refiner-v1/reports"
OUT_CSV       = OUT_RPT_DIR / "crop_s6a_full_validation_summary.csv"
OUT_JSON      = OUT_RPT_DIR / "crop_s6a_full_validation_summary.json"
OUT_MD        = OUT_RPT_DIR / "crop_s6a_full_validation_summary.md"

# 기존 파일 미수정 확인 대상 경로 (항목 25)
WATCH_DIRS = [
    BASE_DIR / "outputs/second-stage-lesion-refiner-v1/crops_s6a_full",
    BASE_DIR / "outputs/second-stage-lesion-refiner-v1/candidates",
    BASE_DIR / "outputs/second-stage-lesion-refiner-v1/evaluation",
]
WATCH_FILES_DIRECT = [
    SUMMARY_CSV,
    SUMMARY_JSON,
    MANIFEST_CSV,
]

# ─────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────
EXPECTED_TOTAL         = 130_659
EXPECTED_PATIENT_COUNT = 154
EXPECTED_POS           = 43_553
EXPECTED_HN            = 87_106
EXPECTED_LUNG1_140     = 2_232
EXPECTED_CROP_SHAPE    = (3, 96, 96)
EXPECTED_SAMPLING_RULE = "S6-A_positive_all_hn_ratio2"
REQUIRED_KEYS = [
    "crop", "label", "sampling_label", "sampling_rule", "patient_id",
    "local_z", "slice_index", "slice_index_valid", "z_source",
    "crop_coords", "orig_bbox",
    "score_original", "score_valid950_weighted", "score_valid950_soft",
    "composite_rank_v2", "position_bin", "z_level", "central_peripheral",
    "lesion_patch_ratio", "roi_inside_ratio", "air_ratio_950", "air_ratio_970",
    "valid_ratio_roi_air950", "valid_ratio_roi_air970",
]


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ─────────────────────────────────────────────
# 항목 25: mtime 스냅샷
# ─────────────────────────────────────────────
def snapshot_mtime() -> Dict[str, float]:
    """보호 대상 파일/폴더의 mtime을 기록한다."""
    snap: Dict[str, float] = {}
    for p in WATCH_FILES_DIRECT:
        if p.exists():
            snap[str(p)] = p.stat().st_mtime
    for d in WATCH_DIRS:
        if d.exists():
            for f in d.rglob("*"):
                if f.is_file():
                    snap[str(f)] = f.stat().st_mtime
    return snap


def compare_mtime(before: Dict[str, float]) -> List[str]:
    """시작 시점 대비 mtime이 변경된 파일 목록 반환."""
    changed: List[str] = []
    for path_str, old_mtime in before.items():
        p = Path(path_str)
        if p.exists():
            new_mtime = p.stat().st_mtime
            if new_mtime != old_mtime:
                changed.append(path_str)
        else:
            changed.append(f"[DELETED] {path_str}")
    return changed


# ─────────────────────────────────────────────
# guard_check
# ─────────────────────────────────────────────
def guard_check(dry_run: bool) -> None:
    errors = []

    if not CROPS_DIR.exists():
        errors.append(f"[GUARD] crops_s6a_full 폴더 없음: {CROPS_DIR}")
    if not SUMMARY_CSV.exists():
        errors.append(f"[GUARD] summary CSV 없음: {SUMMARY_CSV}")
    if not SUMMARY_JSON.exists():
        errors.append(f"[GUARD] summary JSON 없음: {SUMMARY_JSON}")
    if not MANIFEST_CSV.exists():
        errors.append(f"[GUARD] manifest CSV 없음: {MANIFEST_CSV}")

    if not dry_run:
        for p in [OUT_CSV, OUT_JSON, OUT_MD]:
            if p.exists():
                errors.append(f"[GUARD] 출력 파일이 이미 존재합니다 (overwrite 방지): {p}")

    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        print("\n[중단] guard 조건 미통과.", file=sys.stderr)
        sys.exit(1)

    # npz 개수 0 확인
    total_npz = sum(
        1 for p in CROPS_DIR.rglob("*.npz") if p.is_file()
    )
    if total_npz == 0:
        print(f"[GUARD] crops_s6a_full 폴더에 npz 파일이 없습니다: {CROPS_DIR}", file=sys.stderr)
        sys.exit(1)

    log(f"[GUARD] 모든 guard 조건 통과. (npz 사전 확인: {total_npz}개)")


# ─────────────────────────────────────────────
# 파일 목록 수집
# ─────────────────────────────────────────────
def collect_npz() -> Dict[str, List[Path]]:
    """환자 폴더별 npz 파일 목록. {patient_id: sorted list of Path}"""
    result: Dict[str, List[Path]] = {}
    for d in sorted(CROPS_DIR.iterdir()):
        if d.is_dir():
            files = sorted(d.glob("*.npz"))
            if files:
                result[d.name] = files
    return result


# ─────────────────────────────────────────────
# 전수 검증 (파일별 루프)
# ─────────────────────────────────────────────
def validate_per_file(
    patient_npz_map: Dict[str, List[Path]],
) -> Tuple[List[dict], Dict[str, List[str]]]:
    """
    전수 검증 루프.
    반환:
      - per_file_records: 파일별 결과 dict 리스트
      - per_patient_issues: {patient_id: [issue_str, ...]}
    """
    per_file_records: List[dict] = []
    per_patient_issues: Dict[str, List[str]] = {}

    total = sum(len(v) for v in patient_npz_map.values())
    checked = 0

    for pid, files in patient_npz_map.items():
        for fpath in files:
            fname = f"{pid}/{fpath.name}"
            issues: List[str] = []

            try:
                d = np.load(str(fpath), allow_pickle=True)
                actual_keys = set(d.files)

                # 항목 9: 필수 key
                missing_keys = [k for k in REQUIRED_KEYS if k not in actual_keys]
                keys_ok = len(missing_keys) == 0
                if not keys_ok:
                    issues.append(f"missing_keys={missing_keys}")

                # 항목 18: slice_index_valid key (9번과 별도)
                siv_ok = "slice_index_valid" in actual_keys
                if not siv_ok:
                    issues.append("slice_index_valid key 없음")

                # 안전하게 값 추출
                crop         = d["crop"]          if "crop"          in actual_keys else None
                label_raw    = d["label"]          if "label"         in actual_keys else None
                sl_raw       = d["sampling_label"] if "sampling_label" in actual_keys else None
                z_source_raw = str(d["z_source"])  if "z_source"      in actual_keys else ""
                local_z_raw  = d["local_z"]        if "local_z"       in actual_keys else None
                crop_coords  = d["crop_coords"]    if "crop_coords"   in actual_keys else None
                orig_bbox    = d["orig_bbox"]      if "orig_bbox"     in actual_keys else None
                rule_raw     = str(d["sampling_rule"]) if "sampling_rule" in actual_keys else ""
                pid_npz      = str(d["patient_id"]) if "patient_id"   in actual_keys else ""

                # 항목 10: shape
                shape_ok = (crop is not None and tuple(crop.shape) == EXPECTED_CROP_SHAPE)
                if not shape_ok:
                    actual_shape = tuple(crop.shape) if crop is not None else None
                    issues.append(f"shape={actual_shape}")

                # 항목 11: dtype
                dtype_ok = (crop is not None and crop.dtype == np.float32)
                if not dtype_ok:
                    issues.append(f"dtype={crop.dtype if crop is not None else None}")

                # 항목 12: NaN/Inf
                nan_ok = (crop is not None and bool(np.isfinite(crop).all()))
                if not nan_ok:
                    has_nan = bool(np.isnan(crop).any()) if crop is not None else False
                    has_inf = bool(np.isinf(crop).any()) if crop is not None else False
                    issues.append(f"nan={has_nan},inf={has_inf}")

                # 항목 13: label 0 또는 1
                label_int = -1
                if label_raw is not None:
                    try:
                        label_int = int(label_raw)
                    except Exception:
                        label_int = -1
                label_val_ok = label_int in (0, 1)
                if not label_val_ok:
                    issues.append(f"label={label_int}")

                # 항목 14: sampling_label 값
                sl_str = str(sl_raw) if sl_raw is not None else ""
                sl_val_ok = sl_str in ("positive", "hard_negative")
                if not sl_val_ok:
                    issues.append(f"sampling_label={sl_str!r}")

                # 항목 15: label ↔ sampling_label 일치
                expected_label = 1 if sl_str == "positive" else (0 if sl_str == "hard_negative" else -1)
                match_ok = (label_int == expected_label)
                if not match_ok:
                    issues.append(f"label_mismatch: label={label_int}, sl={sl_str}")

                # 항목 16: z_source
                zsrc_ok = (z_source_raw == "local_z")
                if not zsrc_ok:
                    issues.append(f"z_source={z_source_raw!r}")

                # 항목 17: local_z >= 0
                lz = -1
                localz_ok = False
                if local_z_raw is not None:
                    try:
                        lz = int(local_z_raw)
                        localz_ok = (lz >= 0)
                    except Exception:
                        localz_ok = False
                if not localz_ok:
                    issues.append(f"local_z={lz}")

                # 항목 19: crop_coords 길이 4
                coords_len_ok = (crop_coords is not None and len(crop_coords) == 4)
                if not coords_len_ok:
                    issues.append(f"crop_coords len={len(crop_coords) if crop_coords is not None else None}")

                # 항목 20: orig_bbox 길이 4
                bbox_len_ok = (orig_bbox is not None and len(orig_bbox) == 4)
                if not bbox_len_ok:
                    issues.append(f"orig_bbox len={len(orig_bbox) if orig_bbox is not None else None}")

                # 항목 21: crop_coords 96×96
                coords_96_ok = False
                if coords_len_ok:
                    y0, x0, y1, x1 = (
                        int(crop_coords[0]), int(crop_coords[1]),
                        int(crop_coords[2]), int(crop_coords[3]),
                    )
                    coords_96_ok = ((y1 - y0) == 96 and (x1 - x0) == 96)
                    if not coords_96_ok:
                        issues.append(f"crop_coords size: h={y1-y0},w={x1-x0}")

                # 항목 22: sampling_rule
                rule_ok = (rule_raw == EXPECTED_SAMPLING_RULE)
                if not rule_ok:
                    issues.append(f"sampling_rule={rule_raw!r}")

                # 항목 23: patient_id 내부값 vs 폴더명
                pid_match_ok = (pid_npz == pid)
                if not pid_match_ok:
                    issues.append(f"patient_id mismatch: folder={pid}, npz={pid_npz}")

                d.close()

            except Exception as exc:
                issues.append(f"load_error={exc}")
                keys_ok = siv_ok = shape_ok = dtype_ok = nan_ok = False
                label_val_ok = sl_val_ok = match_ok = zsrc_ok = localz_ok = False
                coords_len_ok = bbox_len_ok = coords_96_ok = rule_ok = pid_match_ok = False
                label_int = lz = -1
                sl_str = z_source_raw = rule_raw = ""

            per_file_records.append({
                "patient_id":      pid,
                "file_name":       fpath.name,
                "keys_ok":         keys_ok,
                "siv_ok":          siv_ok,
                "shape_ok":        shape_ok,
                "dtype_ok":        dtype_ok,
                "nan_ok":          nan_ok,
                "label_val_ok":    label_val_ok,
                "sl_val_ok":       sl_val_ok,
                "label_match_ok":  match_ok,
                "zsrc_ok":         zsrc_ok,
                "localz_ok":       localz_ok,
                "coords_len_ok":   coords_len_ok,
                "bbox_len_ok":     bbox_len_ok,
                "coords_96_ok":    coords_96_ok,
                "rule_ok":         rule_ok,
                "pid_match_ok":    pid_match_ok,
                "label_int":       label_int,
                "sampling_label":  sl_str,
                "local_z":         lz,
                "z_source":        z_source_raw,
                "issues":          "|".join(issues),
            })

            if issues:
                per_patient_issues.setdefault(pid, [])
                per_patient_issues[pid].append(f"{fpath.name}: {' | '.join(issues)}")

            checked += 1
            if checked % 10_000 == 0:
                log(f"  진행: {checked:,}/{total:,}")

    log(f"  전수 검증 완료: {checked:,}개")
    return per_file_records, per_patient_issues


# ─────────────────────────────────────────────
# 항목 24: 파일명 연속성 검증
# ─────────────────────────────────────────────
def check_filename_sequence(patient_npz_map: Dict[str, List[Path]]) -> Tuple[bool, Dict[str, str]]:
    """
    각 환자 폴더에서 파일명이 000000.npz, 000001.npz, ... 으로 연속되는지 확인.
    반환: (전체 통과 여부, {patient_id: error_msg})
    """
    bad_patients: Dict[str, str] = {}
    for pid, files in patient_npz_map.items():
        stems = []
        for f in files:
            stem = f.stem
            if not stem.isdigit():
                bad_patients[pid] = f"비숫자 파일명: {f.name}"
                break
            stems.append(int(stem))
        else:
            stems_sorted = sorted(stems)
            expected = list(range(len(stems_sorted)))
            if stems_sorted != expected:
                bad_patients[pid] = f"불연속: 기대 0~{len(stems_sorted)-1}, 실제={stems_sorted[:5]}..."
    return len(bad_patients) == 0, bad_patients


# ─────────────────────────────────────────────
# 집계 헬퍼
# ─────────────────────────────────────────────
def _agg(records: List[dict], field: str) -> dict:
    fail = [r["patient_id"] + "/" + r["file_name"] for r in records if not r[field]]
    total = len(records)
    return {
        "pass_count": total - len(fail),
        "fail_count": len(fail),
        "fail_files": fail[:50],  # 최대 50개만 기록
        "pass": len(fail) == 0,
    }


# ─────────────────────────────────────────────
# 출력 파일 생성
# ─────────────────────────────────────────────
def write_csv(checks: List[dict]) -> None:
    """check_id, check_name, expected, actual, status 형식으로 저장."""
    OUT_RPT_DIR.mkdir(parents=True, exist_ok=True)
    import csv as csv_mod
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv_mod.DictWriter(f, fieldnames=["check_id", "check_name", "expected", "actual", "status"])
        writer.writeheader()
        writer.writerows(checks)
    log(f"[저장] {OUT_CSV}")


def write_json(run_mode: str, total_npz: int, checks: List[dict], per_patient_issues: Dict[str, List[str]]) -> None:
    OUT_RPT_DIR.mkdir(parents=True, exist_ok=True)
    fail_count = sum(1 for c in checks if c["status"] == "FAIL")
    warn_count = sum(1 for c in checks if c["status"] == "WARN")
    overall = "PASS" if fail_count == 0 else "FAIL"
    out = {
        "run_mode": run_mode,
        "timestamp": datetime.now().isoformat(),
        "total_npz_scanned": total_npz,
        "checks": checks,
        "overall": overall,
        "fail_count": fail_count,
        "warn_count": warn_count,
        "per_patient_issues": per_patient_issues,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(f"[저장] {OUT_JSON}")


def write_md(run_mode: str, total_npz: int, checks: List[dict], per_patient_issues: Dict[str, List[str]]) -> None:
    OUT_RPT_DIR.mkdir(parents=True, exist_ok=True)
    fail_count = sum(1 for c in checks if c["status"] == "FAIL")
    overall = "PASS" if fail_count == 0 else "FAIL"
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# S6-A Full Crop Validation Summary",
        "",
        f"- 날짜: {now_str}",
        f"- 모드: {run_mode}",
        f"- 총 npz 수: {total_npz:,}",
        "",
        f"## Overall: {overall}",
        "",
        "## Check Results",
        "",
        "| check_id | check_name | expected | actual | status |",
        "|----------|------------|----------|--------|--------|",
    ]
    for c in checks:
        lines.append(
            f"| {c['check_id']} | {c['check_name']} | {c['expected']} | {c['actual']} | {c['status']} |"
        )

    if per_patient_issues:
        lines += [
            "",
            "## Per-Patient Issues",
            "",
        ]
        for pid, issue_list in sorted(per_patient_issues.items()):
            lines.append(f"### {pid}")
            for iss in issue_list[:20]:
                lines.append(f"- {iss}")
            if len(issue_list) > 20:
                lines.append(f"- ... 외 {len(issue_list)-20}개")
            lines.append("")

    lines += [""]
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"[저장] {OUT_MD}")


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="S6-A full crop 전수 검증")
    parser.add_argument("--dry-run", action="store_true", help="파일 저장 없이 결과만 출력")
    args = parser.parse_args()

    run_mode = "dry-run" if args.dry_run else "full"

    print("=" * 65)
    print("  validate_s6a_crop_full.py")
    print(f"  모드: {run_mode}")
    print(f"  시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # guard
    guard_check(dry_run=args.dry_run)

    # 항목 25: 시작 시점 mtime 스냅샷
    log("[항목 25] 보호 대상 파일 mtime 스냅샷 기록 중...")
    mtime_before = snapshot_mtime()
    log(f"  스냅샷 대상: {len(mtime_before)}개 파일")

    t_start = time.time()

    # ── npz 파일 목록 수집 ──
    log("npz 파일 목록 수집 중...")
    patient_npz_map = collect_npz()
    total_npz = sum(len(v) for v in patient_npz_map.values())
    actual_patient_count = len(patient_npz_map)
    log(f"환자 폴더: {actual_patient_count}개, 총 npz: {total_npz:,}개")

    # ── 외부 파일 로드 ──
    log("summary CSV 로드 중...")
    summary_df = pd.read_csv(SUMMARY_CSV)
    summary_csv_total = len(summary_df)

    log("summary JSON 로드 중...")
    with open(SUMMARY_JSON, encoding="utf-8") as f:
        summary_json_data = json.load(f)
    summary_json_total = summary_json_data.get("total_crops", -1)

    log("manifest CSV 로드 중...")
    manifest_df = pd.read_csv(MANIFEST_CSV)
    manifest_total = len(manifest_df)

    # summary CSV 환자별 npz 수
    summary_per_patient: Dict[str, int] = {}
    if "patient_id" in summary_df.columns:
        for pid, grp in summary_df.groupby("patient_id"):
            summary_per_patient[str(pid)] = len(grp)

    # manifest 환자별 row 수
    manifest_per_patient: Dict[str, int] = {}
    if "patient_id" in manifest_df.columns:
        for pid, grp in manifest_df.groupby("patient_id"):
            manifest_per_patient[str(pid)] = len(grp)

    # ── 항목 24: 파일명 연속성 (per-file 루프 전에 처리) ──
    log("[항목 24] 파일명 연속성 검증 중...")
    seq_pass, seq_bad = check_filename_sequence(patient_npz_map)

    # ── 항목 9~23 전수 검증 ──
    log("[항목 9~23] 전수 검증 시작 (10,000개 단위 진행률 출력)...")
    per_file_records, per_patient_issues = validate_per_file(patient_npz_map)

    # 집계
    a9  = _agg(per_file_records, "keys_ok")
    a10 = _agg(per_file_records, "shape_ok")
    a11 = _agg(per_file_records, "dtype_ok")
    a12 = _agg(per_file_records, "nan_ok")
    a13 = _agg(per_file_records, "label_val_ok")
    a14 = _agg(per_file_records, "sl_val_ok")
    a15 = _agg(per_file_records, "label_match_ok")
    a16 = _agg(per_file_records, "zsrc_ok")
    a17 = _agg(per_file_records, "localz_ok")
    a18 = _agg(per_file_records, "siv_ok")
    a19 = _agg(per_file_records, "coords_len_ok")
    a20 = _agg(per_file_records, "bbox_len_ok")
    a21 = _agg(per_file_records, "coords_96_ok")
    a22 = _agg(per_file_records, "rule_ok")
    a23 = _agg(per_file_records, "pid_match_ok")

    # 분포
    npz_pos = sum(1 for r in per_file_records if r["sampling_label"] == "positive")
    npz_hn  = sum(1 for r in per_file_records if r["sampling_label"] == "hard_negative")

    # 환자별 npz 수 vs summary/manifest 일치
    patient_count_mismatch_summary: List[str] = []
    patient_count_mismatch_manifest: List[str] = []
    for pid, files in patient_npz_map.items():
        npz_cnt = len(files)
        if pid in summary_per_patient and summary_per_patient[pid] != npz_cnt:
            patient_count_mismatch_summary.append(
                f"{pid}: npz={npz_cnt}, summary={summary_per_patient[pid]}"
            )
        if pid in manifest_per_patient and manifest_per_patient[pid] != npz_cnt:
            patient_count_mismatch_manifest.append(
                f"{pid}: npz={npz_cnt}, manifest={manifest_per_patient[pid]}"
            )

    # LUNG1-140 npz 수
    lung1_140_actual = len(patient_npz_map.get("LUNG1-140", []))

    # 항목 25: mtime 비교
    changed_files = compare_mtime(mtime_before)
    readonly_ok = len(changed_files) == 0

    elapsed = round(time.time() - t_start, 2)

    # ─────────────────────────────────────────────
    # checks 리스트 구성 (CSV/JSON/MD 공통 형식)
    # ─────────────────────────────────────────────
    def st(flag: bool) -> str:
        return "PASS" if flag else "FAIL"

    checks: List[dict] = [
        {
            "check_id": 1,
            "check_name": "npz 총 개수",
            "expected": EXPECTED_TOTAL,
            "actual": total_npz,
            "status": st(total_npz == EXPECTED_TOTAL),
        },
        {
            "check_id": 2,
            "check_name": "환자 폴더 수",
            "expected": EXPECTED_PATIENT_COUNT,
            "actual": actual_patient_count,
            "status": st(actual_patient_count == EXPECTED_PATIENT_COUNT),
        },
        {
            "check_id": 3,
            "check_name": "summary CSV 총 행 수 vs npz 개수",
            "expected": total_npz,
            "actual": summary_csv_total,
            "status": st(summary_csv_total == total_npz),
        },
        {
            "check_id": 4,
            "check_name": "summary JSON total_crops vs npz 개수",
            "expected": total_npz,
            "actual": summary_json_total,
            "status": st(summary_json_total == total_npz),
        },
        {
            "check_id": 5,
            "check_name": "manifest row 수 vs npz 개수",
            "expected": total_npz,
            "actual": manifest_total,
            "status": st(manifest_total == total_npz),
        },
        {
            "check_id": 6,
            "check_name": f"positive={EXPECTED_POS}, hard_negative={EXPECTED_HN}",
            "expected": f"pos={EXPECTED_POS},hn={EXPECTED_HN}",
            "actual": f"pos={npz_pos},hn={npz_hn}",
            "status": st(npz_pos == EXPECTED_POS and npz_hn == EXPECTED_HN),
        },
        {
            "check_id": 7,
            "check_name": "환자별 npz 수 vs summary/manifest 일치",
            "expected": "mismatch=0",
            "actual": f"summary_mismatch={len(patient_count_mismatch_summary)},manifest_mismatch={len(patient_count_mismatch_manifest)}",
            "status": st(len(patient_count_mismatch_summary) == 0 and len(patient_count_mismatch_manifest) == 0),
        },
        {
            "check_id": 8,
            "check_name": "LUNG1-140 npz 수",
            "expected": EXPECTED_LUNG1_140,
            "actual": lung1_140_actual,
            "status": st(lung1_140_actual == EXPECTED_LUNG1_140),
        },
        {
            "check_id": 9,
            "check_name": "필수 key 존재 (24개)",
            "expected": "fail=0",
            "actual": f"fail={a9['fail_count']}",
            "status": st(a9["pass"]),
        },
        {
            "check_id": 10,
            "check_name": "crop shape (3,96,96)",
            "expected": "fail=0",
            "actual": f"fail={a10['fail_count']}",
            "status": st(a10["pass"]),
        },
        {
            "check_id": 11,
            "check_name": "crop dtype float32",
            "expected": "fail=0",
            "actual": f"fail={a11['fail_count']}",
            "status": st(a11["pass"]),
        },
        {
            "check_id": 12,
            "check_name": "crop NaN/Inf 0개",
            "expected": "fail=0",
            "actual": f"fail={a12['fail_count']}",
            "status": st(a12["pass"]),
        },
        {
            "check_id": 13,
            "check_name": "label 값 0 또는 1",
            "expected": "fail=0",
            "actual": f"fail={a13['fail_count']}",
            "status": st(a13["pass"]),
        },
        {
            "check_id": 14,
            "check_name": "sampling_label 값 유효",
            "expected": "fail=0",
            "actual": f"fail={a14['fail_count']}",
            "status": st(a14["pass"]),
        },
        {
            "check_id": 15,
            "check_name": "label ↔ sampling_label 일치",
            "expected": "fail=0",
            "actual": f"fail={a15['fail_count']}",
            "status": st(a15["pass"]),
        },
        {
            "check_id": 16,
            "check_name": "z_source=local_z",
            "expected": "fail=0",
            "actual": f"fail={a16['fail_count']}",
            "status": st(a16["pass"]),
        },
        {
            "check_id": 17,
            "check_name": "local_z >= 0",
            "expected": "fail=0",
            "actual": f"fail={a17['fail_count']}",
            "status": st(a17["pass"]),
        },
        {
            "check_id": 18,
            "check_name": "slice_index_valid key 존재",
            "expected": "fail=0",
            "actual": f"fail={a18['fail_count']}",
            "status": st(a18["pass"]),
        },
        {
            "check_id": 19,
            "check_name": "crop_coords 길이 4",
            "expected": "fail=0",
            "actual": f"fail={a19['fail_count']}",
            "status": st(a19["pass"]),
        },
        {
            "check_id": 20,
            "check_name": "orig_bbox 길이 4",
            "expected": "fail=0",
            "actual": f"fail={a20['fail_count']}",
            "status": st(a20["pass"]),
        },
        {
            "check_id": 21,
            "check_name": "crop_coords 96×96",
            "expected": "fail=0",
            "actual": f"fail={a21['fail_count']}",
            "status": st(a21["pass"]),
        },
        {
            "check_id": 22,
            "check_name": f"sampling_rule={EXPECTED_SAMPLING_RULE}",
            "expected": "fail=0",
            "actual": f"fail={a22['fail_count']}",
            "status": st(a22["pass"]),
        },
        {
            "check_id": 23,
            "check_name": "폴더명 vs 내부 patient_id 일치",
            "expected": "fail=0",
            "actual": f"fail={a23['fail_count']}",
            "status": st(a23["pass"]),
        },
        {
            "check_id": 24,
            "check_name": "파일명 000000.npz 연속 형식",
            "expected": "bad_patients=0",
            "actual": f"bad_patients={len(seq_bad)}",
            "status": st(seq_pass),
        },
        {
            "check_id": 25,
            "check_name": "기존 파일 미수정 (read-only)",
            "expected": "changed=0",
            "actual": f"changed={len(changed_files)}",
            "status": st(readonly_ok),
        },
    ]

    # 콘솔 출력
    fail_count = sum(1 for c in checks if c["status"] == "FAIL")
    overall = "PASS" if fail_count == 0 else "FAIL"

    print()
    print("=" * 65)
    print("  검증 결과 요약")
    print("=" * 65)
    for c in checks:
        mark = c["status"]
        print(f"  [{mark}] #{c['check_id']:02d} {c['check_name']}: expected={c['expected']}, actual={c['actual']}")
    print("-" * 65)
    print(f"  최종 판정: {overall}  (FAIL={fail_count}개)")
    print(f"  소요 시간: {elapsed}초")
    print("=" * 65)

    if changed_files:
        print("\n[경고] 보호 대상 파일 변경 감지:")
        for cf in changed_files:
            print(f"  {cf}")

    if seq_bad:
        print("\n[항목 24] 파일명 불연속 환자:")
        for pid, msg in seq_bad.items():
            print(f"  {pid}: {msg}")

    if args.dry_run:
        print(f"\n[DRY-RUN] 파일 저장 없음. 실제 실행은 --dry-run 없이 진행하세요.")
        return

    log("\n결과 파일 저장 중...")
    write_csv(checks)
    write_json(run_mode, total_npz, checks, per_patient_issues)
    write_md(run_mode, total_npz, checks, per_patient_issues)

    log(f"\n=== 완료: {overall} (소요 {elapsed}초) ===")


if __name__ == "__main__":
    main()
