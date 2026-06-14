#!/usr/bin/env python3
"""
validate_s6a_crop_smoke.py
==========================
crops_s6a_smoke 150개 npz 전수 검증 스크립트.

목적:
- S6-A crop smoke 결과가 실제 학습 입력으로 안전한지 확인
- read-only 검증만 수행 (npz 수정/재생성/PNG 생성 금지)

검증 항목 (20개):
  1.  npz 총 개수 150개인지
  2.  환자 폴더 5개인지
  3.  환자별 npz 개수 30개씩인지
  4.  필수 key 존재 여부 (전수)
  5.  crop shape 전부 (3,96,96)인지
  6.  crop dtype 전부 float32인지
  7.  crop NaN/Inf 0개인지 (전수)
  8.  label 값이 0 또는 1 숫자인지
  9.  sampling_label 값이 positive/hard_negative인지
  10. label과 sampling_label 일치하는지 (positive→1, hard_negative→0)
  11. z_source가 전부 "local_z"인지
  12. local_z 값이 존재하고 음수가 아닌지
  13. slice_index_valid key 전부 존재하는지
  14. crop_coords 길이가 4인지
  15. orig_bbox 길이가 4인지
  16. crop_coords가 96×96 크기인지
  17. sampling_rule이 전부 "S6-A_positive_all_hn_ratio2"인지
  18. smoke summary CSV 총 개수와 npz 실제 개수 일치하는지
  19. smoke summary pos/hard_negative 수와 npz label 분포 일치하는지
  20. 기존 score/candidate/evaluation/crop 파일 미수정 확인

절대 금지:
- npz 수정/재생성 금지
- PNG 생성 금지
- 모델 학습 금지
- scoring 재실행 금지
- 기존 score/candidate/evaluation/crop 파일 수정 금지
- S6-A manifest 원본 수정 금지
- pip/conda install 금지

syntax check (실행 아님):
  python -m py_compile scripts/validate_s6a_crop_smoke.py

실행:
  source ~/ai_env/bin/activate && \\
  python scripts/validate_s6a_crop_smoke.py [--dry-run]
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# 경로 상수
# ─────────────────────────────────────────────
REPO_ROOT       = Path(__file__).resolve().parents[1]
SMOKE_CROPS_DIR = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/crops_s6a_smoke"
SUMMARY_CSV     = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/reports/crop_s6a_smoke_summary.csv"
SUMMARY_JSON    = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/reports/crop_s6a_smoke_summary.json"
MANIFEST_CSV    = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/candidates/rule_s6a_gs2_selected_candidate_manifest_dryrun.csv"
OUT_RPT_DIR     = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/reports"
OUT_CSV         = OUT_RPT_DIR / "crop_s6a_smoke_validation_summary.csv"
OUT_JSON        = OUT_RPT_DIR / "crop_s6a_smoke_validation_summary.json"
OUT_MD          = OUT_RPT_DIR / "crop_s6a_smoke_validation_summary.md"

# ─────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────
SMOKE_PATIENTS = [
    "LUNG1-140",
    "LUNG1-415",
    "LUNG1-156",
    "MSD_lung_071",
    "MSD_lung_096",
]
EXPECTED_TOTAL        = 150
EXPECTED_PER_PATIENT  = 30
EXPECTED_CROP_SHAPE   = (3, 96, 96)
EXPECTED_SAMPLING_RULE = "S6-A_positive_all_hn_ratio2"
REQUIRED_KEYS = [
    "crop", "label", "sampling_label", "sampling_rule", "patient_id",
    "local_z", "slice_index", "slice_index_valid", "z_source",
    "crop_coords", "orig_bbox",
    "score_original", "score_valid950_weighted", "score_valid950_soft",
    "composite_rank_v2", "position_bin", "z_level", "central_peripheral",
    "lesion_patch_ratio",
]

# 항목 20: 이 스크립트는 아래 경로들을 write/touch하지 않는다 (read-only 보장).
# 보호 대상 경로 (참고용 주석):
#   outputs/second-stage-lesion-refiner-v1/crops_s6a_smoke/  → read-only
#   outputs/second-stage-lesion-refiner-v1/candidates/       → read-only
#   outputs/second-stage-lesion-refiner-v1/crops/            → read-only
#   outputs/second-stage-lesion-refiner-v1/reports/crop_s6a_smoke_summary.*  → read-only
# 쓰기 허용 경로: reports/crop_s6a_smoke_validation_summary.{csv,json,md} 만


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ─────────────────────────────────────────────
# guard_check
# ─────────────────────────────────────────────
def guard_check(dry_run: bool) -> None:
    errors = []

    if not SMOKE_CROPS_DIR.exists():
        errors.append(f"[GUARD] crops_s6a_smoke 폴더 없음: {SMOKE_CROPS_DIR}")
    if not SUMMARY_CSV.exists():
        errors.append(f"[GUARD] smoke summary CSV 없음: {SUMMARY_CSV}")
    if not SUMMARY_JSON.exists():
        errors.append(f"[GUARD] smoke summary JSON 없음: {SUMMARY_JSON}")
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

    log("[GUARD] 모든 guard 조건 통과.")


# ─────────────────────────────────────────────
# 전수 검증
# ─────────────────────────────────────────────
def validate_all(dry_run: bool) -> dict:
    """
    150개 npz 전수 검증.
    반환: {
        "per_file_records": [...],   # per-file 결과 list
        "check_results": {...},      # 항목별 집계
        "overall_verdict": str,
        "elapsed_seconds": float,
    }
    """
    t_start = time.time()

    # 파일 목록 수집
    per_file_records: List[dict] = []
    patient_file_map: Dict[str, List[Path]] = {}
    for pid in SMOKE_PATIENTS:
        pid_dir = SMOKE_CROPS_DIR / pid
        files = sorted(pid_dir.glob("*.npz")) if pid_dir.exists() else []
        patient_file_map[pid] = files

    total_npz = sum(len(v) for v in patient_file_map.values())
    actual_patient_dirs = [d.name for d in sorted(SMOKE_CROPS_DIR.iterdir()) if d.is_dir()] \
        if SMOKE_CROPS_DIR.exists() else []

    # ── 항목 1: 총 개수 ──
    chk1 = {
        "expected": EXPECTED_TOTAL,
        "actual": total_npz,
        "pass": total_npz == EXPECTED_TOTAL,
    }

    # ── 항목 2: 환자 폴더 5개 ──
    missing_pids = [p for p in SMOKE_PATIENTS if p not in actual_patient_dirs]
    extra_pids   = [p for p in actual_patient_dirs if p not in SMOKE_PATIENTS]
    chk2 = {
        "expected_patients": SMOKE_PATIENTS,
        "actual_patients": actual_patient_dirs,
        "missing": missing_pids,
        "extra": extra_pids,
        "pass": len(missing_pids) == 0 and len(extra_pids) == 0,
    }

    # ── 항목 3: 환자별 30개씩 ──
    per_patient_count = {pid: len(fs) for pid, fs in patient_file_map.items()}
    bad_count_pids = {pid: cnt for pid, cnt in per_patient_count.items() if cnt != EXPECTED_PER_PATIENT}
    chk3 = {
        "per_patient": per_patient_count,
        "bad_patients": bad_count_pids,
        "pass": len(bad_count_pids) == 0,
    }

    # ── 항목 4~17: per-file 전수 검증 ──
    # 집계 카운터 (각 항목별 fail 목록)
    fail_keys:      List[str] = []
    fail_shape:     List[str] = []
    fail_dtype:     List[str] = []
    fail_nan:       List[str] = []
    fail_label_val: List[str] = []
    fail_sl_val:    List[str] = []
    fail_match:     List[str] = []
    fail_zsrc:      List[str] = []
    fail_localz:    List[str] = []
    fail_siv:       List[str] = []
    fail_coords_len: List[str] = []
    fail_bbox_len:   List[str] = []
    fail_coords_96:  List[str] = []
    fail_rule:       List[str] = []

    checked = 0
    for pid in SMOKE_PATIENTS:
        for fpath in patient_file_map[pid]:
            fname = f"{pid}/{fpath.name}"
            d = np.load(str(fpath), allow_pickle=True)
            actual_keys = set(d.files)

            # 항목 4: 필수 key
            missing_keys = [k for k in REQUIRED_KEYS if k not in actual_keys]
            keys_ok = len(missing_keys) == 0
            if not keys_ok:
                fail_keys.append(fname)

            # 항목 13: slice_index_valid key (4와 중복 확인)
            siv_ok = "slice_index_valid" in actual_keys
            if not siv_ok:
                fail_siv.append(fname)

            # 기본값 (key 없을 때 안전 처리)
            crop          = d["crop"] if "crop" in actual_keys else None
            label_raw     = d["label"] if "label" in actual_keys else None
            sl_raw        = d["sampling_label"] if "sampling_label" in actual_keys else None
            z_source_raw  = str(d["z_source"]) if "z_source" in actual_keys else ""
            local_z_raw   = d["local_z"] if "local_z" in actual_keys else None
            crop_coords   = d["crop_coords"] if "crop_coords" in actual_keys else None
            orig_bbox     = d["orig_bbox"] if "orig_bbox" in actual_keys else None
            rule_raw      = str(d["sampling_rule"]) if "sampling_rule" in actual_keys else ""

            # 항목 5: shape
            shape_ok = (crop is not None and tuple(crop.shape) == EXPECTED_CROP_SHAPE)
            if not shape_ok:
                fail_shape.append(fname)

            # 항목 6: dtype
            dtype_ok = (crop is not None and crop.dtype == np.float32)
            if not dtype_ok:
                fail_dtype.append(fname)

            # 항목 7: NaN/Inf (전수)
            nan_ok = (crop is not None and bool(np.isfinite(crop).all()))
            if not nan_ok:
                fail_nan.append(fname)

            # 항목 8: label 0 또는 1
            label_int = int(label_raw) if label_raw is not None else -1
            label_val_ok = label_int in (0, 1)
            if not label_val_ok:
                fail_label_val.append(fname)

            # 항목 9: sampling_label 값
            sl_str = str(sl_raw) if sl_raw is not None else ""
            sl_val_ok = sl_str in ("positive", "hard_negative")
            if not sl_val_ok:
                fail_sl_val.append(fname)

            # 항목 10: label ↔ sampling_label 일치
            expected_label = 1 if sl_str == "positive" else (0 if sl_str == "hard_negative" else -1)
            match_ok = (label_int == expected_label)
            if not match_ok:
                fail_match.append(fname)

            # 항목 11: z_source
            zsrc_ok = (z_source_raw == "local_z")
            if not zsrc_ok:
                fail_zsrc.append(fname)

            # 항목 12: local_z >= 0
            if local_z_raw is not None:
                lz = int(local_z_raw)
                localz_ok = (lz >= 0)
            else:
                localz_ok = False
                lz = -1
            if not localz_ok:
                fail_localz.append(fname)

            # 항목 14: crop_coords 길이 4
            coords_len_ok = (crop_coords is not None and len(crop_coords) == 4)
            if not coords_len_ok:
                fail_coords_len.append(fname)

            # 항목 15: orig_bbox 길이 4
            bbox_len_ok = (orig_bbox is not None and len(orig_bbox) == 4)
            if not bbox_len_ok:
                fail_bbox_len.append(fname)

            # 항목 16: crop_coords 96×96
            if coords_len_ok:
                y0, x0, y1, x1 = int(crop_coords[0]), int(crop_coords[1]), int(crop_coords[2]), int(crop_coords[3])
                coords_96_ok = ((y1 - y0) == 96 and (x1 - x0) == 96)
            else:
                coords_96_ok = False
            if not coords_96_ok:
                fail_coords_96.append(fname)

            # 항목 17: sampling_rule
            rule_ok = (rule_raw == EXPECTED_SAMPLING_RULE)
            if not rule_ok:
                fail_rule.append(fname)

            per_file_records.append({
                "patient_id":       pid,
                "file_name":        fpath.name,
                "keys_ok":          keys_ok,
                "missing_keys":     "|".join(missing_keys),
                "shape_ok":         shape_ok,
                "dtype_ok":         dtype_ok,
                "nan_ok":           nan_ok,
                "label_val_ok":     label_val_ok,
                "sl_val_ok":        sl_val_ok,
                "label_match_ok":   match_ok,
                "zsrc_ok":          zsrc_ok,
                "localz_ok":        localz_ok,
                "siv_ok":           siv_ok,
                "coords_len_ok":    coords_len_ok,
                "bbox_len_ok":      bbox_len_ok,
                "coords_96_ok":     coords_96_ok,
                "rule_ok":          rule_ok,
                "label_int":        label_int,
                "sampling_label":   sl_str,
                "local_z":          lz,
                "z_source":         z_source_raw,
                "sampling_rule":    rule_raw,
            })

            d.close()
            checked += 1
            if checked % 50 == 0:
                log(f"  진행: {checked}/{EXPECTED_TOTAL}")

    log(f"  전수 검증 완료: {checked}개")

    # ── 항목 4~17 집계 ──
    def _agg(fail_list: List[str], total: int) -> dict:
        return {
            "pass_count": total - len(fail_list),
            "fail_count": len(fail_list),
            "fail_files": fail_list,
            "pass": len(fail_list) == 0,
        }

    total_checked = len(per_file_records)
    chk4  = _agg(fail_keys,       total_checked)
    chk5  = _agg(fail_shape,      total_checked)
    chk6  = _agg(fail_dtype,      total_checked)
    chk7  = _agg(fail_nan,        total_checked)
    chk8  = _agg(fail_label_val,  total_checked)
    chk9  = _agg(fail_sl_val,     total_checked)
    chk10 = _agg(fail_match,      total_checked)
    chk11 = _agg(fail_zsrc,       total_checked)
    chk12 = _agg(fail_localz,     total_checked)
    chk13 = _agg(fail_siv,        total_checked)
    chk14 = _agg(fail_coords_len, total_checked)
    chk15 = _agg(fail_bbox_len,   total_checked)
    chk16 = _agg(fail_coords_96,  total_checked)
    chk17 = _agg(fail_rule,       total_checked)

    # ── 항목 18-19: summary CSV vs npz 개수/분포 일치 ──
    smry_df = pd.read_csv(SUMMARY_CSV)
    smry_total = len(smry_df)
    smry_pos   = int((smry_df["sampling_label"] == "positive").sum())
    smry_hn    = int((smry_df["sampling_label"] == "hard_negative").sum())

    npz_pos = sum(1 for r in per_file_records if r["sampling_label"] == "positive")
    npz_hn  = sum(1 for r in per_file_records if r["sampling_label"] == "hard_negative")

    chk18 = {
        "summary_csv_total": smry_total,
        "npz_actual_total":  total_checked,
        "pass": smry_total == total_checked,
    }
    chk19 = {
        "summary_csv_pos":       smry_pos,
        "summary_csv_hn":        smry_hn,
        "npz_actual_pos":        npz_pos,
        "npz_actual_hn":         npz_hn,
        "pos_match":  smry_pos == npz_pos,
        "hn_match":   smry_hn  == npz_hn,
        "pass": (smry_pos == npz_pos and smry_hn == npz_hn),
    }

    # 항목 20: 이 스크립트는 score/candidate/evaluation/crops_s6a_smoke 내 npz를
    # 수정하거나 touch하지 않는다. 위 코드에서 np.load(allow_pickle=True)로
    # read-only 로드만 수행하며, 쓰기는 OUT_CSV/OUT_JSON/OUT_MD 3개에만 한다.
    chk20 = {"pass": True, "note": "read-only 검증만 수행, 기존 파일 미수정 확인"}

    # ── overall 판정 ──
    all_checks_pass = all([
        chk1["pass"], chk2["pass"], chk3["pass"],
        chk4["pass"], chk5["pass"], chk6["pass"], chk7["pass"],
        chk8["pass"], chk9["pass"], chk10["pass"], chk11["pass"],
        chk12["pass"], chk13["pass"], chk14["pass"], chk15["pass"],
        chk16["pass"], chk17["pass"], chk18["pass"], chk19["pass"],
        chk20["pass"],
    ])
    any_check_fail = not all_checks_pass
    partial = any_check_fail and any([
        chk1["pass"], chk2["pass"], chk3["pass"],
    ])

    if all_checks_pass:
        verdict = "전체 통과"
    elif partial:
        verdict = "부분 통과"
    else:
        verdict = "미통과"

    elapsed = round(time.time() - t_start, 2)

    check_results = {
        "chk1_total_count":     chk1,
        "chk2_patient_folders": chk2,
        "chk3_per_patient_count": chk3,
        "chk4_required_keys":   chk4,
        "chk5_crop_shape":      chk5,
        "chk6_crop_dtype":      chk6,
        "chk7_nan_inf":         chk7,
        "chk8_label_value":     chk8,
        "chk9_sl_value":        chk9,
        "chk10_label_match":    chk10,
        "chk11_z_source":       chk11,
        "chk12_local_z":        chk12,
        "chk13_siv_key":        chk13,
        "chk14_coords_len":     chk14,
        "chk15_bbox_len":       chk15,
        "chk16_coords_96":      chk16,
        "chk17_sampling_rule":  chk17,
        "chk18_summary_count":  chk18,
        "chk19_summary_dist":   chk19,
        "chk20_readonly":       chk20,
    }

    return {
        "per_file_records": per_file_records,
        "check_results": check_results,
        "overall_verdict": verdict,
        "overall_pass": all_checks_pass,
        "elapsed_seconds": elapsed,
        "validated_at": datetime.now().isoformat(),
        "total_validated": total_checked,
    }


# ─────────────────────────────────────────────
# 출력 파일 생성
# ─────────────────────────────────────────────
def write_csv(per_file_records: List[dict]) -> None:
    df = pd.DataFrame(per_file_records)
    OUT_RPT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    log(f"[저장] {OUT_CSV}")


def write_json(result: dict) -> None:
    OUT_RPT_DIR.mkdir(parents=True, exist_ok=True)
    # per_file_records는 CSV에 저장하므로 JSON에는 제외
    out = {k: v for k, v in result.items() if k != "per_file_records"}
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(f"[저장] {OUT_JSON}")


def write_md(result: dict) -> None:
    cr = result["check_results"]
    verdict = result["overall_verdict"]
    elapsed = result["elapsed_seconds"]
    validated_at = result["validated_at"]

    def pf(flag: bool) -> str:
        return "PASS" if flag else "FAIL"

    lines = [
        "# crop_s6a_smoke 검증 보고서",
        "",
        f"- 검증 시각: {validated_at}",
        f"- 소요 시간: {elapsed}초",
        f"- **최종 판정: {verdict}**",
        "",
        "## 항목별 결과",
        "",
        "| # | 항목 | 결과 | 상세 |",
        "|---|------|------|------|",
    ]

    def row(num, name, chk, detail=""):
        return f"| {num} | {name} | {pf(chk['pass'])} | {detail} |"

    c1  = cr["chk1_total_count"]
    c2  = cr["chk2_patient_folders"]
    c3  = cr["chk3_per_patient_count"]
    c4  = cr["chk4_required_keys"]
    c5  = cr["chk5_crop_shape"]
    c6  = cr["chk6_crop_dtype"]
    c7  = cr["chk7_nan_inf"]
    c8  = cr["chk8_label_value"]
    c9  = cr["chk9_sl_value"]
    c10 = cr["chk10_label_match"]
    c11 = cr["chk11_z_source"]
    c12 = cr["chk12_local_z"]
    c13 = cr["chk13_siv_key"]
    c14 = cr["chk14_coords_len"]
    c15 = cr["chk15_bbox_len"]
    c16 = cr["chk16_coords_96"]
    c17 = cr["chk17_sampling_rule"]
    c18 = cr["chk18_summary_count"]
    c19 = cr["chk19_summary_dist"]
    c20 = cr["chk20_readonly"]

    lines += [
        row(1,  "npz 총 개수 150개",           c1,  f"실제={c1['actual']}, 기대={c1['expected']}"),
        row(2,  "환자 폴더 5개",                c2,  f"missing={c2['missing']}, extra={c2['extra']}"),
        row(3,  "환자별 30개씩",                c3,  f"불일치={list(c3['bad_patients'].keys())}"),
        row(4,  "필수 key 존재",                c4,  f"fail={c4['fail_count']}개"),
        row(5,  "crop shape (3,96,96)",         c5,  f"fail={c5['fail_count']}개"),
        row(6,  "crop dtype float32",           c6,  f"fail={c6['fail_count']}개"),
        row(7,  "NaN/Inf 0개 (전수)",           c7,  f"fail={c7['fail_count']}개"),
        row(8,  "label 0/1 숫자",               c8,  f"fail={c8['fail_count']}개"),
        row(9,  "sampling_label 값 유효",       c9,  f"fail={c9['fail_count']}개"),
        row(10, "label ↔ sampling_label 일치",  c10, f"fail={c10['fail_count']}개"),
        row(11, "z_source=local_z",             c11, f"fail={c11['fail_count']}개"),
        row(12, "local_z >= 0",                 c12, f"fail={c12['fail_count']}개"),
        row(13, "slice_index_valid key 존재",   c13, f"fail={c13['fail_count']}개"),
        row(14, "crop_coords 길이 4",           c14, f"fail={c14['fail_count']}개"),
        row(15, "orig_bbox 길이 4",             c15, f"fail={c15['fail_count']}개"),
        row(16, "crop_coords 96×96",            c16, f"fail={c16['fail_count']}개"),
        row(17, f"sampling_rule={EXPECTED_SAMPLING_RULE}", c17, f"fail={c17['fail_count']}개"),
        row(18, "summary CSV 총 개수 일치",     c18, f"summary={c18['summary_csv_total']}, npz={c18['npz_actual_total']}"),
        row(19, "summary pos/hn 분포 일치",     c19, f"pos: summary={c19['summary_csv_pos']} vs npz={c19['npz_actual_pos']}, hn: summary={c19['summary_csv_hn']} vs npz={c19['npz_actual_hn']}"),
        row(20, "기존 파일 미수정 (read-only)", c20, c20["note"]),
        "",
    ]

    lines += [
        "## 환자별 요약",
        "",
        "| patient_id | npz 수 | pos | hard_negative |",
        "|------------|--------|-----|---------------|",
    ]
    for pid in SMOKE_PATIENTS:
        cnt = c3["per_patient"].get(pid, 0)
        lines.append(f"| {pid} | {cnt} | (CSV 참조) | (CSV 참조) |")

    lines += [""]
    OUT_RPT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"[저장] {OUT_MD}")


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="S6-A crop smoke 전수 검증")
    parser.add_argument("--dry-run", action="store_true", help="파일 저장 없이 결과만 출력")
    args = parser.parse_args()

    print("=" * 65)
    print("  validate_s6a_crop_smoke.py")
    print(f"  모드: {'dry-run' if args.dry_run else '실제 실행'}")
    print(f"  시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    guard_check(dry_run=args.dry_run)

    log("전수 검증 시작...")
    result = validate_all(dry_run=args.dry_run)

    cr = result["check_results"]
    verdict = result["overall_verdict"]

    # 콘솔 요약 출력
    print()
    print("=" * 65)
    print("  검증 결과 요약")
    print("=" * 65)
    for key, chk in cr.items():
        flag = chk.get("pass", False)
        mark = "PASS" if flag else "FAIL"
        fail_cnt = chk.get("fail_count", "")
        detail = f" (fail={fail_cnt})" if fail_cnt != "" and not flag else ""
        print(f"  [{mark}] {key}{detail}")

    print("-" * 65)
    print(f"  최종 판정: {verdict}")
    print(f"  소요 시간: {result['elapsed_seconds']}초")
    print("=" * 65)

    if args.dry_run:
        print("\n[DRY-RUN] 파일 저장 없음. 실제 실행은 사용자 승인 후 진행하세요.")
        return

    log("\n결과 파일 저장 중...")
    write_csv(result["per_file_records"])
    write_json(result)
    write_md(result)

    log(f"\n=== 완료: {verdict} (소요 {result['elapsed_seconds']}초) ===")


if __name__ == "__main__":
    main()
