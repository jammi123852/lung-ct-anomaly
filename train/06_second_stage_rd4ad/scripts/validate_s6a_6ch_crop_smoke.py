#!/usr/bin/env python3
"""
validate_s6a_6ch_crop_smoke.py
==============================
crops_s6a_6ch_smoke에 생성된 150개 npz를 전수 검증한다.

실행 모드:
- --dry-run: 전수 검증 수행, summary 저장 없음
- 인자 없음: 전수 검증 + summary CSV/JSON/MD 저장

진행 순서:
1. --dry-run 먼저 실행
2. 결과 확인 후 사용자 승인
3. 실제 저장은 인자 없이 실행

절대 금지:
- npz 수정/재생성 금지
- crops_s6a_full/ 접근 금지
- crops_s6a_6ch_smoke/ 수정/삭제 금지
- full 6ch crop 생성 금지
- 학습/추론 금지
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# 경로 상수
# ─────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]

SMOKE_CROPS_DIR   = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/crops_s6a_6ch_smoke"
SMOKE_SUMMARY_JSON = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/reports/crop_s6a_6ch_smoke_summary.json"
MANIFEST_CSV      = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/candidates/rule_s6a_gs2_selected_candidate_manifest_dryrun.csv"
STAGE_SPLIT_CSV   = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1_balanced.csv"

OUT_RPT_DIR = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/reports"
OUT_CSV  = OUT_RPT_DIR / "crop_s6a_6ch_smoke_validation_summary.csv"
OUT_JSON = OUT_RPT_DIR / "crop_s6a_6ch_smoke_validation_summary.json"
OUT_MD   = OUT_RPT_DIR / "crop_s6a_6ch_smoke_validation_summary.md"

# 접근 금지 경로 (존재 확인용 상수만 — 내용 접근 금지)
LEGACY_CROPS_S6A_FULL = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/crops_s6a_full"
CROPS_S6A_6CH_FULL    = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/crops_s6a_6ch_full"

# ─────────────────────────────────────────────
# 기대값
# ─────────────────────────────────────────────
EXPECTED_TOTAL      = 150
EXPECTED_PATIENTS   = ["LUNG1-001", "LUNG1-004", "LUNG1-008", "MSD_lung_001", "MSD_lung_003"]
EXPECTED_PER_PATIENT = 30
EXPECTED_POSITIVE   = 75
EXPECTED_HN         = 75
EXPECTED_SHAPE      = (6, 96, 96)
EXPECTED_DTYPE      = np.float32

REQUIRED_KEYS = [
    "image", "label", "sampling_label", "sampling_rule", "patient_id",
    "local_z", "slice_index", "slice_index_valid", "z_source", "crop_coords",
    "orig_bbox", "score_original", "score_valid950_weighted", "score_valid950_soft",
    "composite_rank_v2", "position_bin", "z_level", "central_peripheral",
    "lesion_patch_ratio", "roi_inside_ratio", "air_ratio_950", "air_ratio_970",
    "valid_ratio_roi_air950", "valid_ratio_roi_air970",
]
FORBIDDEN_KEYS = ["crop"]


# ─────────────────────────────────────────────
# guard_check
# ─────────────────────────────────────────────
def guard_check(dry_run: bool) -> None:
    errors = []

    # guard 1: smoke crop 폴더 없으면 중단
    if not SMOKE_CROPS_DIR.exists():
        errors.append(f"[GUARD 1] smoke crop 폴더 없음: {SMOKE_CROPS_DIR}")

    # guard 2: smoke summary JSON 없으면 중단
    if not SMOKE_SUMMARY_JSON.exists():
        errors.append(f"[GUARD 2] smoke summary JSON 없음: {SMOKE_SUMMARY_JSON}")

    # guard 3: manifest 없으면 중단
    if not MANIFEST_CSV.exists():
        errors.append(f"[GUARD 3] manifest 없음: {MANIFEST_CSV}")

    # guard 4: stage split 없으면 중단
    if not STAGE_SPLIT_CSV.exists():
        errors.append(f"[GUARD 4] stage split 없음: {STAGE_SPLIT_CSV}")

    # guard 5: 출력 validation summary 파일이 이미 있으면 중단 (dry-run은 제외)
    if not dry_run:
        for p in [OUT_CSV, OUT_JSON, OUT_MD]:
            if p.exists():
                errors.append(f"[GUARD 5] validation summary 이미 존재: {p}")

    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        print("\n[중단] guard 조건 미통과.", file=sys.stderr)
        sys.exit(1)

    print("[GUARD] 모든 guard 조건 통과.")


# ─────────────────────────────────────────────
# 전수 검증
# ─────────────────────────────────────────────
def validate_all() -> tuple:
    records = []
    global_issues = []

    # ── 폴더/파일 수 검증 ─────────────────────────────────
    patient_dirs  = sorted([d for d in SMOKE_CROPS_DIR.iterdir() if d.is_dir()])
    actual_patients = [d.name for d in patient_dirs]
    npz_files     = list(SMOKE_CROPS_DIR.rglob("*.npz"))
    actual_total  = len(npz_files)

    # V01: 총 npz 수
    check_total = (actual_total == EXPECTED_TOTAL)
    if not check_total:
        global_issues.append(f"[V01] 총 npz 수 불일치: 실제={actual_total}, 기대={EXPECTED_TOTAL}")

    # V02: 환자 폴더 수 / ID
    check_n_patients = (len(patient_dirs) == len(EXPECTED_PATIENTS))
    if not check_n_patients:
        global_issues.append(f"[V02] 환자 폴더 수 불일치: 실제={len(patient_dirs)}, 기대={len(EXPECTED_PATIENTS)}")
    missing_patients = [p for p in EXPECTED_PATIENTS if p not in actual_patients]
    extra_patients   = [p for p in actual_patients if p not in EXPECTED_PATIENTS]
    if missing_patients:
        global_issues.append(f"[V02] 누락 환자: {missing_patients}")
    if extra_patients:
        global_issues.append(f"[V02] 예상 외 환자: {extra_patients}")

    # V03: 환자별 30개
    per_patient_counts = {d.name: len(list(d.glob("*.npz"))) for d in patient_dirs}
    check_per_patient = all(c == EXPECTED_PER_PATIENT for c in per_patient_counts.values())
    if not check_per_patient:
        for pid, cnt in per_patient_counts.items():
            if cnt != EXPECTED_PER_PATIENT:
                global_issues.append(f"[V03] {pid}: npz 수={cnt}, 기대={EXPECTED_PER_PATIENT}")

    # V04: smoke summary total_crops와 실제 npz 수 일치
    with open(SMOKE_SUMMARY_JSON, "r", encoding="utf-8") as fj:
        smoke_summary = json.load(fj)
    summary_total = smoke_summary.get("total_crops", -1)
    check_summary_total = (summary_total == actual_total)
    if not check_summary_total:
        global_issues.append(f"[V04] summary total_crops={summary_total} vs 실제={actual_total}")

    # stage split 로드
    split_df = pd.read_csv(STAGE_SPLIT_CSV)
    holdout_set = set(split_df.loc[split_df["stage_split"] == "stage2_holdout", "patient_id"])

    # ── 파일별 전수 검증 ───────────────────────────────────
    total_positive = 0
    total_hn       = 0

    for npz_path in sorted(SMOKE_CROPS_DIR.rglob("*.npz")):
        rec = {
            "npz_path":          str(npz_path.relative_to(REPO_ROOT)),
            "patient_id_folder": npz_path.parent.name,
            "issues":            [],
        }

        try:
            f = np.load(str(npz_path), allow_pickle=True)
        except Exception as e:
            rec["issues"].append(f"[LOAD] npz 로드 실패: {e}")
            rec["n_issues"] = len(rec["issues"])
            rec["pass"] = False
            records.append(rec)
            continue

        keys = list(f.keys())

        # V06: 필수 key 존재 여부
        missing_keys = [k for k in REQUIRED_KEYS if k not in keys]
        if missing_keys:
            rec["issues"].append(f"[V06] 필수 key 누락: {missing_keys}")

        # V07: crop key 없어야 함
        forbidden_found = [k for k in FORBIDDEN_KEYS if k in keys]
        if forbidden_found:
            rec["issues"].append(f"[V07] 금지 key 존재: {forbidden_found}")

        # V08~11: image 검증
        if "image" in keys:
            image = f["image"]

            if image.shape != EXPECTED_SHAPE:
                rec["issues"].append(f"[V08] shape 오류: {image.shape} (기대 {EXPECTED_SHAPE})")

            if image.dtype != EXPECTED_DTYPE:
                rec["issues"].append(f"[V09] dtype 오류: {image.dtype} (기대 float32)")

            img_min = float(image.min())
            img_max = float(image.max())
            rec["image_min"] = img_min
            rec["image_max"] = img_max
            if img_min < -1e-6 or img_max > 1.0 + 1e-6:
                rec["issues"].append(f"[V10] 범위 이탈: min={img_min:.6f}, max={img_max:.6f}")

            n_nan = int(np.isnan(image).sum())
            n_inf = int(np.isinf(image).sum())
            rec["n_nan"] = n_nan
            rec["n_inf"] = n_inf
            if n_nan > 0 or n_inf > 0:
                rec["issues"].append(f"[V11] NaN={n_nan}, Inf={n_inf}")
        else:
            rec["image_min"] = None
            rec["image_max"] = None
            rec["n_nan"]     = None
            rec["n_inf"]     = None

        # V12~14: label / sampling_label 검증
        label_val         = None
        sampling_label_val = None

        if "label" in keys:
            label_val = int(f["label"])
            rec["label"] = label_val
            if label_val not in (0, 1):
                rec["issues"].append(f"[V12] label 값 오류: {label_val} (기대 0/1)")

        if "sampling_label" in keys:
            sampling_label_val = str(f["sampling_label"])
            rec["sampling_label"] = sampling_label_val
            if sampling_label_val not in ("positive", "hard_negative"):
                rec["issues"].append(f"[V13] sampling_label 값 오류: {sampling_label_val}")

        if label_val is not None and sampling_label_val is not None:
            expected_label = 1 if sampling_label_val == "positive" else 0
            if label_val != expected_label:
                rec["issues"].append(
                    f"[V14] label/sampling_label 불일치: label={label_val}, sampling_label={sampling_label_val}"
                )
            if sampling_label_val == "positive":
                total_positive += 1
            elif sampling_label_val == "hard_negative":
                total_hn += 1

        # V15: z_source
        if "z_source" in keys:
            z_source_val = str(f["z_source"])
            rec["z_source"] = z_source_val
            if z_source_val != "local_z":
                rec["issues"].append(f"[V15] z_source 오류: {z_source_val} (기대 local_z)")

        # V16: local_z 음수/NaN
        if "local_z" in keys:
            local_z_raw = f["local_z"]
            local_z_float = float(local_z_raw)
            if np.isnan(local_z_float):
                rec["local_z"] = None
                rec["issues"].append(f"[V16] local_z NaN")
            elif int(local_z_float) < 0:
                rec["local_z"] = int(local_z_float)
                rec["issues"].append(f"[V16] local_z 음수: {int(local_z_float)}")
            else:
                rec["local_z"] = int(local_z_float)

        # V17: slice_index_valid key 존재 여부 (V06에 포함)

        # V18: crop_coords 길이 4 / V20: 96×96 크기
        if "crop_coords" in keys:
            cc = f["crop_coords"]
            rec["crop_coords_len"] = len(cc)
            if len(cc) != 4:
                rec["issues"].append(f"[V18] crop_coords 길이 오류: {len(cc)} (기대 4)")
            else:
                h = int(cc[2]) - int(cc[0])
                w = int(cc[3]) - int(cc[1])
                rec["crop_h"] = h
                rec["crop_w"] = w
                if h != 96 or w != 96:
                    rec["issues"].append(f"[V20] crop 크기 오류: h={h}, w={w} (기대 96×96)")
        else:
            rec["crop_coords_len"] = None
            rec["crop_h"]          = None
            rec["crop_w"]          = None

        # V19: orig_bbox 길이 4
        if "orig_bbox" in keys:
            ob = f["orig_bbox"]
            rec["orig_bbox_len"] = len(ob)
            if len(ob) != 4:
                rec["issues"].append(f"[V19] orig_bbox 길이 오류: {len(ob)} (기대 4)")
        else:
            rec["orig_bbox_len"] = None

        # V21: patient_id 폴더명과 npz 내부 patient_id 일치
        if "patient_id" in keys:
            pid_npz = str(f["patient_id"])
            rec["patient_id_npz"] = pid_npz
            if pid_npz != npz_path.parent.name:
                rec["issues"].append(
                    f"[V21] patient_id 불일치: 폴더={npz_path.parent.name}, npz={pid_npz}"
                )

        # guard 6 (파일 수준): stage2_holdout 즉시 중단
        if "patient_id" in keys:
            pid_val = str(f["patient_id"])
            if pid_val in holdout_set:
                print(f"[ABORT] stage2_holdout 감지: {pid_val}", file=sys.stderr)
                sys.exit(1)

        rec["n_issues"] = len(rec["issues"])
        rec["pass"]     = (rec["n_issues"] == 0)
        records.append(rec)

    # V05: smoke summary positive/hard_negative와 실제 분포 일치
    summary_positive = smoke_summary.get("positive_crops", -1)
    summary_hn       = smoke_summary.get("hard_negative_crops", -1)
    check_pos = (total_positive == summary_positive)
    check_hn  = (total_hn == summary_hn)
    if not check_pos:
        global_issues.append(f"[V05] positive 수 불일치: 실제={total_positive}, summary={summary_positive}")
    if not check_hn:
        global_issues.append(f"[V05] hard_negative 수 불일치: 실제={total_hn}, summary={summary_hn}")

    # V22~23: stage split 기반 검증
    manifest_df = pd.read_csv(MANIFEST_CSV, usecols=["patient_id", "stage_split"])
    for pid in EXPECTED_PATIENTS:
        pid_rows = manifest_df[manifest_df["patient_id"] == pid]
        if pid_rows.empty:
            global_issues.append(f"[V22] {pid}: manifest에 없음")
            continue
        stage = pid_rows["stage_split"].iloc[0]
        if stage != "stage1_dev":
            global_issues.append(f"[V22] {pid}: stage={stage} (기대 stage1_dev)")
        if pid in holdout_set:
            global_issues.append(f"[V23] {pid}: stage2_holdout 감지")

    # V24: crops_s6a_full 비접촉 확인 (존재 여부만)
    legacy_exists = LEGACY_CROPS_S6A_FULL.exists()

    # V25: full crop 폴더 미존재 확인
    full_6ch_exists = CROPS_S6A_6CH_FULL.exists()

    # summary 딕셔너리
    n_pass = sum(1 for r in records if r.get("pass", False))
    n_fail = len(records) - n_pass

    summary = {
        "validated_at":                 datetime.now().isoformat(),
        "smoke_crop_dir":               str(SMOKE_CROPS_DIR),
        "actual_total_npz":             actual_total,
        "expected_total_npz":           EXPECTED_TOTAL,
        "check_total_npz":              check_total,
        "actual_n_patients":            len(patient_dirs),
        "expected_n_patients":          len(EXPECTED_PATIENTS),
        "check_n_patients":             check_n_patients,
        "per_patient_counts":           per_patient_counts,
        "check_per_patient":            check_per_patient,
        "check_summary_total":          check_summary_total,
        "actual_positive":              total_positive,
        "actual_hard_negative":         total_hn,
        "expected_positive":            EXPECTED_POSITIVE,
        "expected_hard_negative":       EXPECTED_HN,
        "check_positive":               check_pos,
        "check_hard_negative":          check_hn,
        "n_pass":                       n_pass,
        "n_fail":                       n_fail,
        "global_issues":                global_issues,
        "legacy_crops_s6a_full_exists": legacy_exists,
        "crops_s6a_6ch_full_exists":    full_6ch_exists,
    }

    return records, summary


# ─────────────────────────────────────────────
# summary 저장
# ─────────────────────────────────────────────
def save_summary(records: list, summary: dict) -> None:
    OUT_RPT_DIR.mkdir(parents=True, exist_ok=True)

    # CSV
    df = pd.DataFrame(records)
    df["issues_str"] = df["issues"].apply(lambda x: " | ".join(x) if x else "")
    df.drop(columns=["issues"], inplace=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[저장] {OUT_CSV}")

    # JSON
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[저장] {OUT_JSON}")

    # MD
    verdict = "전체 통과" if summary["n_fail"] == 0 and not summary["global_issues"] else "미통과"
    lines = [
        "# S6-A 6ch Smoke Crop Validation Summary",
        "",
        f"- **검증 일시**: {summary['validated_at']}",
        f"- **판정**: {verdict}",
        f"- **총 npz**: {summary['actual_total_npz']} / {summary['expected_total_npz']} (기대)",
        f"- **환자 수**: {summary['actual_n_patients']} / {summary['expected_n_patients']}",
        f"- **positive**: {summary['actual_positive']} / {summary['expected_positive']}",
        f"- **hard_negative**: {summary['actual_hard_negative']} / {summary['expected_hard_negative']}",
        f"- **통과**: {summary['n_pass']} / **실패**: {summary['n_fail']}",
        "",
        "## 전역 이슈",
    ]
    if summary["global_issues"]:
        for issue in summary["global_issues"]:
            lines.append(f"- {issue}")
    else:
        lines.append("- 없음")
    lines += [
        "",
        "## 기타",
        f"- crops_s6a_full 비접촉: {'폴더 존재(내용 미접근)' if summary['legacy_crops_s6a_full_exists'] else '폴더 없음'}",
        f"- crops_s6a_6ch_full 미존재: {'이미 존재(경고)' if summary['crops_s6a_6ch_full_exists'] else '미존재(정상)'}",
    ]
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[저장] {OUT_MD}")


# ─────────────────────────────────────────────
# dry-run 결과 출력
# ─────────────────────────────────────────────
def print_dry_run_report(records: list, summary: dict) -> None:
    verdict = "전체 통과" if summary["n_fail"] == 0 and not summary["global_issues"] else "미통과"
    print()
    print("=" * 70)
    print("  [DRY-RUN] S6-A 6ch Smoke Crop Validation 결과")
    print("=" * 70)
    print(f"  판정                   : {verdict}")
    print(f"  총 npz                 : {summary['actual_total_npz']} / {summary['expected_total_npz']} (기대)")
    print(f"  환자 수                : {summary['actual_n_patients']} / {summary['expected_n_patients']}")
    print(f"  환자별 30개            : {'전부 30개' if summary['check_per_patient'] else '불일치 있음'}")
    print(f"  summary total 일치     : {summary['check_summary_total']}")
    print(f"  positive               : {summary['actual_positive']} / {summary['expected_positive']}")
    print(f"  hard_negative          : {summary['actual_hard_negative']} / {summary['expected_hard_negative']}")
    print(f"  통과 파일              : {summary['n_pass']}")
    print(f"  실패 파일              : {summary['n_fail']}")

    print()
    if summary["global_issues"]:
        print("  [전역 이슈]")
        for issue in summary["global_issues"]:
            print(f"    {issue}")
    else:
        print("  [전역 이슈] 없음")

    failed = [r for r in records if not r.get("pass", True)]
    if failed:
        print(f"\n  [실패 파일 {len(failed)}개]")
        for r in failed[:20]:
            print(f"    {r['npz_path']}")
            for iss in r["issues"]:
                print(f"      → {iss}")
        if len(failed) > 20:
            print(f"    ... ({len(failed) - 20}개 더)")
    else:
        print("\n  [실패 파일] 없음")

    print()
    print(f"  crops_s6a_full 비접촉  : {'폴더 존재(내용 미접근)' if summary['legacy_crops_s6a_full_exists'] else '폴더 없음'}")
    print(f"  crops_s6a_6ch_full     : {'이미 존재(경고)' if summary['crops_s6a_6ch_full_exists'] else '미존재(정상)'}")
    print()
    print("  [DRY-RUN] summary 파일 저장 없음.")
    print("  실제 저장 명령 (사용자 승인 후):")
    print("    source ~/ai_env/bin/activate && \\")
    print("    python scripts/validate_s6a_6ch_crop_smoke.py")
    print("=" * 70)


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="S6-A 6ch smoke crop 전수 검증")
    parser.add_argument("--dry-run", action="store_true", help="검증 수행, summary 저장 없음")
    args = parser.parse_args()

    print("=" * 70)
    print("  validate_s6a_6ch_crop_smoke.py")
    print(f"  모드: {'dry-run' if args.dry_run else '실행(summary 저장)'}")
    print(f"  시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    guard_check(dry_run=args.dry_run)

    print("  전수 검증 중...")
    records, summary = validate_all()

    if args.dry_run:
        print_dry_run_report(records, summary)
    else:
        save_summary(records, summary)
        verdict = "전체 통과" if summary["n_fail"] == 0 and not summary["global_issues"] else "미통과"
        print(f"\n[완료] 판정: {verdict}")
        print(f"  총 npz={summary['actual_total_npz']}, 통과={summary['n_pass']}, 실패={summary['n_fail']}")
        print(f"  전역 이슈: {len(summary['global_issues'])}건")


if __name__ == "__main__":
    main()
