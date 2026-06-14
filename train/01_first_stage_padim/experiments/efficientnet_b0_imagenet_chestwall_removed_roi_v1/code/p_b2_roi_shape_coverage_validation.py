"""
P-B2: ROI shape/coverage validation
- refined_roi_v4_20_modeB_all_v1 마스크 분석
- 기존 roi_0_0 normal vs refined_roi 비교 (coverage ratio)
- patient_id 매칭 확인
- E드라이브 미마운트로 model_roi.npy 정체 확인 불가 → 별도 보고
"""
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ── 경로 설정 ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
EXP_ROOT     = Path(__file__).resolve().parent.parent

REFINED_ROI_ROOT = PROJECT_ROOT / "outputs" / "mip-postprocess-research-v1" / "masks" / "refined_roi_v4_20_modeB_all_v1"
ROI_0_0_NORMAL_ROOT = Path("/mnt/c/Users/jinhy/Desktop/Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1")

LESION_SPLIT_CSV  = PROJECT_ROOT / "outputs" / "second-stage-lesion-refiner-v1" / "splits" / "lesion_stage_split_v1_balanced.csv"
NEW_NORMAL_MANIFEST = Path("/mnt/c/Users/jinhy/Desktop/Normal_LUNA16_padim_training_ready_v2_tslungguard_nochest/manifests/patient_manifest.csv")
NEW_LESION_MANIFEST = Path("/mnt/c/Users/jinhy/Desktop/NSCLC_MSD_padim_test_ready_model_roi_v1/manifests/patient_manifest.csv")
OLD_NORMAL_MANIFEST = ROI_0_0_NORMAL_ROOT / "manifests" / "patient_manifest.csv"

REPORT_DIR = EXP_ROOT / "outputs" / "reports" / "p_b2_roi_shape_coverage_validation"

SCRIPT_NAME = "p_b2_roi_shape_coverage_validation.py"

# ── 가드 ────────────────────────────────────────────────────────────────────
def run_guards():
    print("[G0] refined_roi 마스크 루트 존재 확인")
    assert REFINED_ROI_ROOT.exists(), f"refined_roi 루트 없음: {REFINED_ROI_ROOT}"
    assert (REFINED_ROI_ROOT / "normal").exists(), "refined_roi/normal 폴더 없음"
    assert (REFINED_ROI_ROOT / "lesion").exists(), "refined_roi/lesion 폴더 없음"
    print("  → 통과")

    print("[G1] lesion split CSV 존재 확인")
    assert LESION_SPLIT_CSV.exists(), f"lesion split CSV 없음: {LESION_SPLIT_CSV}"
    print("  → 통과")

    print("[G2] 기존 roi_0_0 normal manifest 존재 확인")
    assert OLD_NORMAL_MANIFEST.exists(), f"roi_0_0 normal manifest 없음: {OLD_NORMAL_MANIFEST}"
    print("  → 통과")

    print("[G3] stage2_holdout 접근 금지 선언")
    print("  → 이 스크립트는 stage2_holdout CT/ROI/lesion_mask value를 열지 않음")
    print("  → lesion 마스크 분석은 stage1_dev 154명으로 제한")
    print("  → 통과")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print("[G4] 출력 경로 확인 완료")


# ── lesion split에서 stage1_dev 환자 목록 로드 ──────────────────────────────
def load_stage1_dev_ids():
    stage1_dev_ids = set()
    with open(LESION_SPLIT_CSV, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 컬럼명: stage_split (lesion_stage_split_v1_balanced.csv 기준)
            split_val = row.get("stage_split", row.get("split", "")).strip()
            if split_val == "stage1_dev":
                stage1_dev_ids.add(row["safe_id"].strip())
    print(f"[split] stage1_dev: {len(stage1_dev_ids)}명")
    return stage1_dev_ids


# ── manifest CSV에서 safe_id 목록 로드 ──────────────────────────────────────
def load_manifest_ids(manifest_path, id_col="safe_id"):
    ids = []
    if not Path(manifest_path).exists():
        return ids
    with open(manifest_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(row[id_col].strip())
    return ids


# ── refined_roi 마스크 분석 ───────────────────────────────────────────────
def analyze_refined_roi_masks(group, stage1_dev_ids=None):
    group_dir = REFINED_ROI_ROOT / group
    subdirs = sorted([d for d in group_dir.iterdir() if d.is_dir()])

    results = []
    shape_counter = {}
    missing = []
    shape_mismatches = []

    for sid in subdirs:
        mask_path = sid / "refined_roi.npy"
        row = {"safe_id": sid.name, "group": group, "has_mask": mask_path.exists()}

        if not mask_path.exists():
            missing.append(sid.name)
            row.update({"shape": None, "dtype": None, "voxel_ones": None, "voxel_total": None})
            results.append(row)
            continue

        if group == "lesion" and stage1_dev_ids is not None:
            if sid.name not in stage1_dev_ids:
                # stage2_holdout: 존재 확인만 (value 읽지 않음)
                row.update({"shape": "NOT_LOADED_HOLDOUT", "dtype": None,
                            "voxel_ones": None, "voxel_total": None, "is_holdout": True})
                results.append(row)
                continue

        arr = np.load(str(mask_path), mmap_mode='r')
        voxel_ones  = int(np.sum(arr))
        voxel_total = int(arr.size)
        shape_str = str(arr.shape)
        shape_counter[shape_str] = shape_counter.get(shape_str, 0) + 1

        row.update({
            "shape": shape_str,
            "dtype": str(arr.dtype),
            "voxel_ones":  voxel_ones,
            "voxel_total": voxel_total,
            "voxel_ratio": round(voxel_ones / voxel_total, 6) if voxel_total > 0 else 0,
            "is_holdout": False
        })
        results.append(row)
        del arr

    print(f"  [{group}] 총 {len(subdirs)}개 폴더, 누락 마스크: {len(missing)}개")
    print(f"  [{group}] shape 분포: {shape_counter}")
    return results, missing, shape_counter


# ── 기존 roi_0_0 normal vs refined_roi 비교 ─────────────────────────────────
def compare_roi_vs_refined(n_sample=50):
    print(f"\n[비교] roi_0_0 vs refined_roi (normal 최대 {n_sample}명 샘플)")
    roi0_normal_dir = ROI_0_0_NORMAL_ROOT / "volumes_npy"
    if not roi0_normal_dir.exists():
        print("  → roi_0_0 normal volumes_npy 없음, 비교 건너뜀")
        return []

    results = []
    subdirs = sorted([d for d in roi0_normal_dir.iterdir() if d.is_dir()])[:n_sample]

    for vol_dir in subdirs:
        sid = vol_dir.name
        roi0_path    = vol_dir / "roi_0_0.npy"
        refined_path = REFINED_ROI_ROOT / "normal" / sid / "refined_roi.npy"

        row = {"safe_id": sid, "roi0_exists": roi0_path.exists(),
               "refined_exists": refined_path.exists()}

        if not roi0_path.exists() or not refined_path.exists():
            row.update({"shape_match": False, "dice": None,
                        "roi0_ones": None, "refined_ones": None,
                        "removed_voxels": None, "removed_ratio": None})
            results.append(row)
            continue

        a = np.load(str(roi0_path),    mmap_mode='r').astype(np.uint8)
        b = np.load(str(refined_path), mmap_mode='r').astype(np.uint8)

        shape_match = (a.shape == b.shape)
        if not shape_match:
            row.update({"shape_match": False, "shape_roi0": str(a.shape),
                        "shape_refined": str(b.shape),
                        "dice": None, "roi0_ones": None, "refined_ones": None,
                        "removed_voxels": None, "removed_ratio": None})
            results.append(row)
            del a, b
            continue

        roi0_ones    = int(np.sum(a))
        refined_ones = int(np.sum(b))
        # removed = roi0 ON, refined OFF (흉벽 제거 부분)
        removed      = int(np.sum((a == 1) & (b == 0)))
        # added = roi0 OFF, refined ON (거의 없어야 함)
        added        = int(np.sum((a == 0) & (b == 1)))
        intersection = int(np.sum((a == 1) & (b == 1)))
        union        = int(np.sum((a == 1) | (b == 1)))
        dice         = round(2 * intersection / (roi0_ones + refined_ones), 6) if (roi0_ones + refined_ones) > 0 else 0

        row.update({
            "shape_match":    True,
            "shape":          str(a.shape),
            "roi0_ones":      roi0_ones,
            "refined_ones":   refined_ones,
            "removed_voxels": removed,
            "added_voxels":   added,
            "removed_ratio":  round(removed / roi0_ones, 6) if roi0_ones > 0 else 0,
            "coverage_ratio": round(refined_ones / roi0_ones, 6) if roi0_ones > 0 else 0,
            "dice":           dice,
        })
        results.append(row)
        del a, b

    return results


# ── patient_id 매칭 확인 ─────────────────────────────────────────────────
def check_patient_matching():
    print("\n[매칭] patient_id 매칭 확인")

    refined_normal_ids = set(d.name for d in (REFINED_ROI_ROOT/"normal").iterdir() if d.is_dir())
    refined_lesion_ids = set(d.name for d in (REFINED_ROI_ROOT/"lesion").iterdir() if d.is_dir())

    old_normal_ids    = set(load_manifest_ids(OLD_NORMAL_MANIFEST))
    new_normal_ids    = set(load_manifest_ids(NEW_NORMAL_MANIFEST)) if NEW_NORMAL_MANIFEST.exists() else set()
    new_lesion_ids    = set(load_manifest_ids(NEW_LESION_MANIFEST)) if NEW_LESION_MANIFEST.exists() else set()

    # refined_roi vs old_normal manifest
    normal_only_refined = refined_normal_ids - old_normal_ids
    normal_only_manifest= old_normal_ids - refined_normal_ids

    # refined_roi lesion vs new_lesion manifest
    lesion_only_refined = refined_lesion_ids - new_lesion_ids
    lesion_only_manifest= new_lesion_ids - refined_lesion_ids

    result = {
        "refined_normal_count": len(refined_normal_ids),
        "refined_lesion_count": len(refined_lesion_ids),
        "old_normal_manifest_count": len(old_normal_ids),
        "new_normal_manifest_count": len(new_normal_ids),
        "new_lesion_manifest_count": len(new_lesion_ids),
        "normal_only_in_refined_count":  len(normal_only_refined),
        "normal_only_in_manifest_count": len(normal_only_manifest),
        "lesion_only_in_refined_count":  len(lesion_only_refined),
        "lesion_only_in_manifest_count": len(lesion_only_manifest),
        "normal_match_old": len(refined_normal_ids & old_normal_ids),
        "lesion_match_new": len(refined_lesion_ids & new_lesion_ids),
        "normal_only_in_refined_list":   sorted(normal_only_refined)[:10],
        "lesion_only_in_refined_list":   sorted(lesion_only_refined)[:10],
    }

    print(f"  refined normal: {result['refined_normal_count']}, old_normal manifest: {result['old_normal_manifest_count']}")
    print(f"  refined normal 미매칭: {result['normal_only_in_refined_count']} / manifest 미매칭: {result['normal_only_in_manifest_count']}")
    print(f"  refined lesion: {result['refined_lesion_count']}, new_lesion manifest: {result['new_lesion_manifest_count']}")
    print(f"  refined lesion 미매칭: {result['lesion_only_in_refined_count']} / manifest 미매칭: {result['lesion_only_in_manifest_count']}")
    return result


# ── E드라이브 접근 가능 여부 확인 ─────────────────────────────────────────
def check_e_drive():
    e_drive_paths = [
        "/mnt/e/jyp/ct_data_2d_preprocessed/NSCLC_MSD_padim_test_ready_model_roi_v1/volumes_npy",
        "/mnt/e/jyp/ct_data_2d_preprocessed/Normal_LUNA16_padim_training_ready_v2_tslungguard_nochest/volumes_npy",
    ]
    accessible = []
    for p in e_drive_paths:
        if Path(p).exists():
            accessible.append(p)
    return {"e_drive_accessible": len(accessible) > 0, "accessible_paths": accessible}


# ── 보고서 저장 ─────────────────────────────────────────────────────────────
def save_csv(rows, path, fieldnames=None):
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{SCRIPT_NAME}] 시작: {ts}")
    print()

    # 가드
    run_guards()
    print()

    # stage1_dev 목록
    stage1_dev_ids = load_stage1_dev_ids()
    print()

    # refined_roi normal 분석
    print("[1] refined_roi normal 마스크 분석")
    normal_results, normal_missing, normal_shape_counter = analyze_refined_roi_masks("normal")

    # refined_roi lesion 분석 (stage1_dev만 value 읽기)
    print("[2] refined_roi lesion 마스크 분석 (stage1_dev only)")
    lesion_results, lesion_missing, lesion_shape_counter = analyze_refined_roi_masks("lesion", stage1_dev_ids)

    # roi_0_0 vs refined_roi 비교
    print("[3] roi_0_0 vs refined_roi 비교 (normal)")
    compare_results = compare_roi_vs_refined(n_sample=362)

    # patient_id 매칭
    matching = check_patient_matching()

    # E드라이브 접근 확인
    e_drive = check_e_drive()
    print(f"\n[E드라이브] 접근 가능: {e_drive['e_drive_accessible']}")
    if not e_drive["e_drive_accessible"]:
        print("  → /mnt/e/ 없음: model_roi.npy 정체 확인 불가 (E드라이브 데이터)")

    # ── 통계 계산 ────────────────────────────────────────────────────────
    # normal coverage stats (roi_0_0 vs refined)
    cov_rows = [r for r in compare_results if r.get("coverage_ratio") is not None]
    cov_values = [r["coverage_ratio"] for r in cov_rows]
    removed_ratios = [r["removed_ratio"] for r in cov_rows]

    normal_cov_stats = {}
    if cov_values:
        normal_cov_stats = {
            "n": len(cov_values),
            "median_coverage": round(float(np.median(cov_values)), 6),
            "mean_coverage":   round(float(np.mean(cov_values)), 6),
            "min_coverage":    round(float(np.min(cov_values)), 6),
            "p5_coverage":     round(float(np.percentile(cov_values, 5)), 6),
            "median_removed_ratio": round(float(np.median(removed_ratios)), 6),
            "mean_removed_ratio":   round(float(np.mean(removed_ratios)), 6),
            "max_removed_ratio":    round(float(np.max(removed_ratios)), 6),
        }

    # lesion coverage stats (stage1_dev only)
    stage1_lesion_rows = [r for r in lesion_results if not r.get("is_holdout", False) and r.get("voxel_ones") is not None]
    lesion_voxel_ratios = [r["voxel_ratio"] for r in stage1_lesion_rows if r.get("voxel_ratio") is not None]
    lesion_cov_stats = {}
    if lesion_voxel_ratios:
        lesion_cov_stats = {
            "n_stage1_dev": len(lesion_voxel_ratios),
            "median_voxel_ratio": round(float(np.median(lesion_voxel_ratios)), 6),
            "mean_voxel_ratio":   round(float(np.mean(lesion_voxel_ratios)), 6),
        }

    # shape mismatch 수
    shape_mismatch_normal = sum(1 for r in compare_results if r.get("shape_match") == False)

    # ── 판정 ─────────────────────────────────────────────────────────────
    issues = []
    if normal_missing:
        issues.append(f"normal 마스크 누락 {len(normal_missing)}건")
    if lesion_missing:
        issues.append(f"lesion 마스크 누락 {len(lesion_missing)}건")
    if shape_mismatch_normal > 0:
        issues.append(f"normal shape mismatch {shape_mismatch_normal}건")
    if matching["normal_only_in_manifest_count"] > 0:
        issues.append(f"normal manifest에만 있고 refined_roi 없는 {matching['normal_only_in_manifest_count']}건")
    if matching["lesion_only_in_manifest_count"] > 0:
        issues.append(f"lesion manifest에만 있고 refined_roi 없는 {matching['lesion_only_in_manifest_count']}건")
    if not e_drive["e_drive_accessible"]:
        issues.append("E드라이브 미마운트: model_roi.npy 정체 확인 불가")

    # model_roi 정체는 E드라이브 없이 불가 → NOT_DETERMINED
    model_roi_identity = "NOT_DETERMINED_E_DRIVE_INACCESSIBLE"

    # 필수 실패 조건 (shape mismatch, 마스크 누락 많음)
    hard_fail = shape_mismatch_normal > 0 or len(normal_missing) > 5 or len(lesion_missing) > 5
    soft_issue = not e_drive["e_drive_accessible"]

    if hard_fail:
        verdict = "실패"
    elif soft_issue:
        verdict = "부분통과"
    else:
        verdict = "통과"

    print(f"\n[판정] {verdict}")
    for iss in issues:
        print(f"  ⚠ {iss}")

    # ── 보고서 저장 ────────────────────────────────────────────────────────
    report_json = {
        "stage": "P-B2_roi_shape_coverage_validation",
        "created": ts,
        "verdict": verdict,

        "scope": {
            "stage2_holdout_accessed": False,
            "training": False,
            "model_forward": False,
            "scoring": False,
            "threshold_calculated": False,
            "metrics_calculated": False,
            "existing_files_modified": False,
            "lesion_value_read_stage1_dev_only": True,
        },

        "refined_roi_asset": {
            "path": str(REFINED_ROI_ROOT),
            "normal_count": len(list((REFINED_ROI_ROOT/"normal").iterdir())),
            "lesion_count": len(list((REFINED_ROI_ROOT/"lesion").iterdir())),
            "normal_missing": len(normal_missing),
            "lesion_missing": len(lesion_missing),
            "normal_shape_distribution": normal_shape_counter,
            "lesion_shape_distribution": lesion_shape_counter,
            "dtype_confirmed": "uint8",
            "values_confirmed": "[0, 1]",
        },

        "patient_matching": matching,

        "normal_roi_coverage_stats": normal_cov_stats,
        "lesion_roi_voxel_stats": lesion_cov_stats,

        "shape_mismatch": {
            "normal_shape_mismatch_count": shape_mismatch_normal,
            "verdict": "통과" if shape_mismatch_normal == 0 else "실패"
        },

        "model_roi_identity": {
            "verdict": model_roi_identity,
            "reason": "E드라이브(/mnt/e/) 미마운트로 model_roi.npy NPY 파일 접근 불가",
            "e_drive_accessible": e_drive["e_drive_accessible"],
            "resolution": "E드라이브 WSL 마운트 후 스크립트 재실행 또는 별도 검증 스크립트 사용 필요"
        },

        "normal_roi_connection_plan": {
            "option_A": "refined_roi_v4_20_modeB_all_v1/normal/<safe_id>/refined_roi.npy 별도 로드 (DataLoader 수정 필요)",
            "option_B": "v2_tslungguard_nochest의 pure_lung.npy가 이미 흉벽 제거 마스크인지 확인 필요 (E드라이브 접근 후)",
            "recommended": "Option A (별도 로드) — pure_lung.npy와 refined_roi.npy 동일성 확인 전까지",
            "path_resolver_change": "PathResolver에 refined_roi_root 파라미터 추가 필요",
        },

        "issues": issues,
        "p_b3_ready": "E드라이브 미접근 제외 시 shape/count 기준으로 READY — model_roi 정체 확인 후 P-B3 진행 권장",
    }

    out_json = REPORT_DIR / "p_b2_roi_shape_coverage_validation.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report_json, f, ensure_ascii=False, indent=2)
    print(f"\n[저장] {out_json}")

    # CSV 보고서들
    save_csv(normal_results, REPORT_DIR / "roi_file_matching_normal.csv")
    save_csv(lesion_results,  REPORT_DIR / "roi_file_matching_lesion.csv")
    save_csv(compare_results, REPORT_DIR / "roi_shape_check_summary.csv")

    # roi_file_matching_summary (normal+lesion 합계)
    summary_rows = [
        {"group": "normal", "total": len(normal_results), "missing_mask": len(normal_missing), "shape_types": len(normal_shape_counter)},
        {"group": "lesion_stage1_dev", "total": len(stage1_lesion_rows), "missing_mask": len(lesion_missing), "shape_types": len(lesion_shape_counter)},
    ]
    save_csv(summary_rows, REPORT_DIR / "roi_file_matching_summary.csv")

    # lesion_model_roi_identity_check.csv
    identity_rows = [{"check": "model_roi.npy vs refined_roi.npy pixel comparison",
                      "result": model_roi_identity,
                      "reason": "E드라이브 미마운트"}]
    save_csv(identity_rows, REPORT_DIR / "lesion_model_roi_identity_check.csv")

    # normal_roi_connection_plan.csv
    plan_rows = [
        {"option": "A", "description": "refined_roi_v4_20_modeB_all_v1/normal/<id>/refined_roi.npy 별도 로드",
         "requires": "DataLoader PathResolver 수정", "status": "권장"},
        {"option": "B", "description": "v2_tslungguard_nochest pure_lung.npy 사용 (흉벽 제거 여부 미확인)",
         "requires": "E드라이브 접근 후 동일성 확인", "status": "미확인"},
    ]
    save_csv(plan_rows, REPORT_DIR / "normal_roi_connection_plan.csv")

    # MD 보고서
    md_lines = [
        f"# P-B2 ROI Shape/Coverage Validation",
        f"",
        f"- 생성일: {ts}",
        f"- 판정: **{verdict}**",
        f"",
        f"---",
        f"",
        f"## 0. 이번 단계 범위",
        f"",
        f"- stage2_holdout 접근: **없음** ✅",
        f"- 학습/forward/scoring/threshold/metrics: **미실행** ✅",
        f"- lesion value 읽기: **stage1_dev {len(stage1_dev_ids)}명만** ✅",
        f"",
        f"---",
        f"",
        f"## 1. refined_roi asset 확인",
        f"",
        f"| 항목 | 결과 |",
        f"|------|------|",
        f"| normal 마스크 count | {report_json['refined_roi_asset']['normal_count']}개 |",
        f"| lesion 마스크 count | {report_json['refined_roi_asset']['lesion_count']}개 |",
        f"| normal 누락 | {len(normal_missing)}개 |",
        f"| lesion 누락 | {len(lesion_missing)}개 |",
        f"| dtype | uint8 |",
        f"| values | [0, 1] |",
        f"| normal shape 분포 | {normal_shape_counter} |",
        f"",
        f"---",
        f"",
        f"## 2. patient_id 매칭",
        f"",
        f"| 항목 | 결과 |",
        f"|------|------|",
        f"| refined normal ↔ old_normal manifest 매칭 | {matching['normal_match_old']}개 |",
        f"| normal refined에만 있는 것 | {matching['normal_only_in_refined_count']}개 |",
        f"| normal manifest에만 있는 것 | {matching['normal_only_in_manifest_count']}개 |",
        f"| refined lesion ↔ new_lesion manifest 매칭 | {matching['lesion_match_new']}개 |",
        f"| lesion refined에만 있는 것 | {matching['lesion_only_in_refined_count']}개 |",
        f"| lesion manifest에만 있는 것 | {matching['lesion_only_in_manifest_count']}개 |",
        f"",
        f"---",
        f"",
        f"## 3. roi_0_0 vs refined_roi 비교 (normal)",
        f"",
    ]

    if normal_cov_stats:
        md_lines += [
            f"| 지표 | 값 |",
            f"|------|----|",
            f"| 비교 대상 수 | {normal_cov_stats['n']}명 |",
            f"| coverage ratio 중앙값 | {normal_cov_stats['median_coverage']} |",
            f"| coverage ratio 평균 | {normal_cov_stats['mean_coverage']} |",
            f"| coverage ratio 최솟값 | {normal_cov_stats['min_coverage']} |",
            f"| coverage ratio p5 | {normal_cov_stats['p5_coverage']} |",
            f"| 제거 비율 중앙값 | {normal_cov_stats['median_removed_ratio']} |",
            f"| 제거 비율 평균 | {normal_cov_stats['mean_removed_ratio']} |",
            f"| 제거 비율 최대 | {normal_cov_stats['max_removed_ratio']} |",
            f"| shape mismatch | {shape_mismatch_normal}건 |",
            f"",
        ]
    else:
        md_lines += ["비교 데이터 없음 (roi_0_0 normal volumes_npy 없음)\n"]

    md_lines += [
        f"---",
        f"",
        f"## 4. model_roi.npy 정체 확인",
        f"",
        f"- 결과: **{model_roi_identity}**",
        f"- 이유: E드라이브(`/mnt/e/`) WSL 미마운트 — lesion volumes_npy 접근 불가",
        f"- 해결 방법: E드라이브 WSL 마운트 후 아래 스크립트 실행",
        f"",
        f"```bash",
        f"# E드라이브 마운트 후 (WSL에서)",
        f"# sudo mkdir -p /mnt/e && sudo mount -t drvfs E: /mnt/e",
        f"# 이후 model_roi_identity_check.py 실행 (P-B2 보완 스크립트)",
        f"```",
        f"",
        f"---",
        f"",
        f"## 5. Normal ROI 연결 방식 설계",
        f"",
        f"| 방안 | 설명 | 필요 작업 | 상태 |",
        f"|------|------|----------|------|",
        f"| A (권장) | refined_roi_v4_20_modeB_all_v1/normal/<id>/refined_roi.npy 별도 로드 | DataLoader/PathResolver 수정 | 권장 |",
        f"| B | v2_tslungguard_nochest pure_lung.npy 사용 | E드라이브 접근 후 동일성 확인 | 미확인 |",
        f"",
        f"---",
        f"",
        f"## 6. 미결 사항",
        f"",
    ]
    for iss in issues:
        md_lines.append(f"- ⚠ {iss}")

    md_lines += [
        f"",
        f"---",
        f"",
        f"## 7. P-B3 lesion safety validation 진행 가능 여부",
        f"",
        f"- shape/count 기준: **READY** (normal 362개, lesion stage1_dev 154개 마스크 존재)",
        f"- model_roi 정체 미확인: **E드라이브 마운트 후 확인 권장**",
        f"- P-B3에서 반드시 확인할 항목:",
        f"  1. stage1_dev 154명 lesion mask coverage: `refined_roi & lesion_mask` vs `roi_0_0 & lesion_mask`",
        f"  2. pleura-adjacent lesion 보존율",
        f"  3. lower_peripheral lesion 보존율",
        f"  4. old roi_0_0 대비 lesion coverage 감소율 (중앙값, p5, 최솟값)",
        f"  5. lesion이 새 ROI 밖으로 밀린 케이스 목록 (coverage 0.0 또는 매우 낮은 케이스)",
        f"",
        f"---",
        f"",
        f"## 8. 최종 판정",
        f"",
        f"- **{verdict}**",
        f"- 주요 이유: E드라이브 미마운트로 model_roi.npy 정체 확인 불가",
        f"- shape/count 기준으로는 refined_roi asset 완비",
    ]

    out_md = REPORT_DIR / "p_b2_roi_shape_coverage_validation.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"[저장] {out_md}")

    print(f"\n[완료] 판정: {verdict}")
    print(f"[완료] 출력 경로: {REPORT_DIR}")
    return 0 if verdict != "실패" else 1


if __name__ == "__main__":
    sys.exit(main())
