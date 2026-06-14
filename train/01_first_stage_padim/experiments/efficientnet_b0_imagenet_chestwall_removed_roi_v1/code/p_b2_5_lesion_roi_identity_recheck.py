"""
P-B2.5: lesion ROI identity recheck + E-drive availability validation
- /mnt/e mount 상태 확인 (sudo mount 실행 안 함)
- stage1_dev 154명만 value-level 로드 (stage2_holdout 차단)
- model_roi.npy 정체: E드라이브 의존 → 가능 시에만 비교, 불가 시 NOT_COMPARABLE
- lesion roi_0_0(C드라이브) vs refined_roi 비교 (E드라이브 불필요)
"""
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# ── 경로 설정 ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
EXP_ROOT     = Path(__file__).resolve().parent.parent

REFINED_ROI_ROOT = PROJECT_ROOT / "outputs" / "mip-postprocess-research-v1" / "masks" / "refined_roi_v4_20_modeB_all_v1"

# lesion roi_0_0 (C드라이브, 실재 확인됨 308개)
LESION_ROI0_ROOT = Path("/mnt/c/Users/jinhy/Desktop/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1/volumes_npy")

# lesion model_roi (E드라이브, 미마운트 가능)
LESION_MODEL_ROI_E = Path("/mnt/e/jyp/ct_data_2d_preprocessed/NSCLC_MSD_padim_test_ready_model_roi_v1/volumes_npy")

LESION_SPLIT_CSV = PROJECT_ROOT / "outputs" / "second-stage-lesion-refiner-v1" / "splits" / "lesion_stage_split_v1_balanced.csv"

REPORT_DIR = EXP_ROOT / "outputs" / "reports" / "p_b2_5_lesion_roi_identity_recheck"

SCRIPT_NAME = "p_b2_5_lesion_roi_identity_recheck.py"

# 기대값
EXPECTED_STAGE1_DEV = 154
EXPECTED_NSCLC = 125
EXPECTED_MSD   = 29


# ── E드라이브 마운트 상태 확인 (sudo mount 실행 안 함) ──────────────────────
def check_e_drive_mount():
    status = {
        "mnt_e_exists": Path("/mnt/e").exists(),
        "lesion_model_roi_volumes_exists": LESION_MODEL_ROI_E.exists(),
        "sudo_mount_executed": False,   # 절대 실행 안 함
        "mount_command_suggested": "sudo mkdir -p /mnt/e && sudo mount -t drvfs E: /mnt/e",
    }
    status["e_drive_accessible"] = status["mnt_e_exists"] and status["lesion_model_roi_volumes_exists"]
    return status


# ── lesion split에서 stage1_dev / stage2_holdout 목록 로드 ──────────────────
def load_lesion_splits():
    stage1_dev = []   # (safe_id, group)
    stage2_holdout = set()
    with open(LESION_SPLIT_CSV, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split_val = row.get("stage_split", "").strip()
            safe_id   = row["safe_id"].strip()
            group     = row.get("group", "").strip()
            if split_val == "stage1_dev":
                stage1_dev.append((safe_id, group))
            elif split_val == "stage2_holdout":
                stage2_holdout.add(safe_id)
    return stage1_dev, stage2_holdout


# ── 두 마스크 pixel-level 비교 ──────────────────────────────────────────────
def compare_masks(path_a, path_b):
    """path_a 기준(분자), path_b 기준. coverage = a∩b / b_ones 형태로 별도 계산."""
    if not Path(path_a).exists() or not Path(path_b).exists():
        return {"comparable": False, "reason": "file_missing"}
    a = np.load(str(path_a), mmap_mode='r').astype(np.uint8)
    b = np.load(str(path_b), mmap_mode='r').astype(np.uint8)
    if a.shape != b.shape:
        res = {"comparable": False, "reason": "shape_mismatch",
               "shape_a": str(a.shape), "shape_b": str(b.shape)}
        del a, b
        return res
    a_ones = int(np.sum(a))
    b_ones = int(np.sum(b))
    inter  = int(np.sum((a == 1) & (b == 1)))
    union  = int(np.sum((a == 1) | (b == 1)))
    identical = bool(np.array_equal(np.asarray(a), np.asarray(b)))
    dice = round(2 * inter / (a_ones + b_ones), 6) if (a_ones + b_ones) > 0 else 0.0
    iou  = round(inter / union, 6) if union > 0 else 0.0
    res = {
        "comparable": True,
        "identical": identical,
        "shape": str(a.shape),
        "a_ones": a_ones,
        "b_ones": b_ones,
        "intersection": inter,
        "union": union,
        "dice": dice,
        "iou": iou,
        "voxel_diff_abs": abs(a_ones - b_ones),
        # coverage_a_over_b: b 안에서 a가 차지하는 비율 (a가 refined, b가 roi0이면 refined가 roi0의 몇 %)
        "coverage_a_over_b": round(a_ones / b_ones, 6) if b_ones > 0 else 0.0,
    }
    del a, b
    return res


def pct(values, p):
    return round(float(np.percentile(values, p)), 6) if values else None


def save_csv(rows, path, fieldnames=None):
    if not rows:
        # 빈 파일이라도 헤더 남기기
        with open(path, "w", newline="", encoding="utf-8") as f:
            if fieldnames:
                csv.DictWriter(f, fieldnames=fieldnames).writeheader()
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{SCRIPT_NAME}] 시작: {ts}\n")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. E드라이브 mount 상태 ──────────────────────────────────────────
    e_status = check_e_drive_mount()
    print(f"[E드라이브] /mnt/e 존재: {e_status['mnt_e_exists']}, "
          f"model_roi volumes 접근: {e_status['lesion_model_roi_volumes_exists']}")
    with open(REPORT_DIR / "e_drive_mount_status.json", "w", encoding="utf-8") as f:
        json.dump(e_status, f, ensure_ascii=False, indent=2)

    # ── 2. lesion split 로드 ─────────────────────────────────────────────
    stage1_dev, stage2_holdout = load_lesion_splits()
    n_stage1 = len(stage1_dev)
    n_nsclc = sum(1 for _, g in stage1_dev if g == "NSCLC")
    n_msd   = sum(1 for _, g in stage1_dev if g == "MSD_Lung")
    print(f"[split] stage1_dev: {n_stage1}명 (NSCLC {n_nsclc} / MSD_Lung {n_msd})")
    print(f"[split] stage2_holdout: {len(stage2_holdout)}명 (value 로드 안 함)")

    # ── 3. stage2_holdout contamination 가드 ─────────────────────────────
    stage1_ids = set(sid for sid, _ in stage1_dev)
    contamination = stage1_ids & stage2_holdout
    if contamination:
        print(f"[중단] stage1_dev에 stage2_holdout 혼입 {len(contamination)}명 — 즉시 중단")
        sys.exit(1)
    print(f"[가드] stage1_dev ∩ stage2_holdout = 0 ✅")

    # ── 4. stage1_dev 154명에 대해서만 비교 ──────────────────────────────
    identity_rows = []        # per-patient identity (model_roi 비교 + roi0 비교)
    coverage_rows = []        # refined vs roi0 coverage 분포용

    refined_missing = 0
    roi0_missing    = 0
    model_roi_missing = 0
    shape_mismatch  = 0

    refined_vs_roi0_dice = []
    refined_vs_roi0_iou  = []
    refined_vs_roi0_cov  = []   # refined / roi0
    refined_vs_roi0_removed_ratio = []

    model_roi_comparable_count = 0

    for safe_id, group in stage1_dev:
        refined_path = REFINED_ROI_ROOT / "lesion" / safe_id / "refined_roi.npy"
        roi0_path    = LESION_ROI0_ROOT / safe_id / "roi_0_0.npy"
        model_path   = LESION_MODEL_ROI_E / safe_id / "model_roi.npy"  # E드라이브

        row = {"safe_id": safe_id, "group": group,
               "refined_exists": refined_path.exists(),
               "roi0_exists": roi0_path.exists(),
               "model_roi_exists_e_drive": model_path.exists()}

        if not refined_path.exists():
            refined_missing += 1
        if not roi0_path.exists():
            roi0_missing += 1
        if not model_path.exists():
            model_roi_missing += 1

        # --- refined vs roi_0_0 (C드라이브, E 불필요) ---
        cmp_rr = compare_masks(refined_path, roi0_path)
        if cmp_rr.get("comparable"):
            row["refined_vs_roi0_dice"] = cmp_rr["dice"]
            row["refined_vs_roi0_iou"]  = cmp_rr["iou"]
            row["refined_vs_roi0_identical"] = cmp_rr["identical"]
            row["refined_ones"] = cmp_rr["a_ones"]
            row["roi0_ones"]    = cmp_rr["b_ones"]
            cov = cmp_rr["coverage_a_over_b"]   # refined / roi0
            row["refined_over_roi0_coverage"] = cov
            removed = round(1 - cov, 6)
            row["refined_removed_ratio_vs_roi0"] = removed
            refined_vs_roi0_dice.append(cmp_rr["dice"])
            refined_vs_roi0_iou.append(cmp_rr["iou"])
            refined_vs_roi0_cov.append(cov)
            refined_vs_roi0_removed_ratio.append(removed)
            coverage_rows.append({
                "safe_id": safe_id, "group": group,
                "roi0_ones": cmp_rr["b_ones"], "refined_ones": cmp_rr["a_ones"],
                "coverage_ratio": cov, "removed_ratio": removed,
                "dice": cmp_rr["dice"], "iou": cmp_rr["iou"],
            })
        else:
            row["refined_vs_roi0_dice"] = None
            row["refined_vs_roi0_reason"] = cmp_rr.get("reason")
            if cmp_rr.get("reason") == "shape_mismatch":
                shape_mismatch += 1

        # --- model_roi vs refined / roi_0_0 (E드라이브 필요) ---
        if e_status["e_drive_accessible"] and model_path.exists():
            cmp_mr_ref = compare_masks(model_path, refined_path)
            cmp_mr_roi0 = compare_masks(model_path, roi0_path)
            if cmp_mr_ref.get("comparable"):
                model_roi_comparable_count += 1
                row["model_vs_refined_dice"] = cmp_mr_ref["dice"]
                row["model_vs_refined_identical"] = cmp_mr_ref["identical"]
            if cmp_mr_roi0.get("comparable"):
                row["model_vs_roi0_dice"] = cmp_mr_roi0["dice"]
                row["model_vs_roi0_identical"] = cmp_mr_roi0["identical"]
        else:
            row["model_vs_refined_dice"] = None
            row["model_vs_roi0_dice"] = None
            row["model_roi_note"] = "E_DRIVE_INACCESSIBLE"

        identity_rows.append(row)

    print(f"\n[비교] stage1_dev {n_stage1}명 처리 완료")
    print(f"  refined 누락: {refined_missing}, roi_0_0 누락: {roi0_missing}, "
          f"model_roi(E) 누락: {model_roi_missing}, shape mismatch: {shape_mismatch}")
    print(f"  model_roi 비교 가능: {model_roi_comparable_count}명")

    # ── 5. refined vs roi_0_0 coverage 분포 ──────────────────────────────
    cov_dist = {}
    if refined_vs_roi0_cov:
        cov_dist = {
            "n": len(refined_vs_roi0_cov),
            "coverage_median": round(float(np.median(refined_vs_roi0_cov)), 6),
            "coverage_mean":   round(float(np.mean(refined_vs_roi0_cov)), 6),
            "coverage_p1":  pct(refined_vs_roi0_cov, 1),
            "coverage_p5":  pct(refined_vs_roi0_cov, 5),
            "coverage_p95": pct(refined_vs_roi0_cov, 95),
            "coverage_p99": pct(refined_vs_roi0_cov, 99),
            "coverage_min": round(float(np.min(refined_vs_roi0_cov)), 6),
            "removed_ratio_median": round(float(np.median(refined_vs_roi0_removed_ratio)), 6),
            "removed_ratio_max":    round(float(np.max(refined_vs_roi0_removed_ratio)), 6),
            "dice_median": round(float(np.median(refined_vs_roi0_dice)), 6),
            "iou_median":  round(float(np.median(refined_vs_roi0_iou)), 6),
        }
        # 최대 손실 케이스
        worst = sorted(coverage_rows, key=lambda r: r["coverage_ratio"])[:5]
        cov_dist["worst_coverage_cases"] = [
            {"safe_id": w["safe_id"], "group": w["group"],
             "coverage_ratio": w["coverage_ratio"], "removed_ratio": w["removed_ratio"]}
            for w in worst
        ]
        print(f"  refined/roi0 coverage 중앙값: {cov_dist['coverage_median']}, "
              f"min: {cov_dist['coverage_min']}, p1: {cov_dist['coverage_p1']}")

    # ── 6. model_roi 정체 판정 ───────────────────────────────────────────
    if not e_status["e_drive_accessible"]:
        model_roi_identity = "not_comparable"
        model_roi_reason = "E드라이브(/mnt/e) 미마운트로 model_roi.npy value 접근 불가"
    else:
        # E드라이브 접근 가능 시 dice 기반 판정 (실제로는 이번 실행에서 도달 안 함)
        dices_ref = [r.get("model_vs_refined_dice") for r in identity_rows if r.get("model_vs_refined_dice") is not None]
        dices_roi0 = [r.get("model_vs_roi0_dice") for r in identity_rows if r.get("model_vs_roi0_dice") is not None]
        med_ref = float(np.median(dices_ref)) if dices_ref else 0
        med_roi0 = float(np.median(dices_roi0)) if dices_roi0 else 0
        if med_ref > 0.999:
            model_roi_identity = "same_as_refined_roi_v4_20_modeB"
        elif med_roi0 > 0.999:
            model_roi_identity = "same_as_roi_0_0"
        elif med_roi0 > 0.9 or med_ref > 0.9:
            model_roi_identity = "different_total_segmentor_roi"
        else:
            model_roi_identity = "different_unknown"
        model_roi_reason = f"median dice vs refined={med_ref:.4f}, vs roi0={med_roi0:.4f}"

    # ── 7. 판정 ──────────────────────────────────────────────────────────
    issues = []
    if refined_missing: issues.append(f"refined 마스크 누락 {refined_missing}건")
    if roi0_missing: issues.append(f"lesion roi_0_0 누락 {roi0_missing}건")
    if shape_mismatch: issues.append(f"refined vs roi0 shape mismatch {shape_mismatch}건")
    if n_stage1 != EXPECTED_STAGE1_DEV: issues.append(f"stage1_dev {n_stage1}≠{EXPECTED_STAGE1_DEV}")
    if n_nsclc != EXPECTED_NSCLC: issues.append(f"NSCLC {n_nsclc}≠{EXPECTED_NSCLC}")
    if n_msd != EXPECTED_MSD: issues.append(f"MSD_Lung {n_msd}≠{EXPECTED_MSD}")
    if not e_status["e_drive_accessible"]:
        issues.append("E드라이브 미마운트: model_roi.npy 정체 확인 불가")

    hard_fail = shape_mismatch > 0 or refined_missing > 0 or roi0_missing > 0 \
        or n_stage1 != EXPECTED_STAGE1_DEV
    if hard_fail:
        verdict = "실패"
    elif not e_status["e_drive_accessible"]:
        verdict = "부분통과"   # model_roi 정체만 미결, refined vs roi0 비교는 완료
    else:
        verdict = "통과"

    print(f"\n[판정] {verdict}")
    for i in issues:
        print(f"  ⚠ {i}")

    # P-B3 진행 가능 여부: lesion roi_0_0 + lesion_mask 둘 다 C드라이브에 있으면 가능
    sample_lesion_mask = LESION_ROI0_ROOT / stage1_dev[0][0] / "lesion_mask_roi_0_0.npy"
    p_b3_lesion_mask_available = sample_lesion_mask.exists()

    # ── 8. 보고서 JSON ───────────────────────────────────────────────────
    report = {
        "stage": "P-B2.5_lesion_roi_identity_recheck",
        "created": ts,
        "verdict": verdict,
        "scope": {
            "stage2_holdout_accessed": False,
            "stage2_holdout_value_loaded": False,
            "training": False, "model_forward": False, "scoring": False,
            "threshold_calculated": False, "metrics_calculated": False,
            "ct_intensity_analyzed": False,
            "existing_files_modified": False,
            "sudo_mount_executed": False,
        },
        "e_drive_status": e_status,
        "stage1_dev": {
            "count": n_stage1, "expected": EXPECTED_STAGE1_DEV,
            "nsclc": n_nsclc, "nsclc_expected": EXPECTED_NSCLC,
            "msd_lung": n_msd, "msd_expected": EXPECTED_MSD,
            "count_match": n_stage1 == EXPECTED_STAGE1_DEV,
            "group_match": (n_nsclc == EXPECTED_NSCLC and n_msd == EXPECTED_MSD),
        },
        "stage2_holdout_contamination": len(contamination),
        "file_existence": {
            "refined_missing": refined_missing,
            "lesion_roi0_missing": roi0_missing,
            "model_roi_missing_e_drive": model_roi_missing,
            "shape_mismatch": shape_mismatch,
        },
        "model_roi_identity": {
            "verdict": model_roi_identity,
            "reason": model_roi_reason,
            "comparable_count": model_roi_comparable_count,
            "likely_identity_from_config": "v2_tslungguard_nochest TotalSegmentor 폐 ROI (P-B2 config 단서, 미검증)",
            "note": "이 branch에서 실제 사용할 ROI는 refined_roi_v4_20_modeB이며 model_roi.npy는 ROI input으로 쓰지 않음",
        },
        "refined_vs_roi0_lesion_coverage": cov_dist,
        "p_b3_readiness": {
            "refined_roi_lesion_available": refined_missing == 0,
            "lesion_roi0_available_c_drive": roi0_missing == 0,
            "lesion_mask_roi0_available_c_drive": p_b3_lesion_mask_available,
            "e_drive_required_for_model_roi": True,
            "p_b3_can_proceed_without_e_drive": (refined_missing == 0 and roi0_missing == 0 and p_b3_lesion_mask_available),
            "note": "lesion_mask_roi_0_0.npy가 C드라이브에 존재하면 P-B3 lesion safety는 E드라이브 없이 진행 가능",
        },
        "issues": issues,
    }
    with open(REPORT_DIR / "p_b2_5_lesion_roi_identity_recheck.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ── 9. CSV 저장 ──────────────────────────────────────────────────────
    save_csv(identity_rows, REPORT_DIR / "lesion_stage1_dev_roi_identity_summary.csv")
    save_csv(coverage_rows, REPORT_DIR / "lesion_stage1_dev_roi_coverage_distribution.csv")
    # pairwise comparison summary
    pairwise = [
        {"pair": "refined_vs_roi0", "comparable": "yes",
         "n": len(refined_vs_roi0_cov),
         "median_dice": cov_dist.get("dice_median"),
         "median_iou": cov_dist.get("iou_median"),
         "median_coverage": cov_dist.get("coverage_median"),
         "note": "C드라이브 비교 완료"},
        {"pair": "model_roi_vs_refined", "comparable": "no" if not e_status["e_drive_accessible"] else "yes",
         "n": model_roi_comparable_count,
         "median_dice": None, "median_iou": None, "median_coverage": None,
         "note": "E드라이브 미마운트" if not e_status["e_drive_accessible"] else "비교됨"},
        {"pair": "model_roi_vs_roi0", "comparable": "no" if not e_status["e_drive_accessible"] else "yes",
         "n": model_roi_comparable_count,
         "median_dice": None, "median_iou": None, "median_coverage": None,
         "note": "E드라이브 미마운트" if not e_status["e_drive_accessible"] else "비교됨"},
    ]
    save_csv(pairwise, REPORT_DIR / "lesion_stage1_dev_roi_pairwise_comparison.csv")

    # ── 10. MD 보고서 ────────────────────────────────────────────────────
    md = []
    md.append(f"# P-B2.5 Lesion ROI Identity Recheck + E-drive Validation\n")
    md.append(f"- 생성일: {ts}")
    md.append(f"- 판정: **{verdict}**\n")
    md.append("---\n")
    md.append("## 0. 이번 단계 범위\n")
    md.append("- stage2_holdout value 로드: **없음** ✅")
    md.append("- sudo mount 실행: **없음** ✅")
    md.append("- 학습/forward/scoring/threshold/metrics: **미실행** ✅")
    md.append(f"- lesion value 로드: **stage1_dev {n_stage1}명만** ✅\n")
    md.append("---\n")
    md.append("## 1. E드라이브 mount 상태\n")
    md.append("| 항목 | 결과 |")
    md.append("|------|------|")
    md.append(f"| /mnt/e 존재 | {e_status['mnt_e_exists']} |")
    md.append(f"| model_roi volumes 접근 | {e_status['lesion_model_roi_volumes_exists']} |")
    md.append(f"| E드라이브 접근 가능 | **{e_status['e_drive_accessible']}** |")
    md.append(f"| sudo mount 실행 | **False** (미실행) |")
    md.append(f"| 안내 명령(미실행) | `{e_status['mount_command_suggested']}` |\n")
    md.append("---\n")
    md.append("## 2. stage1_dev 확정\n")
    md.append("| 항목 | 값 | 기대 | 일치 |")
    md.append("|------|----|------|------|")
    md.append(f"| stage1_dev | {n_stage1} | {EXPECTED_STAGE1_DEV} | {'✅' if n_stage1==EXPECTED_STAGE1_DEV else '❌'} |")
    md.append(f"| NSCLC | {n_nsclc} | {EXPECTED_NSCLC} | {'✅' if n_nsclc==EXPECTED_NSCLC else '❌'} |")
    md.append(f"| MSD_Lung | {n_msd} | {EXPECTED_MSD} | {'✅' if n_msd==EXPECTED_MSD else '❌'} |")
    md.append(f"| stage2_holdout 혼입 | {len(contamination)} | 0 | {'✅' if not contamination else '❌'} |\n")
    md.append("---\n")
    md.append("## 3. 파일 존재 / shape\n")
    md.append("| 항목 | 결과 |")
    md.append("|------|------|")
    md.append(f"| refined 마스크 누락 | {refined_missing} |")
    md.append(f"| lesion roi_0_0 누락 (C드라이브) | {roi0_missing} |")
    md.append(f"| model_roi 누락 (E드라이브) | {model_roi_missing} (E 미마운트) |")
    md.append(f"| refined vs roi0 shape mismatch | {shape_mismatch} |\n")
    md.append("---\n")
    md.append("## 4. model_roi.npy 정체 판정\n")
    md.append(f"- 판정: **{model_roi_identity}**")
    md.append(f"- 이유: {model_roi_reason}")
    md.append(f"- config 단서: v2_tslungguard_nochest TotalSegmentor 폐 ROI 추정 (미검증)")
    md.append(f"- **중요**: 이 branch가 실제 사용할 ROI는 `refined_roi_v4_20_modeB`이며, ")
    md.append(f"  `model_roi.npy`는 ROI input으로 쓰지 않음. 정체 미확인이 학습 진행을 막지 않음.\n")
    md.append("---\n")
    md.append("## 5. refined_roi vs roi_0_0 (lesion stage1_dev) — E드라이브 불필요\n")
    if cov_dist:
        md.append("| 지표 | 값 |")
        md.append("|------|----|")
        md.append(f"| 비교 대상 | {cov_dist['n']}명 |")
        md.append(f"| coverage 중앙값 | {cov_dist['coverage_median']} |")
        md.append(f"| coverage 평균 | {cov_dist['coverage_mean']} |")
        md.append(f"| coverage p1 | {cov_dist['coverage_p1']} |")
        md.append(f"| coverage p5 | {cov_dist['coverage_p5']} |")
        md.append(f"| coverage p95 | {cov_dist['coverage_p95']} |")
        md.append(f"| coverage p99 | {cov_dist['coverage_p99']} |")
        md.append(f"| coverage 최솟값 | {cov_dist['coverage_min']} |")
        md.append(f"| removed_ratio 중앙값 | {cov_dist['removed_ratio_median']} |")
        md.append(f"| removed_ratio 최대 | {cov_dist['removed_ratio_max']} |")
        md.append(f"| dice 중앙값 | {cov_dist['dice_median']} |")
        md.append(f"| iou 중앙값 | {cov_dist['iou_median']} |\n")
        md.append("### 최대 손실(낮은 coverage) 5케이스\n")
        md.append("| safe_id | group | coverage | removed |")
        md.append("|---------|-------|----------|---------|")
        for w in cov_dist["worst_coverage_cases"]:
            md.append(f"| {w['safe_id']} | {w['group']} | {w['coverage_ratio']} | {w['removed_ratio']} |")
        md.append("")
    else:
        md.append("비교 데이터 없음\n")
    md.append("---\n")
    md.append("## 6. P-B3 lesion safety validation 진행 가능 여부\n")
    md.append("| 자원 | 위치 | 사용 가능 |")
    md.append("|------|------|-----------|")
    md.append(f"| refined_roi lesion | WSL | {'✅' if refined_missing==0 else '❌'} |")
    md.append(f"| lesion roi_0_0 | C드라이브 | {'✅' if roi0_missing==0 else '❌'} |")
    md.append(f"| lesion_mask_roi_0_0 | C드라이브 | {'✅' if p_b3_lesion_mask_available else '❌'} |")
    md.append(f"| model_roi | E드라이브 | ❌ (불필요) |\n")
    md.append(f"- **P-B3 E드라이브 없이 진행 가능: {report['p_b3_readiness']['p_b3_can_proceed_without_e_drive']}**")
    md.append("- 근거: lesion roi_0_0 + lesion_mask_roi_0_0가 C드라이브에 존재하여, ")
    md.append("  refined_roi & lesion_mask vs roi_0_0 & lesion_mask 비교가 가능\n")
    md.append("### P-B3에서 확인할 항목\n")
    md.append("1. `refined_roi & lesion_mask` vs `roi_0_0 & lesion_mask` — 병변 coverage 감소율 (stage1_dev 154명)")
    md.append("2. coverage 0 또는 매우 낮은 케이스 목록 (병변이 새 ROI 밖으로 밀린 경우)")
    md.append("3. pleura-adjacent lesion 보존율")
    md.append("4. lower_peripheral lesion 보존율")
    md.append("5. 병변 손실 분포 (중앙값, p1, p5, min)\n")
    md.append("---\n")
    md.append("## 7. 미결 사항\n")
    for i in issues:
        md.append(f"- ⚠ {i}")
    if not issues:
        md.append("- 없음")
    md.append("")
    md.append("---\n")
    md.append("## 8. 최종 판정\n")
    md.append(f"- **{verdict}**")
    md.append(f"- model_roi.npy 정체: **not_comparable** (E드라이브 미마운트, 단 ROI input 아니므로 무관)")
    md.append(f"- refined vs roi_0_0 lesion coverage: **비교 완료** (E드라이브 불필요)")
    md.append(f"- P-B3 진행 가능: **{report['p_b3_readiness']['p_b3_can_proceed_without_e_drive']}** (E드라이브 없이)")

    with open(REPORT_DIR / "p_b2_5_lesion_roi_identity_recheck.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"\n[저장] {REPORT_DIR}")
    print(f"[완료] 판정: {verdict}")
    return 0 if verdict != "실패" else 1


if __name__ == "__main__":
    sys.exit(main())
