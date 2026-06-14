"""
B1-E1: Oracle-like Vessel Mask Preflight
EfficientNet-B0 v4_20 ROI branch 기준 oracle-like vessel mask 생성 가능성 확인.

이 스크립트는:
- 실제 suppression 미적용
- 원본 score CSV / threshold / model / ROI / CT 파일 수정 없음
- GPU 미사용, 재학습 없음
- stage2_holdout 접근 금지
- output root가 이미 존재하면 즉시 중단

oracle-like vessel mask 정의:
  lesion 환자: refined_roi_v4_20 내 HU>=0 voxel 중 lesion mask와 겹치지 않는 영역
  normal 환자: refined_roi_v4_20 내 HU>=0 voxel
  ★ 진짜 혈관 GT가 아니라 oracle-like upper-bound 후보임.
"""

import os
import sys
import json
import csv
import time
import shutil
import numpy as np
from pathlib import Path

# ─── ALLOW GUARD ───────────────────────────────────────────────────────────────
ALLOW_REAL_PROCESSING = True   # False → dry-run guard (py_compile / bare-run 차단)
# ───────────────────────────────────────────────────────────────────────────────

PROJ_ROOT = Path(__file__).resolve().parents[1]

# ─── 경로 상수 ──────────────────────────────────────────────────────────────────
LESION_VOLUMES_ROOT = Path(
    "/mnt/c/Users/jinhy/Desktop"
    "/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1"
    "/volumes_npy"
)
NORMAL_VOLUMES_ROOT = Path(
    "/mnt/c/Users/jinhy/Desktop"
    "/Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1"
    "/volumes_npy"
)
V4_20_LESION_ROI_ROOT = (
    PROJ_ROOT / "outputs" / "mip-postprocess-research-v1"
    / "masks" / "refined_roi_v4_20_modeB_all_v1" / "lesion"
)
V4_20_NORMAL_ROI_ROOT = (
    PROJ_ROOT / "outputs" / "mip-postprocess-research-v1"
    / "masks" / "refined_roi_v4_20_modeB_all_v1" / "normal"
)
LESION_SCORE_ROOT = (
    PROJ_ROOT / "experiments" / "efficientnet_b0_imagenet_chestwall_removed_roi_v1"
    / "outputs" / "scores" / "lesion_stage1_dev_by_patient"
)
NORMAL_TEST_SCORE_ROOT = (
    PROJ_ROOT / "experiments" / "efficientnet_b0_imagenet_chestwall_removed_roi_v1"
    / "outputs" / "scores" / "normal_test_by_patient"
)
LESION_SPLIT_CSV = (
    PROJ_ROOT / "outputs" / "second-stage-lesion-refiner-v1"
    / "splits" / "lesion_stage_split_v1.csv"
)
NORMAL_SPLIT_CSV = Path(
    "/mnt/c/Users/jinhy/Desktop"
    "/Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1"
    "/manifests/train_val_test_split.csv"
)
THRESHOLD_JSON = (
    PROJ_ROOT / "experiments" / "efficientnet_b0_imagenet_chestwall_removed_roi_v1"
    / "outputs" / "evaluation" / "normal_val_thresholds" / "normal_val_threshold.json"
)
DIST_NPZ = (
    PROJ_ROOT / "experiments" / "efficientnet_b0_imagenet_chestwall_removed_roi_v1"
    / "outputs" / "models" / "distributions" / "position_bin_stats.npz"
)

OUTPUT_ROOT = (
    PROJ_ROOT / "outputs" / "position-aware-padim-v1"
    / "oracle_vessel_suppression_effb0_v4_20_dev_only"
    / "b1e1_oracle_mask_preflight_v1"
)

# ─── 대상 환자 (lesion stage1_dev score 상위 5명 + normal_test score 상위 3명) ──
LESION_TARGETS = [
    # (patient_id, safe_id, group)
    ("LUNG1-306",    "NSCLC_LUNG1-306__09b6eb87c0",            "NSCLC"),
    ("LUNG1-020",    "NSCLC_LUNG1-020__b843f4f3dc",            "NSCLC"),
    ("LUNG1-383",    "NSCLC_LUNG1-383__0d1d368a6b",            "NSCLC"),
    ("MSD_lung_073", "MSD_Lung_MSD_lung_073__48b988b3d6",      "MSD_Lung"),
    ("MSD_lung_069", "MSD_Lung_MSD_lung_069__02b753ea9d",      "MSD_Lung"),
]
NORMAL_TARGETS = [
    # (patient_id, safe_id, group)
    (
        "subset2_1.3.6.1.4.1.14519.5.2.1.6279.6001.102133688497886810253331438797",
        "subset2_1.3.6.1.4.1.14519.5.2.1.6279.6001.102133688497886810253331438797__37e91b0fbb",
        "LUNA16",
    ),
    (
        "subset1_1.3.6.1.4.1.14519.5.2.1.6279.6001.197987940182806628828566429132",
        "subset1_1.3.6.1.4.1.14519.5.2.1.6279.6001.197987940182806628828566429132__9b83070aaa",
        "LUNA16",
    ),
    ("normal004", "normal004__9190565aec", "LUNA16_normal"),
]


def abort(msg: str) -> None:
    print(f"[ABORT] {msg}", file=sys.stderr)
    sys.exit(2)


def mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else -1.0


def shape_str(arr: np.ndarray) -> str:
    return "x".join(str(d) for d in arr.shape)


def main() -> None:
    if not ALLOW_REAL_PROCESSING:
        abort("ALLOW_REAL_PROCESSING=False: dry-run guard 활성. 이 스크립트는 직접 실행 불가.")

    # ── output root 존재 확인 ───────────────────────────────────────────────────
    if OUTPUT_ROOT.exists():
        abort(f"output root가 이미 존재합니다. 덮어쓰지 않습니다: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)

    # ── 원본 파일 mtime 스냅샷 ─────────────────────────────────────────────────
    protected_files = [THRESHOLD_JSON, DIST_NPZ, LESION_SPLIT_CSV]
    mtime_before = {str(p): mtime(p) for p in protected_files}

    # ── stage2 holdout denylist 구성 ──────────────────────────────────────────
    holdout_pids: set = set()
    holdout_sids: set = set()
    with open(LESION_SPLIT_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("stage_split", "") == "stage2_holdout":
                holdout_pids.add(row["patient_id"].strip())
                holdout_sids.add(row["safe_id"].strip())

    # ── 대상 목록 구성 ─────────────────────────────────────────────────────────
    targets = []
    for pid, sid, grp in LESION_TARGETS:
        targets.append({
            "patient_id": pid, "safe_id": sid, "group": grp,
            "role": "lesion_candidate", "split": "stage1_dev",
            "source_dataset": grp,
        })
    for pid, sid, grp in NORMAL_TARGETS:
        targets.append({
            "patient_id": pid, "safe_id": sid, "group": grp,
            "role": "normal_control", "split": "normal_test",
            "source_dataset": grp,
        })

    # ── 각 환자 검증 및 계산 ───────────────────────────────────────────────────
    rows = []
    errors = []
    n_usable = 0

    for t in targets:
        pid   = t["patient_id"]
        sid   = t["safe_id"]
        role  = t["role"]

        # stage2_holdout 교집합 확인
        in_holdout = (pid in holdout_pids) or (sid in holdout_sids)
        t["stage2_holdout_intersection_flag"] = in_holdout
        if in_holdout:
            errors.append({"patient_id": pid, "stage": "holdout_check",
                           "msg": "stage2_holdout 교집합 발견: 즉시 FAIL"})
            abort(f"stage2_holdout 교집합 발견: {pid}")

        # 경로 결정
        if role == "lesion_candidate":
            ct_path         = LESION_VOLUMES_ROOT / sid / "ct_hu.npy"
            roi_path        = V4_20_LESION_ROI_ROOT / sid / "refined_roi.npy"
            lesion_path     = LESION_VOLUMES_ROOT / sid / "lesion_mask_roi_0_0.npy"
            score_csv_path  = LESION_SCORE_ROOT / f"{pid}.csv"
        else:
            ct_path         = NORMAL_VOLUMES_ROOT / sid / "ct_hu.npy"
            roi_path        = V4_20_NORMAL_ROI_ROOT / sid / "refined_roi.npy"
            lesion_path     = None
            score_csv_path  = NORMAL_TEST_SCORE_ROOT / f"{pid}.csv"

        ct_exists         = ct_path.exists()
        roi_exists        = roi_path.exists()
        lesion_exists     = (lesion_path.exists() if lesion_path else None)
        score_exists      = score_csv_path.exists()

        ct_shape_str      = ""
        roi_shape_str     = ""
        lesion_shape_str  = ""
        shape_match       = False
        roi_voxel_count   = 0
        hu_ge0_count      = 0
        lesion_voxel_count = 0
        oracle_vessel_count = 0
        oracle_ratio        = 0.0
        usable              = False
        err_msg             = ""

        try:
            if not ct_exists:
                raise FileNotFoundError(f"CT 없음: {ct_path}")
            if not roi_exists:
                raise FileNotFoundError(f"ROI 없음: {roi_path}")
            if role == "lesion_candidate" and not lesion_exists:
                raise FileNotFoundError(f"lesion mask 없음: {lesion_path}")
            if not score_exists:
                raise FileNotFoundError(f"score CSV 없음: {score_csv_path}")

            ct  = np.load(str(ct_path),  mmap_mode='r')
            roi = np.load(str(roi_path), mmap_mode='r')
            ct_shape_str  = shape_str(ct)
            roi_shape_str = shape_str(roi)

            if ct.shape != roi.shape:
                raise ValueError(f"shape mismatch: ct={ct.shape} roi={roi.shape}")

            if role == "lesion_candidate":
                les = np.load(str(lesion_path), mmap_mode='r')
                lesion_shape_str = shape_str(les)
                if les.shape != ct.shape:
                    raise ValueError(f"lesion shape mismatch: les={les.shape} ct={ct.shape}")
            shape_match = True

            # ROI 내부 voxel 계산 (bool)
            roi_bool = np.asarray(roi) > 0
            roi_voxel_count = int(roi_bool.sum())

            # HU >= 0 inside ROI
            ct_arr = np.asarray(ct)
            hu_ge0_mask = roi_bool & (ct_arr >= 0)
            hu_ge0_count = int(hu_ge0_mask.sum())

            # oracle-like vessel mask
            if role == "lesion_candidate":
                les_arr = np.asarray(les)
                les_bool = les_arr > 0
                oracle_mask = hu_ge0_mask & (~les_bool)
                lesion_voxel_count = int((roi_bool & les_bool).sum())
            else:
                oracle_mask = hu_ge0_mask
                lesion_voxel_count = 0

            oracle_vessel_count = int(oracle_mask.sum())
            oracle_ratio = (oracle_vessel_count / roi_voxel_count
                            if roi_voxel_count > 0 else 0.0)

            usable = True
            n_usable += 1

        except Exception as e:
            err_msg = str(e)
            errors.append({"patient_id": pid, "stage": "per_patient", "msg": err_msg})

        rows.append({
            "patient_id":                   pid,
            "safe_id":                      sid,
            "split":                        t["split"],
            "source_dataset":               t["source_dataset"],
            "role":                         role,
            "ct_path":                      str(ct_path),
            "roi_path":                     str(roi_path),
            "lesion_mask_path":             str(lesion_path) if lesion_path else "",
            "score_csv_path":               str(score_csv_path),
            "ct_exists":                    ct_exists,
            "roi_exists":                   roi_exists,
            "lesion_mask_exists":           lesion_exists if lesion_exists is not None else "",
            "score_csv_exists":             score_exists,
            "ct_shape":                     ct_shape_str,
            "roi_shape":                    roi_shape_str,
            "lesion_mask_shape":            lesion_shape_str,
            "shape_match":                  shape_match,
            "roi_voxel_count":              roi_voxel_count,
            "hu_ge_0_voxel_count_inside_roi": hu_ge0_count,
            "lesion_voxel_count":           lesion_voxel_count,
            "oracle_like_vessel_voxel_count": oracle_vessel_count,
            "oracle_like_vessel_ratio_in_roi": round(oracle_ratio, 6),
            "stage2_holdout_intersection_flag": in_holdout,
            "usable_for_b1e2":              usable,
            "error_msg":                    err_msg,
        })

    # ── mtime 사후 검증 ─────────────────────────────────────────────────────────
    mtime_violations = []
    for path_str, before in mtime_before.items():
        after = mtime(Path(path_str))
        if before != after:
            mtime_violations.append(f"{path_str}: {before} → {after}")
    if mtime_violations:
        abort("원본 파일 mtime 변경 감지:\n" + "\n".join(mtime_violations))

    # ── CSV 저장 ────────────────────────────────────────────────────────────────
    targets_csv = OUTPUT_ROOT / "b1e1_oracle_mask_preflight_targets.csv"
    errors_csv  = OUTPUT_ROOT / "b1e1_oracle_mask_preflight_errors.csv"

    fieldnames = [
        "patient_id","safe_id","split","source_dataset","role",
        "ct_path","roi_path","lesion_mask_path","score_csv_path",
        "ct_exists","roi_exists","lesion_mask_exists","score_csv_exists",
        "ct_shape","roi_shape","lesion_mask_shape","shape_match",
        "roi_voxel_count","hu_ge_0_voxel_count_inside_roi",
        "lesion_voxel_count","oracle_like_vessel_voxel_count",
        "oracle_like_vessel_ratio_in_roi",
        "stage2_holdout_intersection_flag","usable_for_b1e2","error_msg",
    ]
    with open(targets_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    err_fields = ["patient_id","stage","msg"]
    with open(errors_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=err_fields)
        w.writeheader()
        w.writerows(errors)

    # ── summary JSON ────────────────────────────────────────────────────────────
    n_total   = len(rows)
    n_lesion  = sum(1 for r in rows if r["role"] == "lesion_candidate")
    n_normal  = sum(1 for r in rows if r["role"] == "normal_control")
    n_fail    = sum(1 for r in rows if not r["usable_for_b1e2"])

    oracle_ratios = [r["oracle_like_vessel_ratio_in_roi"]
                     for r in rows if r["usable_for_b1e2"]]

    summary = {
        "step": "B1-E1",
        "branch": "efficientnet_b0_imagenet_chestwall_removed_roi_v1",
        "roi_source": "refined_roi_v4_20_modeB_all_v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_total": n_total,
        "n_lesion_candidate": n_lesion,
        "n_normal_control": n_normal,
        "n_usable_for_b1e2": n_usable,
        "n_fail": n_fail,
        "n_error": len(errors),
        "stage2_holdout_intersection": 0,
        "oracle_ratio_min":  round(min(oracle_ratios), 6) if oracle_ratios else None,
        "oracle_ratio_max":  round(max(oracle_ratios), 6) if oracle_ratios else None,
        "oracle_ratio_mean": round(float(np.mean(oracle_ratios)), 6) if oracle_ratios else None,
        "mtime_violations": len(mtime_violations),
        "score_modified": False,
        "threshold_recalculated": False,
        "model_modified": False,
        "roi_modified": False,
        "ct_modified": False,
        "suppression_applied": False,
        "stage2_holdout_accessed": False,
        "gpu_used": False,
        "all_checks_passed": (n_fail == 0 and len(mtime_violations) == 0 and len(errors) == 0),
        "next_step_b1e2_ready": (n_usable > 0),
    }

    summary_json = OUTPUT_ROOT / "b1e1_oracle_mask_preflight_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── 보고서 MD ───────────────────────────────────────────────────────────────
    report_lines = [
        "# B1-E1 Oracle-like Vessel Mask Preflight 보고서",
        "",
        f"생성일시: {summary['created']}",
        f"branch: {summary['branch']}",
        f"ROI source: {summary['roi_source']}",
        "",
        "## 1. 실험 성격 고지",
        "",
        "- **이번 실험은 real vessel segmentation 검증이 아니라 oracle-like upper-bound 실험이다.**",
        "- HU>=0 기반 mask는 진짜 혈관 GT가 아니며, 조영제 혈관 / 종격동 구조물 / 흉벽 bright tissue가 섞일 수 있다.",
        "- 이번 단계에서는 **suppression을 적용하지 않았다.** score CSV / threshold / model 파일을 수정하지 않았다.",
        "- oracle mask는 '만약 혈관 위치를 완벽히 안다면 FP를 얼마나 줄일 수 있는가'를 가늠하기 위한 dev-only 상한선이다.",
        "",
        "## 2. 검증 결과 요약",
        "",
        f"| 항목 | 값 |",
        f"|---|---|",
        f"| 전체 대상 | {n_total}명 |",
        f"| lesion_candidate | {n_lesion}명 |",
        f"| normal_control | {n_normal}명 |",
        f"| usable_for_b1e2 | {n_usable}명 |",
        f"| 실패 | {n_fail}명 |",
        f"| stage2_holdout 교집합 | 0 |",
        f"| mtime 위반 | {len(mtime_violations)} |",
        f"| oracle_ratio 범위 | {summary['oracle_ratio_min']} ~ {summary['oracle_ratio_max']} |",
        f"| oracle_ratio 평균 | {summary['oracle_ratio_mean']} |",
        "",
        "## 3. 환자별 oracle-like vessel mask 수치",
        "",
        "| patient_id | role | roi_voxels | hu_ge0_in_roi | lesion_voxels | oracle_vessel | oracle_ratio | usable |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        report_lines.append(
            f"| {r['patient_id']} | {r['role']} | {r['roi_voxel_count']:,} "
            f"| {r['hu_ge_0_voxel_count_inside_roi']:,} | {r['lesion_voxel_count']:,} "
            f"| {r['oracle_like_vessel_voxel_count']:,} "
            f"| {r['oracle_like_vessel_ratio_in_roi']:.4f} | {r['usable_for_b1e2']} |"
        )

    report_lines += [
        "",
        "## 4. 오류 목록",
        "",
        ("없음." if not errors else "\n".join(
            f"- {e['patient_id']}: [{e['stage']}] {e['msg']}" for e in errors
        )),
        "",
        "## 5. 다음 단계 안내",
        "",
        "- **B1-E2**: score CSV의 patch 좌표(y0,x0,y1,x1,local_z)와 oracle-like vessel mask의 voxel overlap을 매핑하는 dry-run.",
        "- B1-E2에서도 **원본 score CSV 수정 금지**. adjusted score는 새 preview CSV에만 생성해야 한다.",
        "- B1-E2는 GPU 불필요, patch 좌표 매핑만 확인.",
        "",
        "## 6. 안전 게이트 확인",
        "",
        "| 항목 | 상태 |",
        "|---|---|",
        "| score CSV 수정 | 미실행 |",
        "| threshold 재계산 | 미실행 |",
        "| model 수정 | 미실행 |",
        "| ROI 파일 수정 | 미실행 |",
        "| CT 파일 수정 | 미실행 |",
        "| suppression 적용 | 미실행 |",
        "| stage2_holdout 접근 | 없음 |",
        "| GPU 사용 | 없음 |",
        "| mtime 위반 | 없음 |",
    ]

    report_md = OUTPUT_ROOT / "b1e1_oracle_mask_preflight_report.md"
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    # ── DONE 파일 ────────────────────────────────────────────────────────────────
    all_pass = summary["all_checks_passed"]
    print(f"\n=== B1-E1 Oracle Mask Preflight 완료 ===")
    print(f"  총 대상: {n_total}명 / 통과: {n_usable}명 / 실패: {n_fail}명")
    if oracle_ratios:
        print(f"  oracle_ratio 범위: {min(oracle_ratios):.4f} ~ {max(oracle_ratios):.4f}")
    print(f"  stage2_holdout 교집합: 0")
    print(f"  mtime 위반: {len(mtime_violations)}")
    print(f"  출력: {OUTPUT_ROOT}")
    print(f"  전체 통과: {all_pass}")

    if all_pass:
        done_file = OUTPUT_ROOT / "DONE"
        done_file.write_text(f"B1-E1 PASS {summary['created']}\n")
        print("  → DONE 파일 생성 완료")
    else:
        print("  → 오류 있음. DONE 파일 미생성.")
        for r in rows:
            if r["error_msg"]:
                print(f"    [{r['patient_id']}] {r['error_msg']}")


if __name__ == "__main__":
    main()
