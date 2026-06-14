"""
P-B3: v4_20-only lesion safety validation on stage1_dev (154명)
- v4_20 refined ROI가 lesion GT mask를 얼마나 보존/손실하는지 voxel-level 검증
- ROI source: refined_roi_v4_20_modeB_all_v1/lesion (v4_20 lock)
- GT mask: C드라이브 원본 lesion_mask_roi_0_0.npy (병변 위치 GT)
- preservation_ratio = (mask & roi) voxel / mask voxel
- AUROC/AUPRC/Dice/recall 계산 안 함. CT intensity 분석 안 함.
"""
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

# ── 경로 ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
EXP_ROOT     = Path(__file__).resolve().parent.parent

V4_20_LESION_ROOT = PROJECT_ROOT / "outputs" / "mip-postprocess-research-v1" / "masks" / "refined_roi_v4_20_modeB_all_v1" / "lesion"
LESION_MASK_ROOT  = Path("/mnt/c/Users/jinhy/Desktop/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1/volumes_npy")
LESION_MASK_FILE  = "lesion_mask_roi_0_0.npy"
LESION_PATCH_DIR  = Path("/mnt/c/Users/jinhy/Desktop/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1/patch_index_by_patient")

LESION_SPLIT_CSV  = PROJECT_ROOT / "outputs" / "second-stage-lesion-refiner-v1" / "splits" / "lesion_stage_split_v1_balanced.csv"
P_B2_6_JSON       = EXP_ROOT / "outputs" / "reports" / "p_b2_6_v4_20_source_lock" / "p_b2_6_v4_20_source_lock.json"

REPORT_DIR = EXP_ROOT / "outputs" / "reports" / "p_b3_lesion_safety_validation"
SCRIPT_NAME = "p_b3_lesion_safety_validation.py"

EXPECTED_STAGE1_DEV = 154
EXPECTED_NSCLC = 125
EXPECTED_MSD   = 29

# 심각 손실 임계
THRESHOLDS = [1.0, 0.99, 0.95, 0.90, 0.80]


def load_p_b2_6():
    if not P_B2_6_JSON.exists():
        return None
    return json.load(open(P_B2_6_JSON, encoding="utf-8"))


def load_lesion_split():
    stage1, holdout = [], set()
    with open(LESION_SPLIT_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            sv = row.get("stage_split", "").strip()
            sid = row["safe_id"].strip()
            g = row.get("group", "").strip()
            if sv == "stage1_dev":
                stage1.append((sid, g))
            elif sv == "stage2_holdout":
                holdout.add(sid)
    return stage1, holdout


def load_patch_position(safe_id):
    """patch CSV에서 has_lesion_patch=1 patch의 central_peripheral / lesion_zone_type 집계."""
    p = LESION_PATCH_DIR / f"{safe_id}.csv"
    if not p.exists():
        return {"available": False}
    cp_counter = Counter()
    zone_counter = Counter()
    n_lesion_patch = 0
    with open(p, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("has_lesion_patch", "").strip() in ("1", "True", "true"):
                n_lesion_patch += 1
                cp = row.get("central_peripheral", "").strip()
                zone = row.get("lesion_zone_type", "").strip()
                if cp:   cp_counter[cp] += 1
                if zone: zone_counter[zone] += 1
    peripheral = cp_counter.get("peripheral", 0)
    central    = cp_counter.get("central", 0)
    peripheral_ratio = round(peripheral / n_lesion_patch, 4) if n_lesion_patch > 0 else None
    return {
        "available": True,
        "n_lesion_patch": n_lesion_patch,
        "peripheral_patch": peripheral,
        "central_patch": central,
        "peripheral_ratio": peripheral_ratio,
        "top_zone": zone_counter.most_common(1)[0][0] if zone_counter else "",
    }


def pct(values, p):
    return round(float(np.percentile(values, p)), 6) if values else None


def save_csv(rows, path, fieldnames=None):
    if not rows:
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

    # ── 1. P-B2.6 입력 검증 ──────────────────────────────────────────────
    pb26 = load_p_b2_6()
    pb26_ok = pb26 is not None and pb26.get("verdict") in ("통과", "부분통과")
    source_locked = pb26 is not None and \
        pb26.get("user_correction_applied", {}).get("official_roi_source") == "refined_roi_v4_20_modeB_all_v1"
    print(f"[P-B2.6] verdict={pb26.get('verdict') if pb26 else None}, "
          f"source_locked={source_locked}")
    if not pb26_ok or not source_locked:
        print("[중단] P-B2.6 전제 불충족")
        sys.exit(1)

    # ── 2. split ─────────────────────────────────────────────────────────
    stage1, holdout = load_lesion_split()
    n_nsclc = sum(1 for _, g in stage1 if g == "NSCLC")
    n_msd   = sum(1 for _, g in stage1 if g == "MSD_Lung")
    stage1_ids = set(s for s, _ in stage1)
    contamination = stage1_ids & holdout
    print(f"[split] stage1_dev {len(stage1)} (NSCLC {n_nsclc} / MSD {n_msd}), holdout {len(holdout)}")
    print(f"[가드] contamination = {len(contamination)}")
    if contamination:
        print("[중단] stage1_dev ∩ stage2_holdout > 0")
        sys.exit(1)

    # ── 3. per-patient lesion safety ─────────────────────────────────────
    per_patient = []
    risk_cases  = []
    preservation_ratios = []
    loss_ratios = []

    roi_missing = 0
    mask_missing = 0
    shape_mismatch = 0
    complete_loss = 0
    total_lesion_voxels_all = 0
    preserved_voxels_all = 0

    for safe_id, group in stage1:
        roi_path  = V4_20_LESION_ROOT / safe_id / "refined_roi.npy"
        mask_path = LESION_MASK_ROOT / safe_id / LESION_MASK_FILE

        row = {"safe_id": safe_id, "group": group}

        if not roi_path.exists():
            roi_missing += 1
            row["status"] = "roi_missing"
            per_patient.append(row); continue
        if not mask_path.exists():
            mask_missing += 1
            row["status"] = "mask_missing"
            per_patient.append(row); continue

        roi  = np.load(str(roi_path),  mmap_mode='r').astype(np.uint8)
        mask = np.load(str(mask_path), mmap_mode='r').astype(np.uint8)

        if roi.shape != mask.shape:
            shape_mismatch += 1
            row.update({"status": "shape_mismatch",
                        "roi_shape": str(roi.shape), "mask_shape": str(mask.shape)})
            per_patient.append(row)
            del roi, mask
            continue

        total = int(np.sum(mask))
        if total == 0:
            row.update({"status": "no_lesion_voxel", "total_lesion_voxels": 0,
                        "preserved": 0, "lost": 0, "preservation_ratio": None})
            per_patient.append(row)
            del roi, mask
            continue

        preserved = int(np.sum((mask == 1) & (roi == 1)))
        lost = total - preserved
        pres_ratio = round(preserved / total, 6)
        loss_ratio = round(1 - pres_ratio, 6)

        total_lesion_voxels_all += total
        preserved_voxels_all += preserved
        preservation_ratios.append(pres_ratio)
        loss_ratios.append(loss_ratio)

        # 위치 보조 정보
        pos = load_patch_position(safe_id)

        row.update({
            "status": "ok",
            "total_lesion_voxels": total,
            "preserved": preserved,
            "lost": lost,
            "preservation_ratio": pres_ratio,
            "loss_ratio": loss_ratio,
            "n_lesion_patch": pos.get("n_lesion_patch"),
            "peripheral_ratio": pos.get("peripheral_ratio"),
            "top_zone": pos.get("top_zone"),
        })
        per_patient.append(row)

        if pres_ratio == 0:
            complete_loss += 1
        # 위험 케이스: preservation < 0.95
        if pres_ratio < 0.95:
            risk_cases.append({
                "safe_id": safe_id, "group": group,
                "preservation_ratio": pres_ratio, "loss_ratio": loss_ratio,
                "total_lesion_voxels": total, "lost": lost,
                "peripheral_ratio": pos.get("peripheral_ratio"),
                "top_zone": pos.get("top_zone"),
            })

        del roi, mask

    n_ok = sum(1 for r in per_patient if r.get("status") == "ok")
    print(f"\n[처리] ok={n_ok}, roi_missing={roi_missing}, mask_missing={mask_missing}, "
          f"shape_mismatch={shape_mismatch}")

    # ── 4. 분포 ──────────────────────────────────────────────────────────
    dist = {}
    if preservation_ratios:
        dist = {
            "n": len(preservation_ratios),
            "preservation_min":    round(float(np.min(preservation_ratios)), 6),
            "preservation_p1":     pct(preservation_ratios, 1),
            "preservation_p5":     pct(preservation_ratios, 5),
            "preservation_median": round(float(np.median(preservation_ratios)), 6),
            "preservation_mean":   round(float(np.mean(preservation_ratios)), 6),
            "preservation_p95":    pct(preservation_ratios, 95),
            "preservation_p99":    pct(preservation_ratios, 99),
            "loss_max":    round(float(np.max(loss_ratios)), 6),
            "loss_median": round(float(np.median(loss_ratios)), 6),
            "loss_mean":   round(float(np.mean(loss_ratios)), 6),
            "aggregate_preservation": round(preserved_voxels_all / total_lesion_voxels_all, 6)
                                      if total_lesion_voxels_all > 0 else None,
        }

    # threshold 케이스 카운트
    thr_counts = {}
    for t in THRESHOLDS:
        thr_counts[f"lt_{t}"] = sum(1 for r in preservation_ratios if r < t)
    thr_counts["eq_0"] = complete_loss

    print(f"[분포] preservation median={dist.get('preservation_median')}, "
          f"min={dist.get('preservation_min')}, p1={dist.get('preservation_p1')}")
    print(f"[케이스] <1.0={thr_counts['lt_1.0']}, <0.95={thr_counts['lt_0.95']}, "
          f"<0.90={thr_counts['lt_0.9']}, <0.80={thr_counts['lt_0.8']}, ==0={complete_loss}")

    # worst-case (preservation 낮은 순 10명)
    ok_rows = [r for r in per_patient if r.get("status") == "ok"]
    worst = sorted(ok_rows, key=lambda r: r["preservation_ratio"])[:10]

    # ── 5. 판정 ───────────────────────────────────────────────────────────
    issues = []
    if roi_missing: issues.append(f"ROI 누락 {roi_missing}")
    if mask_missing: issues.append(f"GT mask 누락 {mask_missing}")
    if shape_mismatch: issues.append(f"shape mismatch {shape_mismatch}")
    if complete_loss: issues.append(f"complete lesion loss {complete_loss}건")
    if len(stage1) != EXPECTED_STAGE1_DEV: issues.append(f"stage1_dev {len(stage1)}≠{EXPECTED_STAGE1_DEV}")

    severe_080 = thr_counts["lt_0.8"]
    severe_090 = thr_counts["lt_0.9"]

    # 판정 로직
    if complete_loss > 0 or shape_mismatch > 0 or roi_missing > 0 or mask_missing > 0 \
       or len(stage1) != EXPECTED_STAGE1_DEV:
        verdict = "실패"
    elif severe_080 == 0 and severe_090 <= 1:
        # 심각 손실(<0.80) 0건, <0.90도 거의 없음 → train smoke 진행 가능
        verdict = "통과"
    else:
        # 일부 손실 있어 위험 케이스 검토 필요
        verdict = "부분통과"

    print(f"\n[판정] {verdict}")
    for i in issues: print(f"  ⚠ {i}")

    p_b4_can_proceed = (verdict in ("통과", "부분통과")) and complete_loss == 0 and shape_mismatch == 0

    # ── 6. JSON ───────────────────────────────────────────────────────────
    report = {
        "stage": "P-B3_v4_20_lesion_safety_validation",
        "created": ts,
        "verdict": verdict,
        "input_validation": {
            "p_b2_6_verdict": pb26.get("verdict"),
            "p_b2_6_ok": pb26_ok,
            "official_roi_source": "refined_roi_v4_20_modeB_all_v1",
            "source_locked": source_locked,
            "model_roi_used": False,
            "e_drive_used": False,
            "roi_0_0_used_as_blocker": False,
        },
        "scope": {
            "stage2_holdout_accessed": False,
            "stage2_holdout_value_loaded": False,
            "training": False, "model_forward": False, "scoring": False,
            "threshold_calculated": False, "metrics_calculated": False,
            "auroc_auprc_dice_recall_computed": False,
            "ct_intensity_analyzed": False,
            "existing_files_modified": False,
        },
        "sources": {
            "roi_source": "refined_roi_v4_20_modeB_all_v1/lesion/<id>/refined_roi.npy",
            "gt_mask_source": "C드라이브 NSCLC_MSD..._roi0_0_..._usable_only_v1/volumes_npy/<id>/lesion_mask_roi_0_0.npy",
            "gt_mask_note": "roi_0_0 기준 clipped GT. preservation_ratio = 기존 roi_0_0 branch 병변 중 v4_20에서 살아남는 비율 = 흉벽 제거로 추가 손실되는 병변량.",
        },
        "stage1_dev": {
            "count": len(stage1), "nsclc": n_nsclc, "msd_lung": n_msd,
            "count_match": len(stage1) == EXPECTED_STAGE1_DEV,
            "group_match": (n_nsclc == EXPECTED_NSCLC and n_msd == EXPECTED_MSD),
        },
        "stage2_holdout_contamination": len(contamination),
        "file_existence": {
            "roi_missing": roi_missing, "mask_missing": mask_missing,
            "shape_mismatch": shape_mismatch, "processed_ok": n_ok,
        },
        "voxel_totals": {
            "total_lesion_voxels": total_lesion_voxels_all,
            "preserved_voxels": preserved_voxels_all,
            "lost_voxels": total_lesion_voxels_all - preserved_voxels_all,
        },
        "preservation_distribution": dist,
        "threshold_case_counts": thr_counts,
        "complete_lesion_loss": complete_loss,
        "severe_loss_lt_0_90": severe_090,
        "severe_loss_lt_0_80": severe_080,
        "worst_cases": [
            {"safe_id": w["safe_id"], "group": w["group"],
             "preservation_ratio": w["preservation_ratio"], "loss_ratio": w["loss_ratio"],
             "total_lesion_voxels": w["total_lesion_voxels"], "lost": w["lost"],
             "peripheral_ratio": w.get("peripheral_ratio")}
            for w in worst
        ],
        "p_b4_readiness": {
            "can_proceed": p_b4_can_proceed,
            "complete_loss_zero": complete_loss == 0,
            "note": "complete loss 0 + shape mismatch 0이면 train smoke 진행 가능. 단 위험 케이스는 학습 후 결과 해석 시 참고.",
        },
        "issues": issues,
    }
    with open(REPORT_DIR / "p_b3_lesion_safety_validation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ── 7. CSV ────────────────────────────────────────────────────────────
    save_csv(per_patient, REPORT_DIR / "lesion_safety_per_patient.csv")
    save_csv(risk_cases,  REPORT_DIR / "lesion_safety_risk_cases.csv",
             fieldnames=["safe_id","group","preservation_ratio","loss_ratio",
                         "total_lesion_voxels","lost","peripheral_ratio","top_zone"])

    summary_rows = [{
        "metric": k, "value": v
    } for k, v in {
        "stage1_dev_count": len(stage1),
        "nsclc": n_nsclc, "msd_lung": n_msd,
        "contamination": len(contamination),
        "roi_missing": roi_missing, "mask_missing": mask_missing,
        "shape_mismatch": shape_mismatch,
        "total_lesion_voxels": total_lesion_voxels_all,
        "preserved_voxels": preserved_voxels_all,
        "aggregate_preservation": dist.get("aggregate_preservation"),
        "preservation_median": dist.get("preservation_median"),
        "preservation_min": dist.get("preservation_min"),
        "complete_loss": complete_loss,
        "lt_0.95": thr_counts["lt_0.95"],
        "lt_0.90": thr_counts["lt_0.9"],
        "lt_0.80": thr_counts["lt_0.8"],
        "verdict": verdict,
    }.items()]
    save_csv(summary_rows, REPORT_DIR / "lesion_safety_summary.csv")

    dist_rows = []
    for t in THRESHOLDS:
        dist_rows.append({"threshold": f"preservation < {t}", "case_count": thr_counts[f"lt_{t}"]})
    dist_rows.append({"threshold": "preservation == 0 (complete loss)", "case_count": complete_loss})
    if dist:
        for k in ["preservation_min","preservation_p1","preservation_p5","preservation_median",
                  "preservation_p95","preservation_p99","aggregate_preservation"]:
            dist_rows.append({"threshold": k, "case_count": dist.get(k)})
    save_csv(dist_rows, REPORT_DIR / "lesion_safety_distribution.csv",
             fieldnames=["threshold","case_count"])

    # ── 8. MD ─────────────────────────────────────────────────────────────
    md = []
    md.append("# P-B3 v4_20-only Lesion Safety Validation (stage1_dev)\n")
    md.append(f"- 생성일: {ts}")
    md.append(f"- 판정: **{verdict}**\n")
    md.append("---\n")
    md.append("## 0. 입력 검증 / scope\n")
    md.append("| 항목 | 결과 |")
    md.append("|------|------|")
    md.append(f"| P-B2.6 verdict | {pb26.get('verdict')} ({'OK' if pb26_ok else 'NG'}) |")
    md.append(f"| 공식 ROI source = v4_20 | {source_locked} ✅ |")
    md.append("| model_roi.npy 사용 | 안 함 ✅ |")
    md.append("| E드라이브 사용 | 안 함 ✅ |")
    md.append("| roi_0_0를 blocker로 사용 | 안 함 ✅ |")
    md.append("| stage2_holdout value 로드 | 없음 ✅ |")
    md.append("| 학습/forward/scoring/threshold/metrics | 미실행 ✅ |")
    md.append("| AUROC/AUPRC/Dice/recall 계산 | 안 함 ✅ |")
    md.append("| CT intensity 분석 | 안 함 ✅ |\n")
    md.append("---\n")
    md.append("## 1. source\n")
    md.append("- ROI: `refined_roi_v4_20_modeB_all_v1/lesion/<id>/refined_roi.npy` (v4_20 lock)")
    md.append("- GT mask: C드라이브 `lesion_mask_roi_0_0.npy` (병변 위치 GT)")
    md.append("- ⚠ GT mask는 roi_0_0 기준 clipped. 따라서 **preservation_ratio = 기존 roi_0_0 branch 병변 중")
    md.append("  v4_20에서 살아남는 비율 = 흉벽 제거로 추가 손실되는 병변량**. (정확한 비교 기준)\n")
    md.append("---\n")
    md.append("## 2. stage1_dev 확정\n")
    md.append("| 항목 | 값 | 기대 | 일치 |")
    md.append("|------|----|------|------|")
    md.append(f"| stage1_dev | {len(stage1)} | {EXPECTED_STAGE1_DEV} | {'✅' if len(stage1)==EXPECTED_STAGE1_DEV else '❌'} |")
    md.append(f"| NSCLC | {n_nsclc} | {EXPECTED_NSCLC} | {'✅' if n_nsclc==EXPECTED_NSCLC else '❌'} |")
    md.append(f"| MSD_Lung | {n_msd} | {EXPECTED_MSD} | {'✅' if n_msd==EXPECTED_MSD else '❌'} |")
    md.append(f"| holdout contamination | {len(contamination)} | 0 | {'✅' if not contamination else '❌'} |\n")
    md.append("---\n")
    md.append("## 3. 파일 존재 / shape\n")
    md.append("| 항목 | 결과 |")
    md.append("|------|------|")
    md.append(f"| ROI 누락 | {roi_missing} |")
    md.append(f"| GT mask 누락 | {mask_missing} |")
    md.append(f"| shape mismatch | {shape_mismatch} |")
    md.append(f"| 정상 처리 | {n_ok}/{len(stage1)} |\n")
    md.append("---\n")
    md.append("## 4. lesion preservation 분포 (154명)\n")
    if dist:
        md.append("| 지표 | 값 |")
        md.append("|------|----|")
        md.append(f"| aggregate preservation (전체 voxel 합) | **{dist['aggregate_preservation']}** |")
        md.append(f"| preservation 중앙값 | {dist['preservation_median']} |")
        md.append(f"| preservation 평균 | {dist['preservation_mean']} |")
        md.append(f"| preservation 최솟값 | {dist['preservation_min']} |")
        md.append(f"| preservation p1 | {dist['preservation_p1']} |")
        md.append(f"| preservation p5 | {dist['preservation_p5']} |")
        md.append(f"| preservation p95 | {dist['preservation_p95']} |")
        md.append(f"| preservation p99 | {dist['preservation_p99']} |")
        md.append(f"| loss 최대 | {dist['loss_max']} |")
        md.append(f"| loss 중앙값 | {dist['loss_median']} |\n")
    md.append("---\n")
    md.append("## 5. 손실 케이스 카운트\n")
    md.append("| 기준 | 환자 수 |")
    md.append("|------|---------|")
    md.append(f"| preservation < 1.00 | {thr_counts['lt_1.0']} |")
    md.append(f"| preservation < 0.99 | {thr_counts['lt_0.99']} |")
    md.append(f"| preservation < 0.95 | {thr_counts['lt_0.95']} |")
    md.append(f"| preservation < 0.90 | {thr_counts['lt_0.9']} |")
    md.append(f"| preservation < 0.80 | {thr_counts['lt_0.8']} |")
    md.append(f"| **preservation == 0 (complete loss)** | **{complete_loss}** |\n")
    md.append("---\n")
    md.append("## 6. worst-case 환자 (preservation 낮은 순 10명)\n")
    md.append("| safe_id | group | preservation | loss | lesion voxel | peripheral_ratio |")
    md.append("|---------|-------|--------------|------|--------------|------------------|")
    for w in worst:
        md.append(f"| {w['safe_id']} | {w['group']} | {w['preservation_ratio']} | "
                  f"{w['loss_ratio']} | {w['total_lesion_voxels']} | {w.get('peripheral_ratio')} |")
    md.append("")
    md.append("> peripheral_ratio = has_lesion_patch 중 central_peripheral=peripheral 비율 (patch 단위 보조 정보)\n")
    md.append("---\n")
    md.append("## 7. 미실행 / 무수정 확인\n")
    md.append("- 학습/forward/scoring/threshold/metrics: **미실행** ✅")
    md.append("- stage2_holdout value 접근: **없음** ✅")
    md.append("- 기존 roi_0_0 / EfficientNet-B0 / P-B1~P-B2.6 결과: **무수정** ✅\n")
    md.append("---\n")
    md.append("## 8. 판정 근거\n")
    md.append(f"- complete lesion loss: **{complete_loss}건**")
    md.append(f"- severe loss (<0.90): {severe_090}건 / (<0.80): {severe_080}건")
    md.append(f"- shape mismatch: {shape_mismatch}건")
    md.append(f"- **판정: {verdict}**\n")
    md.append("---\n")
    md.append("## 9. P-B4 train smoke 진행 가능 여부\n")
    md.append(f"- **가능: {p_b4_can_proceed}**")
    md.append("- 근거: complete lesion loss 0건 + shape mismatch 0건")
    md.append("- 단, 위험 케이스(preservation 낮은 환자)는 학습 후 lesion 미탐 해석 시 참고\n")
    md.append("### P-B4 프롬프트 초안\n")
    md.append("```")
    md.append("P-B4 normal train smoke limit1 진행해줘.")
    md.append("ROI: refined_roi_v4_20_modeB_all_v1/normal/<id>/refined_roi.npy (v4_20 lock)")
    md.append("CT: 원본 ct_hu.npy (normal v2_tslungguard_nochest)")
    md.append("normal train 290명 중 1명만 smoke. DataLoader가 external refined ROI를 로드하도록 설계.")
    md.append("금지: full train, scoring, threshold, metrics, stage2_holdout")
    md.append("```\n")
    md.append("---\n")
    md.append("## 10. 최종 판정\n")
    md.append(f"- **{verdict}**")
    if dist:
        md.append(f"- 전체 lesion voxel 중 v4_20 ROI 보존율(aggregate): **{dist['aggregate_preservation']}**")
    md.append(f"- complete loss: {complete_loss}건 / severe(<0.80): {severe_080}건")
    md.append(f"- P-B4 진행 가능: **{p_b4_can_proceed}**")

    with open(REPORT_DIR / "p_b3_lesion_safety_validation.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"\n[저장] {REPORT_DIR}")
    print(f"[완료] 판정: {verdict}")
    return 0 if verdict != "실패" else 1


if __name__ == "__main__":
    sys.exit(main())
