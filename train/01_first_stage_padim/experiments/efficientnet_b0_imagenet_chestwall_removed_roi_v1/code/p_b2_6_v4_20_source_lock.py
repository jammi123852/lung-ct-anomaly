"""
P-B2.6: v4_20_modeB ROI source lock + branch correction
- 공식 ROI source를 refined_roi_v4_20_modeB_all_v1로 고정
- model_roi.npy / E드라이브 / roi_0_0 의존성을 필수 조건에서 제거
- v4_20 내부 실제 파일 구조 정리 (lesion mask 부재 사실 확인 포함)
- normal/lesion path rule 확정, P-B3 재정의
"""
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# ── 경로 ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
EXP_ROOT     = Path(__file__).resolve().parent.parent

V4_20_ROOT = PROJECT_ROOT / "outputs" / "mip-postprocess-research-v1" / "masks" / "refined_roi_v4_20_modeB_all_v1"
NORMAL_SPLIT_JSON = PROJECT_ROOT / "outputs" / "position-aware-padim-v1" / "splits" / "normal_v1.json"
LESION_SPLIT_CSV  = PROJECT_ROOT / "outputs" / "second-stage-lesion-refiner-v1" / "splits" / "lesion_stage_split_v1_balanced.csv"

# lesion mask 후보 source (v4_20 안에 없음 → 원본 C드라이브). 비교/필수 아님, 기록 목적.
LESION_MASK_FALLBACK_DIR = Path("/mnt/c/Users/jinhy/Desktop/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1/volumes_npy")
LESION_MASK_FALLBACK_FILE = "lesion_mask_roi_0_0.npy"

REPORT_DIR = EXP_ROOT / "outputs" / "reports" / "p_b2_6_v4_20_source_lock"
SCRIPT_NAME = "p_b2_6_v4_20_source_lock.py"

EXPECTED_NORMAL = 362
EXPECTED_LESION = 308
EXPECTED_TRAIN, EXPECTED_VAL, EXPECTED_TEST = 290, 36, 36
EXPECTED_STAGE1_DEV = 154


def load_normal_split():
    d = json.load(open(NORMAL_SPLIT_JSON, encoding="utf-8"))
    p2s = d.get("patient_to_safe_id", {})
    train = [p2s.get(p, p) for p in d.get("train", [])]
    val   = [p2s.get(p, p) for p in d.get("val", [])]
    test  = [p2s.get(p, p) for p in d.get("test", [])]
    return train, val, test


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


def list_v4_20(group):
    g = V4_20_ROOT / group
    return sorted([d.name for d in g.iterdir() if d.is_dir()])


def file_names_in(group):
    """group 폴더 내 각 환자 폴더의 파일명 집합 합계."""
    names = {}
    g = V4_20_ROOT / group
    for d in g.iterdir():
        if d.is_dir():
            for f in d.iterdir():
                names[f.name] = names.get(f.name, 0) + 1
    return names


def get_shape(group, safe_id, fname="refined_roi.npy"):
    p = V4_20_ROOT / group / safe_id / fname
    if not p.exists():
        return None
    return tuple(np.load(str(p), mmap_mode='r').shape)


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

    # ── 1. v4_20 asset inventory ─────────────────────────────────────────
    normal_ids = list_v4_20("normal")
    lesion_ids = list_v4_20("lesion")
    normal_files = file_names_in("normal")
    lesion_files = file_names_in("lesion")

    print(f"[inventory] normal 폴더 {len(normal_ids)}개, 파일종류={normal_files}")
    print(f"[inventory] lesion 폴더 {len(lesion_ids)}개, 파일종류={lesion_files}")

    # v4_20 내부에 lesion mask 파일이 있는가?
    v4_20_has_lesion_mask = any("mask" in n.lower() and "lesion" in n.lower() for n in lesion_files)
    v4_20_lesion_mask_count = sum(c for n, c in lesion_files.items()
                                  if "mask" in n.lower() and "lesion" in n.lower())
    print(f"[inventory] v4_20 내부 lesion mask 파일: {'있음' if v4_20_has_lesion_mask else '없음(refined_roi.npy만 존재)'}")

    # ── 2. split 로드 ─────────────────────────────────────────────────────
    train, val, test = load_normal_split()
    stage1, holdout = load_lesion_split()
    n_nsclc = sum(1 for _, g in stage1 if g == "NSCLC")
    n_msd   = sum(1 for _, g in stage1 if g == "MSD_Lung")
    print(f"[split] normal train {len(train)} / val {len(val)} / test {len(test)}")
    print(f"[split] lesion stage1_dev {len(stage1)} (NSCLC {n_nsclc} / MSD {n_msd}), holdout {len(holdout)}")

    # ── 3. 매칭 (normal train/val/test ↔ v4_20 normal) ───────────────────
    normal_set = set(normal_ids)
    def match_report(ids, label):
        present = [i for i in ids if i in normal_set]
        missing = [i for i in ids if i not in normal_set]
        return {"label": label, "total": len(ids), "matched": len(present), "missing": len(missing),
                "missing_list": missing[:5]}

    train_m = match_report(train, "normal_train")
    val_m   = match_report(val,   "normal_val")
    test_m  = match_report(test,  "normal_test")

    # lesion stage1_dev ↔ v4_20 lesion
    lesion_set = set(lesion_ids)
    stage1_ids = [s for s, _ in stage1]
    stage1_matched = [i for i in stage1_ids if i in lesion_set]
    stage1_missing = [i for i in stage1_ids if i not in lesion_set]

    print(f"[매칭] train {train_m['matched']}/{train_m['total']}, "
          f"val {val_m['matched']}/{val_m['total']}, test {test_m['matched']}/{test_m['total']}")
    print(f"[매칭] lesion stage1_dev {len(stage1_matched)}/{len(stage1_ids)}")

    # ── 4. stage2_holdout 미접근 가드 ────────────────────────────────────
    # holdout safe_id는 value 로드 안 함. shape 확인 대상에서 제외.
    holdout_in_v4_lesion = len(holdout & lesion_set)  # 폴더 존재는 셀 수 있음 (value 아님)
    print(f"[가드] v4_20 lesion 중 holdout 폴더 수: {holdout_in_v4_lesion} (value 로드 안 함)")

    # ── 5. shape 확인 (normal 전수 + lesion stage1_dev 전수, holdout 제외) ─
    normal_shapes = {}
    normal_shape_none = 0
    for sid in normal_ids:
        sh = get_shape("normal", sid)
        if sh is None:
            normal_shape_none += 1
        else:
            normal_shapes[str(sh)] = normal_shapes.get(str(sh), 0) + 1

    lesion_shapes = {}
    lesion_shape_none = 0
    for sid in stage1_matched:  # stage1_dev만 (holdout 제외)
        sh = get_shape("lesion", sid)
        if sh is None:
            lesion_shape_none += 1
        else:
            lesion_shapes[str(sh)] = lesion_shapes.get(str(sh), 0) + 1

    # 모든 shape이 (Z,512,512) 형식인지 확인
    def hw_ok(shape_counter):
        for s in shape_counter:
            t = eval(s)
            if len(t) != 3 or t[1] != 512 or t[2] != 512:
                return False
        return True
    normal_hw_ok = hw_ok(normal_shapes)
    lesion_hw_ok = hw_ok(lesion_shapes)
    print(f"[shape] normal 512x512 일관: {normal_hw_ok}, lesion 512x512 일관: {lesion_hw_ok}")

    # ── 6. lesion mask fallback 존재 여부 (참고 기록, 필수 아님) ───────────
    sample_mask = LESION_MASK_FALLBACK_DIR / stage1_ids[0] / LESION_MASK_FALLBACK_FILE
    fallback_mask_available = sample_mask.exists()
    fallback_mask_count = 0
    if LESION_MASK_FALLBACK_DIR.exists():
        for sid in stage1_ids:
            if (LESION_MASK_FALLBACK_DIR / sid / LESION_MASK_FALLBACK_FILE).exists():
                fallback_mask_count += 1
    print(f"[mask] v4_20 외부 lesion mask(C드라이브) stage1_dev 가용: {fallback_mask_count}/{len(stage1_ids)}")

    # ── 7. 판정 ───────────────────────────────────────────────────────────
    issues = []
    if len(normal_ids) != EXPECTED_NORMAL: issues.append(f"normal {len(normal_ids)}≠{EXPECTED_NORMAL}")
    if len(lesion_ids) != EXPECTED_LESION: issues.append(f"lesion {len(lesion_ids)}≠{EXPECTED_LESION}")
    if train_m["missing"] or val_m["missing"] or test_m["missing"]:
        issues.append("normal split ↔ v4_20 매칭 누락 존재")
    if stage1_missing: issues.append(f"lesion stage1_dev 매칭 누락 {len(stage1_missing)}건")
    if not normal_hw_ok or not lesion_hw_ok: issues.append("ROI HW 512x512 불일치 존재")
    if not v4_20_has_lesion_mask:
        issues.append("v4_20 내부에 lesion mask 없음 (refined_roi.npy만) — 사용자 전제 보정 필요")

    # hard fail 조건
    hard_fail = (len(normal_ids) != EXPECTED_NORMAL or len(lesion_ids) != EXPECTED_LESION
                 or train_m["missing"] or val_m["missing"] or test_m["missing"] or stage1_missing
                 or not normal_hw_ok or not lesion_hw_ok)
    if hard_fail:
        verdict = "실패"
    elif not v4_20_has_lesion_mask:
        verdict = "부분통과"  # ROI lock은 성공, lesion mask source 보정 필요
    else:
        verdict = "통과"

    print(f"\n[판정] {verdict}")
    for i in issues:
        print(f"  ⚠ {i}")

    # P-B3 진행 가능: v4_20 lesion ROI(필수) + lesion mask source(외부) 둘 다 있으면 가능
    p_b3_can_proceed = (not stage1_missing) and fallback_mask_count == len(stage1_ids)

    # ── 8. JSON 보고서 ────────────────────────────────────────────────────
    report = {
        "stage": "P-B2.6_v4_20_modeB_source_lock",
        "created": ts,
        "verdict": verdict,
        "user_correction_applied": {
            "model_roi_npy_used": False,
            "e_drive_used": False,
            "roi_0_0_comparison_required": False,
            "official_roi_source": "refined_roi_v4_20_modeB_all_v1",
        },
        "scope": {
            "stage2_holdout_accessed": False,
            "stage2_holdout_value_loaded": False,
            "training": False, "model_forward": False, "scoring": False,
            "threshold_calculated": False, "metrics_calculated": False,
            "existing_files_modified": False,
        },
        "v4_20_inventory": {
            "root": str(V4_20_ROOT),
            "normal_folder_count": len(normal_ids),
            "lesion_folder_count": len(lesion_ids),
            "normal_file_types": normal_files,
            "lesion_file_types": lesion_files,
            "v4_20_has_internal_lesion_mask": v4_20_has_lesion_mask,
            "v4_20_internal_lesion_mask_count": v4_20_lesion_mask_count,
        },
        "official_sources": {
            "normal_train_roi": "refined_roi_v4_20_modeB_all_v1/normal/<safe_id>/refined_roi.npy",
            "lesion_safety_roi": "refined_roi_v4_20_modeB_all_v1/lesion/<safe_id>/refined_roi.npy",
            "lesion_mask_source": "v4_20 내부에 없음 → 외부 원본 lesion_mask_roi_0_0.npy (C드라이브, GT 위치용, ROI 비교 아님)",
            "lesion_mask_fallback_path": str(LESION_MASK_FALLBACK_DIR / "<safe_id>" / LESION_MASK_FALLBACK_FILE),
        },
        "split_matching": {
            "normal_train": train_m, "normal_val": val_m, "normal_test": test_m,
            "lesion_stage1_dev": {"total": len(stage1_ids), "matched": len(stage1_matched),
                                  "missing": len(stage1_missing)},
            "lesion_stage1_dev_nsclc": n_nsclc, "lesion_stage1_dev_msd": n_msd,
        },
        "shape_check": {
            "normal_shape_distribution_count": len(normal_shapes),
            "normal_hw_512_consistent": normal_hw_ok,
            "normal_shape_none": normal_shape_none,
            "lesion_stage1_dev_shape_distribution_count": len(lesion_shapes),
            "lesion_hw_512_consistent": lesion_hw_ok,
            "lesion_shape_none": lesion_shape_none,
            "note": "ROI는 볼륨별 Z 상이(정상). CT shape 매칭은 P-B4 smoke에서 검증.",
        },
        "stage2_holdout": {
            "status": "LOCKED",
            "holdout_folders_in_v4_lesion": holdout_in_v4_lesion,
            "value_loaded": False,
        },
        "lesion_mask_clip_check": {
            "v4_20_internal_lesion_mask": v4_20_has_lesion_mask,
            "verdict": "N/A — v4_20 내부에 lesion mask 부재. clip 여부는 P-B3에서 외부 mask vs refined ROI로 확인",
        },
        "lesion_mask_fallback": {
            "source": "C드라이브 roi_0_0 폴더 lesion_mask_roi_0_0.npy",
            "stage1_dev_available": fallback_mask_count,
            "stage1_dev_total": len(stage1_ids),
            "note": "이건 roi_0_0 ROI 비교가 아니라 병변 GT 위치 마스크. E드라이브 불필요.",
        },
        "dataloader_change": {
            "do_not_expect_model_roi_npy": True,
            "load_external_refined_roi": True,
            "rule": "DataLoader/PathResolver가 ct_hu.npy(원본)와 refined_roi.npy(v4_20)를 별도 경로에서 로드하도록 설계",
        },
        "p_b3_readiness": {
            "lesion_roi_v4_20_available": not stage1_missing,
            "lesion_mask_external_available": fallback_mask_count == len(stage1_ids),
            "e_drive_required": False,
            "can_proceed": p_b3_can_proceed,
        },
        "prior_unresolved_reclassified": [
            {"item": "model_roi.npy 정체 확인", "old_status": "P-B2/P-B2.5 미결",
             "new_status": "branch scope 밖 (ROI input 아님, 사용 안 함)"},
            {"item": "E드라이브 /mnt/e 마운트", "old_status": "P-B2/P-B2.5 blocker",
             "new_status": "branch scope 밖 (v4_20 + C드라이브 mask로 충분)"},
            {"item": "lesion roi_0_0 대비 ROI coverage 비교", "old_status": "P-B2.5 부분 수행",
             "new_status": "참고 분석으로 보류 (학습 필수 조건 아님)"},
        ],
        "issues": issues,
    }
    with open(REPORT_DIR / "p_b2_6_v4_20_source_lock.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ── 9. CSV ────────────────────────────────────────────────────────────
    inv_rows = [
        {"group": "normal", "folder_count": len(normal_ids),
         "file_types": ";".join(normal_files.keys()),
         "has_lesion_mask": False, "roi_file": "refined_roi.npy"},
        {"group": "lesion", "folder_count": len(lesion_ids),
         "file_types": ";".join(lesion_files.keys()),
         "has_lesion_mask": v4_20_has_lesion_mask, "roi_file": "refined_roi.npy"},
    ]
    save_csv(inv_rows, REPORT_DIR / "v4_20_asset_inventory.csv")

    path_rules = [
        {"purpose": "normal_train_roi", "split": "train(290)",
         "path_rule": "v4_20/normal/<safe_id>/refined_roi.npy", "source": "v4_20", "required": "yes"},
        {"purpose": "normal_val_roi", "split": "val(36)",
         "path_rule": "v4_20/normal/<safe_id>/refined_roi.npy", "source": "v4_20", "required": "yes"},
        {"purpose": "normal_test_roi", "split": "test(36)",
         "path_rule": "v4_20/normal/<safe_id>/refined_roi.npy", "source": "v4_20", "required": "yes"},
        {"purpose": "lesion_safety_roi", "split": "stage1_dev(154)",
         "path_rule": "v4_20/lesion/<safe_id>/refined_roi.npy", "source": "v4_20", "required": "yes"},
        {"purpose": "lesion_gt_mask", "split": "stage1_dev(154)",
         "path_rule": "C:/.../roi0_0_..._usable_only_v1/volumes_npy/<safe_id>/lesion_mask_roi_0_0.npy",
         "source": "C드라이브 원본(GT 위치, ROI 비교 아님)", "required": "P-B3만"},
        {"purpose": "ct_hu", "split": "all",
         "path_rule": "원본 ct_hu.npy (normal: v2_tslungguard_nochest / lesion: roi_0_0 폴더)",
         "source": "원본", "required": "P-B4 학습부터"},
    ]
    save_csv(path_rules, REPORT_DIR / "v4_20_path_rules.csv")

    scope_rows = [
        {"item": "model_roi.npy 사용", "before": "P-B2/P-B2.5 비교 시도", "after": "사용 안 함 (scope 밖)"},
        {"item": "E드라이브 /mnt/e", "before": "P-B2/P-B2.5 blocker", "after": "사용 안 함 (scope 밖)"},
        {"item": "roi_0_0 ROI coverage 비교", "before": "P-B2.5 부분 수행", "after": "참고 보류 (필수 아님)"},
        {"item": "lesion mask source", "before": "v4_20 내부 가정", "after": "v4_20에 없음 → C드라이브 원본 GT mask"},
        {"item": "공식 ROI source", "before": "혼재", "after": "refined_roi_v4_20_modeB_all_v1 단일 lock"},
    ]
    save_csv(scope_rows, REPORT_DIR / "v4_20_branch_scope_correction.csv")

    # ── 10. MD ────────────────────────────────────────────────────────────
    md = []
    md.append("# P-B2.6 v4_20_modeB ROI Source Lock + Branch Correction\n")
    md.append(f"- 생성일: {ts}")
    md.append(f"- 판정: **{verdict}**\n")
    md.append("---\n")
    md.append("## 0. 사용자 정정 반영\n")
    md.append("| 항목 | 반영 |")
    md.append("|------|------|")
    md.append("| model_roi.npy 사용 | **안 함** ✅ |")
    md.append("| E드라이브 `/mnt/e` 사용 | **안 함** ✅ |")
    md.append("| roi_0_0 비교 필수 조건 | **제거** (참고 분석으로 보류) ✅ |")
    md.append("| 공식 ROI source | **refined_roi_v4_20_modeB_all_v1 단일 lock** ✅ |\n")
    md.append("---\n")
    md.append("## 1. v4_20 asset inventory\n")
    md.append("| group | 폴더 수 | 파일 종류 | lesion mask 내장 |")
    md.append("|-------|---------|-----------|------------------|")
    md.append(f"| normal | {len(normal_ids)} | {';'.join(normal_files.keys())} | — |")
    md.append(f"| lesion | {len(lesion_ids)} | {';'.join(lesion_files.keys())} | **{'있음' if v4_20_has_lesion_mask else '없음'}** |\n")
    md.append("> ⚠ **핵심 보정**: v4_20 내부에는 `refined_roi.npy`(ROI 마스크)만 있고, **lesion mask는 없습니다**.")
    md.append("> 사용자 전제(\"v4_20 안에 병변 mask도 있음\")와 다르므로, 병변 GT mask는 외부 원본을 사용합니다.\n")
    md.append("---\n")
    md.append("## 2. 공식 source 확정\n")
    md.append("| 용도 | path rule | source |")
    md.append("|------|-----------|--------|")
    md.append("| normal train/val/test ROI | `v4_20/normal/<safe_id>/refined_roi.npy` | v4_20 |")
    md.append("| lesion safety ROI | `v4_20/lesion/<safe_id>/refined_roi.npy` | v4_20 |")
    md.append("| lesion GT mask | `C:/.../roi0_0_usable_only_v1/volumes_npy/<safe_id>/lesion_mask_roi_0_0.npy` | C드라이브 원본 |")
    md.append("| CT (학습 입력) | 원본 ct_hu.npy | 원본 (P-B4부터) |\n")
    md.append("> lesion GT mask는 **ROI 비교가 아니라 병변 위치 GT**입니다. E드라이브 불필요(C드라이브 존재).\n")
    md.append("---\n")
    md.append("## 3. split ↔ v4_20 매칭\n")
    md.append("| split | 총원 | 매칭 | 누락 |")
    md.append("|-------|------|------|------|")
    md.append(f"| normal_train | {train_m['total']} | {train_m['matched']} | {train_m['missing']} |")
    md.append(f"| normal_val | {val_m['total']} | {val_m['matched']} | {val_m['missing']} |")
    md.append(f"| normal_test | {test_m['total']} | {test_m['matched']} | {test_m['missing']} |")
    md.append(f"| lesion_stage1_dev | {len(stage1_ids)} | {len(stage1_matched)} | {len(stage1_missing)} |")
    md.append(f"| (stage1_dev 구성) | NSCLC {n_nsclc} / MSD {n_msd} | | |\n")
    md.append("---\n")
    md.append("## 4. shape 확인\n")
    md.append("| group | shape 종류 수 | 512×512 일관 | 로드 실패 |")
    md.append("|-------|---------------|--------------|-----------|")
    md.append(f"| normal (362) | {len(normal_shapes)} | {normal_hw_ok} | {normal_shape_none} |")
    md.append(f"| lesion stage1_dev (154) | {len(lesion_shapes)} | {lesion_hw_ok} | {lesion_shape_none} |\n")
    md.append("> ROI Z는 볼륨별 상이(정상). CT shape 매칭은 P-B4 smoke에서 검증.")
    md.append("> stage2_holdout 154명 ROI shape는 **읽지 않음** (LOCKED).\n")
    md.append("---\n")
    md.append("## 5. lesion mask clipped 확인\n")
    md.append("- v4_20 내부에 lesion mask 없음 → 이 단계에서 clip 판정 불가 (N/A)")
    md.append("- 외부 GT mask(C드라이브) 사용 시, P-B3에서 `mask & refined_roi` vs `mask` 비교로 clip/손실 측정\n")
    md.append("---\n")
    md.append("## 6. DataLoader 수정 방향\n")
    md.append("- `model_roi.npy` 기대 구조 **사용 안 함**")
    md.append("- ct_hu.npy(원본)와 refined_roi.npy(v4_20)를 **별도 경로에서 로드**")
    md.append("- PathResolver에 `refined_roi_root` 파라미터 추가 필요 (P-B4 설계)\n")
    md.append("---\n")
    md.append("## 7. P-B2/P-B2.5 미결 사항 재분류\n")
    md.append("| 항목 | 이전 | 변경 |")
    md.append("|------|------|------|")
    md.append("| model_roi.npy 정체 | 미결 | branch scope 밖 (ROI input 아님) |")
    md.append("| E드라이브 마운트 | blocker | branch scope 밖 (불필요) |")
    md.append("| roi_0_0 coverage 비교 | 부분 수행 | 참고 보류 (필수 아님) |\n")
    md.append("---\n")
    md.append("## 8. stage2_holdout\n")
    md.append("- 상태: **LOCKED** 유지")
    md.append(f"- v4_20 lesion 중 holdout 폴더 수 {holdout_in_v4_lesion} (폴더 카운트만, value 로드 0)\n")
    md.append("---\n")
    md.append("## 9. 미실행 확인\n")
    md.append("- 학습/forward/scoring/threshold/metrics: **미실행** ✅")
    md.append("- stage2_holdout value 접근: **없음** ✅")
    md.append("- 기존 roi_0_0 / EfficientNet-B0 / P-B1·P-B2·P-B2.5 결과: **무수정** ✅\n")
    md.append("---\n")
    md.append("## 10. P-B3 v4_20-only lesion safety validation 가능 여부\n")
    md.append(f"- **가능: {p_b3_can_proceed}**")
    md.append(f"  - lesion ROI(v4_20) stage1_dev: {len(stage1_matched)}/{len(stage1_ids)}")
    md.append(f"  - lesion GT mask(C드라이브) stage1_dev: {fallback_mask_count}/{len(stage1_ids)}")
    md.append("  - E드라이브 불필요\n")
    md.append("### P-B3 프롬프트 초안\n")
    md.append("```")
    md.append("P-B3 v4_20-only lesion safety validation on stage1_dev only 진행해줘.")
    md.append("")
    md.append("목표: v4_20 lesion refined ROI가 병변(GT mask)을 얼마나 보존/손실하는지 검증.")
    md.append("ROI source: refined_roi_v4_20_modeB_all_v1/lesion/<id>/refined_roi.npy (v4_20 고정)")
    md.append("GT mask: C드라이브 roi0_0 폴더 lesion_mask_roi_0_0.npy (병변 위치 GT, ROI 비교 아님)")
    md.append("")
    md.append("금지: 학습/forward/scoring/threshold/metrics, stage2_holdout, model_roi.npy, E드라이브")
    md.append("")
    md.append("검증 (stage1_dev 154명만):")
    md.append("1. mask & refined_roi voxel / mask voxel → 병변 보존율 (per patient)")
    md.append("2. 병변 손실 분포 (중앙값, p1, p5, min)")
    md.append("3. 완전 손실(보존율 0) 또는 극저 보존 케이스 목록")
    md.append("4. pleura-adjacent / lower_peripheral 병변 보존율 (position_bin)")
    md.append("5. clip 여부: mask가 refined_roi 밖에 얼마나 존재하는지")
    md.append("")
    md.append("출력: outputs/reports/p_b3_v4_20_lesion_safety_validation/")
    md.append("```\n")
    md.append("---\n")
    md.append("## 11. 최종 판정\n")
    md.append(f"- **{verdict}**")
    md.append("- v4_20 ROI source lock: **완료** ✅")
    md.append("- model_roi / E드라이브 / roi_0_0 의존성: **필수 조건에서 제거** ✅")
    md.append(f"- 보정: v4_20 내부 lesion mask 부재 → 외부 GT mask 사용으로 명시")
    md.append(f"- P-B3 진행 가능: **{p_b3_can_proceed}** (E드라이브 없이)")

    with open(REPORT_DIR / "p_b2_6_v4_20_source_lock.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"\n[저장] {REPORT_DIR}")
    print(f"[완료] 판정: {verdict}")
    return 0 if verdict != "실패" else 1


if __name__ == "__main__":
    sys.exit(main())
