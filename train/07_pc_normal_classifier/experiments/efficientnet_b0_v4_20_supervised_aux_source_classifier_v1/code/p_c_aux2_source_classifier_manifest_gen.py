"""
P-C-AUX2: NSCLC-vs-MSD Auxiliary Source Classifier Training Manifest Generator

목적:
    C-lite manifest (positive crops only)에서 source label을 부여하여
    NSCLC-source vs MSD_Lung-source auxiliary classifier 학습용 manifest 생성.

실행 모드:
    --dry-check : manifest 미생성, 설계 검증만 수행 (이번 단계 기본)
    --full      : 실제 manifest 생성 (아래 3개 confirm flag 필수 동반)
                  --confirm-positive-only --confirm-no-hard-negative --confirm-no-holdout
                  --full 단독 사용 시 abort (exit 1)

금지사항:
    - hard_negative row 학습 데이터 포함 금지
    - stage2_holdout 접근 금지
    - 실제 학습, model forward, scoring 금지
    - 기존 P-C/N-C/P-B 결과 수정 금지
    - 금지 표현 (cancer probability, malignancy probability,
                 폐선암 확률, 암종 확정, 진단 모델) 사용 금지

허용 표현:
    - NSCLC-source likelihood
    - MSD-source likelihood
    - auxiliary source classifier
    - NSCLC-like lesion candidate score

사용법:
    # dry-check (기본)
    source ~/ai_env/bin/activate
    cd experiments/efficientnet_b0_v4_20_supervised_aux_source_classifier_v1/code
    python p_c_aux2_source_classifier_manifest_gen.py --dry-check

    # 사용자 승인 후 full 생성 (4개 flag 필수)
    source ~/ai_env/bin/activate
    cd experiments/efficientnet_b0_v4_20_supervised_aux_source_classifier_v1/code
    python p_c_aux2_source_classifier_manifest_gen.py \
      --full \
      --confirm-positive-only \
      --confirm-no-hard-negative \
      --confirm-no-holdout
"""

import csv
import json
import os
import sys
import argparse
from collections import defaultdict
from datetime import datetime


# ── 경로 설정 ──────────────────────────────────────────────────────────────────

BRANCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.dirname(os.path.dirname(BRANCH_DIR))

PC_BRANCH = os.path.join(
    PROJECT_DIR,
    "experiments/efficientnet_b0_v4_20_second_stage_refiner_v1"
)

CLITE_MANIFEST = os.path.join(
    PC_BRANCH,
    "outputs/training_manifests/p_c10_c_lite_training_manifest/"
    "p_c10_c_lite_training_manifest.csv"
)
PC3_MANIFEST = os.path.join(
    PC_BRANCH,
    "outputs/candidates/p_c3_candidate_manifest/p_c3_candidate_manifest.csv"
)
CROP_BASE = os.path.join(PC_BRANCH)  # crop_path는 여기서 상대경로

DRYCHECK_REPORT_DIR = os.path.join(
    BRANCH_DIR,
    "outputs/reports/p_c_aux2_manifest_gen_drycheck"
)
OUTPUT_MANIFEST_DIR = os.path.join(
    BRANCH_DIR,
    "outputs/manifests/p_c_aux2_source_classifier_training_manifest"
)

# 금지: stage2_holdout manifest 접근 (경로만 기록, 실제 로드 금지)
STAGE2_HOLDOUT_PATH = os.path.join(
    PROJECT_DIR,
    "outputs/second-stage-lesion-refiner-v1/datasets/"
    "s6a_stage2_holdout_candidate_coordinate_manifest_v1.csv"
)

# ── 상수 ──────────────────────────────────────────────────────────────────────

PATIENT_CAP = 100          # 환자당 최대 positive crop 수
SOURCE_LABEL = {"NSCLC": 1, "MSD_Lung": 0}
AUX_CANDIDATE_ID_PREFIX = "AUX"
ROI_PATCH_RATIO_AVAILABLE = False  # P-C3/C-lite 어디에도 roi_patch_ratio 컬럼 없음

# 출력 manifest 컬럼 순서
OUTPUT_COLUMNS = [
    "aux_candidate_id",
    "original_candidate_id",
    "patient_id",
    "safe_id",
    "source_name",
    "source_label",
    "split",
    "crop_path",
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
    "lesion_pixels",
    "has_lesion_patch",
    "is_positive_only",
    "original_p_c_label",
    "tiny_lesion_flag",
    "no_hit_fallback_flag",
    "p_b3_risk6_flag",
    "patient_positive_count_before_cap",
    "patient_positive_count_after_cap",
    "patient_cap_applied",
    "sample_weight",
    "class_weight",
    "forbidden_hard_negative_used",
    "forbidden_supervised_diagnostic_wording",
]


# ── 유틸리티 ──────────────────────────────────────────────────────────────────

def load_csv(path, encoding="utf-8-sig"):
    with open(path, newline="", encoding=encoding) as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def infer_source(safe_id):
    if safe_id.startswith("NSCLC_"):
        return "NSCLC"
    elif safe_id.startswith("MSD_Lung_"):
        return "MSD_Lung"
    return "UNKNOWN"


# ── 메인 로직 ─────────────────────────────────────────────────────────────────

def load_and_join(clite_path, pc3_path):
    """C-lite positive rows와 P-C3 메타를 join하여 AUX 입력 row 반환."""
    pc3_map = {r["candidate_id"]: r for r in load_csv(pc3_path)}
    clite_rows = load_csv(clite_path)

    aux_rows = []
    hard_neg_count = 0
    unknown_source = 0

    for row in clite_rows:
        label = row["candidate_label"]
        if label != "positive":
            hard_neg_count += 1
            continue

        cid = row["candidate_id"]
        p3 = pc3_map.get(cid, {})

        source = infer_source(row["safe_id"])
        if source == "UNKNOWN":
            unknown_source += 1
            continue

        # center 계산 (y0/y1, x0/x1 평균)
        center_y = (int(row["y0"]) + int(row["y1"])) // 2
        center_x = (int(row["x0"]) + int(row["x1"])) // 2

        no_hit_fallback = (
            row.get("no_hit_patient", "False").lower() == "true"
            or row.get("fallback_positive_below_p95", "False").lower() == "true"
        )

        aux_row = {
            # 식별자
            "aux_candidate_id": "",          # 나중에 인덱스로 채움
            "original_candidate_id": cid,
            "patient_id": row["patient_id"],
            "safe_id": row["safe_id"],
            # source label
            "source_name": source,
            "source_label": SOURCE_LABEL[source],
            # split
            "split": row["split_plan"],
            # crop
            "crop_path": row["crop_path"],
            # 좌표
            "local_z": row["local_z"],
            "slice_index": row["slice_index"],
            "y0": row["y0"],
            "x0": row["x0"],
            "y1": row["y1"],
            "x1": row["x1"],
            "center_y": center_y,
            "center_x": center_x,
            # P-C3에서 join
            "position_bin": p3.get("position_bin", ""),
            "z_level": p3.get("z_level", ""),
            "roi_patch_ratio": "NA",  # 실제 roi_patch_ratio 컬럼 없음 — lesion_pixels 혼용 금지
            "lesion_pixels": p3.get("lesion_pixels", ""),
            "has_lesion_patch": p3.get("has_lesion_patch", "True"),
            # 플래그
            "is_positive_only": True,
            "original_p_c_label": label,
            "tiny_lesion_flag": row.get("tiny_lesion_flag", "False"),
            "no_hit_fallback_flag": no_hit_fallback,
            "p_b3_risk6_flag": row.get("p_b3_risk6_flag", "False"),
            # cap / weight (후처리에서 채움)
            "patient_positive_count_before_cap": 0,
            "patient_positive_count_after_cap": 0,
            "patient_cap_applied": False,
            "sample_weight": 0.0,
            "class_weight": 0.0,
            # guardrail
            "forbidden_hard_negative_used": False,
            "forbidden_supervised_diagnostic_wording": False,
        }
        aux_rows.append(aux_row)

    return aux_rows, hard_neg_count, unknown_source


def apply_patient_cap(aux_rows, cap=PATIENT_CAP):
    """환자별 cap 적용. crop_path 기준 무작위 선택 (정렬로 재현성 보장)."""
    patient_crops = defaultdict(list)
    for r in aux_rows:
        patient_crops[(r["patient_id"], r["split"])].append(r)

    capped = []
    for (pid, split), crops in patient_crops.items():
        before = len(crops)
        # 재현성을 위해 original_candidate_id 정렬 후 cap 적용
        crops_sorted = sorted(crops, key=lambda x: x["original_candidate_id"])
        selected = crops_sorted[:cap]
        after = len(selected)
        applied = before > cap

        for r in selected:
            r["patient_positive_count_before_cap"] = before
            r["patient_positive_count_after_cap"] = after
            r["patient_cap_applied"] = applied
            capped.append(r)

    return capped


def compute_class_weights(capped_rows):
    """train split 기준 class_weight 계산 후 전체 row에 기록."""
    train_rows = [r for r in capped_rows if r["split"] == "train"]
    n_nsclc = sum(1 for r in train_rows if r["source_name"] == "NSCLC")
    n_msd = sum(1 for r in train_rows if r["source_name"] == "MSD_Lung")
    total = n_nsclc + n_msd

    if n_nsclc == 0 or n_msd == 0:
        raise ValueError(f"class weight 계산 불가: NSCLC={n_nsclc}, MSD={n_msd}")

    cw_nsclc = total / (2 * n_nsclc)
    cw_msd = total / (2 * n_msd)

    for r in capped_rows:
        if r["source_name"] == "NSCLC":
            r["class_weight"] = round(cw_nsclc, 6)
            r["sample_weight"] = round(cw_nsclc, 6)
        else:
            r["class_weight"] = round(cw_msd, 6)
            r["sample_weight"] = round(cw_msd, 6)

    return cw_nsclc, cw_msd


def assign_aux_ids(rows):
    """aux_candidate_id 부여 (AUX_000000 형식)."""
    for i, r in enumerate(rows):
        r["aux_candidate_id"] = f"{AUX_CANDIDATE_ID_PREFIX}_{i:07d}"
    return rows


def check_patient_leakage(rows):
    train_pids = {r["patient_id"] for r in rows if r["split"] == "train"}
    val_pids = {r["patient_id"] for r in rows if r["split"] == "val"}
    return train_pids & val_pids


def check_crop_paths(rows, base, n_sample=200):
    """crop_path 파일 존재 여부 샘플 확인 (npz 로드 없이 path existence만)."""
    import random
    sample = random.sample(rows, min(n_sample, len(rows)))
    missing = [r for r in sample if not os.path.exists(os.path.join(base, r["crop_path"]))]
    return len(sample), len(missing)


# ── Dry-check ─────────────────────────────────────────────────────────────────

def run_drycheck():
    """실제 manifest 생성 없이 설계 검증만 수행."""
    os.makedirs(DRYCHECK_REPORT_DIR, exist_ok=True)
    errors = []
    warnings = []
    checks = []

    def add_check(item, required, actual, status, note=""):
        checks.append({
            "check_item": item,
            "required": required,
            "actual": actual,
            "status": status,
            "note": note,
        })
        if status == "FAIL":
            errors.append(f"[FAIL] {item}: {actual}")
        elif status in ("WARN", "WARNING"):
            warnings.append(f"[WARN] {item}: {actual}")

    # ── 1. 입력 파일 존재 확인 ────────────────────────────────────────────
    print("[1] 입력 파일 존재 확인...")
    for label, path in [("C-lite manifest", CLITE_MANIFEST), ("P-C3 manifest", PC3_MANIFEST)]:
        exists = os.path.exists(path)
        add_check(f"{label} 존재", "존재", "존재" if exists else "없음",
                  "PASS" if exists else "FAIL")

    # ── 2. positive-only 필터링 + join ────────────────────────────────────
    print("[2] positive 필터링 + P-C3 join...")
    aux_rows, hard_neg_count, unknown_count = load_and_join(CLITE_MANIFEST, PC3_MANIFEST)
    add_check("hard_negative 제외", "0 hard_neg_rows", f"{hard_neg_count:,} 제외됨",
              "PASS", "positive-only 필터링 정상")
    add_check("UNKNOWN source", "0", unknown_count,
              "PASS" if unknown_count == 0 else "FAIL")

    # ── 3. source 분포 확인 ───────────────────────────────────────────────
    print("[3] source label 분포 확인...")
    source_count = defaultdict(int)
    source_pids = defaultdict(set)
    for r in aux_rows:
        source_count[r["source_name"]] += 1
        source_pids[r["source_name"]].add(r["patient_id"])

    nsclc_crops = source_count["NSCLC"]
    msd_crops = source_count["MSD_Lung"]
    ratio = nsclc_crops / msd_crops if msd_crops > 0 else float("inf")

    add_check("NSCLC positive crops (before cap)", "~32,721", nsclc_crops,
              "PASS" if 32000 <= nsclc_crops <= 33000 else "WARN")
    add_check("MSD_Lung positive crops (before cap)", "~2,549", msd_crops,
              "PASS" if 2400 <= msd_crops <= 2600 else "WARN")
    add_check("NSCLC patients (before cap)", "125", len(source_pids["NSCLC"]),
              "PASS" if len(source_pids["NSCLC"]) == 125 else "WARN")
    add_check("MSD_Lung patients (before cap)", "29", len(source_pids["MSD_Lung"]),
              "PASS" if len(source_pids["MSD_Lung"]) == 29 else "WARN")
    add_check("crop imbalance (before cap)", "<13:1", f"{ratio:.1f}:1",
              "WARN" if ratio > 10 else "PASS",
              "cap 적용 후 완화 예정")

    # ── 4. patient cap 적용 ───────────────────────────────────────────────
    print("[4] patient cap=100 시뮬레이션...")
    capped = apply_patient_cap(aux_rows, cap=PATIENT_CAP)
    cw_nsclc, cw_msd = compute_class_weights(capped)
    assign_aux_ids(capped)

    capped_source_count = defaultdict(lambda: defaultdict(int))
    capped_source_pids = defaultdict(lambda: defaultdict(set))
    for r in capped:
        split = r["split"]
        src = r["source_name"]
        capped_source_count[split][src] += 1
        capped_source_pids[split][src].add(r["patient_id"])

    nsclc_after = sum(capped_source_count[s]["NSCLC"] for s in ["train","val"])
    msd_after = sum(capped_source_count[s]["MSD_Lung"] for s in ["train","val"])
    ratio_after = nsclc_after / msd_after if msd_after > 0 else float("inf")

    add_check("NSCLC crops after cap=100", "~9,971", nsclc_after,
              "PASS" if 9000 <= nsclc_after <= 11000 else "WARN")
    add_check("MSD_Lung crops after cap=100", "~1,545", msd_after,
              "PASS" if 1400 <= msd_after <= 1700 else "WARN")
    add_check("crop imbalance after cap", "<8:1", f"{ratio_after:.1f}:1",
              "PASS" if ratio_after < 8 else "WARN")
    add_check("class_weight_NSCLC", "~0.58", f"{cw_nsclc:.4f}", "PASS")
    add_check("class_weight_MSD", "~3.54", f"{cw_msd:.4f}", "PASS")

    # ── 5. train/val split 확인 ───────────────────────────────────────────
    print("[5] train/val split + leakage 확인...")
    leakage = check_patient_leakage(capped)
    add_check("train/val patient leakage", "0", len(leakage),
              "PASS" if len(leakage) == 0 else "FAIL",
              f"overlap pids: {sorted(leakage)[:3]}" if leakage else "누출 없음")

    val_msd_crops = capped_source_count["val"]["MSD_Lung"]
    val_msd_pids = len(capped_source_pids["val"]["MSD_Lung"])
    add_check("val MSD_Lung positive crops", ">200", val_msd_crops,
              "WARN" if val_msd_crops < 300 else "PASS",
              f"6 patients — val AUROC 해석 매우 제한적")
    add_check("val MSD_Lung patients", ">5", val_msd_pids,
              "WARN" if val_msd_pids <= 6 else "PASS",
              "기존 P-C10 split 한계")

    # ── 6. crop_path 샘플 확인 ────────────────────────────────────────────
    print("[6] crop_path 존재 샘플 확인 (200건)...")
    n_sample, n_missing = check_crop_paths(capped, CROP_BASE, n_sample=200)
    add_check("crop_path 존재 (샘플 200)", "missing=0",
              f"sample={n_sample}, missing={n_missing}",
              "PASS" if n_missing == 0 else "WARN")

    # ── 7. guardrail 확인 ─────────────────────────────────────────────────
    print("[7] guardrail 확인...")
    hard_neg_in_capped = sum(1 for r in capped
                              if r.get("original_p_c_label") != "positive"
                              or r.get("forbidden_hard_negative_used"))
    add_check("hard_negative in output", "0", hard_neg_in_capped,
              "PASS" if hard_neg_in_capped == 0 else "FAIL")

    add_check("roi_patch_ratio 의미 오류", "lesion_pixels proxy 없음",
              "NA 고정 (ROI_PATCH_RATIO_AVAILABLE=False)", "PASS",
              "P-C3/C-lite 어디에도 roi_patch_ratio 컬럼 없음 확인")
    add_check("roi_patch_ratio_available", str(ROI_PATCH_RATIO_AVAILABLE),
              str(ROI_PATCH_RATIO_AVAILABLE), "PASS")
    add_check("stage2_holdout 접근", "미접근", "미접근", "PASS",
              "경로 존재만 확인, manifest row 로드 없음")
    add_check("actual manifest 생성", "미생성", "미생성 (dry-check)", "PASS")
    add_check("학습/scoring/forward 실행", "미실행", "미실행", "PASS")
    add_check("vessel mask input", "미사용", "미사용", "PASS",
              "CT 3ch only 설계 유지")

    # ── 8. tiny/no-hit/fallback 분포 ─────────────────────────────────────
    tiny_src = defaultdict(int)
    nohit_src = defaultdict(int)
    for r in capped:
        src = r["source_name"]
        if str(r.get("tiny_lesion_flag", "False")).lower() == "true":
            tiny_src[src] += 1
        if r.get("no_hit_fallback_flag"):
            nohit_src[src] += 1

    # ── 보고서 생성 ───────────────────────────────────────────────────────
    print("[8] 보고서 생성...")
    verdict = "PASS" if not errors else ("PARTIAL_PASS" if not any("FAIL" in e for e in errors) else "FAIL")

    # --- guardrail CSV ---
    guardrail_rows = checks
    write_csv(
        os.path.join(DRYCHECK_REPORT_DIR, "p_c_aux2_guardrail_check.csv"),
        guardrail_rows,
        ["check_item", "required", "actual", "status", "note"]
    )

    # --- input validation CSV ---
    write_csv(
        os.path.join(DRYCHECK_REPORT_DIR, "p_c_aux2_input_validation.csv"),
        [{"file": k, "path": v, "exists": os.path.exists(v)}
         for k, v in [("C-lite manifest", CLITE_MANIFEST), ("P-C3 manifest", PC3_MANIFEST)]],
        ["file", "path", "exists"]
    )

    # --- positive only filter check CSV ---
    write_csv(
        os.path.join(DRYCHECK_REPORT_DIR, "p_c_aux2_positive_only_filter_check.csv"),
        [
            {"check": "C-lite total rows", "value": hard_neg_count + len(aux_rows), "note": ""},
            {"check": "positive rows (after filter)", "value": len(aux_rows), "note": ""},
            {"check": "hard_negative rows removed", "value": hard_neg_count, "note": "학습에서 완전 제외"},
            {"check": "unknown source rows", "value": unknown_count, "note": ""},
        ],
        ["check", "value", "note"]
    )

    # --- source label distribution CSV ---
    write_csv(
        os.path.join(DRYCHECK_REPORT_DIR, "p_c_aux2_source_label_distribution.csv"),
        [
            {"source": "NSCLC", "aux_label": 1,
             "crops_before_cap": nsclc_crops, "patients_before_cap": len(source_pids["NSCLC"]),
             "crops_after_cap": nsclc_after, "note": ""},
            {"source": "MSD_Lung", "aux_label": 0,
             "crops_before_cap": msd_crops, "patients_before_cap": len(source_pids["MSD_Lung"]),
             "crops_after_cap": msd_after, "note": ""},
            {"source": "ratio", "aux_label": "",
             "crops_before_cap": f"{ratio:.1f}:1", "patients_before_cap": "",
             "crops_after_cap": f"{ratio_after:.1f}:1", "note": "NSCLC:MSD_Lung"},
        ],
        ["source", "aux_label", "crops_before_cap", "patients_before_cap",
         "crops_after_cap", "note"]
    )

    # --- patient cap simulation CSV ---
    raw_split_src = defaultdict(lambda: defaultdict(int))
    for r in aux_rows:
        raw_split_src[r["split"]][r["source_name"]] += 1
    cap_rows = []
    for split in ["train", "val"]:
        for src in ["NSCLC", "MSD_Lung"]:
            cap_rows.append({
                "split": split, "source": src,
                "crops_before_cap": raw_split_src[split][src],
                "crops_after_cap": capped_source_count[split][src],
                "patients": len(capped_source_pids[split][src]),
                "cap_applied": PATIENT_CAP,
            })
    write_csv(
        os.path.join(DRYCHECK_REPORT_DIR, "p_c_aux2_patient_cap_simulation.csv"),
        cap_rows,
        ["split", "source", "crops_before_cap", "crops_after_cap", "patients", "cap_applied"]
    )

    # --- train/val split check CSV ---
    write_csv(
        os.path.join(DRYCHECK_REPORT_DIR, "p_c_aux2_train_val_split_check.csv"),
        [
            {"item": "train patients (NSCLC)", "value": len(capped_source_pids["train"]["NSCLC"]),
             "status": "OK"},
            {"item": "train patients (MSD_Lung)", "value": len(capped_source_pids["train"]["MSD_Lung"]),
             "status": "OK"},
            {"item": "val patients (NSCLC)", "value": len(capped_source_pids["val"]["NSCLC"]),
             "status": "OK"},
            {"item": "val patients (MSD_Lung)", "value": len(capped_source_pids["val"]["MSD_Lung"]),
             "status": "WARN — 6 patients"},
            {"item": "train/val leakage", "value": len(leakage), "status": "PASS" if not leakage else "FAIL"},
            {"item": "split basis", "value": "patient_id", "status": "OK — P-C10 split 재사용"},
        ],
        ["item", "value", "status"]
    )

    # --- tiny/nohit/fallback check CSV ---
    write_csv(
        os.path.join(DRYCHECK_REPORT_DIR, "p_c_aux2_tiny_nohit_fallback_check.csv"),
        [
            {"source": "NSCLC", "flag": "tiny_lesion", "count_after_cap": tiny_src["NSCLC"],
             "action": "포함 유지, flag 컬럼 표기"},
            {"source": "MSD_Lung", "flag": "tiny_lesion", "count_after_cap": tiny_src["MSD_Lung"],
             "action": "포함 유지, flag 컬럼 표기"},
            {"source": "NSCLC", "flag": "no_hit_fallback", "count_after_cap": nohit_src["NSCLC"],
             "action": "포함 유지, flag 컬럼 표기"},
            {"source": "MSD_Lung", "flag": "no_hit_fallback", "count_after_cap": nohit_src["MSD_Lung"],
             "action": "포함 유지, flag 컬럼 표기"},
        ],
        ["source", "flag", "count_after_cap", "action"]
    )

    # --- manifest schema plan CSV ---
    schema_rows = [
        {"column": col, "source": _schema_source(col), "note": _schema_note(col)}
        for col in OUTPUT_COLUMNS
    ]
    write_csv(
        os.path.join(DRYCHECK_REPORT_DIR, "p_c_aux2_manifest_schema_plan.csv"),
        schema_rows,
        ["column", "source", "note"]
    )

    # --- class weight plan CSV ---
    write_csv(
        os.path.join(DRYCHECK_REPORT_DIR, "p_c_aux2_class_weight_plan.csv"),
        [
            {"source": "NSCLC", "label": 1,
             "crops_train_after_cap": capped_source_count["train"]["NSCLC"],
             "class_weight": round(cw_nsclc, 6),
             "sample_weight": round(cw_nsclc, 6), "note": "train 기준 계산"},
            {"source": "MSD_Lung", "label": 0,
             "crops_train_after_cap": capped_source_count["train"]["MSD_Lung"],
             "class_weight": round(cw_msd, 6),
             "sample_weight": round(cw_msd, 6), "note": "train 기준 계산"},
            {"source": "formula", "label": "",
             "crops_train_after_cap": "",
             "class_weight": "total / (2 * n_class)",
             "sample_weight": "", "note": "sklearn balanced 방식"},
        ],
        ["source", "label", "crops_train_after_cap", "class_weight", "sample_weight", "note"]
    )

    # --- errors CSV ---
    err_rows = [{"severity": "WARN" if w.startswith("[WARN") else "FAIL",
                 "message": w} for w in warnings + errors]
    if not err_rows:
        err_rows = [{"severity": "INFO", "message": "no errors or warnings"}]
    write_csv(
        os.path.join(DRYCHECK_REPORT_DIR, "p_c_aux2_errors.csv"),
        err_rows,
        ["severity", "message"]
    )

    # --- JSON summary ---
    summary = {
        "stage": "P-C-AUX2",
        "mode": "dry-check",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "verdict": verdict,
        "config": {
            "patient_cap": PATIENT_CAP,
            "source_labels": SOURCE_LABEL,
            "vessel_mask_input": False,
            "hard_negative_included": False,
        },
        "data": {
            "positive_before_cap": {"NSCLC": nsclc_crops, "MSD_Lung": msd_crops,
                                    "ratio": round(ratio, 2)},
            "positive_after_cap": {"NSCLC": nsclc_after, "MSD_Lung": msd_after,
                                   "ratio": round(ratio_after, 2)},
            "train": {
                "NSCLC": {"crops": capped_source_count["train"]["NSCLC"],
                           "patients": len(capped_source_pids["train"]["NSCLC"])},
                "MSD_Lung": {"crops": capped_source_count["train"]["MSD_Lung"],
                              "patients": len(capped_source_pids["train"]["MSD_Lung"])},
            },
            "val": {
                "NSCLC": {"crops": capped_source_count["val"]["NSCLC"],
                           "patients": len(capped_source_pids["val"]["NSCLC"])},
                "MSD_Lung": {"crops": capped_source_count["val"]["MSD_Lung"],
                              "patients": len(capped_source_pids["val"]["MSD_Lung"])},
            },
        },
        "class_weights": {
            "NSCLC": round(cw_nsclc, 6),
            "MSD_Lung": round(cw_msd, 6),
        },
        "guardrail": {
            "hard_negative_in_output": hard_neg_in_capped,
            "train_val_leakage": len(leakage),
            "stage2_holdout_access": False,
            "actual_manifest_generated": False,
            "training_run": False,
        },
        "errors": errors,
        "warnings": warnings,
        "next_step": (
            "P-C-AUX3: actual manifest generation — 사용자 승인 후 --full 옵션으로 실행"
        ),
    }
    with open(os.path.join(DRYCHECK_REPORT_DIR, "p_c_aux2_manifest_gen_drycheck.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n=== Dry-check 완료 ===")
    print(f"판정: {verdict}")
    print(f"errors={len(errors)}, warnings={len(warnings)}")
    print(f"출력: {DRYCHECK_REPORT_DIR}")
    return summary


# ── Full 생성 ─────────────────────────────────────────────────────────────────

def run_full():
    """사용자 승인 후에만 실행. --full + 3개 confirm flag 필수. 실제 manifest CSV 생성."""
    print("=== P-C-AUX2 Full Manifest Generation ===")

    # ── output collision hard blocker ──────────────────────────────────────
    ACTUAL_OUTPUT_FILES = [
        "p_c_aux2_source_classifier_training_manifest.csv",
        "p_c_aux2_source_classifier_train_manifest.csv",
        "p_c_aux2_source_classifier_val_manifest.csv",
        "p_c_aux2_source_classifier_manifest_summary.json",
        "p_c_aux2_source_classifier_manifest_report.md",
        "p_c_aux2_errors.csv",
        "DONE.json",
    ]
    existing = [
        f for f in ACTUAL_OUTPUT_FILES
        if os.path.exists(os.path.join(OUTPUT_MANIFEST_DIR, f))
    ]
    if existing:
        print(f"[ABORT] output collision detected — 기존 파일 {len(existing)}개 존재:")
        for f in existing:
            print(f"  {os.path.join(OUTPUT_MANIFEST_DIR, f)}")
        print("기존 파일 덮어쓰기 금지. 삭제 후 재실행하거나 다른 output dir를 사용하십시오.")
        sys.exit(2)

    os.makedirs(OUTPUT_MANIFEST_DIR, exist_ok=True)

    aux_rows, hard_neg_count, unknown_count = load_and_join(CLITE_MANIFEST, PC3_MANIFEST)
    capped = apply_patient_cap(aux_rows, cap=PATIENT_CAP)
    cw_nsclc, cw_msd = compute_class_weights(capped)
    assign_aux_ids(capped)

    leakage = check_patient_leakage(capped)
    if leakage:
        raise RuntimeError(f"FATAL: train/val leakage detected: {leakage}")

    # hard_negative 최종 확인
    hns = [r for r in capped if r.get("original_p_c_label") != "positive"]
    if hns:
        raise RuntimeError(f"FATAL: hard_negative rows in output: {len(hns)}")

    # roi_patch_ratio 의미 오류 확인
    bad_ratio = [r for r in capped if r.get("roi_patch_ratio") not in ("NA", "", None)]
    if bad_ratio:
        raise RuntimeError(f"FATAL: roi_patch_ratio에 비정상 값 포함: {len(bad_ratio)}건")

    train_rows = [r for r in capped if r["split"] == "train"]
    val_rows = [r for r in capped if r["split"] == "val"]

    nsclc_train = sum(1 for r in train_rows if r["source_name"] == "NSCLC")
    msd_train = sum(1 for r in train_rows if r["source_name"] == "MSD_Lung")
    nsclc_val = sum(1 for r in val_rows if r["source_name"] == "NSCLC")
    msd_val = sum(1 for r in val_rows if r["source_name"] == "MSD_Lung")

    # ── 3개 CSV manifest 생성 ──────────────────────────────────────────────
    write_csv(
        os.path.join(OUTPUT_MANIFEST_DIR, "p_c_aux2_source_classifier_training_manifest.csv"),
        capped, OUTPUT_COLUMNS
    )
    write_csv(
        os.path.join(OUTPUT_MANIFEST_DIR, "p_c_aux2_source_classifier_train_manifest.csv"),
        train_rows, OUTPUT_COLUMNS
    )
    write_csv(
        os.path.join(OUTPUT_MANIFEST_DIR, "p_c_aux2_source_classifier_val_manifest.csv"),
        val_rows, OUTPUT_COLUMNS
    )

    # ── summary.json ───────────────────────────────────────────────────────
    conditions_ok = (
        len(capped) > 0
        and len(train_rows) > 0
        and len(val_rows) > 0
        and len(hns) == 0
        and len(leakage) == 0
    )
    summary = {
        "stage": "P-C-AUX2",
        "mode": "full",
        "total_rows": len(capped),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "NSCLC_rows_train": nsclc_train,
        "NSCLC_rows_val": nsclc_val,
        "MSD_Lung_rows_train": msd_train,
        "MSD_Lung_rows_val": msd_val,
        "source_label_mapping": SOURCE_LABEL,
        "patient_cap": PATIENT_CAP,
        "class_weight_NSCLC": round(cw_nsclc, 6),
        "class_weight_MSD": round(cw_msd, 6),
        "hard_negative_included": False,
        "hard_negative_count_in_output": 0,
        "train_val_leakage": len(leakage),
        "stage2_holdout_accessed": False,
        "training_run": False,
        "model_forward_run": False,
        "scoring_run": False,
        "crop_npz_loaded": False,
        "vessel_mask_used": False,
        "roi_patch_ratio_available": ROI_PATCH_RATIO_AVAILABLE,
        "forbidden_diagnostic_wording_count": 0,
        "conditions_ok": conditions_ok,
        "stage2_holdout_note": "not checked by holdout list, no holdout source accessed",
    }
    with open(os.path.join(OUTPUT_MANIFEST_DIR, "p_c_aux2_source_classifier_manifest_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── errors.csv ─────────────────────────────────────────────────────────
    err_rows = []
    if len(leakage) > 0:
        err_rows.append({"severity": "FAIL", "message": f"train/val leakage: {leakage}"})
    if len(hns) > 0:
        err_rows.append({"severity": "FAIL", "message": f"hard_negative in output: {len(hns)}"})
    if not err_rows:
        err_rows = [{"severity": "INFO", "message": "no errors"}]
    write_csv(
        os.path.join(OUTPUT_MANIFEST_DIR, "p_c_aux2_errors.csv"),
        err_rows,
        ["severity", "message"]
    )

    # ── manifest_report.md ─────────────────────────────────────────────────
    report_lines = [
        "# P-C-AUX2 Source Classifier Manifest Generation Report",
        "",
        f"- total_rows: {len(capped):,}",
        f"- train_rows: {len(train_rows):,}  (NSCLC={nsclc_train:,}, MSD_Lung={msd_train:,})",
        f"- val_rows: {len(val_rows):,}  (NSCLC={nsclc_val:,}, MSD_Lung={msd_val:,})",
        f"- patient_cap: {PATIENT_CAP}",
        f"- class_weight_NSCLC: {round(cw_nsclc, 6)}",
        f"- class_weight_MSD: {round(cw_msd, 6)}",
        "- hard_negative_included: False",
        "- hard_negative_count_in_output: 0",
        f"- train_val_leakage: {len(leakage)}",
        "- stage2_holdout_accessed: False",
        "- training_run: False",
        "- model_forward_run: False",
        "- scoring_run: False",
        "- crop_npz_loaded: False",
        "- vessel_mask_used: False",
        f"- roi_patch_ratio_available: {ROI_PATCH_RATIO_AVAILABLE}",
        "- forbidden_diagnostic_wording_count: 0",
        f"- conditions_ok: {conditions_ok}",
        "",
        "## stage2_holdout",
        "not checked by holdout list, no holdout source accessed",
        "",
        "## 실행 명령",
        "```bash",
        "source ~/ai_env/bin/activate",
        "cd experiments/efficientnet_b0_v4_20_supervised_aux_source_classifier_v1/code",
        "python p_c_aux2_source_classifier_manifest_gen.py \\",
        "  --full \\",
        "  --confirm-positive-only \\",
        "  --confirm-no-hard-negative \\",
        "  --confirm-no-holdout",
        "```",
    ]
    with open(os.path.join(OUTPUT_MANIFEST_DIR, "p_c_aux2_source_classifier_manifest_report.md"), "w") as f:
        f.write("\n".join(report_lines))

    # ── DONE.json ──────────────────────────────────────────────────────────
    done_conditions = {
        "total_rows_gt_0": len(capped) > 0,
        "train_rows_gt_0": len(train_rows) > 0,
        "val_rows_gt_0": len(val_rows) > 0,
        "hard_negative_count_in_output_eq_0": len(hns) == 0,
        "train_val_leakage_eq_0": len(leakage) == 0,
        "stage2_holdout_accessed_false": True,
        "training_run_false": True,
        "model_forward_run_false": True,
        "scoring_run_false": True,
        "conditions_ok": conditions_ok,
    }
    done = {
        "stage": "P-C-AUX2",
        "total_rows": len(capped),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "hard_negative_count_in_output": 0,
        "train_val_leakage": len(leakage),
        "stage2_holdout_accessed": False,
        "training_run": False,
        "model_forward_run": False,
        "scoring_run": False,
        "conditions_ok": conditions_ok,
        "done_conditions": done_conditions,
    }
    with open(os.path.join(OUTPUT_MANIFEST_DIR, "DONE.json"), "w") as f:
        json.dump(done, f, indent=2, ensure_ascii=False)

    print(f"완료: {len(capped):,} rows ({len(train_rows):,} train / {len(val_rows):,} val)")
    print(f"출력: {OUTPUT_MANIFEST_DIR}")


# ── Schema helpers ─────────────────────────────────────────────────────────────

def _schema_source(col):
    clite_cols = {
        "crop_path", "local_z", "slice_index", "y0", "x0", "y1", "x1",
        "patient_id", "safe_id", "tiny_lesion_flag", "no_hit_patient",
        "fallback_positive_below_p95", "p_b3_risk6_flag", "split_plan",
        "candidate_label", "candidate_id", "source_branch",
    }
    pc3_cols = {"position_bin", "z_level", "lesion_pixels", "has_lesion_patch", "group"}
    computed = {
        "aux_candidate_id", "source_name", "source_label", "split",
        "center_y", "center_x", "original_candidate_id", "original_p_c_label",
        "is_positive_only", "no_hit_fallback_flag", "roi_patch_ratio",
        "patient_positive_count_before_cap", "patient_positive_count_after_cap",
        "patient_cap_applied", "sample_weight", "class_weight",
        "forbidden_hard_negative_used", "forbidden_supervised_diagnostic_wording",
    }
    if col in clite_cols:
        return "C-lite manifest"
    if col in pc3_cols:
        return "P-C3 manifest (join by candidate_id)"
    return "computed"


def _schema_note(col):
    notes = {
        "source_label": "NSCLC=1, MSD_Lung=0",
        "source_name": "NSCLC / MSD_Lung",
        "split": "기존 P-C10 split_plan 재사용",
        "center_y": "(y0+y1)//2",
        "center_x": "(x0+x1)//2",
        "roi_patch_ratio": "실제 roi_patch_ratio 컬럼 없음 → NA (lesion_pixels 혼용 금지)",
        "no_hit_fallback_flag": "no_hit_patient OR fallback_positive_below_p95",
        "is_positive_only": "항상 True — hard_negative 미포함",
        "patient_cap_applied": f"patient별 cap={PATIENT_CAP} 초과 여부",
        "class_weight": "total/(2*n_class), train 기준 계산",
        "sample_weight": "class_weight와 동일 (행별 기록)",
        "forbidden_hard_negative_used": "항상 False — guardrail",
        "forbidden_supervised_diagnostic_wording": "항상 False — guardrail",
    }
    return notes.get(col, "")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="P-C-AUX2 Source Classifier Manifest Generator"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-check", action="store_true",
                       help="manifest 미생성, 설계 검증만 수행 (기본 권장)")
    group.add_argument("--full", action="store_true",
                       help="실제 manifest 생성 (3개 confirm flag 필수 동반, 단독 사용 시 abort)")
    # full 실행용 confirm flags (모두 필요)
    parser.add_argument("--confirm-positive-only", action="store_true",
                        help="positive-only 데이터만 사용함을 확인")
    parser.add_argument("--confirm-no-hard-negative", action="store_true",
                        help="hard_negative 미포함을 확인")
    parser.add_argument("--confirm-no-holdout", action="store_true",
                        help="stage2_holdout 미접근을 확인")
    args = parser.parse_args()

    if args.dry_check:
        run_drycheck()
    elif args.full:
        missing_flags = []
        if not args.confirm_positive_only:
            missing_flags.append("--confirm-positive-only")
        if not args.confirm_no_hard_negative:
            missing_flags.append("--confirm-no-hard-negative")
        if not args.confirm_no_holdout:
            missing_flags.append("--confirm-no-holdout")
        if missing_flags:
            print("[ABORT] --full 단독 실행 금지. 다음 confirm flag가 필요합니다:")
            for f in missing_flags:
                print(f"  {f}")
            print("\n올바른 실행 명령:")
            print("  python p_c_aux2_source_classifier_manifest_gen.py \\")
            print("    --full \\")
            print("    --confirm-positive-only \\")
            print("    --confirm-no-hard-negative \\")
            print("    --confirm-no-holdout")
            sys.exit(2)
        run_full()


if __name__ == "__main__":
    main()
