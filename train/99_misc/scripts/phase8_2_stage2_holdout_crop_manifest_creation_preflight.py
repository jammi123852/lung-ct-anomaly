#!/usr/bin/env python3
"""
phase8_2_stage2_holdout_crop_manifest_creation_preflight.py
============================================================
Phase 8.2: stage2_holdout 154명 dedicated crop/manifest 생성 전 preflight.

목적:
  - stage2_holdout 154명 전체에 대해 dedicated crop root와 dedicated manifest를
    생성하기 전 생성 계획을 확정한다.
  - 이번 단계는 creation preflight only다.

절대 금지:
  - 실제 crop 생성 금지
  - 실제 manifest 생성 금지
  - crop 복사 금지
  - npz 로드 금지
  - CT/ROI/mask npy 로드 금지
  - model forward 금지
  - scoring 금지
  - metric 계산 금지
  - threshold 계산 금지
  - training 금지
  - NSCLC/MSD root 내용 접근 금지 (1단계 stat/listdir만 허용)

실행 모드:
  --run 없음 : dry-run (파일 미생성, 계획만 출력)
  --run      : CSV/JSON/MD 생성

실행 명령:
  source ~/ai_env/bin/activate && \\
  python scripts/phase8_2_stage2_holdout_crop_manifest_creation_preflight.py --run

syntax check (실행 아님):
  python -m py_compile scripts/phase8_2_stage2_holdout_crop_manifest_creation_preflight.py
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

# ─────────────────────────────────────────────
# 경로 상수
# ─────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]

# 입력 파일
PHASE8_0_JSON = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/review_annotations/phase8_0_stage2_holdout_final_eval_preflight_v1/phase8_0_stage2_holdout_final_eval_preflight_v1.json"
PHASE8_1_JSON = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/review_annotations/phase8_1_stage2_holdout_manifest_crop_asset_preflight_v1/phase8_1_stage2_holdout_manifest_crop_asset_preflight_v1.json"
PHASE8_1B_JSON = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/review_annotations/phase8_1b_stage2_holdout_excluded_rows_audit_v1/phase8_1b_stage2_holdout_excluded_rows_audit_v1.json"
SPLIT_CSV = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1.csv"
PHASE6_1B_MANIFEST_CSV = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/review_annotations/phase6_1b_s6a_stage1_dev_filtered_manifest_v1/phase6_1b_s6a_stage1_dev_filtered_manifest_v1.csv"
PHASE6_1B_EXCLUDED_CSV = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/review_annotations/phase6_1b_s6a_stage1_dev_filtered_manifest_v1/phase6_1b_s6a_stage2_holdout_excluded_rows_v1.csv"
PATHS_CONFIG = REPO_ROOT / "configs/paths.local.yaml"

# dedicated output 후보 경로
DEDICATED_CROP_ROOT = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/crops_stage2_holdout_6ch_dedicated_v1"
DEDICATED_MANIFEST = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/datasets/s6a_stage2_holdout_filtered_manifest_v1.csv"

# 기존 crop root (read-only 확인만)
EXISTING_CROP_ROOT = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/crops_s6a_6ch_full"

# output root (preflight 결과 저장)
OUT_DIR = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/review_annotations/phase8_2_stage2_holdout_crop_manifest_creation_preflight_v1"
OUT_CSV = OUT_DIR / "phase8_2_stage2_holdout_crop_manifest_creation_preflight_v1.csv"
OUT_JSON = OUT_DIR / "phase8_2_stage2_holdout_crop_manifest_creation_preflight_v1.json"
OUT_MD = OUT_DIR / "phase8_2_stage2_holdout_crop_manifest_creation_preflight_report_v1.md"

# 기대 상수
EXPECTED_STAGE2_PATIENT_COUNT = 154
EXPECTED_CONTAMINATED_PATIENTS = ["LUNG1-295", "LUNG1-415"]
EXPECTED_CROP_SHAPE = "(6,96,96)"
EXPECTED_INPUT_CHANNELS = 6
EXPECTED_CROP_SIZE = 96


# ─────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _pass_fail(condition: bool) -> str:
    return "PASS" if condition else "FAIL"


# ─────────────────────────────────────────────
# Section A: prior phase readiness
# ─────────────────────────────────────────────

def check_prior_phase_readiness() -> tuple[list[dict], dict]:
    rows = []
    summary = {}

    # phase8_0
    d0 = _load_json(PHASE8_0_JSON)
    obs_8_0 = d0.get("readiness_for_stage2_holdout_eval", "NOT_FOUND")
    exp_8_0 = "BLOCKED_MISSING_STAGE2_ASSETS"
    status_8_0 = _pass_fail(obs_8_0 == exp_8_0)
    rows.append({
        "section": "A",
        "item": "Phase 8.0 readiness_for_stage2_holdout_eval",
        "expected": exp_8_0,
        "observed": obs_8_0,
        "status": status_8_0,
        "note": "phase8_0 최종 상태"
    })
    summary["phase8_0_readiness"] = obs_8_0
    summary["phase8_0_status"] = status_8_0

    # phase8_1
    d1 = _load_json(PHASE8_1_JSON)
    obs_8_1_ready = d1.get("readiness_for_phase8_2", "NOT_FOUND")
    exp_8_1_ready = "READY_FOR_PHASE8_2_STAGE2_MANIFEST_CREATION"
    status_8_1_ready = _pass_fail(obs_8_1_ready == exp_8_1_ready)
    rows.append({
        "section": "A",
        "item": "Phase 8.1 readiness_for_phase8_2",
        "expected": exp_8_1_ready,
        "observed": obs_8_1_ready,
        "status": status_8_1_ready,
        "note": "phase8_1 manifest/crop asset preflight 최종 상태"
    })
    summary["phase8_1_readiness"] = obs_8_1_ready
    summary["phase8_1_readiness_status"] = status_8_1_ready

    obs_8_1_opt = d1.get("recommended_option", "NOT_FOUND")
    exp_8_1_opt = "C"
    status_8_1_opt = _pass_fail(obs_8_1_opt == exp_8_1_opt)
    rows.append({
        "section": "A",
        "item": "Phase 8.1 recommended_option",
        "expected": exp_8_1_opt,
        "observed": obs_8_1_opt,
        "status": status_8_1_opt,
        "note": "Option C = generate from original CT/ROI"
    })
    summary["phase8_1_recommended_option"] = obs_8_1_opt
    summary["phase8_1_option_status"] = status_8_1_opt

    # phase8_1b
    d1b = _load_json(PHASE8_1B_JSON)
    obs_1b_audit = d1b.get("audit_status", "NOT_FOUND")
    exp_1b_audit = "PASS_READY_FOR_PHASE8_2"
    status_1b_audit = _pass_fail(obs_1b_audit == exp_1b_audit)
    rows.append({
        "section": "A",
        "item": "Phase 8.1b audit_status",
        "expected": exp_1b_audit,
        "observed": obs_1b_audit,
        "status": status_1b_audit,
        "note": "excluded rows leakage audit 최종 상태"
    })
    summary["phase8_1b_audit_status"] = obs_1b_audit
    summary["phase8_1b_audit_check"] = status_1b_audit

    # stage2_holdout patient count
    df_split = pd.read_csv(SPLIT_CSV)
    s2_count = int((df_split["stage_split"] == "stage2_holdout").sum())
    exp_count = EXPECTED_STAGE2_PATIENT_COUNT
    status_count = _pass_fail(s2_count == exp_count)
    rows.append({
        "section": "A",
        "item": "stage2_holdout patient count",
        "expected": str(exp_count),
        "observed": str(s2_count),
        "status": status_count,
        "note": "split CSV 기준 patient 수"
    })
    summary["stage2_holdout_patient_count"] = s2_count
    summary["stage2_patient_count_status"] = status_count

    # contamination audit (LUNG1-295, LUNG1-415)
    obs_contaminated = d1b.get("excluded_patient_ids", [])
    exp_contaminated = EXPECTED_CONTAMINATED_PATIENTS
    status_contam = _pass_fail(sorted(obs_contaminated) == sorted(exp_contaminated))
    rows.append({
        "section": "A",
        "item": "contamination audit: excluded_patient_ids",
        "expected": str(sorted(exp_contaminated)),
        "observed": str(sorted(obs_contaminated)),
        "status": status_contam,
        "note": "LUNG1-295/LUNG1-415 오염 환자 확인"
    })
    summary["contaminated_patient_ids"] = obs_contaminated
    summary["contamination_audit_status"] = status_contam

    # in_stage2_holdout 확인
    obs_in_s2 = d1b.get("excluded_patients_in_stage2_holdout", [])
    status_in_s2 = _pass_fail(sorted(obs_in_s2) == sorted(exp_contaminated))
    rows.append({
        "section": "A",
        "item": "contaminated patients in stage2_holdout",
        "expected": str(sorted(exp_contaminated)),
        "observed": str(sorted(obs_in_s2)),
        "status": status_in_s2,
        "note": "오염 환자가 stage2_holdout에만 있고 stage1_dev에는 없음"
    })

    # not in stage1_dev 확인
    obs_in_s1 = d1b.get("excluded_patients_in_stage1_dev", [])
    status_not_in_s1 = _pass_fail(len(obs_in_s1) == 0)
    rows.append({
        "section": "A",
        "item": "contaminated patients NOT in stage1_dev",
        "expected": "[]",
        "observed": str(obs_in_s1),
        "status": status_not_in_s1,
        "note": "stage1_dev에 오염 환자 없음 확인"
    })

    all_pass = all(r["status"] == "PASS" for r in rows)
    summary["section_A_overall"] = "PASS" if all_pass else "FAIL"

    return rows, summary


# ─────────────────────────────────────────────
# Section B: source script inventory
# ─────────────────────────────────────────────

def check_source_script_inventory() -> tuple[list[dict], list[dict]]:
    rows = []
    summary_list = []

    scripts_dir = REPO_ROOT / "scripts"

    # 탐색 패턴 및 후보 역할 정의
    # (pattern, candidate_role, allowed_use, forbidden_use)
    search_patterns = [
        ("*s6a*crop*6ch*.py", "S6-A 6ch crop 생성", "crop pipeline 참조, config 확인", "직접 실행 금지"),
        ("*crop*s6a*6ch*.py", "S6-A 6ch crop 생성", "crop pipeline 참조, config 확인", "직접 실행 금지"),
        ("generate_s6a_crop_full_6ch.py", "S6-A 6ch full crop 생성 (기존 stage1_dev용)", "pipeline 참조, config 확인", "직접 실행 금지, stage2_holdout 입력 금지"),
        ("generate_s6a_crop_full.py", "S6-A full crop 생성 (비 6ch)", "참조 가능", "직접 실행 금지"),
        ("generate_s6a_crop_smoke.py", "S6-A smoke crop 생성", "참조 가능", "직접 실행 금지"),
        ("*dataset_index*.py", "dataset index 생성", "manifest schema 참조", "직접 실행 금지"),
        ("*manifest*creation*.py", "manifest 생성", "schema 참조", "직접 실행 금지"),
        ("rule_s6a_manifest_gen.py", "S6-A manifest 생성", "manifest 생성 로직 참조", "직접 실행 금지"),
        ("phase6_1b_s6a_stage1_dev_filtered_manifest.py", "phase6_1b filtered manifest (stage1_dev용)", "schema 참조", "stage2_holdout 재사용 금지"),
        ("validate_s6a_6ch_crop_full.py", "S6-A 6ch crop validation", "참조 가능", "직접 실행 금지"),
        ("smoke_s6a_6ch_dataloader.py", "S6-A 6ch dataloader smoke", "참조 가능", "직접 실행 금지"),
    ]

    # train/val split / scoring / evaluation → excluded 표시
    excluded_patterns = [
        ("train_s6a_rd4ad_verifier.py", "학습 스크립트", "excluded (학습 금지)"),
        ("phase7_4_v1v1_crop_level_metrics.py", "evaluation/metric", "excluded (metric 금지)"),
        ("phase6_2b_s6a_model_forward_smoke.py", "model forward smoke", "excluded (model forward 금지)"),
        ("*score*.py", "scoring 관련", "excluded (scoring 금지)"),
        ("*eval*.py", "evaluation 관련", "excluded (metric 금지)"),
        ("*train*.py", "학습 관련", "excluded (학습 금지)"),
        ("*split*.py", "split 관련", "excluded (split 수정 금지)"),
    ]

    seen_scripts = set()

    # 후보 script 탐색
    for pattern, role, allowed, forbidden in search_patterns:
        if os.path.isabs(pattern) or "/" in pattern:
            matches = [str(scripts_dir / pattern)]
        else:
            matches = glob.glob(str(scripts_dir / pattern))

        if not matches:
            rows.append({
                "section": "B",
                "script_path": str(scripts_dir / pattern),
                "exists": False,
                "candidate_role": role,
                "allowed_use": allowed,
                "forbidden_use": forbidden,
                "status": "NOT_FOUND",
                "note": f"패턴 {pattern} 매칭 없음"
            })
            summary_list.append({
                "script_path": str(scripts_dir / pattern),
                "exists": False,
                "candidate_role": role,
                "status": "NOT_FOUND"
            })
        else:
            for m in matches:
                if m in seen_scripts:
                    continue
                seen_scripts.add(m)
                exists = os.path.exists(m)
                rows.append({
                    "section": "B",
                    "script_path": m,
                    "exists": exists,
                    "candidate_role": role,
                    "allowed_use": allowed,
                    "forbidden_use": forbidden,
                    "status": "FOUND" if exists else "NOT_FOUND",
                    "note": "read-only 확인만"
                })
                summary_list.append({
                    "script_path": m,
                    "exists": exists,
                    "candidate_role": role,
                    "status": "FOUND" if exists else "NOT_FOUND"
                })

    # excluded scripts 확인
    for pattern, role, reason in excluded_patterns:
        matches = glob.glob(str(scripts_dir / pattern))
        for m in matches:
            if m in seen_scripts:
                continue
            seen_scripts.add(m)
            rows.append({
                "section": "B",
                "script_path": m,
                "exists": True,
                "candidate_role": role,
                "allowed_use": "없음",
                "forbidden_use": reason,
                "status": "EXCLUDED",
                "note": "이번 phase에서 사용 금지"
            })

    return rows, summary_list


# ─────────────────────────────────────────────
# Section C: source data path preflight
# ─────────────────────────────────────────────

def check_source_data_path() -> tuple[list[dict], dict]:
    rows = []
    summary = {}

    # paths.local.yaml 로드
    paths_cfg = _load_yaml(PATHS_CONFIG)
    vol_root_str = paths_cfg.get("nsclc_msd_usable_only_v2", "")
    vol_root = Path(vol_root_str) if vol_root_str else None

    # 1. paths.local.yaml 존재 확인
    cfg_exists = PATHS_CONFIG.exists()
    rows.append({
        "section": "C",
        "source_type": "paths config",
        "candidate_path_or_manifest": str(PATHS_CONFIG),
        "exists_or_resolvable": cfg_exists,
        "allowed_check": "파일 존재 확인",
        "forbidden_check": "내용 수정 금지",
        "status": "PASS" if cfg_exists else "FAIL",
        "note": "CT/ROI path 설정 파일"
    })

    # 2. nsclc_msd_usable_only_v2 경로 확인
    vol_root_exists = vol_root is not None and vol_root.exists()
    rows.append({
        "section": "C",
        "source_type": "CT/ROI source root (nsclc_msd_usable_only_v2)",
        "candidate_path_or_manifest": vol_root_str or "NOT_SET",
        "exists_or_resolvable": vol_root_exists,
        "allowed_check": "os.path.exists, 1단계 listdir",
        "forbidden_check": "npy load 금지, recursive scan 금지, 내용 확인 금지",
        "status": "PASS" if vol_root_exists else "FAIL",
        "note": "paths.local.yaml nsclc_msd_usable_only_v2"
    })
    summary["vol_root"] = vol_root_str
    summary["vol_root_exists"] = vol_root_exists

    # 3. volumes_npy 폴더 존재 확인
    if vol_root_exists:
        volumes_npy = vol_root / "volumes_npy"
        vnpy_exists = volumes_npy.exists()
        vnpy_count = len(os.listdir(volumes_npy)) if vnpy_exists else 0
        rows.append({
            "section": "C",
            "source_type": "volumes_npy subfolder",
            "candidate_path_or_manifest": str(volumes_npy),
            "exists_or_resolvable": vnpy_exists,
            "allowed_check": "os.path.exists, len(listdir) 1단계만",
            "forbidden_check": "npy load 금지, 내용 확인 금지",
            "status": "PASS" if vnpy_exists else "FAIL",
            "note": f"patient dir 수: {vnpy_count}"
        })
        summary["volumes_npy_exists"] = vnpy_exists
        summary["volumes_npy_patient_dir_count"] = vnpy_count
    else:
        rows.append({
            "section": "C",
            "source_type": "volumes_npy subfolder",
            "candidate_path_or_manifest": "NOT_CHECKED",
            "exists_or_resolvable": False,
            "allowed_check": "N/A",
            "forbidden_check": "N/A",
            "status": "SKIP",
            "note": "vol_root 미존재로 skip"
        })
        summary["volumes_npy_exists"] = False

    # 4. split CSV 기반 stage2_holdout 환자 경로 resolvable 확인
    #    (실제 ct_hu.npy 로드 금지, safe_id → directory 존재 여부만 확인)
    df_split = pd.read_csv(SPLIT_CSV)
    s2 = df_split[df_split["stage_split"] == "stage2_holdout"][["patient_id", "safe_id"]].copy()

    if vol_root_exists:
        volumes_npy = vol_root / "volumes_npy"
        vol_dirs = set(os.listdir(volumes_npy))
        found_count = sum(1 for sid in s2["safe_id"] if sid in vol_dirs)
        missing_count = len(s2) - found_count
        missing_patients = [
            pid for pid, sid in zip(s2["patient_id"], s2["safe_id"])
            if sid not in vol_dirs
        ]
        path_status = "PASS" if missing_count == 0 else "FAIL"
        rows.append({
            "section": "C",
            "source_type": "stage2_holdout patient CT/ROI dir (safe_id match)",
            "candidate_path_or_manifest": str(volumes_npy),
            "exists_or_resolvable": missing_count == 0,
            "allowed_check": "safe_id in volumes_npy listdir (1단계)",
            "forbidden_check": "npy load 금지, ct_hu.npy 로드 금지",
            "status": path_status,
            "note": f"found {found_count}/154, missing {missing_count}" + (
                f" — missing: {missing_patients[:3]}" if missing_patients else ""
            )
        })
        summary["stage2_ct_roi_found"] = found_count
        summary["stage2_ct_roi_missing"] = missing_count
        summary["stage2_ct_roi_path_status"] = path_status
    else:
        rows.append({
            "section": "C",
            "source_type": "stage2_holdout patient CT/ROI dir",
            "candidate_path_or_manifest": "NOT_CHECKED",
            "exists_or_resolvable": False,
            "allowed_check": "N/A",
            "forbidden_check": "N/A",
            "status": "SKIP",
            "note": "vol_root 미존재로 skip"
        })
        summary["stage2_ct_roi_path_status"] = "SKIP"

    # 5. crops_s6a_6ch_full (기존 crop root) 1단계 stat 확인
    crop_full_exists = EXISTING_CROP_ROOT.exists()
    crop_full_count = 0
    if crop_full_exists:
        crop_full_count = len(os.listdir(EXISTING_CROP_ROOT))
    rows.append({
        "section": "C",
        "source_type": "crops_s6a_6ch_full (기존 crop root, stage1_dev 152 + 오염 2명)",
        "candidate_path_or_manifest": str(EXISTING_CROP_ROOT),
        "exists_or_resolvable": crop_full_exists,
        "allowed_check": "os.path.exists, 1단계 listdir count",
        "forbidden_check": "npz 로드 금지, 내용 확인 금지",
        "status": "PASS" if crop_full_exists else "FAIL",
        "note": f"patient dir 수: {crop_full_count} (stage2_holdout용 새 생성에는 사용 금지)"
    })
    summary["existing_crop_root_count"] = crop_full_count

    # 6. phase6_1b manifest CSV 존재 확인 (입력 참조용)
    mf_exists = PHASE6_1B_MANIFEST_CSV.exists()
    rows.append({
        "section": "C",
        "source_type": "phase6_1b filtered manifest (stage1_dev용, 재사용 금지)",
        "candidate_path_or_manifest": str(PHASE6_1B_MANIFEST_CSV),
        "exists_or_resolvable": mf_exists,
        "allowed_check": "파일 존재 확인, schema 참조",
        "forbidden_check": "stage2_holdout manifest 재사용 금지, npz path 직접 인용 금지",
        "status": "PASS" if mf_exists else "FAIL",
        "note": "schema 참조 전용, stage2_holdout dedicated 생성에는 신규 생성 필요"
    })

    # 7. excluded rows CSV 존재 확인
    ex_exists = PHASE6_1B_EXCLUDED_CSV.exists()
    rows.append({
        "section": "C",
        "source_type": "stage2_holdout excluded rows CSV",
        "candidate_path_or_manifest": str(PHASE6_1B_EXCLUDED_CSV),
        "exists_or_resolvable": ex_exists,
        "allowed_check": "파일 존재 확인, row count 참조",
        "forbidden_check": "내용 기반 threshold/scoring 금지",
        "status": "PASS" if ex_exists else "FAIL",
        "note": "1,222 rows, LUNG1-295/LUNG1-415 포함 stage2_holdout 기존 누락 rows"
    })

    all_pass = all(r["status"] in ("PASS", "SKIP") for r in rows)
    summary["section_C_overall"] = "PASS" if all_pass else "FAIL"

    return rows, summary


# ─────────────────────────────────────────────
# Section D: dedicated output plan
# ─────────────────────────────────────────────

def check_dedicated_output_plan() -> tuple[list[dict], dict]:
    rows = []
    summary = {}

    checks = [
        (
            "dedicated_crop_root",
            DEDICATED_CROP_ROOT,
            "crops_stage2_holdout_6ch_dedicated_v1/",
            "mkdir(parents=True, exist_ok=False) — Phase 8.2 run 승인 후 생성",
            "exist_ok=False; output_root 이미 있으면 즉시 중단"
        ),
        (
            "dedicated_manifest",
            DEDICATED_MANIFEST,
            "s6a_stage2_holdout_filtered_manifest_v1.csv",
            "Phase 8.2 run 승인 후 신규 생성",
            "exist_ok=False; 파일 이미 있으면 즉시 중단"
        ),
        (
            "preflight_output_dir",
            OUT_DIR,
            "phase8_2_stage2_holdout_crop_manifest_creation_preflight_v1/",
            "이번 preflight --run 실행 시 생성",
            "exist_ok=False; 이미 있으면 즉시 중단"
        ),
    ]

    for output_type, path, path_label, planned_action, overwrite_guard in checks:
        exists = path.exists()
        status = "OK_NOT_EXISTS" if not exists else "CONFLICT_EXISTS"
        rows.append({
            "section": "D",
            "output_type": output_type,
            "output_path": str(path),
            "exists": exists,
            "planned_action": planned_action,
            "overwrite_guard": overwrite_guard,
            "status": status
        })
        summary[output_type] = {
            "path": str(path),
            "exists": exists,
            "status": status
        }

    # 기존 output과 충돌 없는지 추가 확인
    conflict_paths = [
        REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/crops_stage2_holdout_6ch_dedicated_v1",
        REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/datasets/s6a_stage2_holdout_filtered_manifest_v1.csv",
    ]
    any_conflict = any(p.exists() for p in conflict_paths)
    summary["any_output_conflict"] = any_conflict
    summary["section_D_overall"] = "PASS" if not any_conflict else "FAIL_CONFLICT"
    summary["exist_ok_false_principle"] = True

    return rows, summary


# ─────────────────────────────────────────────
# Section E: manifest schema design
# ─────────────────────────────────────────────

def check_manifest_schema_design() -> tuple[list[dict], list[dict]]:
    schema_fields = [
        ("row_id", True, "int, 0-indexed", "생성 시 자동 부여", ""),
        ("patient_id", True, "LUNG1-XXX 또는 MSD_lung_XXX", "split CSV patient_id", ""),
        ("npz_path", True, "dedicated crop root 기준 상대경로 또는 절대경로", "Phase 8.2 run 시 생성", "Phase 8.3 validation 전까지 미검증"),
        ("label", True, "0 (normal) 또는 1 (lesion)", "원본 manifest 또는 split 기반", ""),
        ("sampling_label", True, "positive 또는 hard_negative", "원본 manifest 기반", ""),
        ("stage_split", True, "stage2_holdout (고정값)", "split CSV", "모든 row stage2_holdout"),
        ("source_manifest", True, "phase6_1b 또는 신규 S6-A index 경로", "Phase 8.2 run 시 기록", "phase6_1b 재사용 금지"),
        ("source_crop_root", True, "crops_stage2_holdout_6ch_dedicated_v1 절대경로", "Phase 8.2 run 시 기록", "기존 crops_s6a_6ch_full 사용 금지"),
        ("asset_scope", True, "dedicated_stage2_holdout (고정값)", "고정", ""),
        ("contamination_check_status", True, "LUNG1-295/LUNG1-415: contaminated_regen_from_ct_roi; 나머지: clean_new_generation", "phase8_1b audit 기반", ""),
        ("approval_required_before_scoring", True, "True (고정값)", "Phase 8.2 원칙", "scoring 전 사용자 승인 필수"),
        ("manifest_status", True, "pending_creation_preflight (현재); created_after_phase8_2_run (생성 후)", "Phase 상태 반영", ""),
        ("crop_shape", True, "(6,96,96)", "generate_s6a_crop_full_6ch.py 기준", ""),
        ("input_channels", True, "6", "6ch (lung 3ch + mediastinal 3ch)", ""),
        ("crop_size", True, "96", "96×96 spatial", ""),
        ("generation_status", True, "pending_generation (현재); generated (생성 후)", "Phase 8.2 run 후 갱신", ""),
        ("issue", False, "빈값 또는 오류 메시지", "생성 중 오류 발생 시 기록", ""),
        ("note", False, "자유 텍스트", "필요 시 기록", ""),
    ]

    rows = []
    summary_list = []

    for field_name, required, expected_val, source, note in schema_fields:
        rows.append({
            "section": "E",
            "field_name": field_name,
            "required": required,
            "expected_value_or_rule": expected_val,
            "source": source,
            "note": note
        })
        summary_list.append({
            "field_name": field_name,
            "required": required,
            "expected_value_or_rule": expected_val
        })

    return rows, summary_list


# ─────────────────────────────────────────────
# Section F: generation safety plan
# ─────────────────────────────────────────────

def check_generation_safety_plan() -> tuple[list[dict], list[dict]]:
    safety_checks = [
        (
            "stage1_dev 환자 포함 금지",
            "stage1_dev patient 0명",
            "Phase 8.2 run 실행 전 split CSV 기반 확인 필수",
            "stage2_holdout dedicated manifest에 stage1_dev 환자 포함 금지"
        ),
        (
            "stage2_holdout 전원 포함",
            "stage2_holdout patient 154명",
            "Phase 8.2 run 실행 전 확인 필수",
            "154명 전원 생성 대상"
        ),
        (
            "LUNG1-295 오염 구 crop 사용 금지",
            "crops_s6a_6ch_full/LUNG1-295 사용 금지, 원본 CT/ROI에서 신규 생성",
            "Phase 8.2 run 스크립트에서 crops_s6a_6ch_full 경로 참조 금지",
            "contaminated old crop 재사용 금지"
        ),
        (
            "LUNG1-415 오염 구 crop 사용 금지",
            "crops_s6a_6ch_full/LUNG1-415 사용 금지, 원본 CT/ROI에서 신규 생성",
            "Phase 8.2 run 스크립트에서 crops_s6a_6ch_full 경로 참조 금지",
            "contaminated old crop 재사용 금지"
        ),
        (
            "phase6_1b filtered manifest 재사용 금지",
            "phase6_1b_s6a_stage1_dev_filtered_manifest_v1.csv 경로 직접 scoring 입력 금지",
            "Phase 8.2 run 스크립트 설계 시 확인 필수",
            "stage1_dev manifest를 stage2_holdout manifest로 재사용하면 leakage"
        ),
        (
            "original S6-A index 직접 scoring 입력 금지",
            "S6-A index/manifest를 scoring 입력으로 직접 사용 금지",
            "Phase 8.2 run 이후 Phase 8.3 validation 완료 전 scoring 금지",
            "dedicated manifest 생성 후 Phase 8.3 validation → 이후 별도 승인"
        ),
        (
            "생성 후 scoring 전 사용자 승인 필수",
            "approval_required_before_scoring = True",
            "Phase 8.2 run 완료 후 별도 승인 후에만 scoring 진행",
            "dedicated manifest에 approval_required_before_scoring=True 고정"
        ),
        (
            "생성 후 npz validation은 Phase 8.3에서",
            "Phase 8.2 run: crop 생성만, npz shape/value 검증은 별도 Phase 8.3",
            "Phase 8.2 run 스크립트에서 npz 로드 금지",
            "Phase 8.3 별도 승인 후 validation 수행"
        ),
        (
            "tmp → final rename 전략",
            "tmp 폴더에 생성 후 완료 시 final 경로로 rename",
            "Phase 8.2 run 스크립트 설계 시 적용 권장",
            "중단 시 incomplete output 방지"
        ),
    ]

    rows = []
    summary_list = []

    for safety_check, expected, required_before_run, note in safety_checks:
        rows.append({
            "section": "F",
            "safety_check": safety_check,
            "expected": expected,
            "required_before_run": required_before_run,
            "note": note
        })
        summary_list.append({
            "safety_check": safety_check,
            "expected": expected,
            "required_before_run": required_before_run
        })

    return rows, summary_list


# ─────────────────────────────────────────────
# Section G: runtime/cost estimate
# ─────────────────────────────────────────────

def check_runtime_cost_estimate() -> tuple[list[dict], dict]:
    rows = []
    summary = {}

    # stage2_holdout lesion_patch_count 합산 (positive crop 추정)
    df_split = pd.read_csv(SPLIT_CSV)
    s2 = df_split[df_split["stage_split"] == "stage2_holdout"].copy()

    patient_count = len(s2)
    lesion_patch_total = int(s2["lesion_patch_count"].sum()) if "lesion_patch_count" in s2.columns else -1

    # hard negative 수는 stage1_dev 비율로 rough estimate
    # stage1_dev positive:hn ≈ phase6_1b manifest에서 확인 가능하지만 로드 필요
    # rough: stage1_dev ratio ≈ 2:1 (hn:pos) 기준 추정 (참고용)
    if lesion_patch_total > 0:
        hn_estimate_rough = lesion_patch_total * 2  # rough 2x
        total_crop_estimate_rough = lesion_patch_total + hn_estimate_rough
    else:
        hn_estimate_rough = -1
        total_crop_estimate_rough = -1

    # 디스크 용량 추정: 6ch × 96 × 96 × 4 bytes (float32) per npz
    bytes_per_crop_raw = 6 * 96 * 96 * 4  # ~221 KB raw
    # npz compressed ≈ 30~50% of raw
    bytes_per_crop_npz = int(bytes_per_crop_raw * 0.4)
    if total_crop_estimate_rough > 0:
        disk_gb_estimate = total_crop_estimate_rough * bytes_per_crop_npz / (1024 ** 3)
    else:
        disk_gb_estimate = -1.0

    # group 분포
    group_dist = s2["group"].value_counts().to_dict() if "group" in s2.columns else {}

    estimates = [
        ("patient count", str(patient_count), "high", "split CSV 기준 stage2_holdout 154명"),
        ("positive crop (lesion_patch_count 합산)", str(lesion_patch_total), "medium", "split CSV lesion_patch_count 합산; 실제 생성 수와 차이 있을 수 있음"),
        ("hard negative crop (rough 2x 추정)", str(hn_estimate_rough), "low", "stage1_dev ratio 참고한 rough estimate; 실제 script 설계 시 확인 필요"),
        ("total crop estimate (rough)", str(total_crop_estimate_rough), "low", "positive + hn rough 합산"),
        ("disk usage estimate (npz compressed)", f"{disk_gb_estimate:.1f} GB" if disk_gb_estimate > 0 else "N/A", "low", "6ch×96×96×float32×40% compression 기준"),
        ("OOM 위험", "없음 (preflight only)", "high", "preflight에서 npy/npz 로드 없음"),
        ("GPU 필요 여부 (preflight)", "없음", "high", "preflight는 path/file stat만"),
        ("GPU 필요 여부 (Phase 8.2 run: crop 생성)", "불필요 (numpy 기반 crop)", "high", "CT npy 로드 후 numpy crop; GPU 없이 가능"),
        ("CT/ROI 접근 필요 여부", "Phase 8.2 run 시 필요", "high", "volumes_npy 경로 접근, npy load 필요 (Phase 8.2 run에서)"),
        ("예상 시간 (preflight)", "1~2분", "high", "file stat/listdir만"),
        ("예상 시간 (Phase 8.2 run: crop 생성)", "수십 분~수 시간 (154명, CT 로드 포함)", "low", "단일 코어 기준 rough; 실제 환경에 따라 다름"),
        ("group 분포 (NSCLC/MSD_Lung)", str(group_dist), "high", "split CSV group 컬럼 기준"),
    ]

    for item, estimate, confidence, note in estimates:
        rows.append({
            "section": "G",
            "item": item,
            "estimate": estimate,
            "confidence": confidence,
            "note": note
        })

    summary["patient_count"] = patient_count
    summary["lesion_patch_total"] = lesion_patch_total
    summary["hn_estimate_rough"] = hn_estimate_rough
    summary["total_crop_estimate_rough"] = total_crop_estimate_rough
    summary["disk_estimate_gb"] = round(disk_gb_estimate, 2) if disk_gb_estimate > 0 else None
    summary["group_distribution"] = group_dist
    summary["oom_risk"] = "none (preflight only)"
    summary["gpu_required_preflight"] = False
    summary["gpu_required_phase8_2_run"] = False
    summary["ct_roi_access_required"] = "Phase 8.2 run에서만 필요"

    return rows, summary


# ─────────────────────────────────────────────
# Section H: readiness decision
# ─────────────────────────────────────────────

def check_readiness_decision(
    sec_a: dict,
    sec_c: dict,
    sec_d: dict,
) -> tuple[list[dict], str, list[str]]:
    blockers = []

    # Section A 체크
    if sec_a.get("section_A_overall") != "PASS":
        blockers.append("BLOCKED_LEAKAGE_RISK: Phase 8.0/8.1/8.1b 사전 readiness 미통과")

    # Section C 체크
    if sec_c.get("stage2_ct_roi_path_status") == "FAIL":
        blockers.append("BLOCKED_MISSING_SOURCE_CT_ROI_PATHS: stage2_holdout CT/ROI 경로 resolvable 실패")
    if sec_c.get("section_C_overall") == "FAIL":
        blockers.append("BLOCKED_MISSING_SOURCE_CT_ROI_PATHS: source data path 확인 실패")

    # Section D 체크
    if sec_d.get("any_output_conflict"):
        blockers.append("BLOCKED_OUTPUT_CONFLICT: dedicated crop root 또는 manifest 이미 존재")

    if not blockers:
        readiness = "READY_FOR_PHASE8_2_RUN_CROP_MANIFEST_CREATION"
        next_action = "Phase 8.2 run: dedicated crop 생성 및 manifest 생성 별도 승인 요청"
    else:
        readiness = blockers[0].split(":")[0]
        next_action = "blocker 해소 후 재 preflight 실행"

    rows = [
        {
            "section": "H",
            "item": "overall_readiness",
            "status": readiness,
            "blocker": "; ".join(blockers) if blockers else "없음",
            "next_required_action": next_action
        }
    ]

    for i, b in enumerate(blockers):
        rows.append({
            "section": "H",
            "item": f"blocker_{i+1}",
            "status": "BLOCKED",
            "blocker": b,
            "next_required_action": "blocker 해소"
        })

    return rows, readiness, blockers


# ─────────────────────────────────────────────
# MD report 생성
# ─────────────────────────────────────────────

def build_md_report(
    sec_a_summary: dict,
    sec_b_summary: list,
    sec_c_summary: dict,
    sec_d_summary: dict,
    sec_g_summary: dict,
    readiness: str,
    blockers: list[str],
) -> str:
    lines = []

    lines.append("# Phase 8.2 stage2_holdout Crop/Manifest Creation Preflight Report")
    lines.append("")
    lines.append(f"**readiness_for_phase8_2_run**: `{readiness}`")
    lines.append("")

    # 1. 목적
    lines.append("## 1. Phase 8.2 Preflight 목적")
    lines.append("")
    lines.append("- stage2_holdout 154명 전체에 대해 dedicated crop root와 dedicated manifest를 생성하기 전 생성 계획 확정")
    lines.append("- 이번 단계는 creation preflight only — 실제 crop/manifest 생성 금지")
    lines.append("- Phase 8.2 실제 생성 실행은 이번 preflight 통과 후 별도 승인으로만 진행")
    lines.append("")

    # 2. 이전 Phase 결과 요약
    lines.append("## 2. Phase 8.0 / 8.1 / 8.1b 결과 요약")
    lines.append("")
    lines.append(f"| Phase | 항목 | 결과 |")
    lines.append(f"|-------|------|------|")
    lines.append(f"| 8.0 | readiness_for_stage2_holdout_eval | `{sec_a_summary.get('phase8_0_readiness')}` ({sec_a_summary.get('phase8_0_status')}) |")
    lines.append(f"| 8.1 | readiness_for_phase8_2 | `{sec_a_summary.get('phase8_1_readiness')}` ({sec_a_summary.get('phase8_1_readiness_status')}) |")
    lines.append(f"| 8.1 | recommended_option | `{sec_a_summary.get('phase8_1_recommended_option')}` ({sec_a_summary.get('phase8_1_option_status')}) |")
    lines.append(f"| 8.1b | audit_status | `{sec_a_summary.get('phase8_1b_audit_status')}` ({sec_a_summary.get('phase8_1b_audit_check')}) |")
    lines.append(f"| 8.1b | stage2_holdout patient count | `{sec_a_summary.get('stage2_holdout_patient_count')}` ({sec_a_summary.get('stage2_patient_count_status')}) |")
    lines.append(f"| 8.1b | contaminated patients | `{sec_a_summary.get('contaminated_patient_ids')}` ({sec_a_summary.get('contamination_audit_status')}) |")
    lines.append(f"| - | Section A overall | **{sec_a_summary.get('section_A_overall')}** |")
    lines.append("")

    # 3. source script inventory
    lines.append("## 3. Source Script Inventory")
    lines.append("")
    found_scripts = [s for s in sec_b_summary if s.get("status") == "FOUND"]
    not_found_scripts = [s for s in sec_b_summary if s.get("status") == "NOT_FOUND"]
    lines.append(f"- **FOUND**: {len(found_scripts)}개")
    lines.append(f"- **NOT_FOUND**: {len(not_found_scripts)}개")
    lines.append("")
    lines.append("| script | role | status |")
    lines.append("|--------|------|--------|")
    for s in sec_b_summary:
        path_short = Path(s["script_path"]).name
        lines.append(f"| {path_short} | {s['candidate_role']} | {s['status']} |")
    lines.append("")

    # 4. source data path preflight
    lines.append("## 4. Source Data Path Preflight")
    lines.append("")
    lines.append(f"- vol_root (nsclc_msd_usable_only_v2): `{sec_c_summary.get('vol_root')}` — exists: {sec_c_summary.get('vol_root_exists')}")
    lines.append(f"- volumes_npy exists: {sec_c_summary.get('volumes_npy_exists')} ({sec_c_summary.get('volumes_npy_patient_dir_count', 'N/A')} dirs)")
    lines.append(f"- stage2_holdout CT/ROI found: {sec_c_summary.get('stage2_ct_roi_found', 'N/A')}/154")
    lines.append(f"- stage2_holdout CT/ROI missing: {sec_c_summary.get('stage2_ct_roi_missing', 'N/A')}")
    lines.append(f"- Section C overall: **{sec_c_summary.get('section_C_overall')}**")
    lines.append("")

    # 5. dedicated output plan
    lines.append("## 5. Dedicated Output Plan")
    lines.append("")
    for key, val in sec_d_summary.items():
        if isinstance(val, dict):
            lines.append(f"- **{key}**: path=`{val.get('path')}`, exists={val.get('exists')}, status=`{val.get('status')}`")
    lines.append(f"- any_output_conflict: {sec_d_summary.get('any_output_conflict')}")
    lines.append(f"- exist_ok=False 원칙: {sec_d_summary.get('exist_ok_false_principle')}")
    lines.append(f"- Section D overall: **{sec_d_summary.get('section_D_overall')}**")
    lines.append("")

    # 6. dedicated manifest schema (핵심 필드만)
    lines.append("## 6. Dedicated Manifest Schema (핵심 필드)")
    lines.append("")
    lines.append("| field | required | expected value/rule |")
    lines.append("|-------|----------|---------------------|")
    key_fields = [
        ("stage_split", "stage2_holdout"),
        ("asset_scope", "dedicated_stage2_holdout"),
        ("approval_required_before_scoring", "True"),
        ("manifest_status", "pending_creation_preflight → created_after_phase8_2_run"),
        ("crop_shape", "(6,96,96)"),
        ("input_channels", "6"),
        ("crop_size", "96"),
        ("contamination_check_status", "clean_new_generation or contaminated_regen_from_ct_roi"),
    ]
    for f, v in key_fields:
        lines.append(f"| {f} | required | {v} |")
    lines.append("")

    # 7. generation safety plan (핵심)
    lines.append("## 7. Generation Safety Plan (핵심)")
    lines.append("")
    lines.append("- stage1_dev patient 0명 포함 (stage2_holdout 전용)")
    lines.append("- LUNG1-295 / LUNG1-415: contaminated old crops 사용 금지 → 원본 CT/ROI에서 신규 생성")
    lines.append("- phase6_1b filtered manifest 재사용 금지")
    lines.append("- original S6-A index 직접 scoring 입력 금지")
    lines.append("- 생성 후 scoring 전 사용자 승인 필수 (approval_required_before_scoring=True)")
    lines.append("- 생성 후 npz shape/value validation → 별도 Phase 8.3")
    lines.append("- tmp → final rename 전략 권장")
    lines.append("")

    # 8. runtime/cost estimate
    lines.append("## 8. Runtime / Cost Estimate")
    lines.append("")
    lines.append(f"| 항목 | 추정 |")
    lines.append(f"|------|------|")
    lines.append(f"| patient count | {sec_g_summary.get('patient_count')} |")
    lines.append(f"| positive crop (lesion_patch_count 합산) | {sec_g_summary.get('lesion_patch_total')} (medium confidence) |")
    lines.append(f"| hard negative crop (rough 2x 추정) | {sec_g_summary.get('hn_estimate_rough')} (low confidence) |")
    lines.append(f"| total crop estimate (rough) | {sec_g_summary.get('total_crop_estimate_rough')} (low confidence) |")
    lines.append(f"| disk usage estimate | {sec_g_summary.get('disk_estimate_gb')} GB (low confidence) |")
    lines.append(f"| group 분포 | {sec_g_summary.get('group_distribution')} |")
    lines.append(f"| OOM 위험 | {sec_g_summary.get('oom_risk')} |")
    lines.append(f"| GPU 필요 (preflight) | {sec_g_summary.get('gpu_required_preflight')} |")
    lines.append(f"| GPU 필요 (Phase 8.2 run) | {sec_g_summary.get('gpu_required_phase8_2_run')} |")
    lines.append(f"| CT/ROI 접근 | {sec_g_summary.get('ct_roi_access_required')} |")
    lines.append("")

    # 9. readiness 판정
    lines.append("## 9. Readiness 판정")
    lines.append("")
    lines.append(f"**{readiness}**")
    lines.append("")
    if blockers:
        lines.append("### Blockers")
        for b in blockers:
            lines.append(f"- {b}")
        lines.append("")

    # 10. 다음 단계
    lines.append("## 10. 다음 단계")
    lines.append("")
    if readiness == "READY_FOR_PHASE8_2_RUN_CROP_MANIFEST_CREATION":
        lines.append("- **Phase 8.2 run**: dedicated crop 생성 및 manifest 생성")
        lines.append("  - 별도 스크립트 작성 후 사용자 승인 요청")
        lines.append("  - `scripts/phase8_2_stage2_holdout_crop_manifest_creation_run.py` (또는 유사 명칭)")
        lines.append("  - 실행 전 생성 script 검토 및 승인 필수")
        lines.append("  - 생성 완료 후 Phase 8.3 npz shape/value validation 진행")
    else:
        lines.append("- blocker 해소 후 Phase 8.2 preflight 재실행")
        for b in blockers:
            lines.append(f"  - {b}")
    lines.append("")

    # 11. 금지 사항
    lines.append("## 11. 금지 사항")
    lines.append("")
    lines.append("- 실제 crop 생성 금지")
    lines.append("- 실제 manifest 생성 금지")
    lines.append("- crop 복사 금지")
    lines.append("- npz 로드 금지")
    lines.append("- CT/ROI/mask npy 로드 금지")
    lines.append("- model forward 금지")
    lines.append("- scoring 금지")
    lines.append("- metric 계산 금지")
    lines.append("- threshold / p95 / p99 계산 금지")
    lines.append("- hit-rate 계산 금지")
    lines.append("- training 금지")
    lines.append("- checkpoint 생성 금지")
    lines.append("- hard negative 최종 채택 금지")
    lines.append("- split CSV 수정 금지")
    lines.append("- 기존 Phase 6/7/8 output 수정 금지")
    lines.append("- NSCLC/MSD root 내용 접근 금지 (1단계 stat만 허용)")
    lines.append("- pip/conda install 금지")
    lines.append("- 외부 다운로드 금지")
    lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# output guard
# ─────────────────────────────────────────────

def check_output_guard() -> None:
    if OUT_DIR.exists():
        print(f"[GUARD] output root already exists: {OUT_DIR}", file=sys.stderr)
        print("[GUARD] 즉시 중단: exist_ok=False 원칙", file=sys.stderr)
        sys.exit(1)
    for f in [OUT_CSV, OUT_JSON, OUT_MD]:
        if f.exists():
            print(f"[GUARD] output file already exists: {f}", file=sys.stderr)
            print("[GUARD] 즉시 중단: exist_ok=False 원칙", file=sys.stderr)
            sys.exit(1)


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 8.2 stage2_holdout crop/manifest creation preflight"
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="실제 CSV/JSON/MD 생성. 없으면 dry-run (파일 미생성)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 8.2 stage2_holdout Crop/Manifest Creation Preflight")
    print("=" * 70)
    print(f"mode: {'--run (파일 생성)' if args.run else 'dry-run (파일 미생성)'}")
    print()

    # output guard: --run 시에만 체크
    if args.run:
        check_output_guard()

    # ── Section A ──
    print("[Section A] prior phase readiness 확인 중...")
    sec_a_rows, sec_a_summary = check_prior_phase_readiness()
    print(f"  → Section A overall: {sec_a_summary.get('section_A_overall')}")

    # ── Section B ──
    print("[Section B] source script inventory 탐색 중...")
    sec_b_rows, sec_b_summary = check_source_script_inventory()
    found_count = sum(1 for s in sec_b_summary if s["status"] == "FOUND")
    print(f"  → FOUND {found_count}개 script")

    # ── Section C ──
    print("[Section C] source data path preflight 중...")
    sec_c_rows, sec_c_summary = check_source_data_path()
    print(f"  → Section C overall: {sec_c_summary.get('section_C_overall')}")
    print(f"  → stage2_holdout CT/ROI found: {sec_c_summary.get('stage2_ct_roi_found', 'N/A')}/154")

    # ── Section D ──
    print("[Section D] dedicated output plan 확인 중...")
    sec_d_rows, sec_d_summary = check_dedicated_output_plan()
    print(f"  → any_output_conflict: {sec_d_summary.get('any_output_conflict')}")
    print(f"  → Section D overall: {sec_d_summary.get('section_D_overall')}")

    # ── Section E ──
    print("[Section E] manifest schema design 정의 중...")
    sec_e_rows, sec_e_summary = check_manifest_schema_design()
    print(f"  → schema fields: {len(sec_e_summary)}개")

    # ── Section F ──
    print("[Section F] generation safety plan 정리 중...")
    sec_f_rows, sec_f_summary = check_generation_safety_plan()
    print(f"  → safety checks: {len(sec_f_summary)}개")

    # ── Section G ──
    print("[Section G] runtime/cost estimate 계산 중...")
    sec_g_rows, sec_g_summary = check_runtime_cost_estimate()
    print(f"  → patient: {sec_g_summary['patient_count']}, positive crop: {sec_g_summary['lesion_patch_total']}")

    # ── Section H ──
    print("[Section H] readiness decision 판정 중...")
    sec_h_rows, readiness, blockers = check_readiness_decision(
        sec_a_summary, sec_c_summary, sec_d_summary
    )
    print(f"  → readiness: {readiness}")
    if blockers:
        for b in blockers:
            print(f"  → BLOCKER: {b}")

    print()
    print(f"readiness_for_phase8_2_run: {readiness}")
    print()

    if not args.run:
        print("[dry-run] 파일 미생성. --run 플래그를 추가하면 CSV/JSON/MD 생성")
        return

    # ── 파일 생성 (--run) ──
    print("[--run] output 생성 중...")

    # output dir 생성 (exist_ok=False 이미 guard에서 확인됨)
    OUT_DIR.mkdir(parents=True, exist_ok=False)

    # CSV 생성
    all_rows = (
        sec_a_rows + sec_b_rows + sec_c_rows + sec_d_rows
        + sec_e_rows + sec_f_rows + sec_g_rows + sec_h_rows
    )
    df_out = pd.DataFrame(all_rows)
    # 저장 직전 재검증
    if OUT_CSV.exists():
        print(f"[GUARD] output CSV already exists: {OUT_CSV}", file=sys.stderr)
        sys.exit(1)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"  CSV: {OUT_CSV}")

    # JSON 생성
    json_out = {
        "input_paths": {
            "phase8_0_json": str(PHASE8_0_JSON),
            "phase8_1_json": str(PHASE8_1_JSON),
            "phase8_1b_json": str(PHASE8_1B_JSON),
            "split_csv": str(SPLIT_CSV),
            "phase6_1b_manifest_csv": str(PHASE6_1B_MANIFEST_CSV),
            "phase6_1b_excluded_csv": str(PHASE6_1B_EXCLUDED_CSV),
        },
        "prior_phase_readiness": sec_a_summary,
        "source_script_inventory": sec_b_summary,
        "source_data_path_preflight": sec_c_summary,
        "dedicated_output_plan": sec_d_summary,
        "manifest_schema_design": sec_e_summary,
        "generation_safety_plan": sec_f_summary,
        "runtime_cost_estimate": sec_g_summary,
        "readiness_for_phase8_2_run": readiness,
        "blockers": blockers,
        "recommended_next_step": (
            "Phase 8.2 run: dedicated crop 생성 및 manifest 생성 별도 승인 요청"
            if not blockers
            else "blocker 해소 후 재 preflight 실행"
        ),
        "notes": {
            "preflight_only": True,
            "no_crop_generation": True,
            "no_manifest_creation": True,
            "no_npz_loading": True,
            "no_ct_roi_npy_loading": True,
            "no_model_forward": True,
            "no_scoring": True,
            "no_metric_calculation": True,
            "no_threshold": True,
            "no_training": True,
        },
    }
    if OUT_JSON.exists():
        print(f"[GUARD] output JSON already exists: {OUT_JSON}", file=sys.stderr)
        sys.exit(1)
    with open(OUT_JSON, "w") as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)
    print(f"  JSON: {OUT_JSON}")

    # MD report 생성
    md_content = build_md_report(
        sec_a_summary=sec_a_summary,
        sec_b_summary=sec_b_summary,
        sec_c_summary=sec_c_summary,
        sec_d_summary=sec_d_summary,
        sec_g_summary=sec_g_summary,
        readiness=readiness,
        blockers=blockers,
    )
    if OUT_MD.exists():
        print(f"[GUARD] output MD already exists: {OUT_MD}", file=sys.stderr)
        sys.exit(1)
    with open(OUT_MD, "w") as f:
        f.write(md_content)
    print(f"  MD:   {OUT_MD}")

    print()
    print("완료.")
    print(f"readiness_for_phase8_2_run: {readiness}")


if __name__ == "__main__":
    main()
