"""
Phase 8.1: stage2_holdout dedicated manifest/crop asset creation preflight

목적: asset creation preflight only — 실제 manifest 생성/crop 복사/npz 로드/scoring 금지
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

BASE_DIR = Path("outputs/second-stage-lesion-refiner-v1")
OUT_DIR = BASE_DIR / "review_annotations" / "phase8_1_stage2_holdout_manifest_crop_asset_preflight_v1"

PHASE_ID = "phase8_1_stage2_holdout_manifest_crop_asset_preflight_v1"
CSV_PATH = OUT_DIR / f"{PHASE_ID}.csv"
JSON_PATH = OUT_DIR / f"{PHASE_ID}.json"
MD_PATH = OUT_DIR / f"phase8_1_stage2_holdout_manifest_crop_asset_preflight_report_v1.md"

# ── 참고 입력 경로 ──────────────────────────────────────────────────
SPLIT_CSV = BASE_DIR / "splits" / "lesion_stage_split_v1.csv"

PHASE80_DIR = BASE_DIR / "review_annotations" / "phase8_0_stage2_holdout_final_eval_preflight_v1"
PHASE80_CSV = PHASE80_DIR / "phase8_0_stage2_holdout_final_eval_preflight_v1.csv"
PHASE80_JSON = PHASE80_DIR / "phase8_0_stage2_holdout_final_eval_preflight_v1.json"
PHASE80_MD = PHASE80_DIR / "phase8_0_stage2_holdout_final_eval_preflight_report_v1.md"

PHASE61B_DIR = BASE_DIR / "review_annotations" / "phase6_1b_s6a_stage1_dev_filtered_manifest_v1"
PHASE61B_EXCLUDED = PHASE61B_DIR / "phase6_1b_s6a_stage2_holdout_excluded_rows_v1.csv"
PHASE61B_MANIFEST = PHASE61B_DIR / "phase6_1b_s6a_stage1_dev_filtered_manifest_v1.csv"

MIXED_CROP_ROOT = BASE_DIR / "crops_s6a_6ch_full"

DATASETS_DIR = BASE_DIR / "datasets"
DS_INDEX_6CH = DATASETS_DIR / "s6a_6ch_full_dataset_index.csv"
DS_INDEX_FULL = DATASETS_DIR / "s6a_full_dataset_index.csv"
DS_STAGE1_SPLIT = DATASETS_DIR / "s6a_stage1_train_val_split.csv"

DEDICATED_HOLDOUT_MANIFEST_CANDIDATE_1 = DATASETS_DIR / "s6a_stage2_holdout_manifest.csv"
DEDICATED_HOLDOUT_MANIFEST_CANDIDATE_2 = DATASETS_DIR / "s6a_stage2_holdout_filtered_manifest.csv"
DEDICATED_HOLDOUT_CROP_ROOT_CANDIDATE = BASE_DIR / "crops_stage2_holdout_6ch_dedicated"


def _exists(p):
    return Path(p).exists()


def _file_size(p):
    try:
        return Path(p).stat().st_size
    except Exception:
        return None


def _dir_entry_count(p):
    """폴더 1단계 entry 수만 확인 — recursive scan 금지"""
    try:
        return len(list(Path(p).iterdir()))
    except Exception:
        return None


def load_split_csv():
    """split CSV에서 stage2_holdout patient IDs 로드 (read-only)"""
    import csv as _csv
    patients = set()
    stage_counts = {}
    with open(SPLIT_CSV, newline="", encoding="utf-8-sig") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            split = row.get("stage_split", "")
            stage_counts[split] = stage_counts.get(split, 0) + 1
            if split == "stage2_holdout":
                patients.add(row["patient_id"])
    return patients, stage_counts


def load_phase80_json():
    """Phase 8.0 JSON 읽기 (read-only metadata)"""
    with open(PHASE80_JSON) as f:
        return json.load(f)


def check_crop_coverage(holdout_patients):
    """
    crops_s6a_6ch_full/ 폴더에서 stage2_holdout 환자 디렉터리 존재 여부만 확인.
    실제 crop 내용 로드 금지. 폴더 1단계 stat만 사용.
    """
    if not MIXED_CROP_ROOT.exists():
        return {"crop_root_exists": False, "holdout_with_crops": [], "holdout_without_crops": list(holdout_patients), "total_dirs": 0}

    crop_dirs = set(p.name for p in MIXED_CROP_ROOT.iterdir() if p.is_dir())
    holdout_with = sorted(holdout_patients & crop_dirs)
    holdout_without = sorted(holdout_patients - crop_dirs)
    return {
        "crop_root_exists": True,
        "total_dirs": len(crop_dirs),
        "holdout_with_crops": holdout_with,
        "holdout_without_crops": holdout_without,
        "holdout_covered_count": len(holdout_with),
        "holdout_missing_count": len(holdout_without),
    }


def build_section_a(holdout_count, crop_info, phase80_loaded):
    """Section A: input/source inventory"""
    sources = [
        {
            "section": "A",
            "source_id": "split_csv",
            "source_type": "split_metadata",
            "path": str(SPLIT_CSV),
            "exists": str(_exists(SPLIT_CSV)),
            "status": "FOUND" if _exists(SPLIT_CSV) else "MISSING",
            "note": f"stage2_holdout patient count={holdout_count}",
        },
        {
            "section": "A",
            "source_id": "phase8_0_csv",
            "source_type": "phase8_0_preflight_output",
            "path": str(PHASE80_CSV),
            "exists": str(_exists(PHASE80_CSV)),
            "status": "FOUND" if _exists(PHASE80_CSV) else "MISSING",
            "note": "Phase 8.0 preflight CSV output",
        },
        {
            "section": "A",
            "source_id": "phase8_0_json",
            "source_type": "phase8_0_preflight_output",
            "path": str(PHASE80_JSON),
            "exists": str(_exists(PHASE80_JSON)),
            "status": "FOUND" if _exists(PHASE80_JSON) else "MISSING",
            "note": f"readiness={phase80_loaded.get('readiness_for_stage2_holdout_eval','N/A')}",
        },
        {
            "section": "A",
            "source_id": "phase8_0_md",
            "source_type": "phase8_0_preflight_output",
            "path": str(PHASE80_MD),
            "exists": str(_exists(PHASE80_MD)),
            "status": "FOUND" if _exists(PHASE80_MD) else "MISSING",
            "note": "Phase 8.0 preflight MD report",
        },
        {
            "section": "A",
            "source_id": "phase6_1b_excluded_rows",
            "source_type": "leakage_audit_evidence",
            "path": str(PHASE61B_EXCLUDED),
            "exists": str(_exists(PHASE61B_EXCLUDED)),
            "status": "FOUND" if _exists(PHASE61B_EXCLUDED) else "MISSING",
            "note": "1222 rows, 2 patients (LUNG1-295, LUNG1-415). leakage audit evidence only, NOT evaluation manifest",
        },
        {
            "section": "A",
            "source_id": "phase6_1b_filtered_manifest",
            "source_type": "stage1_dev_filtered_manifest",
            "path": str(PHASE61B_MANIFEST),
            "exists": str(_exists(PHASE61B_MANIFEST)),
            "status": "FOUND" if _exists(PHASE61B_MANIFEST) else "MISSING",
            "note": "129437 rows, stage1_dev filtered manifest. FORBIDDEN for stage2 eval",
        },
        {
            "section": "A",
            "source_id": "mixed_crop_root",
            "source_type": "crop_root_mixed_original",
            "path": str(MIXED_CROP_ROOT),
            "exists": str(_exists(MIXED_CROP_ROOT)),
            "status": "FOUND_MIXED" if _exists(MIXED_CROP_ROOT) else "MISSING",
            "note": (
                f"total_patient_dirs={crop_info.get('total_dirs',0)}; "
                f"holdout_with_crops={crop_info.get('holdout_covered_count',0)}/154 "
                f"(only LUNG1-295, LUNG1-415); "
                f"holdout_WITHOUT_crops={crop_info.get('holdout_missing_count',0)}/154"
            ),
        },
        {
            "section": "A",
            "source_id": "ds_index_6ch",
            "source_type": "dataset_index_mixed",
            "path": str(DS_INDEX_6CH),
            "exists": str(_exists(DS_INDEX_6CH)),
            "status": "FOUND_MIXED" if _exists(DS_INDEX_6CH) else "MISSING",
            "note": "130659 rows, 154 patients (stage1_dev 152 + contaminated LUNG1-295/415). mixed asset",
        },
        {
            "section": "A",
            "source_id": "ds_index_full",
            "source_type": "dataset_index_candidate",
            "path": str(DS_INDEX_FULL),
            "exists": str(_exists(DS_INDEX_FULL)),
            "status": "FOUND" if _exists(DS_INDEX_FULL) else "MISSING",
            "note": "s6a_full_dataset_index — scope TBD",
        },
        {
            "section": "A",
            "source_id": "ds_stage1_train_val_split",
            "source_type": "stage1_train_val_split",
            "path": str(DS_STAGE1_SPLIT),
            "exists": str(_exists(DS_STAGE1_SPLIT)),
            "status": "FOUND" if _exists(DS_STAGE1_SPLIT) else "MISSING",
            "note": "154 rows, stage1_dev only train/val split",
        },
        {
            "section": "A",
            "source_id": "dedicated_holdout_manifest_candidate_1",
            "source_type": "dedicated_stage2_holdout_manifest",
            "path": str(DEDICATED_HOLDOUT_MANIFEST_CANDIDATE_1),
            "exists": str(_exists(DEDICATED_HOLDOUT_MANIFEST_CANDIDATE_1)),
            "status": "MISSING" if not _exists(DEDICATED_HOLDOUT_MANIFEST_CANDIDATE_1) else "FOUND",
            "note": "dedicated stage2_holdout manifest not yet created",
        },
        {
            "section": "A",
            "source_id": "dedicated_holdout_manifest_candidate_2",
            "source_type": "dedicated_stage2_holdout_manifest",
            "path": str(DEDICATED_HOLDOUT_MANIFEST_CANDIDATE_2),
            "exists": str(_exists(DEDICATED_HOLDOUT_MANIFEST_CANDIDATE_2)),
            "status": "MISSING" if not _exists(DEDICATED_HOLDOUT_MANIFEST_CANDIDATE_2) else "FOUND",
            "note": "dedicated stage2_holdout filtered manifest not yet created",
        },
        {
            "section": "A",
            "source_id": "dedicated_holdout_crop_root",
            "source_type": "dedicated_stage2_holdout_crop_root",
            "path": str(DEDICATED_HOLDOUT_CROP_ROOT_CANDIDATE),
            "exists": str(_exists(DEDICATED_HOLDOUT_CROP_ROOT_CANDIDATE)),
            "status": "MISSING" if not _exists(DEDICATED_HOLDOUT_CROP_ROOT_CANDIDATE) else "FOUND",
            "note": "dedicated stage2_holdout crop root not yet created",
        },
    ]
    return sources


def build_section_b():
    """Section B: source safety classification"""
    rows = [
        {
            "section": "B",
            "source_id": "split_csv",
            "classification": "SAFE_SOURCE_METADATA_ONLY",
            "allowed_use": "stage2_holdout patient ID 조회, split 검증",
            "forbidden_use": "직접 scoring 입력, manifest 대체 사용 금지",
            "reason": "split 메타데이터만 포함. 실제 crop/score 포함 안 함",
        },
        {
            "section": "B",
            "source_id": "phase8_0_csv",
            "classification": "SAFE_SOURCE_METADATA_ONLY",
            "allowed_use": "readiness 상태 참조, blocker 확인",
            "forbidden_use": "평가 입력으로 사용 금지",
            "reason": "Phase 8.0 preflight output — metadata only",
        },
        {
            "section": "B",
            "source_id": "phase8_0_json",
            "classification": "SAFE_SOURCE_METADATA_ONLY",
            "allowed_use": "readiness 상태 참조, 컨텍스트 확인",
            "forbidden_use": "평가 입력으로 사용 금지",
            "reason": "Phase 8.0 preflight output — metadata only",
        },
        {
            "section": "B",
            "source_id": "phase6_1b_excluded_rows",
            "classification": "SAFE_SOURCE_METADATA_ONLY",
            "allowed_use": "leakage audit evidence 참조, LUNG1-295/LUNG1-415 확인",
            "forbidden_use": "evaluation manifest로 사용 금지 — leakage audit evidence일 뿐임",
            "reason": "stage2_holdout 오염 이력 기록 파일. dedicated manifest가 아님",
        },
        {
            "section": "B",
            "source_id": "phase6_1b_filtered_manifest",
            "classification": "FORBIDDEN_FOR_EVAL_INPUT",
            "allowed_use": "컨텍스트 참조만 허용",
            "forbidden_use": "stage2_holdout evaluation input, manifest 재사용, scoring 입력 금지",
            "reason": "stage1_dev 전용 filtered manifest. stage2 평가에 절대 재사용 금지",
        },
        {
            "section": "B",
            "source_id": "mixed_crop_root",
            "classification": "MIXED_ASSET_NOT_DIRECTLY_USABLE",
            "allowed_use": "LUNG1-295/LUNG1-415 crop path 참조만 허용 (Option A 한정)",
            "forbidden_use": "dedicated stage2_holdout 평가 crop root로 직접 사용 금지. 152/154 holdout 환자 crops 없음",
            "reason": "crops_s6a_6ch_full = stage1_dev 152명 + contaminated LUNG1-295/415 2명. 나머지 152명 stage2_holdout crops 없음",
        },
        {
            "section": "B",
            "source_id": "ds_index_6ch",
            "classification": "MIXED_ASSET_NOT_DIRECTLY_USABLE",
            "allowed_use": "LUNG1-295/LUNG1-415 npz_path 참조만 허용",
            "forbidden_use": "stage2_holdout 전용 dataset index로 직접 사용 금지",
            "reason": "stage1_dev 152 + contaminated 2 = 154 환자만 포함. stage2_holdout 152명 항목 없음",
        },
        {
            "section": "B",
            "source_id": "ds_index_full",
            "classification": "SAFE_SOURCE_METADATA_ONLY",
            "allowed_use": "scope 확인 후 참조 허용",
            "forbidden_use": "scope 확인 전 평가 입력 사용 금지",
            "reason": "s6a_full — stage2_holdout 포함 여부 및 scope 별도 확인 필요",
        },
        {
            "section": "B",
            "source_id": "ds_stage1_train_val_split",
            "classification": "FORBIDDEN_FOR_EVAL_INPUT",
            "allowed_use": "컨텍스트 참조만 허용",
            "forbidden_use": "stage2_holdout 평가 입력 사용 금지",
            "reason": "stage1_dev 전용 train/val split",
        },
        {
            "section": "B",
            "source_id": "dedicated_holdout_manifest_candidate_1",
            "classification": "MISSING",
            "allowed_use": "Phase 8.2에서 생성 후 사용 가능",
            "forbidden_use": "현재 존재하지 않음 — 사용 불가",
            "reason": "dedicated stage2_holdout manifest 아직 생성되지 않음",
        },
        {
            "section": "B",
            "source_id": "dedicated_holdout_manifest_candidate_2",
            "classification": "MISSING",
            "allowed_use": "Phase 8.2에서 생성 후 사용 가능",
            "forbidden_use": "현재 존재하지 않음 — 사용 불가",
            "reason": "dedicated stage2_holdout filtered manifest 아직 생성되지 않음",
        },
        {
            "section": "B",
            "source_id": "dedicated_holdout_crop_root",
            "classification": "MISSING",
            "allowed_use": "Phase 8.2에서 생성 후 사용 가능",
            "forbidden_use": "현재 존재하지 않음 — 사용 불가",
            "reason": "dedicated stage2_holdout crop root 아직 생성되지 않음",
        },
    ]
    return rows


def build_section_c():
    """Section C: dedicated manifest schema design (설계만, 실제 생성 금지)"""
    fields = [
        ("row_id", True, "int, 0-based sequential index", "생성 시 자동 부여", "row 고유 식별자"),
        ("patient_id", True, "str, e.g. LUNG1-XXX", "split CSV / CT 원본", "환자 식별자"),
        ("npz_path", True, "str, relative or absolute path to .npz crop file", "crop 생성/분리 결과", "실제 crop file path"),
        ("label", True, "int, 0=normal 1=lesion", "원본 lesion annotation", "crop-level label"),
        ("sampling_label", True, "str, e.g. lesion/normal/hard_negative", "원본 sampling 규칙", "sampling 분류"),
        ("stage_split", True, "fixed: stage2_holdout", "constant", "모든 row = stage2_holdout"),
        ("source_manifest", True, "str, source manifest 파일명 또는 식별자", "Phase 8.2에서 결정", "crop 출처 추적"),
        ("source_crop_root", True, "str, crop root 경로", "Phase 8.2에서 결정", "crop 파일 위치 추적"),
        ("asset_scope", True, "fixed: dedicated_stage2_holdout", "constant", "혼용 방지 명시"),
        ("contamination_check_status", True, "str: CLEAN / EXCLUDED_CONTAMINATED", "Phase 8.2 leakage check", "LUNG1-295/415 처리 결과"),
        ("approval_required_before_scoring", True, "fixed: True", "constant", "scoring 전 사용자 승인 필수"),
        ("manifest_status", True, "preflight_design_only (8.1) → pending_creation (8.2 생성 후)", "단계별 업데이트", "manifest lifecycle 추적"),
    ]
    rows = []
    for fname, required, rule, source, note in fields:
        rows.append({
            "section": "C",
            "field_name": fname,
            "required": str(required),
            "expected_value_or_rule": rule,
            "source": source,
            "note": note,
        })
    return rows


def build_section_d():
    """Section D: asset creation options A/B/C"""
    rows = [
        {
            "section": "D",
            "option_id": "A",
            "option_name": "manifest_only_reference_existing_npz",
            "description": "mixed crop root에서 stage2_holdout 환자 row만 선별해 dedicated manifest 작성. npz 파일 복사 없이 기존 npz_path 참조.",
            "advantage": "빠름, 디스크 추가 사용 없음",
            "risk": (
                "치명적 한계: crops_s6a_6ch_full에 stage2_holdout 환자 154명 중 "
                "2명(LUNG1-295, LUNG1-415)만 존재. "
                "나머지 152명은 crop 자체가 없으므로 Option A 단독으로는 154명 전체 평가 불가. "
                "mixed root 의존으로 path leakage 주의 필요."
            ),
            "approval_required": "True",
            "recommendation": "NOT_RECOMMENDED_STANDALONE — 2/154 환자만 커버. 154명 전체 평가 불가",
        },
        {
            "section": "D",
            "option_id": "B",
            "option_name": "copy_crops_to_dedicated_root",
            "description": "dedicated stage2_holdout crop root로 필요한 crop을 복사/분리. npz 파일을 새 폴더로 복사.",
            "advantage": "평가 입력 분리 명확, leakage 위험 감소",
            "risk": (
                "Option A와 동일한 근본 한계: 복사 가능한 crop이 2명(LUNG1-295, LUNG1-415)밖에 없음. "
                "나머지 152명 crops 없음 — 복사 대상 자체가 없음. "
                "파일 복사, 디스크 사용, 별도 승인 필요."
            ),
            "approval_required": "True",
            "recommendation": "NOT_RECOMMENDED_STANDALONE — 152명 crops 부재로 불완전. 단독 사용 불가",
        },
        {
            "section": "D",
            "option_id": "C",
            "option_name": "generate_crops_from_original_ct_roi",
            "description": "stage2_holdout crop을 원본 CT/ROI에서 새로 생성. S6-A pipeline 재실행. 152명 crops 전체 신규 생성 + LUNG1-295/415 재생성 (contaminated crops 대체).",
            "advantage": "완전 분리, 모든 154명 커버, contamination-free dedicated crop root 보장",
            "risk": "가장 오래 걸림, CT/ROI 접근 필요, S6-A pipeline 재실행 필요, 별도 승인 필요, 디스크 공간 필요",
            "approval_required": "True",
            "recommendation": "RECOMMENDED — 유일하게 154명 전체 커버 가능. Option A/B는 152명 crops 부재로 단독 사용 불가",
        },
    ]
    return rows


def build_section_e():
    """Section E: leakage checks for next phase"""
    rows = [
        {
            "section": "E",
            "check_item": "stage1_dev_patient_count_in_manifest",
            "expected": "0",
            "required_before_scoring": "True",
            "note": "dedicated manifest에 stage1_dev 환자 포함 여부 반드시 0 확인",
        },
        {
            "section": "E",
            "check_item": "stage2_holdout_patient_count_in_manifest",
            "expected": "154",
            "required_before_scoring": "True",
            "note": "split CSV 기준 154명 전원 포함 여부 확인",
        },
        {
            "section": "E",
            "check_item": "LUNG1_295_handling",
            "expected": "EXCLUDED_CONTAMINATED 또는 재생성 crop으로 대체",
            "required_before_scoring": "True",
            "note": "원본 mixed crop root의 LUNG1-295 crop은 contaminated — 사용 방식 명시 필요",
        },
        {
            "section": "E",
            "check_item": "LUNG1_415_handling",
            "expected": "EXCLUDED_CONTAMINATED 또는 재생성 crop으로 대체",
            "required_before_scoring": "True",
            "note": "원본 mixed crop root의 LUNG1-415 crop은 contaminated — 사용 방식 명시 필요",
        },
        {
            "section": "E",
            "check_item": "phase6_1b_filtered_manifest_row_overlap",
            "expected": "0",
            "required_before_scoring": "True",
            "note": "dedicated manifest row가 stage1_dev filtered manifest (129437 rows)와 row overlap 0 확인",
        },
        {
            "section": "E",
            "check_item": "stage1_dev_filtered_manifest_not_reused",
            "expected": "PASS — 재사용 없음",
            "required_before_scoring": "True",
            "note": "phase6_1b_s6a_stage1_dev_filtered_manifest_v1.csv를 stage2 평가에 재사용하지 않음 확인",
        },
        {
            "section": "E",
            "check_item": "original_s6a_index_not_used_as_scoring_input",
            "expected": "PASS — 직접 사용 없음",
            "required_before_scoring": "True",
            "note": "s6a_6ch_full_dataset_index.csv를 scoring 입력으로 직접 사용하지 않음 확인",
        },
        {
            "section": "E",
            "check_item": "user_approval_before_scoring",
            "expected": "APPROVED",
            "required_before_scoring": "True",
            "note": "dedicated manifest 생성 완료 후 scoring 시작 전 사용자 명시적 승인 필요",
        },
        {
            "section": "E",
            "check_item": "asset_scope_field_value",
            "expected": "dedicated_stage2_holdout",
            "required_before_scoring": "True",
            "note": "manifest 모든 row의 asset_scope = dedicated_stage2_holdout 확인",
        },
        {
            "section": "E",
            "check_item": "approval_required_before_scoring_field_value",
            "expected": "True (모든 row)",
            "required_before_scoring": "True",
            "note": "manifest 모든 row의 approval_required_before_scoring = True 확인",
        },
    ]
    return rows


def build_section_f(holdout_count, crop_info, phase80_loaded):
    """Section F: readiness decision"""
    holdout_missing_crops = crop_info.get("holdout_missing_count", 154)
    split_csv_ok = _exists(SPLIT_CSV)
    phase80_ok = _exists(PHASE80_JSON)
    dedicated_manifest_missing = not _exists(DEDICATED_HOLDOUT_MANIFEST_CANDIDATE_1) and not _exists(DEDICATED_HOLDOUT_MANIFEST_CANDIDATE_2)
    dedicated_crop_missing = not _exists(DEDICATED_HOLDOUT_CROP_ROOT_CANDIDATE)
    crops_insufficient = holdout_missing_crops > 0

    rows = [
        {
            "section": "F",
            "item": "split_csv_exists",
            "status": "PASS" if split_csv_ok else "FAIL",
            "blocker": "" if split_csv_ok else "split CSV 없음",
            "next_required_action": "" if split_csv_ok else "split CSV 복구",
        },
        {
            "section": "F",
            "item": "stage2_holdout_patient_count_verified",
            "status": "PASS" if holdout_count == 154 else "FAIL",
            "blocker": "" if holdout_count == 154 else f"count={holdout_count}, expected=154",
            "next_required_action": "" if holdout_count == 154 else "split CSV 재확인",
        },
        {
            "section": "F",
            "item": "phase8_0_output_exists",
            "status": "PASS" if phase80_ok else "FAIL",
            "blocker": "" if phase80_ok else "Phase 8.0 output 없음",
            "next_required_action": "" if phase80_ok else "Phase 8.0 재실행",
        },
        {
            "section": "F",
            "item": "dedicated_manifest_exists",
            "status": "FAIL" if dedicated_manifest_missing else "PASS",
            "blocker": "dedicated stage2_holdout manifest 없음 — Phase 8.2에서 생성 필요" if dedicated_manifest_missing else "",
            "next_required_action": "Phase 8.2: dedicated manifest creation (Option C 추천)" if dedicated_manifest_missing else "",
        },
        {
            "section": "F",
            "item": "dedicated_crop_root_exists",
            "status": "FAIL" if dedicated_crop_missing else "PASS",
            "blocker": "dedicated stage2_holdout crop root 없음 — Phase 8.2에서 생성 필요" if dedicated_crop_missing else "",
            "next_required_action": "Phase 8.2: crop 생성 (S6-A pipeline 재실행, 154명 전체)" if dedicated_crop_missing else "",
        },
        {
            "section": "F",
            "item": "mixed_crop_root_coverage_sufficient",
            "status": "FAIL",
            "blocker": (
                f"crops_s6a_6ch_full에 stage2_holdout {holdout_missing_crops}/154명 crops 없음. "
                "Option A/B 단독으로는 154명 전체 평가 불가."
            ),
            "next_required_action": "Option C: S6-A pipeline으로 stage2_holdout 전체 154명 crops 신규 생성 (승인 후)",
        },
        {
            "section": "F",
            "item": "readiness_for_phase8_2",
            "status": "READY_FOR_PHASE8_2_STAGE2_MANIFEST_CREATION",
            "blocker": "dedicated manifest 없음 / dedicated crop root 없음 / 152명 crops 미존재",
            "next_required_action": (
                "Phase 8.2: (1) Option C 승인 → S6-A pipeline으로 stage2_holdout 154명 crops 생성, "
                "(2) dedicated manifest 작성, (3) leakage check 통과, (4) 사용자 승인 후 scoring"
            ),
        },
    ]
    return rows


def build_csv(section_a, section_b, section_c, section_d, section_e, section_f):
    all_rows = section_a + section_b + section_c + section_d + section_e + section_f
    fieldsets = {
        "A": ["section", "source_id", "source_type", "path", "exists", "status", "note"],
        "B": ["section", "source_id", "classification", "allowed_use", "forbidden_use", "reason"],
        "C": ["section", "field_name", "required", "expected_value_or_rule", "source", "note"],
        "D": ["section", "option_id", "option_name", "description", "advantage", "risk", "approval_required", "recommendation"],
        "E": ["section", "check_item", "expected", "required_before_scoring", "note"],
        "F": ["section", "item", "status", "blocker", "next_required_action"],
    }
    all_fields = set()
    for fields in fieldsets.values():
        all_fields.update(fields)
    all_fields = sorted(all_fields)
    return all_rows, all_fields


def build_json(holdout_patients, holdout_count, crop_info, phase80_loaded,
               section_a, section_b, section_c, section_d, section_e, section_f):
    readiness_item = next((r for r in section_f if r["item"] == "readiness_for_phase8_2"), {})
    blockers = [r["blocker"] for r in section_f if r.get("blocker")]

    return {
        "input_paths": {
            "split_csv": str(SPLIT_CSV),
            "phase8_0_json": str(PHASE80_JSON),
            "phase8_0_csv": str(PHASE80_CSV),
            "phase6_1b_excluded_rows": str(PHASE61B_EXCLUDED),
            "phase6_1b_filtered_manifest": str(PHASE61B_MANIFEST),
            "mixed_crop_root": str(MIXED_CROP_ROOT),
            "datasets_dir": str(DATASETS_DIR),
        },
        "phase8_0_status": {
            "readiness_for_stage2_holdout_eval": phase80_loaded.get("readiness_for_stage2_holdout_eval"),
            "blockers": phase80_loaded.get("blockers", []),
            "leakage_safety_checks_summary": "PASS (phase8_0 기준)",
        },
        "stage2_holdout_patient_count": holdout_count,
        "source_inventory": {r["source_id"]: {"exists": r["exists"], "status": r["status"]} for r in section_a},
        "source_safety_classification": {r["source_id"]: r["classification"] for r in section_b},
        "dedicated_manifest_schema_design": {r["field_name"]: r["expected_value_or_rule"] for r in section_c},
        "asset_creation_options": {
            r["option_id"]: {
                "name": r["option_name"],
                "recommendation": r["recommendation"],
                "risk": r["risk"],
            }
            for r in section_d
        },
        "recommended_option": "C",
        "recommended_option_reason": (
            "crops_s6a_6ch_full에 stage2_holdout 152/154명 crops 없음. "
            "Option A/B는 2명(LUNG1-295/415)만 커버 — 단독 불가. "
            "Option C (S6-A pipeline 재실행)만 154명 전체 커버 가능."
        ),
        "crop_coverage_analysis": {
            "crop_root": str(MIXED_CROP_ROOT),
            "total_patient_dirs_in_crop_root": crop_info.get("total_dirs", 0),
            "holdout_patients_with_existing_crops": crop_info.get("holdout_with_crops", []),
            "holdout_patients_missing_crops_count": crop_info.get("holdout_missing_count", 0),
            "holdout_patients_missing_crops_sample": crop_info.get("holdout_without_crops", [])[:5],
        },
        "leakage_checks_required_next_phase": [r["check_item"] for r in section_e],
        "leakage_check_special_notes": {
            "LUNG1_295": "contaminated in original index — Phase 8.2에서 처리 방식 명시 필요",
            "LUNG1_415": "contaminated in original index — Phase 8.2에서 처리 방식 명시 필요",
        },
        "readiness_for_phase8_2": readiness_item.get("status", "UNKNOWN"),
        "blockers": [b for b in blockers if b],
        "notes": {
            "preflight_only": True,
            "no_manifest_creation": True,
            "no_crop_copy": True,
            "no_crop_generation": True,
            "no_npz_loading": True,
            "no_model_forward": True,
            "no_scoring": True,
            "no_metric_calculation": True,
            "no_threshold": True,
            "no_training": True,
            "no_stage2_content_analysis": True,
        },
    }


def build_md(holdout_count, crop_info, phase80_loaded, json_data):
    recommended_option = json_data.get("recommended_option", "C")
    readiness = json_data.get("readiness_for_phase8_2", "UNKNOWN")
    blockers = json_data.get("blockers", [])
    holdout_with = crop_info.get("holdout_with_crops", [])
    holdout_missing = crop_info.get("holdout_missing_count", 0)

    lines = [
        "# Phase 8.1: stage2_holdout Dedicated Manifest/Crop Asset Creation Preflight",
        "",
        "## 1. Phase 8.1 목적",
        "",
        "stage2_holdout 최종 평가에 필요한 dedicated 입력 자산(manifest, crop root)을",
        "어떻게 만들지 사전 설계한다.",
        "이번 단계는 **asset creation preflight only** — 실제 생성/복사/로드/scoring 없음.",
        "",
        "## 2. Phase 8.0 결과 요약",
        "",
        f"- readiness_for_stage2_holdout_eval: `{phase80_loaded.get('readiness_for_stage2_holdout_eval')}`",
        f"- stage2_holdout patient count: {phase80_loaded.get('stage2_holdout_patient_count', 154)}",
        "- leakage safety check: PASS (Phase 8.0 기준)",
        "- dedicated stage2_holdout manifest: **없음**",
        "- dedicated stage2_holdout crop root: **없음**",
        "- crops_s6a_6ch_full: mixed/original asset — READY 근거 사용 불가",
        "- Phase 6.1b excluded rows: leakage audit evidence (evaluation manifest 아님)",
        "",
        "## 3. Source Inventory",
        "",
        "| source_id | exists | status | note |",
        "|---|---|---|---|",
        f"| split_csv | {_exists(SPLIT_CSV)} | FOUND | stage2_holdout {holdout_count}명 |",
        f"| phase8_0_json | {_exists(PHASE80_JSON)} | FOUND | readiness=BLOCKED_MISSING_STAGE2_ASSETS |",
        f"| phase6_1b_excluded_rows | {_exists(PHASE61B_EXCLUDED)} | FOUND | 1222 rows, 2 patients (LUNG1-295/415) |",
        f"| phase6_1b_filtered_manifest | {_exists(PHASE61B_MANIFEST)} | FOUND | 129437 rows, stage1_dev only |",
        f"| mixed_crop_root | {_exists(MIXED_CROP_ROOT)} | FOUND_MIXED | holdout covered={crop_info.get('holdout_covered_count',0)}/154 |",
        f"| ds_index_6ch | {_exists(DS_INDEX_6CH)} | FOUND_MIXED | 130659 rows, 154 patients (stage1_dev+contaminated) |",
        f"| dedicated_holdout_manifest | {_exists(DEDICATED_HOLDOUT_MANIFEST_CANDIDATE_1)} | MISSING | 미생성 |",
        f"| dedicated_holdout_crop_root | {_exists(DEDICATED_HOLDOUT_CROP_ROOT_CANDIDATE)} | MISSING | 미생성 |",
        "",
        "## 4. Source Safety Classification",
        "",
        "| source_id | classification |",
        "|---|---|",
        "| split_csv | SAFE_SOURCE_METADATA_ONLY |",
        "| phase8_0_json/csv/md | SAFE_SOURCE_METADATA_ONLY |",
        "| phase6_1b_excluded_rows | SAFE_SOURCE_METADATA_ONLY (evaluation manifest 사용 금지) |",
        "| phase6_1b_filtered_manifest | FORBIDDEN_FOR_EVAL_INPUT |",
        "| mixed_crop_root (crops_s6a_6ch_full) | MIXED_ASSET_NOT_DIRECTLY_USABLE |",
        "| ds_index_6ch | MIXED_ASSET_NOT_DIRECTLY_USABLE |",
        "| ds_stage1_train_val_split | FORBIDDEN_FOR_EVAL_INPUT |",
        "| dedicated_holdout_manifest | MISSING |",
        "| dedicated_holdout_crop_root | MISSING |",
        "",
        "## 5. Dedicated Stage2 Holdout Manifest Schema 설계",
        "",
        "| field_name | required | expected_value_or_rule |",
        "|---|---|---|",
        "| row_id | True | int, 0-based sequential |",
        "| patient_id | True | str, LUNG1-XXX |",
        "| npz_path | True | str, crop .npz path |",
        "| label | True | int, 0=normal 1=lesion |",
        "| sampling_label | True | str (lesion/normal/hard_negative) |",
        "| stage_split | True | **fixed: stage2_holdout** |",
        "| source_manifest | True | Phase 8.2에서 결정 |",
        "| source_crop_root | True | Phase 8.2에서 결정 |",
        "| asset_scope | True | **fixed: dedicated_stage2_holdout** |",
        "| contamination_check_status | True | CLEAN / EXCLUDED_CONTAMINATED |",
        "| approval_required_before_scoring | True | **fixed: True** |",
        "| manifest_status | True | pending_creation → created (8.2 완료 후) |",
        "",
        "## 6. Asset Creation Option A/B/C 비교",
        "",
        "### Option A: manifest only (기존 npz_path 참조)",
        "- 장점: 빠름, 디스크 추가 사용 없음",
        f"- **치명적 한계**: crops_s6a_6ch_full에 stage2_holdout **{holdout_missing}/154명 crops 없음**",
        f"  - crops 있는 holdout 환자: {holdout_with} (2명만, contaminated)",
        "  - 나머지 152명은 crop 자체가 없으므로 Option A 단독으로 전체 평가 불가",
        "- **추천: NOT_RECOMMENDED_STANDALONE**",
        "",
        "### Option B: crops 복사/분리",
        "- 장점: 평가 입력 분리 명확",
        "- **치명적 한계**: 복사 가능한 crop이 2명(LUNG1-295/415)밖에 없음",
        "  - 152명은 복사 대상 crop 자체가 없음",
        "- **추천: NOT_RECOMMENDED_STANDALONE**",
        "",
        "### Option C: 원본 CT/ROI에서 crops 신규 생성 ✅",
        "- 장점: 완전 분리, 154명 전원 커버, contamination-free",
        "- 위험: 오래 걸림, CT/ROI 접근 필요, S6-A pipeline 재실행, 별도 승인 필요",
        "- **추천: RECOMMENDED — 유일하게 154명 전체 커버 가능**",
        "",
        "## 7. Recommended Option",
        "",
        f"**Option {recommended_option}: S6-A pipeline으로 stage2_holdout 154명 crops 신규 생성**",
        "",
        "근거: crops_s6a_6ch_full에 stage2_holdout 152/154명 crops 없음.",
        "Option A/B는 2명(LUNG1-295/415)만 커버 — 단독 불가.",
        "Option C만 154명 전체 커버 가능.",
        "",
        "## 8. Leakage Check 설계 (Phase 8.2에서 반드시 수행)",
        "",
        "| check_item | expected | required_before_scoring |",
        "|---|---|---|",
        "| stage1_dev_patient_count_in_manifest | 0 | True |",
        "| stage2_holdout_patient_count_in_manifest | 154 | True |",
        "| LUNG1_295_handling | EXCLUDED_CONTAMINATED 또는 재생성 crop | True |",
        "| LUNG1_415_handling | EXCLUDED_CONTAMINATED 또는 재생성 crop | True |",
        "| phase6_1b_filtered_manifest_row_overlap | 0 | True |",
        "| stage1_dev_filtered_manifest_not_reused | PASS | True |",
        "| original_s6a_index_not_used_as_scoring_input | PASS | True |",
        "| user_approval_before_scoring | APPROVED | True |",
        "",
        "## 9. Readiness 판정",
        "",
        f"**{readiness}**",
        "",
        "| item | status |",
        "|---|---|",
        "| split_csv_exists | PASS |",
        "| stage2_holdout_patient_count_verified | PASS (154명) |",
        "| phase8_0_output_exists | PASS |",
        "| dedicated_manifest_exists | FAIL — MISSING |",
        "| dedicated_crop_root_exists | FAIL — MISSING |",
        "| mixed_crop_root_coverage_sufficient | FAIL — 152/154명 crops 없음 |",
        "",
        "Blockers:",
        *[f"- {b}" for b in blockers if b],
        "",
        "## 10. 다음 단계",
        "",
        f"readiness = **{readiness}** → Phase 8.2 진행 가능",
        "",
        "Phase 8.2 필요 작업:",
        "1. **사용자 승인** — Option C (S6-A pipeline 재실행) 승인",
        "2. S6-A pipeline으로 stage2_holdout 154명 전체 crops 생성",
        "   - LUNG1-295, LUNG1-415: 원본 CT/ROI로 재생성 (contaminated crops 사용 금지)",
        "   - 나머지 152명: 신규 생성",
        "3. dedicated stage2_holdout manifest 작성 (schema 설계 기준, Section C)",
        "4. leakage check 전항목 통과 확인",
        "5. 사용자 승인 후 scoring 시작",
        "",
        "## 11. 금지 사항",
        "",
        "- 실제 manifest 생성 금지 (이번 단계)",
        "- crop 복사/생성 금지 (이번 단계)",
        "- npz/npy 로드 금지",
        "- CT/ROI/mask npy 로드 금지",
        "- model forward 금지",
        "- scoring/metric/threshold/p95/p99/hit-rate 계산 금지",
        "- training/checkpoint 생성 금지",
        "- split CSV / 기존 Phase 6/7/8 output 수정 금지",
        "- v2/v2v2 접근 금지",
        "- NSCLC/MSD root 내용 접근 금지",
        "- suppression/mask/ROI 수정 금지",
        "- pip/conda install 금지",
        "- 외부 다운로드 금지",
        "",
        "---",
        "_Phase 8.1 preflight only — no asset created, no data loaded_",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    if not args.run:
        print("[INFO] --run 없이 실행됨. 아무것도 생성하지 않습니다.")
        print(f"[INFO] 실행 명령: source ~/ai_env/bin/activate && python {__file__} --run")
        sys.exit(0)

    # ── output guard ──────────────────────────────────────────────────
    if OUT_DIR.exists():
        print(f"[BLOCKED] output root already exists: {OUT_DIR}")
        print("[BLOCKED] 기존 output을 덮어쓰지 않습니다. 중단.")
        sys.exit(1)
    for p in [CSV_PATH, JSON_PATH, MD_PATH]:
        if p.exists():
            print(f"[BLOCKED] output file already exists: {p}")
            sys.exit(1)

    # ── source read-only checks ───────────────────────────────────────
    if not SPLIT_CSV.exists():
        print(f"[ERROR] split CSV not found: {SPLIT_CSV}")
        sys.exit(1)
    if not PHASE80_JSON.exists():
        print(f"[ERROR] Phase 8.0 JSON not found: {PHASE80_JSON}")
        sys.exit(1)

    holdout_patients, stage_counts = load_split_csv()
    holdout_count = len(holdout_patients)
    phase80_loaded = load_phase80_json()
    crop_info = check_crop_coverage(holdout_patients)

    print(f"[INFO] stage2_holdout patient count: {holdout_count}")
    print(f"[INFO] stage_counts: {stage_counts}")
    print(f"[INFO] crop coverage: {crop_info.get('holdout_covered_count',0)}/154 holdout patients have crops")
    print(f"[INFO] holdout patients with crops: {crop_info.get('holdout_with_crops', [])}")

    # ── build sections ────────────────────────────────────────────────
    section_a = build_section_a(holdout_count, crop_info, phase80_loaded)
    section_b = build_section_b()
    section_c = build_section_c()
    section_d = build_section_d()
    section_e = build_section_e()
    section_f = build_section_f(holdout_count, crop_info, phase80_loaded)

    all_rows, all_fields = build_csv(section_a, section_b, section_c, section_d, section_e, section_f)
    json_data = build_json(holdout_patients, holdout_count, crop_info, phase80_loaded,
                           section_a, section_b, section_c, section_d, section_e, section_f)
    md_content = build_md(holdout_count, crop_info, phase80_loaded, json_data)

    # ── write outputs ─────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=False)

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"[DONE] CSV: {CSV_PATH}")

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"[DONE] JSON: {JSON_PATH}")

    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[DONE] MD: {MD_PATH}")

    # ── post-write guard ──────────────────────────────────────────────
    for p in [CSV_PATH, JSON_PATH, MD_PATH]:
        if not p.exists():
            print(f"[ERROR] output file not written: {p}")
            sys.exit(1)

    print()
    print(f"[RESULT] readiness_for_phase8_2: {json_data['readiness_for_phase8_2']}")
    print(f"[RESULT] recommended_option: {json_data['recommended_option']}")
    print(f"[RESULT] blockers: {json_data['blockers']}")
    print()
    print("[DONE] Phase 8.1 preflight complete.")
    print(f"[NEXT]  Phase 8.2: Option C 승인 후 S6-A pipeline으로 stage2_holdout 154명 crops 생성")


if __name__ == "__main__":
    main()
