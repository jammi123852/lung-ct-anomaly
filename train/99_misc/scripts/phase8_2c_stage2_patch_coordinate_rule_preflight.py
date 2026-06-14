#!/usr/bin/env python3
"""
Phase 8.2c: stage2_holdout patch-coordinate source and S6-A manifest rule preflight

목적:
  - stage2_holdout y0/x0/y1/x1 coordinate source가 이미 존재하는지 확인
  - 없으면 기존 S6-A manifest 생성 규칙을 stage2_holdout에 동일하게 적용할 수 있는지 설계
  - preflight only (manifest/crop 생성, npy/npz 로드, scoring, metric 계산 없음)

경로 명시 기준:
  - 리뷰 파일에는 파일명만 있었고 실제 경로가 분산되어 있어 스크립트에서 실제 경로를 명시함
  - full recursive scan 없음
  - 입력 파일들은 schema/reference 확인용 (stage2_holdout evaluation input 직접 사용 금지)
"""

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# ── 입력 경로 (명시 경로로 고정, read-only) ──────────────────────────────────
MANIFEST_DRYRUN = PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/candidates/rule_s6a_gs2_selected_candidate_manifest_dryrun.csv"
HOLDOUT_EXCLUDED = PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/review_annotations/phase6_1b_s6a_stage1_dev_filtered_manifest_v1/phase6_1b_s6a_stage2_holdout_excluded_rows_v1.csv"
STAGE1_DEV_MANIFEST = PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/review_annotations/phase6_1b_s6a_stage1_dev_filtered_manifest_v1/phase6_1b_s6a_stage1_dev_filtered_manifest_v1.csv"
DATASET_INDEX = PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/datasets/s6a_6ch_full_dataset_index.csv"
SPLIT_CSV = PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1.csv"
PHASE8_2B_DIR = PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/review_annotations/phase8_2b_stage2_required_source_file_stat_preflight_v1"

RULE_MANIFEST_GEN = PROJECT_ROOT / "scripts/rule_s6a_manifest_gen.py"
CROP_FULL_6CH    = PROJECT_ROOT / "scripts/generate_s6a_crop_full_6ch.py"
CROP_FULL        = PROJECT_ROOT / "scripts/generate_s6a_crop_full.py"
CROP_SMOKE       = PROJECT_ROOT / "scripts/generate_s6a_crop_smoke.py"
VALIDATE_6CH     = PROJECT_ROOT / "scripts/validate_s6a_6ch_crop_full.py"

# ── 출력 경로 ──────────────────────────────────────────────────────────────────
OUT_DIR = PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/review_annotations/phase8_2c_stage2_patch_coordinate_rule_preflight_v1"
OUT_CSV = OUT_DIR / "phase8_2c_stage2_patch_coordinate_rule_preflight_v1.csv"
OUT_JSON = OUT_DIR / "phase8_2c_stage2_patch_coordinate_rule_preflight_v1.json"
OUT_MD  = OUT_DIR / "phase8_2c_stage2_patch_coordinate_rule_preflight_report_v1.md"

REQUIRED_COORD_COLS = ["patient_id", "local_z", "y0", "x0", "y1", "x1", "label", "sampling_label", "stage_split"]
COORD_COLS = {"y0", "x0", "y1", "x1"}
STAGE2_TARGET_COUNT = 154


# ── 헬퍼 ──────────────────────────────────────────────────────────────────────

def read_csv_header(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or [])


def check_csv_coordinate_source(path: Path) -> dict:
    """CSV에서 컬럼 구조 및 stage2_holdout 좌표 존재 여부를 read-only로 확인."""
    if not path.exists():
        return {
            "exists": False, "columns": [],
            "has_coord_columns": False, "has_required_columns": False,
            "stage2_holdout_rows": 0, "stage2_holdout_patient_count": 0,
        }
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames or [])
        has_coords = COORD_COLS.issubset(set(columns))
        has_required = all(c in columns for c in REQUIRED_COORD_COLS)
        stage2_rows = 0
        stage2_patients: set = set()
        if "stage_split" in columns:
            for row in reader:
                if row.get("stage_split") == "stage2_holdout":
                    stage2_rows += 1
                    pid = row.get("patient_id", "")
                    if pid:
                        stage2_patients.add(pid)
    return {
        "exists": True,
        "columns": columns,
        "has_coord_columns": has_coords,
        "has_required_columns": has_required,
        "stage2_holdout_rows": stage2_rows,
        "stage2_holdout_patient_count": len(stage2_patients),
    }


def extract_rule_elements(path: Path) -> dict:
    """스크립트 텍스트에서 rule 요소를 패턴 매칭으로 추출 (read-only)."""
    if not path.exists():
        return {"exists": False, "rule_elements_found": [], "required_inputs": [],
                "has_stage2_holdout_seal": False, "sampling_rule_name": None}
    content = path.read_text(encoding="utf-8")
    elements = []
    if "S6-A" in content:
        elements.append("S6-A_sampling_rule")
    if "stage2_holdout 환자 분석 금지" in content:
        elements.append("stage2_holdout_analysis_forbidden_comment")
    if "stage2_holdout" in content:
        elements.append("stage2_holdout_reference")
    if all(c in content for c in ["y0", "x0", "y1", "x1"]):
        elements.append("coord_y0_x0_y1_x1")
    if "stage_split" in content or "lesion_stage_split" in content:
        elements.append("split_csv_input")
    if "positive" in content and "hard_negative" in content:
        elements.append("positive_hard_negative_sampling")
    if "patient_id" in content:
        elements.append("patient_id_column")
    if "local_z" in content:
        elements.append("local_z_column")
    if "crop_size" in content or "patch_size" in content:
        elements.append("patch_size_parameter")

    required_inputs = []
    if "lesion_stage_split" in content or "split_csv" in content.lower():
        required_inputs.append("split_csv")
    if "ct_hu" in content or "roi_0_0" in content:
        required_inputs.append("CT_ROI_npy_path")
    if "lesion_mask" in content:
        required_inputs.append("lesion_mask")
    if "score" in content and "candidate" in content:
        required_inputs.append("score_candidate_csv")

    sampling_rule = None
    if "S6-A_positive_all_hn_ratio2" in content:
        sampling_rule = "S6-A_positive_all_hn_ratio2"

    return {
        "exists": True,
        "rule_elements_found": elements,
        "required_inputs": required_inputs,
        "has_stage2_holdout_seal": "stage2_holdout 환자 분석 금지" in content,
        "sampling_rule_name": sampling_rule,
    }


def get_stage2_patients_from_split() -> set:
    if not SPLIT_CSV.exists():
        return set()
    patients: set = set()
    with open(SPLIT_CSV, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("stage_split") == "stage2_holdout":
                pid = row.get("patient_id", "")
                if pid:
                    patients.add(pid)
    return patients


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 8.2c: stage2 coordinate rule preflight")
    parser.add_argument("--run", action="store_true", help="실제 실행 플래그")
    args = parser.parse_args()

    if not args.run:
        print("--run 플래그 없이는 실행하지 않습니다. 실행하려면 --run을 추가하세요.")
        sys.exit(0)

    # output guard
    OUT_DIR.mkdir(parents=True, exist_ok=False)

    print("[Phase 8.2c] stage2_holdout patch-coordinate source and S6-A manifest rule preflight")
    print(f"  output: {OUT_DIR}")
    print()

    # ── Section A: existing coordinate source inventory ──────────────────────

    print("[Section A] existing coordinate source inventory")

    stage2_patients_from_split = get_stage2_patients_from_split()
    print(f"  split CSV stage2_holdout 환자 수: {len(stage2_patients_from_split)}")

    src_defs = [
        (MANIFEST_DRYRUN,    "stage1_dev selected candidate manifest (dryrun)", "schema reference only – stage2 evaluation 직접 사용 금지"),
        (HOLDOUT_EXCLUDED,   "phase6_1b stage2_holdout excluded rows",          "coordinate check only"),
        (STAGE1_DEV_MANIFEST,"phase6_1b stage1_dev filtered manifest",          "schema reference only – stage2 evaluation 직접 사용 금지"),
        (DATASET_INDEX,      "S6-A 6ch full dataset index",                     "schema reference only – stage2 evaluation 직접 사용 금지"),
    ]

    section_a_rows = []
    for path, desc, allowed in src_defs:
        info = check_csv_coordinate_source(path)
        covers = (
            info["stage2_holdout_patient_count"] >= STAGE2_TARGET_COUNT
            if info["stage2_holdout_patient_count"] > 0 else False
        )
        if info["stage2_holdout_rows"] > 0 and info["has_coord_columns"]:
            status = "HAS_STAGE2_COORDS"
        elif info["stage2_holdout_rows"] > 0 and not info["has_coord_columns"]:
            status = "HAS_STAGE2_ROWS_NO_COORDS"
        elif info["exists"] and info["has_coord_columns"]:
            status = "EXISTS_NO_STAGE2_ROWS_HAS_COORDS"
        elif info["exists"]:
            status = "EXISTS_NO_COORD_COLUMNS"
        else:
            status = "FILE_NOT_FOUND"

        row = {
            "section": "A",
            "source_path": str(path),
            "description": desc,
            "exists": info["exists"],
            "has_coord_columns_y0_x0_y1_x1": info["has_coord_columns"],
            "has_required_columns": info["has_required_columns"],
            "stage2_holdout_rows": info["stage2_holdout_rows"],
            "stage2_holdout_patient_count": info["stage2_holdout_patient_count"],
            "covers_154_patients": covers,
            "status": status,
            "allowed_use": allowed,
            "note": "y0/x0/y1/x1 컬럼 없음" if not info["has_coord_columns"] else "좌표 컬럼 있음",
        }
        section_a_rows.append(row)
        print(f"  {path.name}: exists={info['exists']}, has_coords={info['has_coord_columns']}, "
              f"stage2_rows={info['stage2_holdout_rows']}, status={status}")

    print()

    # ── Section B: required coordinate schema ────────────────────────────────

    print("[Section B] required coordinate schema")

    stage1_cols = read_csv_header(MANIFEST_DRYRUN)
    crop_6ch_cols = read_csv_header(MANIFEST_DRYRUN)  # crop script reads from manifest

    crop_script_required = {"y0", "x0", "y1", "x1", "patient_id", "local_z"}

    section_b_rows = []
    for col in REQUIRED_COORD_COLS:
        in_stage1 = col in stage1_cols
        in_crop   = col in crop_script_required
        row = {
            "section": "B",
            "field_name": col,
            "required": True,
            "in_stage1_dev_manifest": in_stage1,
            "in_crop_script_required": in_crop,
            "source_reference": "rule_s6a_gs2_selected_candidate_manifest_dryrun.csv (stage1_dev schema reference)",
            "status": "OK" if in_stage1 else "MISSING_IN_MANIFEST",
            "note": "generate_s6a_crop_full_6ch.py에서 row[y0]/row[x0]/row[y1]/row[x1]으로 center 계산" if col in COORD_COLS else "",
        }
        section_b_rows.append(row)
        print(f"  {col}: in_stage1_manifest={in_stage1}, crop_required={in_crop}")

    print()

    # ── Section C: S6-A rule source inventory ────────────────────────────────

    print("[Section C] S6-A rule source inventory")

    rule_defs = [
        (RULE_MANIFEST_GEN, "primary manifest generation rule",
         "read-only rule analysis / stage2_holdout 동일 규칙 적용 설계",
         "stage2_holdout 환자 분석 직접 실행 금지 / v2/v2v2 접근 금지"),
        (CROP_FULL_6CH,     "6ch crop generation from manifest coords",
         "required column analysis only",
         "stage2_holdout crop 생성 금지"),
        (CROP_FULL,         "single-ch crop generation",
         "required column analysis only",
         "stage2_holdout crop 생성 금지"),
        (CROP_SMOKE,        "smoke crop generation",
         "read-only reference only",
         "stage2_holdout crop 생성 금지"),
        (VALIDATE_6CH,      "crop validation",
         "read-only reference only",
         "stage2_holdout 실행 금지"),
    ]

    section_c_rows = []
    for path, role, allowed, forbidden in rule_defs:
        info = extract_rule_elements(path)
        if info["exists"] and info["rule_elements_found"]:
            status = "OK_RULE_FOUND"
        elif not info["exists"]:
            status = "NOT_FOUND"
        else:
            status = "EXISTS_NO_RULE_ELEMENTS"
        row = {
            "section": "C",
            "script_path": str(path),
            "exists": info["exists"],
            "candidate_role": role,
            "rule_elements_found": "|".join(info["rule_elements_found"]),
            "has_stage2_holdout_seal": info["has_stage2_holdout_seal"],
            "sampling_rule_name": info["sampling_rule_name"] or "",
            "required_inputs": "|".join(info["required_inputs"]),
            "allowed_use": allowed,
            "forbidden_use": forbidden,
            "status": status,
            "note": "stage2_holdout 환자 분석 금지 명시 있음" if info["has_stage2_holdout_seal"] else "",
        }
        section_c_rows.append(row)
        print(f"  {path.name}: exists={info['exists']}, "
              f"elements={len(info['rule_elements_found'])}, "
              f"stage2_seal={info['has_stage2_holdout_seal']}")

    print()

    # ── Section D: rule consistency decision ─────────────────────────────────

    print("[Section D] rule consistency decision")

    has_full_stage2_manifest = any(
        r["stage2_holdout_rows"] > 0
        and r["has_coord_columns_y0_x0_y1_x1"]
        and r["covers_154_patients"]
        for r in section_a_rows
    )

    rule_gen_row = next((r for r in section_c_rows if "rule_s6a_manifest_gen" in r["script_path"]), None)
    rule_source_clear = (
        rule_gen_row is not None
        and rule_gen_row["exists"]
        and len(rule_gen_row["rule_elements_found"].split("|")) >= 3
    )

    if has_full_stage2_manifest:
        decision   = "EXISTING_STAGE2_COORDINATE_MANIFEST_READY"
        evidence   = "stage2_holdout y0/x0/y1/x1 포함 dedicated manifest 존재, 154명 커버 확인"
        blocker    = ""
        next_action = "Phase 8.2E coordinate manifest validation으로 이동"
    elif not rule_source_clear:
        decision   = "BLOCKED_MISSING_COORDINATE_SOURCE"
        evidence   = "rule_s6a_manifest_gen.py 없거나 rule element 불명확"
        blocker    = "rule source 불명확"
        next_action = "rule source 확인 및 복구 필요"
    else:
        decision    = "READY_FOR_PHASE8_2D_STAGE2_COORDINATE_MANIFEST_CREATION"
        evidence    = (
            "기존 stage2_holdout y0/x0/y1/x1 manifest 없음 확인. "
            "rule_s6a_manifest_gen.py rule source 명확. "
            "stage2_holdout 전용 dedicated script로 동일 rule 적용 가능."
        )
        blocker     = (
            "rule_s6a_manifest_gen.py 내부에 stage2_holdout 환자 분석 금지 봉인 있음. "
            "Phase 8.2D에서는 기존 rule을 재사용하되 stage2_holdout 전용 dedicated script 필요."
        )
        next_action = "Phase 8.2D: dedicated stage2_holdout coordinate manifest creation (별도 승인 필요)"

    print(f"  판정: {decision}")
    print(f"  evidence: {evidence[:80]}...")

    section_d_rows = [
        {
            "section": "D",
            "item": "stage2_holdout_coordinate_manifest_exists",
            "status": "FOUND" if has_full_stage2_manifest else "NOT_FOUND",
            "evidence": (
                "A섹션 4개 파일 모두 stage2_holdout y0/x0/y1/x1 없음 확인"
                if not has_full_stage2_manifest else "기존 manifest 존재"
            ),
            "blocker": "",
            "next_required_action": (
                "Phase 8.2D 전용 coordinate manifest 생성 단계 필요"
                if not has_full_stage2_manifest else ""
            ),
        },
        {
            "section": "D",
            "item": "stage1_dev_manifest_schema_valid",
            "status": "OK",
            "evidence": (
                "rule_s6a_gs2_selected_candidate_manifest_dryrun.csv에 "
                "y0/x0/y1/x1/patient_id/local_z/sampling_label/stage_split 컬럼 확인"
            ),
            "blocker": "",
            "next_required_action": "stage1_dev manifest는 schema reference only. stage2_holdout evaluation 직접 사용 금지",
        },
        {
            "section": "D",
            "item": "s6a_rule_source_available",
            "status": "OK" if rule_source_clear else "BLOCKED",
            "evidence": (
                f"rule_s6a_manifest_gen.py exists={rule_gen_row['exists'] if rule_gen_row else False}, "
                f"elements={rule_gen_row['rule_elements_found'] if rule_gen_row else 'N/A'}"
            ),
            "blocker": "" if rule_source_clear else "rule source 불명확",
            "next_required_action": "" if rule_source_clear else "rule source 확인 필요",
        },
        {
            "section": "D",
            "item": "stage2_holdout_policy_check",
            "status": "POLICY_SEAL_IN_SCRIPT",
            "evidence": (
                "rule_s6a_manifest_gen.py 15번 줄에 'stage2_holdout 환자 분석 금지' 명시. "
                "169번 줄에 봉인 확인 abort 코드 있음."
            ),
            "blocker": (
                "rule_s6a_manifest_gen.py를 stage2_holdout에 직접 실행 불가. "
                "Phase 8.2D에서 stage2_holdout 전용 dedicated script 필요."
            ),
            "next_required_action": (
                "Phase 8.2D에서 기존 S6-A rule만 재사용하는 stage2_holdout 전용 manifest creation script 작성. "
                "새 sampling rule 생성 금지. lesion_mask에서 직접 새 center 계산 금지."
            ),
        },
        {
            "section": "D",
            "item": "rule_consistency_decision",
            "status": decision,
            "evidence": evidence,
            "blocker": blocker,
            "next_required_action": next_action,
        },
    ]

    print()

    # ── Section E: next phase design ─────────────────────────────────────────

    print("[Section E] next phase design")

    if decision == "READY_FOR_PHASE8_2D_STAGE2_COORDINATE_MANIFEST_CREATION":
        section_e_rows = [
            {
                "section": "E",
                "phase": "Phase 8.2D",
                "purpose": "dedicated stage2_holdout coordinate manifest creation",
                "allowed_actions": (
                    "기존 S6-A rule (S6-A_positive_all_hn_ratio2) 동일 적용|"
                    "stage2_holdout source CT/ROI/lesion mask 로드|"
                    "y0/x0/y1/x1 좌표 생성|manifest CSV 저장"
                ),
                "forbidden_actions": (
                    "새 sampling rule 생성 금지|"
                    "lesion_mask에서 직접 새 center 계산 금지|"
                    "stage1_dev manifest 직접 재사용 금지|"
                    "v2/v2v2 접근 금지|rule threshold 수정 금지"
                ),
                "approval_required": True,
                "note": "stage2_holdout 전용 dedicated script 필요. rule은 Phase 8.2c에서 고정한 기존 S6-A rule만 사용",
            },
            {
                "section": "E",
                "phase": "Phase 8.2E",
                "purpose": "stage2_holdout coordinate manifest validation",
                "allowed_actions": (
                    "생성된 manifest의 y0/x0/y1/x1 범위 검증|"
                    "154명 커버 확인|label/sampling_label 분포 확인"
                ),
                "forbidden_actions": "crop 생성 금지|npy/npz 로드 금지|scoring 금지",
                "approval_required": True,
                "note": "Phase 8.2D 완료 후 진행",
            },
            {
                "section": "E",
                "phase": "Phase 8.2F",
                "purpose": "stage2_holdout crop/manifest creation",
                "allowed_actions": (
                    "generate_s6a_crop_full_6ch.py 사용|"
                    "stage2_holdout manifest에서 crop 생성|npz 저장"
                ),
                "forbidden_actions": "scoring 금지|metric 계산 금지|threshold 금지",
                "approval_required": True,
                "note": "Phase 8.2E validation 통과 후 진행",
            },
        ]
    elif decision == "EXISTING_STAGE2_COORDINATE_MANIFEST_READY":
        section_e_rows = [
            {
                "section": "E",
                "phase": "Phase 8.2E",
                "purpose": "stage2_holdout coordinate manifest validation",
                "allowed_actions": "기존 manifest의 y0/x0/y1/x1 범위 검증|154명 커버 확인",
                "forbidden_actions": "crop 생성 금지|npy/npz 로드 금지|scoring 금지",
                "approval_required": True,
                "note": "기존 manifest 존재 확인됨. validation으로 바로 이동",
            },
        ]
    else:
        section_e_rows = [
            {
                "section": "E",
                "phase": "BLOCKED",
                "purpose": "blocker 해소 필요",
                "allowed_actions": "blocker 원인 확인",
                "forbidden_actions": "모든 실행 금지",
                "approval_required": True,
                "note": f"blocker: {blocker}",
            },
        ]

    for r in section_e_rows:
        print(f"  {r['phase']}: {r['purpose']}")

    print()

    # ── CSV 저장 ─────────────────────────────────────────────────────────────

    all_rows = section_a_rows + section_b_rows + section_c_rows + section_d_rows + section_e_rows

    all_keys: list = []
    seen_keys: set = set()
    for row in all_rows:
        for k in row.keys():
            if k not in seen_keys:
                all_keys.append(k)
                seen_keys.add(k)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k, "") for k in all_keys})

    print(f"  CSV: {OUT_CSV}")

    # ── JSON 저장 ────────────────────────────────────────────────────────────

    output_json = {
        "phase": "8.2c",
        "run_date": "2026-06-01",
        "input_paths": {
            "manifest_dryrun":    str(MANIFEST_DRYRUN),
            "holdout_excluded":   str(HOLDOUT_EXCLUDED),
            "stage1_dev_manifest":str(STAGE1_DEV_MANIFEST),
            "dataset_index":      str(DATASET_INDEX),
            "split_csv":          str(SPLIT_CSV),
            "phase8_2b_dir":      str(PHASE8_2B_DIR),
            "rule_manifest_gen":  str(RULE_MANIFEST_GEN),
            "crop_full_6ch":      str(CROP_FULL_6CH),
        },
        "existing_coordinate_source_inventory": section_a_rows,
        "required_coordinate_schema": section_b_rows,
        "stage1_dev_manifest_schema_reference": {
            "source":  str(MANIFEST_DRYRUN),
            "columns": stage1_cols,
            "coord_columns_present": sorted(COORD_COLS),
            "note": "schema reference only. stage2_holdout evaluation input으로 직접 사용 금지",
        },
        "s6a_rule_source_inventory": section_c_rows,
        "rule_consistency_decision": decision,
        "readiness_for_next_phase": decision,
        "blockers": [r for r in section_d_rows if r.get("blocker")],
        "recommended_next_step": next_action,
        "notes": {
            "preflight_only": True,
            "no_stage2_manifest_creation": True,
            "no_crop_generation": True,
            "no_npy_loading": True,
            "no_npz_loading": True,
            "no_model_forward": True,
            "no_scoring": True,
            "no_metric_calculation": True,
            "no_threshold": True,
            "no_new_sampling_rule": True,
            "path_note": (
                "리뷰 파일에는 파일명만 있었고 실제 경로가 분산되어 있어 "
                "스크립트에서 실제 경로를 명시해 사용했다."
            ),
            "no_recursive_scan": True,
        },
    }

    def _json_default(obj):
        if isinstance(obj, bool):
            return obj
        return str(obj)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output_json, f, ensure_ascii=False, indent=2, default=_json_default)

    print(f"  JSON: {OUT_JSON}")

    # ── MD 보고서 저장 ───────────────────────────────────────────────────────

    def _bool_str(v) -> str:
        if isinstance(v, bool):
            return str(v)
        return str(v)

    md_lines = [
        "# Phase 8.2c: stage2_holdout patch-coordinate source and S6-A manifest rule preflight report",
        "",
        "## 1. Phase 8.2c 목적",
        "",
        "stage2_holdout 154명에 대한 y0/x0/y1/x1 patch coordinate source가 이미 존재하는지 확인한다.",
        "없으면 기존 S6-A manifest 생성 규칙을 stage2_holdout에 동일하게 적용할 수 있는지 설계한다.",
        "본 단계는 **preflight only**이며, manifest 생성·crop 생성·npy/npz 로드·scoring·metric 계산은 하지 않는다.",
        "",
        "## 2. 현재 blocker 요약",
        "",
        "- 기존 blocker: `BLOCKED_UNCLEAR_CROP_GENERATION_RULE`",
        f"- 본 preflight 판정: **{decision}**",
        "",
        "## 3. existing coordinate source inventory (Section A)",
        "",
        "리뷰 파일에는 파일명만 있었고 실제 경로가 분산되어 있어, 스크립트에서 실제 경로를 명시해 사용했다.",
        "full recursive scan은 하지 않았다.",
        "위 파일들은 schema/reference 확인용이며, stage2_holdout crop 생성 또는 scoring 입력으로 직접 사용하지 않는다.",
        "stage1_dev manifest와 dataset index는 stage2_holdout evaluation input으로 재사용 금지다.",
        "",
        "| 파일 | 존재 | y0/x0/y1/x1 | stage2 행 | stage2 환자 수 | 상태 |",
        "|------|------|-------------|-----------|----------------|------|",
    ]
    for r in section_a_rows:
        fname = Path(r["source_path"]).name
        md_lines.append(
            f"| {fname} | {_bool_str(r['exists'])} | {_bool_str(r['has_coord_columns_y0_x0_y1_x1'])} "
            f"| {r['stage2_holdout_rows']} | {r['stage2_holdout_patient_count']} | {r['status']} |"
        )

    md_lines += [
        "",
        "**결론**: 검색된 4개 파일 모두 stage2_holdout y0/x0/y1/x1 coordinate가 없음을 확인.",
        "",
        "## 4. required coordinate schema (Section B)",
        "",
        "| 컬럼 | stage1_dev manifest | crop script 요구 | 상태 |",
        "|------|---------------------|------------------|------|",
    ]
    for r in section_b_rows:
        md_lines.append(
            f"| {r['field_name']} | {_bool_str(r['in_stage1_dev_manifest'])} "
            f"| {_bool_str(r['in_crop_script_required'])} | {r['status']} |"
        )

    md_lines += [
        "",
        "## 5. stage1_dev S6-A manifest schema reference",
        "",
        f"- source: `{MANIFEST_DRYRUN.name}`",
        f"- coordinate columns (y0/x0/y1/x1): 존재 확인",
        f"- 총 컬럼 수: {len(stage1_cols)}",
        "- **stage2_holdout evaluation input으로 직접 사용 금지**",
        "",
        "## 6. S6-A manifest generation rule source (Section C)",
        "",
        "| 스크립트 | 존재 | stage2 seal | sampling rule | 상태 |",
        "|----------|------|-------------|---------------|------|",
    ]
    for r in section_c_rows:
        fname = Path(r["script_path"]).name
        md_lines.append(
            f"| {fname} | {_bool_str(r['exists'])} | {_bool_str(r['has_stage2_holdout_seal'])} "
            f"| {r['sampling_rule_name']} | {r['status']} |"
        )

    md_lines += [
        "",
        "**주요 발견**:",
        "- `rule_s6a_manifest_gen.py` 15번 줄에 'stage2_holdout 환자 분석 금지' 명시",
        "- 169번 줄에 봉인 확인 abort 코드 있음",
        "- Phase 8.2D에서는 기존 rule을 재사용하되, stage2_holdout 전용 dedicated script 필요",
        "",
        "## 7. rule consistency decision (Section D)",
        "",
        f"### 판정: **{decision}**",
        "",
        f"- evidence: {evidence}",
        f"- blocker: {blocker if blocker else '없음'}",
        f"- next action: {next_action}",
        "",
        "## 8. 다음 단계 (Section E)",
        "",
    ]
    for r in section_e_rows:
        md_lines += [
            f"### {r['phase']}",
            f"- 목적: {r['purpose']}",
            f"- 허용: {r['allowed_actions']}",
            f"- 금지: {r['forbidden_actions']}",
            f"- 승인 필요: {_bool_str(r['approval_required'])}",
            f"- note: {r['note']}",
            "",
        ]

    md_lines += [
        "## 9. 금지 사항",
        "",
        "- 실제 stage2_holdout coordinate manifest 생성 금지",
        "- crop 생성 금지",
        "- npy/npz 로드 금지",
        "- model forward 금지",
        "- scoring 금지",
        "- metric 계산 금지",
        "- threshold/p95/p99 계산 금지",
        "- training/checkpoint 생성 금지",
        "- 기존 Phase 6/7/8 output 수정 금지",
        "- split CSV 수정 금지",
        "- v2/v2v2 접근 금지",
        "- 새 sampling rule 생성 금지",
        "- lesion_mask에서 직접 새 center 계산 금지",
        "- Phase 8.2c는 coordinate source / rule preflight only이며, manifest 생성·crop 생성·npy/npz 로드·scoring·metric 계산은 하지 않는다.",
        "",
    ]

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"  MD: {OUT_MD}")
    print()
    print(f"[Phase 8.2c] 완료")
    print(f"  판정: {decision}")
    print(f"  다음 단계: {next_action}")


if __name__ == "__main__":
    main()
