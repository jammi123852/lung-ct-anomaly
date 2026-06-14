#!/usr/bin/env python3
"""
phase8_2b_stage2_required_source_file_stat_preflight.py
=========================================================
Phase 8.2b: stage2_holdout 154명 required source file stat preflight.

목적:
- Phase 8.2 preflight는 safe_id 폴더 존재만 확인했음.
- 이 스크립트는 실제 crop 생성에 필요한 source 파일 존재 여부를 stat-only로 확인.
- npy 로드, crop 생성, manifest 생성, model forward, scoring, metric 계산은 하지 않는다.

실행 방법:
  source ~/ai_env/bin/activate && python scripts/phase8_2b_stage2_required_source_file_stat_preflight.py --run

금지:
- np.load 금지 (stat만)
- npz 로드 금지
- crop 생성 금지
- manifest 생성 금지
- model forward 금지
- scoring 금지
- metric 계산 금지
- threshold 계산 금지
- training 금지
- NSCLC/MSD root recursive scan 금지
- pip/conda install 금지
- 외부 다운로드 금지
- split CSV 수정 금지
- paths.local.yaml 수정 금지
- 기존 Phase 6/7/8 output 수정 금지

syntax check (실행 아님):
  python -m py_compile scripts/phase8_2b_stage2_required_source_file_stat_preflight.py
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

# ─────────────────────────────────────────────
# 경로 상수
# ─────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]

PHASE8_2_JSON = (
    REPO_ROOT
    / "outputs/second-stage-lesion-refiner-v1/review_annotations"
    / "phase8_2_stage2_holdout_crop_manifest_creation_preflight_v1"
    / "phase8_2_stage2_holdout_crop_manifest_creation_preflight_v1.json"
)
SPLIT_CSV = (
    REPO_ROOT
    / "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1.csv"
)
PATHS_CONFIG = REPO_ROOT / "configs/paths.local.yaml"

CANDIDATE_SCRIPTS = {
    "generate_s6a_crop_full_6ch": REPO_ROOT / "scripts/generate_s6a_crop_full_6ch.py",
    "generate_s6a_crop_full": REPO_ROOT / "scripts/generate_s6a_crop_full.py",
    "generate_s6a_crop_smoke": REPO_ROOT / "scripts/generate_s6a_crop_smoke.py",
    "validate_s6a_6ch_crop_full": REPO_ROOT / "scripts/validate_s6a_6ch_crop_full.py",
}

OUT_DIR = (
    REPO_ROOT
    / "outputs/second-stage-lesion-refiner-v1/review_annotations"
    / "phase8_2b_stage2_required_source_file_stat_preflight_v1"
)
OUT_CSV = OUT_DIR / "phase8_2b_stage2_required_source_file_stat_preflight_v1.csv"
OUT_JSON = OUT_DIR / "phase8_2b_stage2_required_source_file_stat_preflight_v1.json"
OUT_MD = OUT_DIR / "phase8_2b_stage2_required_source_file_stat_preflight_report_v1.md"

CONTAMINATED_PATIENTS = ["LUNG1-295", "LUNG1-415"]

# stat-only로 확인할 후보 파일명 패턴
KNOWN_SOURCE_FILE_PATTERNS = [
    "ct_hu.npy",
    "roi_0_0.npy",
    "lesion_mask_roi_0_0.npy",
    "meta.json",
]


# ─────────────────────────────────────────────
# --run guard
# ─────────────────────────────────────────────
if "--run" not in sys.argv:
    print("사용법: python scripts/phase8_2b_stage2_required_source_file_stat_preflight.py --run")
    print("이 스크립트는 --run 인자 없이 실행되지 않습니다.")
    sys.exit(1)


# ─────────────────────────────────────────────
# output guard: exist_ok=False
# ─────────────────────────────────────────────
try:
    OUT_DIR.mkdir(parents=True, exist_ok=False)
except FileExistsError:
    print(f"[ABORT] output dir 이미 존재: {OUT_DIR}")
    print("기존 output을 보호하기 위해 중단합니다. 수동 삭제 후 재실행하세요.")
    sys.exit(1)


# ─────────────────────────────────────────────
# Section A: required source file schema 확인
# ─────────────────────────────────────────────
def section_a_required_file_schema() -> tuple[list[dict], list[str]]:
    """
    기존 S6-A crop 생성 스크립트들을 read-only로 파싱하여
    실제 필요한 source 파일명을 추출한다.
    반환: (rows, unique_required_files)
    """
    rows = []
    found_files_union: set[str] = set()

    for script_key, script_path in CANDIDATE_SCRIPTS.items():
        if not script_path.exists():
            rows.append({
                "section": "A",
                "item": f"script_exists:{script_key}",
                "source_script": str(script_path),
                "required_file": "",
                "evidence": "파일 없음",
                "status": "SCRIPT_NOT_FOUND",
                "note": "스크립트가 존재하지 않음",
            })
            continue

        try:
            text = script_path.read_text(encoding="utf-8")
        except Exception as e:
            rows.append({
                "section": "A",
                "item": f"script_read:{script_key}",
                "source_script": str(script_path),
                "required_file": "",
                "evidence": str(e),
                "status": "SCRIPT_READ_ERROR",
                "note": "스크립트 읽기 실패",
            })
            continue

        script_found: set[str] = set()
        for fname in KNOWN_SOURCE_FILE_PATTERNS:
            # fname이 문자열 리터럴로 등장하는지 확인
            pattern = re.escape(fname)
            matches = [(m.start(), text[:m.start()].count("\n") + 1)
                       for m in re.finditer(pattern, text)]
            if matches:
                evidence_lines = ", ".join(f"L{ln}" for _, ln in matches[:5])
                script_found.add(fname)
                found_files_union.add(fname)
                rows.append({
                    "section": "A",
                    "item": f"required_file:{script_key}:{fname}",
                    "source_script": script_key,
                    "required_file": fname,
                    "evidence": evidence_lines,
                    "status": "FOUND_IN_SCRIPT",
                    "note": "",
                })

        if not script_found:
            rows.append({
                "section": "A",
                "item": f"no_known_source_file:{script_key}",
                "source_script": script_key,
                "required_file": "",
                "evidence": "알려진 source 파일명 패턴 미발견",
                "status": "NO_KNOWN_SOURCE_FILE",
                "note": "validate 스크립트는 npz만 처리 — source file 직접 접근 없음일 수 있음",
            })

    unique_required = sorted(found_files_union)

    if not unique_required:
        rows.append({
            "section": "A",
            "item": "schema_decision",
            "source_script": "",
            "required_file": "",
            "evidence": "모든 스크립트에서 알려진 source 파일명 미발견",
            "status": "BLOCKED_UNCLEAR_REQUIRED_SOURCE_FILES",
            "note": "required source files를 특정할 수 없음",
        })
    else:
        rows.append({
            "section": "A",
            "item": "schema_decision",
            "source_script": "",
            "required_file": "|".join(unique_required),
            "evidence": f"union of all scripts: {unique_required}",
            "status": "SCHEMA_CONFIRMED",
            "note": "generate_s6a_crop_full_6ch는 ct_hu.npy만, generate_s6a_crop_full/smoke는 4파일 모두 사용",
        })

    return rows, unique_required


# ─────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────
def stat_file(path: Path) -> tuple[bool, int]:
    """파일 존재 여부와 크기(bytes)를 stat으로 확인. np.load 금지."""
    try:
        st = path.stat()
        return True, st.st_size
    except (FileNotFoundError, OSError):
        return False, -1


def make_stat_row(section: str, patient_id: str, safe_id: str,
                  required_file: str, source_root: Path) -> dict:
    path = source_root / "volumes_npy" / safe_id / required_file
    exists, size_bytes = stat_file(path)
    if not exists:
        status = "MISSING"
        note = "파일 없음"
    elif size_bytes == 0:
        status = "ZERO_BYTE"
        note = "파일 크기 0 byte"
    else:
        status = "OK"
        note = ""
    return {
        "section": section,
        "patient_id": patient_id,
        "safe_id": safe_id,
        "required_file": required_file,
        "path": str(path),
        "exists": exists,
        "size_bytes": size_bytes,
        "status": status,
        "note": note,
    }


# ─────────────────────────────────────────────
# Section B: stage2_holdout source file stat
# ─────────────────────────────────────────────
def section_b_stage2_stat(
    required_files: list[str],
    source_root: Path,
    split_df: pd.DataFrame,
) -> list[dict]:
    stage2 = split_df[split_df["stage_split"] == "stage2_holdout"].copy()
    rows = []
    for _, row in stage2.iterrows():
        pid = row["patient_id"]
        sid = row["safe_id"]
        for fname in required_files:
            rows.append(make_stat_row("B", pid, sid, fname, source_root))
    return rows


# ─────────────────────────────────────────────
# Section C: contaminated patient source stat
# ─────────────────────────────────────────────
def section_c_contaminated_stat(
    required_files: list[str],
    source_root: Path,
    split_df: pd.DataFrame,
) -> list[dict]:
    rows = []
    for pid in CONTAMINATED_PATIENTS:
        match = split_df[split_df["patient_id"] == pid]
        if match.empty:
            for fname in required_files:
                rows.append({
                    "section": "C",
                    "patient_id": pid,
                    "safe_id": "",
                    "required_file": fname,
                    "path": "",
                    "exists": False,
                    "size_bytes": -1,
                    "status": "PATIENT_NOT_IN_SPLIT_CSV",
                    "note": f"{pid}가 split CSV에 없음",
                })
        else:
            sid = match.iloc[0]["safe_id"]
            for fname in required_files:
                rows.append(make_stat_row("C", pid, sid, fname, source_root))
    return rows


# ─────────────────────────────────────────────
# Section D: summary by required file
# ─────────────────────────────────────────────
def section_d_summary(
    required_files: list[str],
    section_b_rows: list[dict],
    expected_count: int,
) -> list[dict]:
    rows = []
    b_df = pd.DataFrame(section_b_rows)
    for fname in required_files:
        sub = b_df[b_df["required_file"] == fname] if not b_df.empty else pd.DataFrame()
        if sub.empty:
            found_count = 0
            missing_count = expected_count
            zero_count = 0
        else:
            found_count = int((sub["status"] == "OK").sum())
            missing_count = int((sub["status"] == "MISSING").sum())
            zero_count = int((sub["status"] == "ZERO_BYTE").sum())
        if missing_count > 0:
            status = "MISSING_PATIENTS"
        elif zero_count > 0:
            status = "ZERO_BYTE_PATIENTS"
        else:
            status = "ALL_FOUND"
        rows.append({
            "section": "D",
            "required_file": fname,
            "expected_patient_count": expected_count,
            "found_patient_count": found_count,
            "missing_patient_count": missing_count,
            "zero_byte_count": zero_count,
            "status": status,
            "note": "",
        })
    return rows


# ─────────────────────────────────────────────
# Section E: readiness decision
# ─────────────────────────────────────────────
def section_e_readiness(
    schema_status: str,
    source_root_exists: bool,
    section_d_rows: list[dict],
) -> tuple[list[dict], str, list[str]]:
    blockers = []

    if not source_root_exists:
        blockers.append("BLOCKED_SOURCE_ROOT_MISSING")
    if schema_status == "BLOCKED_UNCLEAR_REQUIRED_SOURCE_FILES":
        blockers.append("BLOCKED_UNCLEAR_REQUIRED_SOURCE_FILES")

    for d in section_d_rows:
        if d["missing_patient_count"] > 0:
            blockers.append(f"BLOCKED_MISSING_REQUIRED_SOURCE_FILES:{d['required_file']}")
        if d["zero_byte_count"] > 0:
            blockers.append(f"BLOCKED_ZERO_BYTE_SOURCE_FILES:{d['required_file']}")

    blockers = list(dict.fromkeys(blockers))  # dedupe, preserve order

    if not blockers:
        readiness = "READY_FOR_PHASE8_2_RUN_CROP_MANIFEST_CREATION"
        next_action = "Phase 8.2 run crop/manifest creation script 설계 및 승인 요청"
    else:
        # 첫 번째 blocker 기준으로 readiness 결정
        if any("SOURCE_ROOT_MISSING" in b for b in blockers):
            readiness = "BLOCKED_SOURCE_ROOT_MISSING"
        elif any("UNCLEAR" in b for b in blockers):
            readiness = "BLOCKED_UNCLEAR_REQUIRED_SOURCE_FILES"
        elif any("MISSING_REQUIRED" in b for b in blockers):
            readiness = "BLOCKED_MISSING_REQUIRED_SOURCE_FILES"
        elif any("ZERO_BYTE" in b for b in blockers):
            readiness = "BLOCKED_ZERO_BYTE_SOURCE_FILES"
        else:
            readiness = "BLOCKED"
        next_action = "blockers 해결 후 재실행"

    rows = [
        {
            "section": "E",
            "item": "source_root_exists",
            "status": "PASS" if source_root_exists else "FAIL",
            "blocker": "BLOCKED_SOURCE_ROOT_MISSING" if not source_root_exists else "",
            "next_required_action": "",
        },
        {
            "section": "E",
            "item": "required_file_schema",
            "status": "PASS" if schema_status == "SCHEMA_CONFIRMED" else "FAIL",
            "blocker": schema_status if schema_status != "SCHEMA_CONFIRMED" else "",
            "next_required_action": "",
        },
        {
            "section": "E",
            "item": "readiness_decision",
            "status": readiness,
            "blocker": "|".join(blockers) if blockers else "",
            "next_required_action": next_action,
        },
    ]
    return rows, readiness, blockers


# ─────────────────────────────────────────────
# JSON 생성
# ─────────────────────────────────────────────
def build_json(
    phase8_2_data: dict,
    required_files: list[str],
    source_root: str,
    source_root_exists: bool,
    section_b_rows: list[dict],
    section_c_rows: list[dict],
    section_d_rows: list[dict],
    readiness: str,
    blockers: list[str],
    stage2_holdout_count: int,
) -> dict:
    b_df = pd.DataFrame(section_b_rows) if section_b_rows else pd.DataFrame()

    # required_file_stat_summary
    stat_summary = {}
    for fname in required_files:
        if b_df.empty:
            stat_summary[fname] = {"found": 0, "missing": 0, "zero_byte": 0}
        else:
            sub = b_df[b_df["required_file"] == fname]
            stat_summary[fname] = {
                "found": int((sub["status"] == "OK").sum()),
                "missing": int((sub["status"] == "MISSING").sum()),
                "zero_byte": int((sub["status"] == "ZERO_BYTE").sum()),
            }

    # missing_required_files: patient_id + safe_id + file 목록
    missing_rows = []
    if not b_df.empty:
        miss = b_df[b_df["status"].isin(["MISSING", "ZERO_BYTE"])]
        for _, r in miss.iterrows():
            missing_rows.append({
                "patient_id": r["patient_id"],
                "safe_id": r["safe_id"],
                "required_file": r["required_file"],
                "status": r["status"],
                "path": r["path"],
            })

    zero_rows = []
    if not b_df.empty:
        zb = b_df[b_df["status"] == "ZERO_BYTE"]
        for _, r in zb.iterrows():
            zero_rows.append({
                "patient_id": r["patient_id"],
                "safe_id": r["safe_id"],
                "required_file": r["required_file"],
                "path": r["path"],
            })

    # contaminated patient status
    contaminated_status = {}
    c_df = pd.DataFrame(section_c_rows) if section_c_rows else pd.DataFrame()
    for pid in CONTAMINATED_PATIENTS:
        if c_df.empty:
            contaminated_status[pid] = {}
        else:
            sub = c_df[c_df["patient_id"] == pid]
            contaminated_status[pid] = {
                r["required_file"]: {
                    "exists": bool(r["exists"]),
                    "size_bytes": int(r["size_bytes"]),
                    "status": r["status"],
                }
                for _, r in sub.iterrows()
            }

    return {
        "input_paths": {
            "phase8_2_preflight_json": str(PHASE8_2_JSON),
            "split_csv": str(SPLIT_CSV),
            "paths_config": str(PATHS_CONFIG),
            "candidate_scripts": {k: str(v) for k, v in CANDIDATE_SCRIPTS.items()},
        },
        "phase8_2_preflight_status": phase8_2_data.get("readiness_for_phase8_2_run", "UNKNOWN"),
        "required_source_files": required_files,
        "stage2_holdout_patient_count": stage2_holdout_count,
        "source_root": source_root,
        "source_root_exists": source_root_exists,
        "required_file_stat_summary": stat_summary,
        "missing_required_files": missing_rows,
        "zero_byte_files": zero_rows,
        "contaminated_patient_source_file_status": contaminated_status,
        "readiness_for_phase8_2_run": readiness,
        "blockers": blockers,
        "notes": {
            "stat_only": True,
            "no_npy_loading": True,
            "no_npz_loading": True,
            "no_crop_generation": True,
            "no_manifest_creation": True,
            "no_model_forward": True,
            "no_scoring": True,
            "no_metric_calculation": True,
            "no_threshold": True,
            "no_training": True,
        },
    }


# ─────────────────────────────────────────────
# MD report 생성
# ─────────────────────────────────────────────
def build_md(result: dict, section_d_rows: list[dict]) -> str:
    readiness = result["readiness_for_phase8_2_run"]
    blockers = result["blockers"]
    required_files = result["required_source_files"]
    source_root = result["source_root"]
    source_root_exists = result["source_root_exists"]
    stage2_count = result["stage2_holdout_patient_count"]
    stat_summary = result["required_file_stat_summary"]
    contaminated = result["contaminated_patient_source_file_status"]
    missing_list = result["missing_required_files"]
    zero_list = result["zero_byte_files"]

    lines = []
    lines.append("# Phase 8.2b: stage2_holdout Required Source File Stat Preflight Report")
    lines.append("")
    lines.append("## 1. Phase 8.2b 목적")
    lines.append("")
    lines.append(
        "Phase 8.2 preflight는 `safe_id in volumes_npy listdir` 방식으로 "
        "stage2_holdout 154명 폴더 존재만 확인했다. "
        "이 preflight는 crop 생성에 실제로 필요한 source 파일들이 "
        "각 safe_id 폴더 안에 존재하는지 stat-only로 검증한다."
    )
    lines.append("")
    lines.append("## 2. 왜 보완이 필요한지")
    lines.append("")
    lines.append(
        "- Phase 8.2 preflight의 `stage2_holdout CT/ROI found 154/154`는 "
        "\"폴더 존재\" 수준으로 제한해서 해석해야 한다."
    )
    lines.append(
        "- 실제 `ct_hu.npy`, `roi_0_0.npy`, `lesion_mask_roi_0_0.npy`, `meta.json` "
        "존재 여부는 별도 확인이 필요하다."
    )
    lines.append("- Phase 8.2 run 전 마지막 파일 존재성 검증이다.")
    lines.append("")
    lines.append("## 3. Required Source File Schema")
    lines.append("")
    lines.append("기존 S6-A crop 생성 스크립트에서 확인된 required files:")
    lines.append("")
    for f in required_files:
        lines.append(f"- `{f}`")
    lines.append("")
    lines.append(
        "> 참고: `generate_s6a_crop_full_6ch.py`는 `ct_hu.npy`만 직접 로드. "
        "`generate_s6a_crop_full.py` / `generate_s6a_crop_smoke.py`는 "
        "`ct_hu.npy + roi_0_0.npy + lesion_mask_roi_0_0.npy + meta.json` 4파일 모두 사용. "
        "Union 방식으로 모든 파일을 required로 간주."
    )
    lines.append("")
    lines.append("## 4. stage2_holdout 154명 Source File Stat 결과")
    lines.append("")
    lines.append(f"- source_root: `{source_root}`")
    lines.append(f"- source_root_exists: {source_root_exists}")
    lines.append(f"- stage2_holdout patient count: {stage2_count}")
    lines.append("")
    lines.append("| required_file | found | missing | zero_byte | status |")
    lines.append("|---|---|---|---|---|")
    for d in section_d_rows:
        lines.append(
            f"| {d['required_file']} | {d['found_patient_count']} "
            f"| {d['missing_patient_count']} | {d['zero_byte_count']} | {d['status']} |"
        )
    lines.append("")
    lines.append("## 5. LUNG1-295 / LUNG1-415 Source File Stat 결과")
    lines.append("")
    for pid in CONTAMINATED_PATIENTS:
        lines.append(f"### {pid}")
        lines.append("")
        pid_data = contaminated.get(pid, {})
        if not pid_data:
            lines.append("- split CSV에서 환자 정보를 찾을 수 없음")
        else:
            lines.append("| required_file | exists | size_bytes | status |")
            lines.append("|---|---|---|---|")
            for fname, info in pid_data.items():
                lines.append(
                    f"| {fname} | {info.get('exists')} "
                    f"| {info.get('size_bytes')} | {info.get('status')} |"
                )
        lines.append("")
    lines.append("## 6. Missing / Zero-byte Summary")
    lines.append("")
    if not missing_list and not zero_list:
        lines.append("- missing: 없음")
        lines.append("- zero_byte: 없음")
    else:
        if missing_list:
            lines.append(f"- missing: {len(missing_list)}건")
            for m in missing_list[:10]:
                lines.append(f"  - {m['patient_id']} / {m['required_file']}: {m['path']}")
            if len(missing_list) > 10:
                lines.append(f"  - ... 외 {len(missing_list) - 10}건")
        if zero_list:
            lines.append(f"- zero_byte: {len(zero_list)}건")
            for z in zero_list[:10]:
                lines.append(f"  - {z['patient_id']} / {z['required_file']}: {z['path']}")
    lines.append("")
    lines.append("## 7. Readiness 판정")
    lines.append("")
    lines.append(f"**{readiness}**")
    lines.append("")
    if blockers:
        lines.append("blockers:")
        for b in blockers:
            lines.append(f"- {b}")
    else:
        lines.append("blockers: 없음")
    lines.append("")
    lines.append("## 8. 다음 단계")
    lines.append("")
    if readiness == "READY_FOR_PHASE8_2_RUN_CROP_MANIFEST_CREATION":
        lines.append(
            "- Phase 8.2 run: dedicated crop 생성 및 manifest 생성 스크립트 설계 및 사용자 승인 요청"
        )
        lines.append(
            "- 생성 대상: `crops_stage2_holdout_6ch_dedicated_v1/` (6ch npz, 154명)"
        )
        lines.append(
            "- LUNG1-295 / LUNG1-415: 오염 구 crop(`crops_s6a_6ch_full`) 사용 금지, "
            "원본 CT/ROI에서 신규 생성"
        )
    else:
        lines.append("- blockers 해결 후 Phase 8.2b 재실행")
        for b in blockers:
            lines.append(f"  - {b}")
    lines.append("")
    lines.append("## 9. 금지 사항 확인")
    lines.append("")
    notes = result.get("notes", {})
    for k, v in notes.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────
def main() -> None:
    print("[Phase 8.2b] required source file stat preflight 시작")

    # 입력 파일 존재 확인
    for label, path in [
        ("PHASE8_2_JSON", PHASE8_2_JSON),
        ("SPLIT_CSV", SPLIT_CSV),
        ("PATHS_CONFIG", PATHS_CONFIG),
    ]:
        if not path.exists():
            print(f"[ABORT] 입력 파일 없음: {label} = {path}")
            sys.exit(1)

    # phase8_2 preflight JSON 로드
    with open(PHASE8_2_JSON, "r", encoding="utf-8") as f:
        phase8_2_data = json.load(f)
    print(f"  phase8_2 readiness: {phase8_2_data.get('readiness_for_phase8_2_run', 'N/A')}")

    # split CSV 로드
    split_df = pd.read_csv(SPLIT_CSV)
    stage2_holdout_df = split_df[split_df["stage_split"] == "stage2_holdout"]
    stage2_holdout_count = len(stage2_holdout_df)
    print(f"  stage2_holdout count: {stage2_holdout_count}")

    # paths.local.yaml 로드
    with open(PATHS_CONFIG, "r", encoding="utf-8") as f:
        paths_cfg = yaml.safe_load(f) or {}
    source_root_str = paths_cfg.get("nsclc_msd_usable_only_v2", "")
    if not source_root_str:
        print("[ABORT] paths.local.yaml에 nsclc_msd_usable_only_v2 키 없음")
        sys.exit(1)
    source_root = Path(source_root_str)
    source_root_exists = source_root.exists()
    print(f"  source_root: {source_root}")
    print(f"  source_root_exists: {source_root_exists}")

    # Section A
    print("[A] required source file schema 확인 중...")
    a_rows, required_files = section_a_required_file_schema()
    schema_status = next(
        (r["status"] for r in a_rows if r.get("item") == "schema_decision"),
        "BLOCKED_UNCLEAR_REQUIRED_SOURCE_FILES",
    )
    print(f"  schema_status: {schema_status}")
    print(f"  required_files: {required_files}")

    # Section B
    print("[B] stage2_holdout 154명 source file stat 중...")
    if required_files and source_root_exists:
        b_rows = section_b_stage2_stat(required_files, source_root, stage2_holdout_df)
    else:
        b_rows = []
    print(f"  stat rows: {len(b_rows)}")

    # Section C
    print("[C] contaminated patient source stat 중...")
    if required_files and source_root_exists:
        c_rows = section_c_contaminated_stat(required_files, source_root, split_df)
    else:
        c_rows = []

    # Section D
    print("[D] required file별 summary 집계 중...")
    d_rows = section_d_summary(required_files, b_rows, stage2_holdout_count)

    # Section E
    print("[E] readiness 판정 중...")
    e_rows, readiness, blockers = section_e_readiness(
        schema_status, source_root_exists, d_rows
    )
    print(f"  readiness: {readiness}")
    print(f"  blockers: {blockers}")

    # 결과 JSON 구성
    result_json = build_json(
        phase8_2_data=phase8_2_data,
        required_files=required_files,
        source_root=str(source_root),
        source_root_exists=source_root_exists,
        section_b_rows=b_rows,
        section_c_rows=c_rows,
        section_d_rows=d_rows,
        readiness=readiness,
        blockers=blockers,
        stage2_holdout_count=stage2_holdout_count,
    )

    # CSV 구성 (Section A/B/C/D/E union)
    all_rows = []
    all_rows.extend(a_rows)
    all_rows.extend(b_rows)
    all_rows.extend(c_rows)
    all_rows.extend(d_rows)
    all_rows.extend(e_rows)
    csv_df = pd.DataFrame(all_rows)

    # MD report 생성
    md_text = build_md(result_json, d_rows)

    # 저장 직전 output file 재검증
    for fpath in [OUT_CSV, OUT_JSON, OUT_MD]:
        if fpath.exists():
            raise RuntimeError(f"[ABORT] 출력 파일 이미 존재: {fpath}")

    # 저장
    csv_df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"  CSV 저장: {OUT_CSV}")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)
    print(f"  JSON 저장: {OUT_JSON}")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"  MD 저장: {OUT_MD}")

    print(f"\n[완료] readiness: {readiness}")
    if blockers:
        print(f"  blockers: {blockers}")


if __name__ == "__main__":
    main()
