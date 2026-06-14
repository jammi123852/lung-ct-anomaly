"""
Phase 2.20d Data Leakage Audit Script
목적: Phase 2.19~2.20d에서 사용된 patient/object가 stage1_dev인지 확인
"""

import os
import sys
import re
import json
import csv
import glob
import datetime

BASE = "/home/jinhy/project/lung-ct-anomaly/outputs/mip-postprocess-research-v1/reports"
SPLIT_FILE = "/home/jinhy/project/lung-ct-anomaly/outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1_balanced.csv"

AUDIT_FILES = [
    os.path.join(BASE, "phase2_19_v2_adaptive_mip_thickness_qa.csv"),
    os.path.join(BASE, "phase2_19e_manual_review_labels_object_level.csv"),
    os.path.join(BASE, "phase2_19g_remaining_manual_review_labels.csv"),
    os.path.join(BASE, "phase2_20b2_updated_vessel_dry_rule_decision_table.csv"),
    os.path.join(BASE, "phase2_20c_dry_decision_visual_review_table.csv"),
    os.path.join(BASE, "phase2_20d_visual_only_dry_decision_overlay_table.csv"),
]

OUTPUT_CSV = os.path.join(BASE, "phase2_20d_data_leakage_audit.csv")
OUTPUT_MD  = os.path.join(BASE, "phase2_20d_data_leakage_audit_summary.md")
OUTPUT_JSON = os.path.join(BASE, "phase2_20d_data_leakage_audit_summary.json")

CONTAMINATION_KEYWORDS = [
    "all 308", "308", "holdout", "stage2_holdout", "full",
    "전체 308", "전체 환자", "recall", "no-hit", "weak case",
    "failure case", "threshold tuning"
]


# ──────────────────────────────────────────────
# 덮어쓰기 방지
# ──────────────────────────────────────────────
def resolve_output_paths(csv_path, md_path, json_path):
    """이미 존재하면 _v2 suffix 부여"""
    def bump(p):
        base, ext = os.path.splitext(p)
        v2 = base + "_v2" + ext
        if os.path.exists(p):
            print(f"[WARNING] {p} 이미 존재 → {v2} 로 저장합니다.")
            return v2
        return p
    return bump(csv_path), bump(md_path), bump(json_path)


# ──────────────────────────────────────────────
# split 로드
# ──────────────────────────────────────────────
def load_split(split_file):
    """patient_id → stage_split dict 반환"""
    mapping = {}
    with open(split_file, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get("patient_id", "").strip()
            stage = row.get("stage_split", "").strip()
            if pid:
                mapping[pid] = stage
    print(f"[INFO] split 로드 완료: {len(mapping)}명")
    return mapping


# ──────────────────────────────────────────────
# object_id에서 patient_id 추출
# ──────────────────────────────────────────────
def extract_patient_from_object_id(object_id):
    """
    p218obj_LUNG1_020_o003 → LUNG1-020
    패턴: p숫자obj_ 제거, _o숫자 제거, 언더스코어→하이픈
    """
    s = object_id.strip()
    # p숫자obj_ 접두사 제거
    s = re.sub(r'^p\d+obj_', '', s)
    # _o숫자 (마지막) 제거
    s = re.sub(r'_o\d+$', '', s)
    # 언더스코어 → 하이픈
    s = s.replace('_', '-')
    return s


# ──────────────────────────────────────────────
# 단일 CSV 감사
# ──────────────────────────────────────────────
def audit_csv_file(filepath, split_map):
    filename = os.path.basename(filepath)
    rows_data = []

    with open(filepath, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        for row in reader:
            rows_data.append(row)

    total_rows = len(rows_data)
    object_ids = set()
    patient_ids = set()

    # row별 patient_id 수집
    row_patient_list = []
    for row in rows_data:
        pid = None

        # 1) patient_id 컬럼 직접 사용
        if "patient_id" in cols and row.get("patient_id", "").strip():
            pid = row["patient_id"].strip()

        # 2) object_id에서 추출
        if pid is None:
            for col in ["object_id", "component_id", "source_object_id"]:
                if col in cols and row.get(col, "").strip():
                    pid = extract_patient_from_object_id(row[col].strip())
                    break

        # 3) panel path에서 추출 시도 (마지막 수단)
        if pid is None:
            for col in ["source_panel_path", "main_panel_path", "supplement_panel_path",
                        "main_panel_exists", "visual_overlay_main_path"]:
                if col in cols and row.get(col, "").strip():
                    # 경로에서 safe_id 또는 patient 패턴 추출 시도
                    m = re.search(r'LUNG[12]-\d+', row[col])
                    if m:
                        pid = m.group(0)
                        break

        row_patient_list.append(pid)

        if pid:
            patient_ids.add(pid)

        # object_id 수집
        for col in ["object_id", "component_id", "source_object_id"]:
            if col in cols and row.get(col, "").strip():
                object_ids.add(row[col].strip())

    # stage_split 매칭
    stage1_dev_rows = 0
    stage2_holdout_rows = 0
    missing_stage_rows = 0
    stage1_dev_patients = set()
    stage2_holdout_patients = set()
    missing_stage_patients = set()

    for pid in row_patient_list:
        if pid is None:
            missing_stage_rows += 1
            missing_stage_patients.add("__no_patient_id__")
            continue
        stage = split_map.get(pid)
        if stage == "stage1_dev":
            stage1_dev_rows += 1
            stage1_dev_patients.add(pid)
        elif stage == "stage2_holdout":
            stage2_holdout_rows += 1
            stage2_holdout_patients.add(pid)
        else:
            missing_stage_rows += 1
            missing_stage_patients.add(pid)

    # audit_status 판정
    if stage2_holdout_rows > 0:
        audit_status = "fail"
    elif missing_stage_rows > 0:
        audit_status = "conditional"
    else:
        audit_status = "pass"

    # risk_note
    risk_notes = []
    if stage2_holdout_rows > 0:
        risk_notes.append(f"stage2_holdout {stage2_holdout_rows}행 발견: {sorted(stage2_holdout_patients)}")
    if missing_stage_rows > 0:
        missing_list = sorted(p for p in missing_stage_patients if p != "__no_patient_id__")
        no_pid_count = sum(1 for p in missing_stage_patients if p == "__no_patient_id__")
        if missing_list:
            risk_notes.append(f"split 미매칭 patient {len(missing_list)}명: {missing_list}")
        if no_pid_count > 0:
            risk_notes.append(f"patient_id 추출 불가 행 {no_pid_count}개")
    risk_note = "; ".join(risk_notes) if risk_notes else "none"

    return {
        "audit_file": filename,
        "file_type": "csv",
        "total_rows": total_rows,
        "unique_patients": len(patient_ids),
        "unique_objects": len(object_ids),
        "stage1_dev_rows": stage1_dev_rows,
        "stage2_holdout_rows": stage2_holdout_rows,
        "missing_stage_rows": missing_stage_rows,
        "stage1_dev_patients": len(stage1_dev_patients),
        "stage2_holdout_patients": len(stage2_holdout_patients),
        "missing_stage_patients": len(missing_stage_patients - {"__no_patient_id__"}),
        "audit_status": audit_status,
        "risk_note": risk_note,
        # 상세용 (CSV에는 포함 안 함)
        "_stage2_holdout_patient_list": sorted(stage2_holdout_patients),
        "_missing_patient_list": sorted(p for p in missing_stage_patients if p != "__no_patient_id__"),
    }


# ──────────────────────────────────────────────
# contamination-risk keyword scan
# ──────────────────────────────────────────────
def scan_keywords(base_dir, keywords):
    """phase2_19*.md, phase2_19*.json, phase2_20*.md, phase2_20*.json 대상"""
    patterns = [
        os.path.join(base_dir, "phase2_19*.md"),
        os.path.join(base_dir, "phase2_19*.json"),
        os.path.join(base_dir, "phase2_20*.md"),
        os.path.join(base_dir, "phase2_20*.json"),
    ]

    scan_results = []
    for pattern in patterns:
        for fpath in sorted(glob.glob(pattern)):
            fname = os.path.basename(fpath)
            found_in_file = {}
            try:
                with open(fpath, encoding="utf-8-sig", errors="replace") as f:
                    content = f.read()
                for kw in keywords:
                    # 대소문자 무관 검색
                    matches = [m.start() for m in re.finditer(re.escape(kw), content, re.IGNORECASE)]
                    if matches:
                        # 컨텍스트 추출 (앞뒤 60자)
                        contexts = []
                        for pos in matches[:3]:  # 최대 3개
                            start = max(0, pos - 60)
                            end = min(len(content), pos + len(kw) + 60)
                            snippet = content[start:end].replace("\n", " ").strip()
                            contexts.append(snippet)
                        found_in_file[kw] = {"count": len(matches), "contexts": contexts}
            except Exception as e:
                found_in_file["__read_error__"] = str(e)

            if found_in_file:
                scan_results.append({
                    "file": fname,
                    "findings": found_in_file
                })

    return scan_results


# ──────────────────────────────────────────────
# CSV 출력
# ──────────────────────────────────────────────
def write_audit_csv(results, out_path):
    fieldnames = [
        "audit_file", "file_type", "total_rows", "unique_patients", "unique_objects",
        "stage1_dev_rows", "stage2_holdout_rows", "missing_stage_rows",
        "stage1_dev_patients", "stage2_holdout_patients", "missing_stage_patients",
        "audit_status", "risk_note"
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"[OUTPUT] audit CSV: {out_path}")


# ──────────────────────────────────────────────
# MD 출력
# ──────────────────────────────────────────────
def write_audit_md(results, scan_results, overall_status, out_path, generated_at):
    holdout_detected = any(r["stage2_holdout_rows"] > 0 for r in results)
    missing_detected = any(r["missing_stage_rows"] > 0 for r in results)

    lines = []
    lines.append("# Phase 2.20d Data Leakage Audit Summary")
    lines.append("")
    lines.append(f"생성일시: {generated_at}")
    lines.append("")
    lines.append("## 1. Audit 목적")
    lines.append("")
    lines.append("Phase 2.19~2.20d에서 사용된 모든 patient/object가 `stage1_dev`인지 확인한다.")
    lines.append("`stage2_holdout` 데이터가 rule 개발 과정에 섞였는지, stage_split 누락이 있는지 점검한다.")
    lines.append("")
    lines.append("## 2. Audit 대상 파일")
    lines.append("")
    for r in results:
        lines.append(f"- `{r['audit_file']}`")
    lines.append("")
    lines.append("## 3. 파일별 판정 결과")
    lines.append("")
    lines.append("| audit_file | total_rows | unique_patients | stage1_dev_rows | stage2_holdout_rows | missing_stage_rows | audit_status |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r['audit_file']} | {r['total_rows']} | {r['unique_patients']} "
            f"| {r['stage1_dev_rows']} | {r['stage2_holdout_rows']} "
            f"| {r['missing_stage_rows']} | **{r['audit_status']}** |"
        )
    lines.append("")
    lines.append("## 4. Stage2 Holdout 발견 여부")
    lines.append("")
    if holdout_detected:
        lines.append("**FAIL: stage2_holdout 데이터가 감지됨**")
        for r in results:
            if r["stage2_holdout_rows"] > 0:
                lines.append(f"- `{r['audit_file']}`: {r['_stage2_holdout_patient_list']}")
    else:
        lines.append("PASS: 모든 파일에서 stage2_holdout 데이터가 감지되지 않았다.")
    lines.append("")
    lines.append("## 5. Missing Stage 여부")
    lines.append("")
    if missing_detected:
        lines.append("CONDITIONAL: split 매칭이 안 된 patient 또는 행이 존재한다.")
        for r in results:
            if r["missing_stage_rows"] > 0:
                lines.append(f"- `{r['audit_file']}`: risk_note = {r['risk_note']}")
    else:
        lines.append("PASS: 모든 행에서 stage_split 매칭 완료.")
    lines.append("")
    lines.append("## 6. Contamination-Risk Keyword Scan 결과")
    lines.append("")
    if not scan_results:
        lines.append("키워드 발견 없음.")
    else:
        for sr in scan_results:
            lines.append(f"### {sr['file']}")
            for kw, info in sr["findings"].items():
                if kw == "__read_error__":
                    lines.append(f"- 읽기 오류: {info}")
                else:
                    lines.append(f"- 키워드 `{kw}`: {info['count']}회 발견")
                    for ctx in info["contexts"]:
                        lines.append(f"  - ...{ctx}...")
            lines.append("")
    lines.append("")
    lines.append("## 7. 최종 권고")
    lines.append("")
    lines.append(f"전체 audit_status: **{overall_status.upper()}**")
    lines.append("")
    if not holdout_detected and not missing_detected:
        lines.append("Phase 2.20d visual review는 stage1_dev-only dry-rule 개발에 사용할 수 있다.")
    elif holdout_detected:
        lines.append("오염된 출력을 rule 개발에 사용하지 말 것. stage1_dev 기준으로 영향받은 출력을 재생성해야 한다.")
    else:
        lines.append("missing stage 매핑을 해결한 후 rule 개발에 사용할 것.")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[OUTPUT] audit MD: {out_path}")


# ──────────────────────────────────────────────
# JSON 출력
# ──────────────────────────────────────────────
def write_audit_json(results, scan_results, overall_status, out_path, generated_at):
    holdout_detected = any(r["stage2_holdout_rows"] > 0 for r in results)
    missing_detected = any(r["missing_stage_rows"] > 0 for r in results)

    files_passed = [r["audit_file"] for r in results if r["audit_status"] == "pass"]
    files_failed = [r["audit_file"] for r in results if r["audit_status"] == "fail"]
    files_conditional = [r["audit_file"] for r in results if r["audit_status"] == "conditional"]

    all_holdout_patients = []
    for r in results:
        all_holdout_patients.extend(r.get("_stage2_holdout_patient_list", []))
    all_holdout_patients = sorted(set(all_holdout_patients))

    # contamination_risk_notes: keyword scan 결과를 간결하게 정리
    contamination_risk_notes = []
    for sr in scan_results:
        for kw, info in sr["findings"].items():
            if kw == "__read_error__":
                continue
            note = f"{sr['file']}: keyword='{kw}' found {info['count']}x"
            contamination_risk_notes.append(note)

    # recommended_next_step
    if holdout_detected:
        recommended = "Do not use contaminated outputs for rule development. Rebuild affected outputs using stage1_dev only."
    elif missing_detected:
        recommended = "Resolve missing stage mapping before using outputs for rule development."
    else:
        recommended = "Phase 2.20d visual review can be used for stage1_dev-only dry-rule development."

    payload = {
        "phase": "2.20d_audit",
        "generated_at": generated_at,
        "audit_status": overall_status,
        "n_files_audited": len(results),
        "files_passed": files_passed,
        "files_failed": files_failed,
        "files_conditional": files_conditional,
        "holdout_detected": holdout_detected,
        "missing_stage_detected": missing_detected,
        "contamination_risk_notes": contamination_risk_notes,
        "stage2_holdout_patients_found": all_holdout_patients,
        "recommended_next_step": recommended,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[OUTPUT] audit JSON: {out_path}")

    return payload


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────
def main():
    generated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[START] Phase 2.20d Data Leakage Audit — {generated_at}")

    # 덮어쓰기 방지
    out_csv, out_md, out_json = resolve_output_paths(OUTPUT_CSV, OUTPUT_MD, OUTPUT_JSON)

    # split 로드
    split_map = load_split(SPLIT_FILE)

    # audit CSV 파일별 감사
    results = []
    for fpath in AUDIT_FILES:
        if not os.path.exists(fpath):
            print(f"[WARNING] 파일 없음 — 건너뜀: {fpath}")
            results.append({
                "audit_file": os.path.basename(fpath),
                "file_type": "csv",
                "total_rows": 0,
                "unique_patients": 0,
                "unique_objects": 0,
                "stage1_dev_rows": 0,
                "stage2_holdout_rows": 0,
                "missing_stage_rows": 0,
                "stage1_dev_patients": 0,
                "stage2_holdout_patients": 0,
                "missing_stage_patients": 0,
                "audit_status": "conditional",
                "risk_note": "파일 없음",
                "_stage2_holdout_patient_list": [],
                "_missing_patient_list": [],
            })
            continue
        print(f"[AUDIT] {os.path.basename(fpath)}")
        r = audit_csv_file(fpath, split_map)
        results.append(r)
        print(f"  → {r['total_rows']}행 / {r['unique_patients']}명 / status={r['audit_status']}")

    # keyword scan
    print("[SCAN] contamination-risk keyword scan 시작")
    scan_results = scan_keywords(BASE, CONTAMINATION_KEYWORDS)
    print(f"[SCAN] 키워드 발견 파일 수: {len(scan_results)}")

    # overall status 결정
    if any(r["audit_status"] == "fail" for r in results):
        overall_status = "fail"
    elif any(r["audit_status"] == "conditional" for r in results):
        overall_status = "conditional"
    else:
        overall_status = "pass"

    # 출력 생성
    write_audit_csv(results, out_csv)
    write_audit_md(results, scan_results, overall_status, out_md, generated_at)
    payload = write_audit_json(results, scan_results, overall_status, out_json, generated_at)

    # ── 최종 보고 ──────────────────────────────
    print("\n" + "="*60)
    print("AUDIT 완료 보고")
    print("="*60)
    print(f"1. exit code: 0 (정상)")
    print(f"2. audit CSV: {out_csv}")
    print(f"3. audit MD:  {out_md}")
    print(f"4. audit JSON: {out_json}")
    print(f"5. audit 대상 파일 수: {len(results)}")
    passed = sum(1 for r in results if r["audit_status"] == "pass")
    failed = sum(1 for r in results if r["audit_status"] == "fail")
    conditional = sum(1 for r in results if r["audit_status"] == "conditional")
    print(f"6. stage1_dev-only 통과 파일 수: {passed}")
    print(f"7. stage2_holdout 발견 파일 수: {failed}")
    print(f"8. missing stage 발견 파일 수: {conditional}")
    print(f"9. stage2_holdout patient 목록: {payload['stage2_holdout_patients_found']}")
    # missing patient 목록
    all_missing = []
    for r in results:
        all_missing.extend(r.get("_missing_patient_list", []))
    all_missing = sorted(set(all_missing))
    print(f"10. missing stage patient 목록: {all_missing}")
    if scan_results:
        print(f"11. contamination-risk keyword scan: {len(scan_results)}개 파일에서 키워드 발견")
        for sr in scan_results:
            kws = [kw for kw in sr["findings"] if kw != "__read_error__"]
            print(f"    - {sr['file']}: {kws}")
    else:
        print("11. contamination-risk keyword scan: 키워드 발견 없음")
    print(f"12. 최종 판정: {overall_status.upper()}")
    print(f"    → {payload['recommended_next_step']}")
    print("13. CT/ROI/mask 로드: 없음 (read-only CSV/MD/JSON만 사용)")
    print("14. outputs 밖 생성: 없음 (모든 출력이 mip-postprocess-research-v1/reports/ 내)")
    print("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
