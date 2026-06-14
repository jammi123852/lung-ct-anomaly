"""
Phase 5.79 Weak 3D Cluster Visual Review Pack Preflight (read-only)
======================================================================
- 입력 4개 파일 read-only 분석만 수행
- PNG/HTML/ZIP/visual pack 절대 생성 금지
- CT/ROI/mask npy 로드 절대 금지
- weak 3D merge 재실행 금지
- stage2_holdout/v2 접근 금지
"""

import sys
import os
import json
import csv
from pathlib import Path
from datetime import datetime

# ============================================================
# 상수 정의
# ============================================================
PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")

P77_CSV = PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/review_annotations/first_stage_padim_cluster_review/phase5_77_weak_3d_merge_dry_run_v1/phase5_77_weak_3d_cluster_summary.csv"
P77_JSON = PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/review_annotations/first_stage_padim_cluster_review/phase5_77_weak_3d_merge_dry_run_v1/phase5_77_weak_3d_cluster_summary.json"
P78_CSV = PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/reports/phase5_78_weak_3d_merge_result_diagnostic_v1/phase5_78_weak_3d_merge_result_diagnostic_v1.csv"
P78_JSON = PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/reports/phase5_78_weak_3d_merge_result_diagnostic_v1/phase5_78_weak_3d_merge_result_diagnostic_v1.json"

OUTPUT_ROOT = PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/reports/phase5_79_weak_3d_visual_pack_preflight_v1"
OUTPUT_MD   = OUTPUT_ROOT / "phase5_79_weak_3d_visual_pack_preflight_v1.md"
OUTPUT_JSON = OUTPUT_ROOT / "phase5_79_weak_3d_visual_pack_preflight_v1.json"
OUTPUT_CSV  = OUTPUT_ROOT / "phase5_79_weak_3d_visual_pack_target_manifest_v1.csv"

VOLUME_ROOT = Path("/mnt/c/Users/jinhy/Desktop/Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1/volumes_npy")

# safe-id 매핑 (manifest에서 확인)
PATIENT_SAFE_ID = {
    "normal004": "normal004__9190565aec",
    "normal013": "normal013__1ee16056c3",
    "normal014": "normal014__142b4ab95d",
}

# 필수 컬럼 목록
REQUIRED_COLS = [
    "diagnostic_group", "cluster3d_id", "patient_id",
    "z_min", "z_max", "z_span", "n_2d_clusters", "n_patches_total",
    "y0_min", "x0_min", "y1_max", "x1_max", "bbox_area",
    "top3_mean_patch_score_3d", "representative_2d_cluster_id",
    "representative_local_z", "representative_y0", "representative_x0",
    "representative_y1", "representative_x1",
    "review_candidate_flag", "overmerge_flag", "large_bbox_overmerge_flag",
    "large_extent_overmerge_flag", "complex_merge_flag", "high_score_ratio_flag",
    "diagnostic_priority", "diagnostic_note",
]

# P77 CSV 좌표 컬럼 (join 대상)
P77_COORD_COLS = [
    "y0_min", "x0_min", "y1_max", "x1_max",
    "representative_y0", "representative_x0", "representative_y1", "representative_x1",
]

# ============================================================
# 안전 가드: stage2_holdout/v2 경로 포함 시 중단
# ============================================================
for path_check in [str(OUTPUT_ROOT)]:
    if "stage2_holdout" in path_check or "/v2" in path_check:
        print(f"[ABORT] 출력 경로에 stage2_holdout 또는 /v2 포함 감지: {path_check}", file=sys.stderr)
        sys.exit(1)

# ============================================================
# 출력 root 사전 존재 시 중단
# ============================================================
if OUTPUT_ROOT.exists():
    print(f"[ABORT] 출력 root가 이미 존재합니다: {OUTPUT_ROOT}", file=sys.stderr)
    print("기존 결과를 보호하기 위해 중단합니다. 폴더를 직접 삭제 후 재실행하세요.", file=sys.stderr)
    sys.exit(1)

# ============================================================
# 입력 파일 존재 검증
# ============================================================
print("[1] 입력 파일 존재 검증...")
missing_inputs = []
for f in [P77_CSV, P77_JSON, P78_CSV, P78_JSON]:
    if not f.exists():
        missing_inputs.append(str(f))
if missing_inputs:
    print(f"[ABORT] 입력 파일 누락: {missing_inputs}", file=sys.stderr)
    sys.exit(1)
print("    P77 CSV/JSON, P78 CSV/JSON 모두 존재 확인.")

# ============================================================
# mtime 기록 (후처리 비교용)
# ============================================================
mtime_before = {}
for label, f in [("P77_CSV", P77_CSV), ("P77_JSON", P77_JSON),
                  ("P78_CSV", P78_CSV), ("P78_JSON", P78_JSON)]:
    mtime_before[label] = os.path.getmtime(f)

# ============================================================
# P78 CSV 로드
# ============================================================
print("[2] P78 CSV 로드...")

def read_csv_as_dicts(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    return rows

p78_rows = read_csv_as_dicts(P78_CSV)
print(f"    P78 CSV 총 행 수: {len(p78_rows)}")

# ============================================================
# diagnostic_group 멤버십 필터 (세미콜론 split 방식)
# ============================================================
print("[3] top9 / overmerge_priority 그룹 필터...")

def group_has(row, label):
    groups = [g.strip() for g in row["diagnostic_group"].split(";")]
    return label in groups

top9_rows = [r for r in p78_rows if group_has(r, "top9_review_candidate")]
priority_rows = [r for r in p78_rows if group_has(r, "overmerge_priority")]
top9_ids = set(r["cluster3d_id"] for r in top9_rows)
priority_ids = set(r["cluster3d_id"] for r in priority_rows)
overlap_ids = top9_ids & priority_ids
unique_ids = top9_ids | priority_ids

n_top9 = len(top9_ids)
n_priority = len(priority_ids)
n_overlap = len(overlap_ids)
n_unique = len(unique_ids)

print(f"    top9_review_candidate: {n_top9}")
print(f"    overmerge_priority:    {n_priority}")
print(f"    overlap:               {n_overlap}")
print(f"    unique_targets:        {n_unique}")

# 데이터 정합성 가드
if not (n_top9 == 9 and n_priority == 10 and n_overlap == 5 and n_unique == 14):
    print(f"[ABORT] 데이터 정합성 가드 실패!", file=sys.stderr)
    print(f"  기대: top9=9, priority=10, overlap=5, unique=14", file=sys.stderr)
    print(f"  실제: top9={n_top9}, priority={n_priority}, overlap={n_overlap}, unique={n_unique}", file=sys.stderr)
    sys.exit(1)
print("    정합성 가드 통과: top9=9, priority=10, overlap=5, unique=14")

# target rows (unique 14개)
target_rows_p78 = {r["cluster3d_id"]: r for r in p78_rows if r["cluster3d_id"] in unique_ids}

# ============================================================
# P77 CSV 로드 (좌표 join용)
# ============================================================
print("[4] P77 CSV 로드 (좌표 join용)...")
p77_rows = read_csv_as_dicts(P77_CSV)
p77_dict = {r["cluster3d_id"]: r for r in p77_rows}
print(f"    P77 CSV 총 행 수: {len(p77_rows)}")

# ============================================================
# 필수 컬럼 확인 + P77 join
# ============================================================
print("[5] 필수 컬럼 확인 및 P77 좌표 join...")

# P78 컬럼 확인
p78_cols = set(p78_rows[0].keys()) if p78_rows else set()
p77_cols = set(p77_rows[0].keys()) if p77_rows else set()

missing_in_p78 = [c for c in REQUIRED_COLS if c not in p78_cols]
missing_in_p77 = [c for c in REQUIRED_COLS if c not in p77_cols]

print(f"    P78 누락 컬럼: {missing_in_p78 if missing_in_p78 else '없음'}")
print(f"    P77 누락 컬럼: {missing_in_p77 if missing_in_p77 else '없음'}")

# P77에서 좌표 컬럼이 빠져있는지 P78을 먼저 확인
coord_cols_missing_in_p78 = [c for c in P77_COORD_COLS if c not in p78_cols]
coord_cols_present_in_p77 = [c for c in P77_COORD_COLS if c in p77_cols]

print(f"    좌표 컬럼 P78 누락: {coord_cols_missing_in_p78}")
print(f"    좌표 컬럼 P77 존재: {coord_cols_present_in_p77}")

# join 수행
join_fail_ids = []
merged_rows = {}
for cid in unique_ids:
    p78_r = target_rows_p78.get(cid)
    if p78_r is None:
        join_fail_ids.append(cid)
        continue
    merged = dict(p78_r)
    # P77 좌표 join
    p77_r = p77_dict.get(cid)
    if p77_r is None:
        join_fail_ids.append(cid)
        continue
    for col in P77_COORD_COLS:
        if col not in merged or not merged[col]:
            merged[col] = p77_r.get(col, "")
        elif col in p77_cols:
            # P78에 값이 있어도 P77을 우선 사용하지 않음 (P78 우선)
            pass
        # P78에 없는 경우 P77에서 채움
        if col not in p78_cols:
            merged[col] = p77_r.get(col, "")
    merged_rows[cid] = merged

if join_fail_ids:
    print(f"[ABORT] P77/P78 join 실패 cluster3d_id: {join_fail_ids}", file=sys.stderr)
    sys.exit(1)
print(f"    join 완료: {len(merged_rows)}개 성공, 실패 없음")

# 최종 필수 컬럼 충족 여부 확인
all_required_present = True
missing_final = []
for c in REQUIRED_COLS:
    sample_missing = sum(1 for row in merged_rows.values() if not row.get(c, "").strip())
    if sample_missing == len(merged_rows):
        missing_final.append(c)
        all_required_present = False

print(f"    최종 필수 컬럼 (14개 row 전부 비어있는 경우): {missing_final if missing_final else '없음'}")

# ============================================================
# 좌표 검증
# ============================================================
print("[6] 좌표 검증 (14개 대상)...")

def try_int(v):
    try:
        f = float(v)
        return int(f), abs(f - round(f)) < 1e-6
    except:
        return None, False

int_fail = []
bbox_fail = []
repr_inside_fail = []
z_span_gt3 = []

for cid, row in merged_rows.items():
    # z 정수성
    for col in ["z_min", "z_max", "representative_local_z"]:
        val, ok = try_int(row.get(col, ""))
        if not ok:
            int_fail.append((cid, col, row.get(col, "")))
    # 좌표 정수성
    for col in ["y0_min", "x0_min", "y1_max", "x1_max"]:
        val, ok = try_int(row.get(col, ""))
        if not ok:
            int_fail.append((cid, col, row.get(col, "")))

    # bbox 유효성
    y0, _ = try_int(row.get("y0_min", ""))
    x0, _ = try_int(row.get("x0_min", ""))
    y1, _ = try_int(row.get("y1_max", ""))
    x1, _ = try_int(row.get("x1_max", ""))
    if y0 is None or x0 is None or y1 is None or x1 is None:
        bbox_fail.append((cid, "parse_fail"))
    elif not ((y1 - y0) > 0 and (x1 - x0) > 0):
        bbox_fail.append((cid, f"h={y1-y0},w={x1-x0}"))

    # representative bbox 내부 검증
    ry0, _ = try_int(row.get("representative_y0", ""))
    rx0, _ = try_int(row.get("representative_x0", ""))
    ry1, _ = try_int(row.get("representative_y1", ""))
    rx1, _ = try_int(row.get("representative_x1", ""))
    if None in [y0, x0, y1, x1, ry0, rx0, ry1, rx1]:
        repr_inside_fail.append((cid, "parse_fail"))
    elif not (ry0 >= y0 and rx0 >= x0 and ry1 <= y1 and rx1 <= x1):
        repr_inside_fail.append((cid, f"repr=[{ry0},{rx0},{ry1},{rx1}] cluster=[{y0},{x0},{y1},{x1}]"))

    # z_span > 3
    z_span_val, _ = try_int(row.get("z_span", "0"))
    if z_span_val is not None and z_span_val > 3:
        z_span_gt3.append(cid)

print(f"    정수성 실패 (z/좌표): {len(int_fail)}건 - {int_fail if int_fail else '없음'}")
print(f"    bbox 유효성 실패: {len(bbox_fail)}건 - {bbox_fail if bbox_fail else '없음'}")
print(f"    representative bbox 내부 실패: {len(repr_inside_fail)}건 - {repr_inside_fail if repr_inside_fail else '없음'}")
print(f"    z_span > 3 후보 수: {len(z_span_gt3)} / 14")
print(f"    z_span > 3 cluster_ids: {z_span_gt3}")

coord_validation_pass = (len(int_fail) == 0 and len(bbox_fail) == 0)

# ============================================================
# visual source 후보 확인 (경로/존재/크기만, 로드 금지)
# ============================================================
print("[7] visual source 후보 확인 (경로/존재/크기만)...")

visual_sources = {}
for patient_id, safe_id in PATIENT_SAFE_ID.items():
    vol_dir = VOLUME_ROOT / safe_id
    ct_path = vol_dir / "ct_hu.npy"
    roi_path = vol_dir / "roi_0_0.npy"
    meta_path = vol_dir / "meta.json"

    ct_exists = ct_path.exists()
    roi_exists = roi_path.exists()
    meta_exists = meta_path.exists()

    ct_size = ct_path.stat().st_size if ct_exists else None
    roi_size = roi_path.stat().st_size if roi_exists else None

    status = "found" if (ct_exists and roi_exists) else ("partial" if (ct_exists or roi_exists) else "not_found")

    visual_sources[patient_id] = {
        "patient_id": patient_id,
        "safe_id": safe_id,
        "volume_dir": str(vol_dir),
        "ct_hu_npy": str(ct_path),
        "ct_hu_exists": ct_exists,
        "ct_hu_size_bytes": ct_size,
        "roi_0_0_npy": str(roi_path),
        "roi_0_0_exists": roi_exists,
        "roi_0_0_size_bytes": roi_size,
        "meta_json_exists": meta_exists,
        "status": status,
        "note": "ct_hu.npy + roi_0_0.npy 존재. pure_lung.npy 없음(roi_0_0.npy로 대체)."
                if status == "found" else "파일 미발견",
        "risk": "npy 로드 시 메모리 주의 (ct_hu.npy ~120-130MB per patient). 로드는 visual pack 단계에서 수행."
    }
    print(f"    {patient_id}: ct_hu={ct_exists}({ct_size}B), roi={roi_exists}({roi_size}B) -> {status}")

visual_source_readiness = "충족" if all(v["status"] == "found" for v in visual_sources.values()) else "미충족"
print(f"    visual_source_readiness: {visual_source_readiness}")

# ============================================================
# patient_distribution 및 z_span 분포
# ============================================================
patient_count = {}
z_span_dist = {}
for cid, row in merged_rows.items():
    pid = row.get("patient_id", "unknown")
    patient_count[pid] = patient_count.get(pid, 0) + 1
    z_span_val, _ = try_int(row.get("z_span", "1"))
    if z_span_val is not None:
        z_span_dist[str(z_span_val)] = z_span_dist.get(str(z_span_val), 0) + 1

# ============================================================
# overmerge 대상 수 집계
# ============================================================
n_targets_overmerge = sum(
    1 for row in merged_rows.values()
    if row.get("overmerge_flag", "").strip().lower() in ("true", "1")
)

# ============================================================
# preflight_readiness 판정
# ============================================================
# READY_FOR_VISUAL_PACK_SCRIPT: 대상 확정 + 좌표 검증 통과 + visual source 후보 1개 이상
n_sources_found = sum(1 for v in visual_sources.values() if v["status"] == "found")

if n_unique == 14 and coord_validation_pass and n_sources_found >= 1:
    if n_sources_found == 3:
        preflight_readiness = "READY_FOR_VISUAL_PACK_SCRIPT"
    else:
        preflight_readiness = "PARTIAL_READY"
elif n_unique == 14 and not coord_validation_pass:
    preflight_readiness = "PARTIAL_READY"
else:
    preflight_readiness = "NOT_READY"

print(f"    preflight_readiness: {preflight_readiness}")

# ============================================================
# unique cluster3d_id 목록 (diagnostic_priority 오름차순)
# ============================================================
sorted_ids = sorted(
    list(unique_ids),
    key=lambda cid: (
        float(merged_rows[cid].get("diagnostic_priority", "99")),
        merged_rows[cid].get("patient_id", ""),
        merged_rows[cid].get("cluster3d_id", "")
    )
)

# ============================================================
# source_group 라벨 부여
# ============================================================
def get_source_group(cid):
    in_top9 = cid in top9_ids
    in_priority = cid in priority_ids
    if in_top9 and in_priority:
        return "top9+overmerge_priority"
    elif in_top9:
        return "top9"
    else:
        return "overmerge_priority"

# ============================================================
# review_order 부여 (1..14)
# ============================================================
review_order_map = {cid: i+1 for i, cid in enumerate(sorted_ids)}

# ============================================================
# visual source 상태 per cluster (patient_id 기준)
# ============================================================
def get_visual_source_info(patient_id):
    info = visual_sources.get(patient_id, {})
    status = info.get("status", "not_found")
    path = info.get("ct_hu_npy", "")
    return status, path

# ============================================================
# 모든 계산 완료 후 mkdir
# ============================================================
print("[8] 출력 폴더 생성...")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
print(f"    생성: {OUTPUT_ROOT}")

# ============================================================
# CSV 출력
# ============================================================
print("[9] target manifest CSV 출력...")

CSV_COLS = [
    "review_order", "source_group", "cluster3d_id", "patient_id",
    "z_min", "z_max", "z_span", "representative_local_z",
    "y0_min", "x0_min", "y1_max", "x1_max",
    "representative_y0", "representative_x0", "representative_y1", "representative_x1",
    "n_2d_clusters", "n_patches_total", "bbox_area", "top3_mean_patch_score_3d",
    "review_candidate_flag", "overmerge_flag", "large_bbox_overmerge_flag",
    "large_extent_overmerge_flag", "complex_merge_flag", "high_score_ratio_flag",
    "diagnostic_priority", "diagnostic_note",
    "visual_source_status", "visual_source_candidate_path",
    "user_label", "user_note",
]

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=CSV_COLS, extrasaction="ignore")
    writer.writeheader()
    for cid in sorted_ids:
        row = merged_rows[cid]
        src_status, src_path = get_visual_source_info(row.get("patient_id", ""))
        out_row = {c: row.get(c, "") for c in CSV_COLS}
        out_row["review_order"] = review_order_map[cid]
        out_row["source_group"] = get_source_group(cid)
        out_row["visual_source_status"] = src_status
        out_row["visual_source_candidate_path"] = src_path
        out_row["user_label"] = ""
        out_row["user_note"] = ""
        writer.writerow(out_row)

print(f"    CSV 저장: {OUTPUT_CSV}")

# ============================================================
# JSON 출력
# ============================================================
print("[10] preflight JSON 출력...")

output_json_data = {
    "output_tag": "phase5_79_weak_3d_visual_pack_preflight_v1",
    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "n_top_review_candidate": n_top9,
    "n_overmerge_priority": n_priority,
    "n_overlap_between_groups": n_overlap,
    "n_unique_visual_targets": n_unique,
    "unique_target_cluster_ids": sorted_ids,
    "patient_distribution": patient_count,
    "z_span_distribution_for_targets": z_span_dist,
    "n_targets_overmerge": n_targets_overmerge,
    "visual_source_candidates": visual_sources,
    "visual_source_readiness": visual_source_readiness,
    "preflight_readiness": preflight_readiness,
    "next_phase_recommendation": (
        "Phase 5.79 visual pack 스크립트 작성 가능. "
        "normal004/013/014 ct_hu.npy + roi_0_0.npy 경로 확정. "
        "14개 cluster3d_id 기준 center slice + representative bbox overlay 생성."
        if preflight_readiness == "READY_FOR_VISUAL_PACK_SCRIPT"
        else "좌표 검증 또는 source 확인 후 재시도"
    ),
    "coord_validation": {
        "int_fail_count": len(int_fail),
        "int_fail_details": [{"cluster3d_id": x[0], "col": x[1], "val": str(x[2])} for x in int_fail],
        "bbox_fail_count": len(bbox_fail),
        "bbox_fail_details": [{"cluster3d_id": x[0], "detail": x[1]} for x in bbox_fail],
        "repr_inside_fail_count": len(repr_inside_fail),
        "repr_inside_fail_details": [{"cluster3d_id": x[0], "detail": x[1]} for x in repr_inside_fail],
        "z_span_gt3_count": len(z_span_gt3),
        "z_span_gt3_ids": z_span_gt3,
    },
    "column_check": {
        "p78_missing_cols": missing_in_p78,
        "p77_missing_cols": missing_in_p77,
        "coord_cols_filled_from_p77": coord_cols_missing_in_p78,
        "final_missing_required_cols": missing_final,
        "readiness": "충족" if (not missing_final and all_required_present) else "부분 충족",
    },
    "notes": {
        "preflight_only": True,
        "no_visual_pack_created": True,
        "no_png_html_zip_created": True,
        "no_volume_loaded": True,
        "phase5_77_readonly": True,
        "phase5_78_readonly": True,
        "no_weak_3d_merge_rerun": True,
        "no_model_forward": True,
        "no_score_recalculation": True,
        "threshold_not_finalized": True,
        "lesion_conclusion_forbidden": True,
        "stage2_holdout_unused": True,
        "v2_unused": True,
        "original_files_unmodified": True,
    }
}

with open(OUTPUT_JSON, "w", encoding="utf-8") as fh:
    json.dump(output_json_data, fh, ensure_ascii=False, indent=2)

print(f"    JSON 저장: {OUTPUT_JSON}")

# ============================================================
# MD 출력
# ============================================================
print("[11] preflight MD 출력...")

# 대상 테이블 구성
target_table_rows = []
for cid in sorted_ids:
    row = merged_rows[cid]
    target_table_rows.append(
        f"| {review_order_map[cid]} | {get_source_group(cid)} | {cid} | {row.get('patient_id','')} | "
        f"{row.get('z_min','')}..{row.get('z_max','')} (span={row.get('z_span','')}) | "
        f"{float(row.get('top3_mean_patch_score_3d', 0)):.4f} | {row.get('diagnostic_priority','')} |"
    )

target_table = "\n".join(target_table_rows)

md_content = f"""# Phase 5.79 Weak 3D Cluster Visual Review Pack Preflight v1

**작성일:** {datetime.now().strftime("%Y-%m-%d")}
**작업 유형:** preflight-only (read-only 분석 + 신규 MD/JSON/CSV 생성만)
**절대 금지:** visual pack/PNG/HTML/ZIP 생성, CT/ROI/mask npy 로드

---

## 1. 목적

Phase 5.77 (weak 3D merge dry-run)와 Phase 5.78 (diagnostic)에서 확정된
14개 visual review 대상 cluster에 대해 visual pack 스크립트 작성 전
전제조건(대상 확정, 좌표 검증, visual source 확인)을 점검한다.

---

## 2. 대상 목록 확정

| 항목 | 수 |
|---|---|
| top9_review_candidate | {n_top9} |
| overmerge_priority | {n_priority} |
| 두 그룹 overlap (교집합) | {n_overlap} |
| unique visual target (합집합) | {n_unique} |

**필터 방식:** diagnostic_group을 ';'로 split 후 멤버십 검사 (== 동등 비교 금지)

### 2.1 대상 14개 목록

| order | source_group | cluster3d_id | patient_id | z 범위 | top3_mean_score | priority |
|---|---|---|---|---|---|---|
{target_table}

---

## 3. 필수 컬럼 확인

- P78 CSV 누락 컬럼: {missing_in_p78 if missing_in_p78 else '없음'}
- P77 CSV 누락 컬럼 (join 대상): {missing_in_p77 if missing_in_p77 else '없음'}
- P78에 없어 P77에서 join한 좌표 컬럼: {coord_cols_missing_in_p78}
- join 실패 cluster3d_id: {'없음' if not join_fail_ids else join_fail_ids}
- 최종 필수 컬럼 전부 비어있는 경우: {missing_final if missing_final else '없음'}
- **컬럼 readiness**: {'충족' if not missing_final and all_required_present else '부분 충족'}

---

## 4. 좌표 검증 결과 (14개 대상)

| 검증 항목 | 통과 수 | 실패 수 | 실패 cluster3d_id |
|---|---|---|---|
| z/좌표 정수성 | {14 - len(set(x[0] for x in int_fail))} | {len(set(x[0] for x in int_fail))} | {list(set(x[0] for x in int_fail)) if int_fail else '없음'} |
| bbox 유효성 (h>0, w>0) | {14 - len(bbox_fail)} | {len(bbox_fail)} | {[x[0] for x in bbox_fail] if bbox_fail else '없음'} |
| representative bbox 내부 | {14 - len(repr_inside_fail)} | {len(repr_inside_fail)} | {[x[0] for x in repr_inside_fail] if repr_inside_fail else '없음'} |
| z_span > 3 후보 | - | {len(z_span_gt3)} | {z_span_gt3 if z_span_gt3 else '없음'} |

---

## 5. Visual Source 후보

| patient_id | safe_id | ct_hu.npy | roi_0_0.npy | 상태 |
|---|---|---|---|---|
| normal004 | normal004__9190565aec | {visual_sources['normal004']['ct_hu_size_bytes']}B | {visual_sources['normal004']['roi_0_0_size_bytes']}B | {visual_sources['normal004']['status']} |
| normal013 | normal013__1ee16056c3 | {visual_sources['normal013']['ct_hu_size_bytes']}B | {visual_sources['normal013']['roi_0_0_size_bytes']}B | {visual_sources['normal013']['status']} |
| normal014 | normal014__142b4ab95d | {visual_sources['normal014']['ct_hu_size_bytes']}B | {visual_sources['normal014']['roi_0_0_size_bytes']}B | {visual_sources['normal014']['status']} |

**volume root:**
`/mnt/c/Users/jinhy/Desktop/Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1/volumes_npy/`

**장점:** ct_hu.npy + roi_0_0.npy 전부 존재. 경로 직접 접근 가능(WSL /mnt/c).

**단점/위험:**
- C 드라이브 접근 속도 (WSL /mnt/c)는 native Linux보다 느릴 수 있음.
- pure_lung.npy 없음 (roi_0_0.npy 사용으로 대체 가능).
- ct_hu.npy 1개당 ~120-130MB. 14개 전부 로드 시 ~1.7GB+. visual pack 단계에서 환자별 순차 로드 필요.
- npy 로드는 visual pack 스크립트 단계에서만 수행 (이번 preflight에서는 로드 금지).

**visual_source_readiness:** {visual_source_readiness}

---

## 6. Visual Pack 설계안 (실제 생성 금지, 구조 제안만)

제안 panel 구성 (cluster3d_id 1개 당 1장):
1. **center slice CT panel** - representative_local_z 기준 ct_hu slice (HU window 적용)
2. **representative bbox overlay** - representative_y0/x0/y1/x1 bbox를 slice에 overlay
3. **3D cluster bbox overlay** - y0_min/x0_min/y1_max/x1_max 전체 bbox를 동일 slice에 overlay
4. **z_min/z_mid/z_max 3-slice context** - cluster의 z 범위 맥락 슬라이스 3장 (각 bbox 포함)
5. **thin MIP 보조 panel** - z_min~z_max 범위 maximum intensity projection (보조, 선택)
6. **score table panel / HTML metadata** - cluster3d_id, top3_mean_patch_score_3d, z_span, overmerge_flag 등 텍스트

---

## 7. Review Label 후보 (guide용)

| label | 설명 |
|---|---|
| pleural_wall | 흉막벽 근처 아티팩트 또는 정상 구조 |
| large_bbox_structure | 큰 bbox 구조 (혈관/기관지 등) |
| vessel_branch | 혈관 분기점 |
| bronchus_air_boundary | 기관지/공기 경계 |
| outside_roi_artifact | ROI 외곽 아티팩트 |
| z_overmerge_ok_continuous_structure | z 방향 연속 구조 (정상 merge) |
| z_overmerge_suspicious_overmerge | z 방향 과도한 merge (의심) |
| unclear | 판단 불명확 |

---

## 8. Preflight Readiness 판정

**preflight_readiness: {preflight_readiness}**

| 조건 | 충족 여부 |
|---|---|
| unique 대상 14개 확정 | {'충족' if n_unique == 14 else '미충족'} |
| 좌표 정수성/bbox 검증 통과 | {'충족' if coord_validation_pass else '미충족 - 세부 내용 4장 참조'} |
| visual source 후보 1개 이상 | {'충족 (3/3)' if n_sources_found == 3 else f'부분 충족 ({n_sources_found}/3)'} |

---

## 9. Next Phase Recommendation

{output_json_data['next_phase_recommendation']}

---

## 10. 해석 제한 사항

- 이 결과는 **sample 3명 dry-run** (normal004, normal013, normal014) 기반이다.
- 입력 score는 **sample-local p99** 기반으로 global threshold가 아니다.
- threshold는 확정되지 않았으며 병변 탐지 결론을 내릴 수 없다.
- stage2_holdout 데이터는 이 단계에서 미검증이다.
- v2 모델은 사용하지 않았다.
- 원본 파일 (P77/P78 CSV/JSON)은 수정 없이 read-only로만 사용됐다.
"""

with open(OUTPUT_MD, "w", encoding="utf-8") as fh:
    fh.write(md_content)

print(f"    MD 저장: {OUTPUT_MD}")

# ============================================================
# mtime 사후 검증
# ============================================================
print("[12] 입력 파일 mtime 사후 검증...")
mtime_ok = True
for label, f in [("P77_CSV", P77_CSV), ("P77_JSON", P77_JSON),
                  ("P78_CSV", P78_CSV), ("P78_JSON", P78_JSON)]:
    after = os.path.getmtime(f)
    if abs(after - mtime_before[label]) > 0.01:
        print(f"    [WARNING] {label} mtime 변경 감지! before={mtime_before[label]}, after={after}", file=sys.stderr)
        mtime_ok = False
    else:
        print(f"    {label}: mtime 변경 없음 (OK)")

# ============================================================
# 완료 요약
# ============================================================
print("\n" + "="*60)
print("Phase 5.79 Preflight 완료 요약")
print("="*60)
print(f"  1. P77 CSV/JSON 존재:         OK")
print(f"  2. P78 CSV/JSON 존재:         OK")
print(f"  3. top9 row 수:               {n_top9} (기대 9)")
print(f"  4. overmerge_priority 수:     {n_priority} (기대 10)")
print(f"  5. unique target 수:          {n_unique} (기대 14)")
print(f"  6. 필수 컬럼 확인:            {'OK' if not missing_final else 'PARTIAL: '+str(missing_final)}")
print(f"  7. 좌표 정수성/bbox 유효성:   {'OK' if coord_validation_pass else 'FAIL: int_fail='+str(len(int_fail))+' bbox_fail='+str(len(bbox_fail))}")
print(f"  8. repr bbox 내부 검증:       {'OK' if not repr_inside_fail else 'FAIL: '+str(len(repr_inside_fail))}")
print(f"  9. visual source 후보:        {n_sources_found}/3 found")
print(f" 10. CT/ROI/mask 로드 없음:     OK (preflight only)")
print(f" 11. CSV/JSON/MD 신규 생성:     OK")
print(f" 12. P77/P78 mtime 미변경:      {'OK' if mtime_ok else 'WARNING'}")
print(f" 13. weak merge 재실행 없음:    OK")
print(f" 14. visual pack 미생성:        OK")
print(f" 15. PNG/HTML/ZIP 미생성:       OK")
print(f" 16. model forward 없음:        OK")
print(f" 17. score 재계산 없음:         OK")
print(f" 18. threshold 확정 없음:       OK")
print(f" 19. 병변 결론 없음:            OK")
print(f" 20. stage2_holdout/v2 없음:    OK")
print()
print(f"  unique cluster3d_id 14개: {sorted_ids}")
print(f"  patient별 target 수: {patient_count}")
print(f"  z_span 분포: {z_span_dist}")
print(f"  visual_source_readiness: {visual_source_readiness}")
print(f"  preflight_readiness: {preflight_readiness}")
print()
print(f"  생성 파일:")
print(f"    {OUTPUT_MD}")
print(f"    {OUTPUT_JSON}")
print(f"    {OUTPUT_CSV}")
print(f"  스크립트 경로: {__file__}")
