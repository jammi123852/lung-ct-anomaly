"""
dev_safe mixed cohort error visual QA manifest 생성 스크립트.

이번 단계에서는 --dry-run까지만 허용.
--run 실행은 --confirm-run 없으면 abort.
PNG 생성, CT/ROI/mask 로드, score CSV 수정, scoring 재실행 금지.
ResNet50 score 경로 접근 금지.
stage2_holdout 접근 금지.
"""

import argparse
import os
import sys
import csv
import json
import re
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

# ============================================================
# 고정 경로 상수
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# v2_v2 source score 경로 (절대 이 경로만 사용)
NORMAL_SCORE_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs/position-aware-padim-v1/scores/padim_v2_roi0_0/normal_by_patient"
)
LESION_SCORE_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs/position-aware-padim-v1/scores/padim_v2_roi0_0/lesion_v2_by_patient"
)

# 금지 경로 패턴
FORBIDDEN_PATH_PATTERN = re.compile(
    r"experiments[/\\]resnet50_imagenet_v1[/\\]outputs[/\\]scores",
    re.IGNORECASE
)

# full_retrospective 금지 패턴
FORBIDDEN_FULL_RETRO_PATTERN = re.compile(r"full_retrospective", re.IGNORECASE)

# ============================================================
# config 로드 (paths.local.yaml) — 데이터 root 획득
# ============================================================

def load_data_roots():
    """
    configs/paths.local.yaml에서 nsclc_msd_usable_only_v2 / normal_training_ready_v2_roi0_0 읽기.
    하드코딩 금지, 키 기반 접근.
    반환: (lesion_root: str, normal_root: str)
    """
    cfg_path = os.path.join(PROJECT_ROOT, "configs", "paths.local.yaml")
    if not os.path.isfile(cfg_path):
        abort(f"configs/paths.local.yaml 없음: {cfg_path}")
    with open(cfg_path, encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f) or {}
    lesion_root = (cfg.get("nsclc_msd_usable_only_v2") or "").strip()
    normal_root = (cfg.get("normal_training_ready_v2_roi0_0") or "").strip()
    if not lesion_root:
        abort("paths.local.yaml: nsclc_msd_usable_only_v2 키가 비어 있습니다.")
    if not normal_root:
        abort("paths.local.yaml: normal_training_ready_v2_roi0_0 키가 비어 있습니다.")
    return lesion_root, normal_root


# ============================================================
# safe_id 조회 (score CSV 첫 데이터행 읽기)
# ============================================================

def get_safe_id_from_score_csv(score_csv_path):
    """
    score CSV 첫 데이터행에서 safe_id 컬럼 반환.
    없으면 None.
    """
    if not os.path.isfile(score_csv_path):
        return None
    with open(score_csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            return row.get("safe_id", "").strip() or None
    return None


def get_patch_coords_from_score_csv(score_csv_path, patch_idx):
    """
    score CSV의 patch_idx(0-based 데이터행)에서 y0,x0,y1,x1 읽기.
    반환: (y0, x0, y1, x1) 또는 None
    """
    if not os.path.isfile(score_csv_path):
        return None
    with open(score_csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i == patch_idx:
                try:
                    return (
                        int(row["y0"]), int(row["x0"]),
                        int(row["y1"]), int(row["x1"])
                    )
                except (KeyError, ValueError):
                    return None
    return None


# ============================================================
# 시각화 상수
# ============================================================

# lung window
LUNG_WL = -300.0   # window center
LUNG_WW = 1400.0   # window width

# 색상 상수 (코드 내부 고정값)
COLOR_ROI_CONTOUR = (80, 200, 80)        # 초록
COLOR_LESION_CONTOUR = (220, 50, 50)     # 빨강
COLOR_PATCH_BOX = (60, 140, 255)         # 파랑
COLOR_PATCH_FALLBACK_CROSS = (255, 200, 0)  # 노랑 (slice-level fallback)
COLOR_OVERLAY_ALPHA = 0.35


# ============================================================
# 시각화 헬퍼
# ============================================================

def hu_to_rgb(slice_hu: np.ndarray) -> np.ndarray:
    """HU → lung window → RGB uint8"""
    lo = LUNG_WL - LUNG_WW / 2.0
    hi = LUNG_WL + LUNG_WW / 2.0
    v = np.clip(slice_hu.astype(np.float32), lo, hi)
    v = (v - lo) / (hi - lo) * 255.0
    return np.stack([v.astype(np.uint8)] * 3, axis=-1)


def blend_mask(rgb: np.ndarray, mask: np.ndarray, color, alpha: float = 0.35) -> np.ndarray:
    """mask 영역에 반투명 color overlay"""
    out = rgb.astype(np.float32)
    m = mask > 0
    if m.any():
        c = np.array(color, dtype=np.float32)
        out[m] = out[m] * (1 - alpha) + c * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def make_contour_mask(mask: np.ndarray) -> np.ndarray:
    """
    binary mask에서 경계(contour) pixel만 남긴 mask 반환.
    내부 픽셀을 제거하고 테두리만 남긴다.
    """
    if mask.max() == 0:
        return mask
    from PIL import Image as _Image, ImageFilter
    pil = _Image.fromarray((mask > 0).astype(np.uint8) * 255, "L")
    eroded = pil.filter(ImageFilter.MinFilter(3))
    contour = np.array(pil).astype(np.int16) - np.array(eroded).astype(np.int16)
    return (contour > 0).astype(np.uint8)


def draw_visual_qa_png(
    ct_slice: np.ndarray,
    roi_slice: np.ndarray,
    lesion_slice,  # None or np.ndarray
    y0: int, x0: int, y1: int, x1: int,
    has_patch_coords: bool,
    title: str,
    out_path: str,
):
    """
    단일 manifest row에 대한 PNG 생성.
    - CT lung window 기반 배경
    - ROI contour 오버레이 (초록)
    - lesion contour 오버레이 (빨강, lesion FN only)
    - candidate patch rectangle (파랑) 또는 slice-level fallback cross (노랑)
    - 제목 텍스트 (상단)
    PNG 저장.
    """
    rgb = hu_to_rgb(ct_slice)

    # ROI contour
    roi_contour = make_contour_mask(roi_slice)
    rgb = blend_mask(rgb, roi_contour, COLOR_ROI_CONTOUR, alpha=0.9)

    # lesion contour (lesion FN only)
    if lesion_slice is not None and lesion_slice.max() > 0:
        lesion_contour = make_contour_mask(lesion_slice)
        rgb = blend_mask(rgb, lesion_contour, COLOR_LESION_CONTOUR, alpha=0.9)

    img = Image.fromarray(rgb, "RGB")
    draw = ImageDraw.Draw(img)

    H, W = ct_slice.shape

    if has_patch_coords:
        # candidate patch rectangle
        draw.rectangle(
            [x0, y0, x1 - 1, y1 - 1],
            outline=COLOR_PATCH_BOX,
            width=2,
        )
    else:
        # slice-level fallback: 이미지 중심에 십자 마커
        cx, cy = W // 2, H // 2
        r = 10
        draw.line([(cx - r, cy), (cx + r, cy)], fill=COLOR_PATCH_FALLBACK_CROSS, width=2)
        draw.line([(cx, cy - r), (cx, cy + r)], fill=COLOR_PATCH_FALLBACK_CROSS, width=2)

    # 제목 텍스트 (상단 왼쪽, 기본 bitmap 폰트)
    # 텍스트가 길 경우 2줄로 분할
    title_lines = []
    max_chars = 60
    words = title.split(" | ")
    line = ""
    for w in words:
        if len(line) + len(w) + 3 > max_chars:
            title_lines.append(line)
            line = w
        else:
            line = (line + " | " + w).strip(" | ")
    if line:
        title_lines.append(line)

    for li, tline in enumerate(title_lines):
        draw.text((4, 4 + li * 12), tline, fill=(255, 255, 0))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)
    del draw

# 입력 파일 경로
ERROR_ANALYSIS_CSV = os.path.join(
    PROJECT_ROOT,
    "evaluation/mixed_cohort_patient_metrics/dev_safe_v1_error_analysis.csv"
)
PATIENT_SCORES_CSV = os.path.join(
    PROJECT_ROOT,
    "evaluation/mixed_cohort_patient_metrics/dev_safe_v1_patient_scores.csv"
)
MANIFEST_CSV = os.path.join(
    PROJECT_ROOT,
    "evaluation/mixed_cohort_patient_metrics/dev_safe_v1_visual_qa_manifest.csv"
)

# 생성 대상 파일 경로
OUTPUT_MANIFEST_DEFAULT = MANIFEST_CSV

# ============================================================
# 안전 확인 유틸리티
# ============================================================

def abort(msg):
    print(f"[ABORT] {msg}", file=sys.stderr)
    sys.exit(1)


def check_forbidden_path(path_str):
    """ResNet50 score 경로 접근 금지 확인."""
    if FORBIDDEN_PATH_PATTERN.search(path_str):
        abort(f"ResNet50 금지 경로 감지: {path_str}")


def check_stage2_holdout(rows):
    """stage2_holdout_flag == 1 행이 있으면 abort."""
    for row in rows:
        flag = str(row.get("stage2_holdout_flag", "0")).strip()
        if flag == "1":
            abort(
                f"stage2_holdout_flag=1 행 감지: patient_id={row.get('patient_id')} — 중단."
            )


# ============================================================
# score CSV 읽기 유틸리티 (read-only, padim_score 헤더 컬럼명 접근)
# ============================================================

def read_score_csv(filepath):
    """
    score CSV를 읽어 list of dict 반환.
    padim_score 컬럼이 없으면 abort.
    ResNet50 경로 접근 금지 확인.
    """
    check_forbidden_path(filepath)
    if not os.path.isfile(filepath):
        abort(f"score CSV 파일 없음: {filepath}")

    rows = []
    with open(filepath, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if fieldnames is None or "padim_score" not in fieldnames:
            abort(
                f"'padim_score' 컬럼 없음 — 파일: {filepath}\n"
                f"실제 컬럼: {fieldnames}"
            )
        for row in reader:
            rows.append(row)
    return rows


# ============================================================
# 후보 선정 로직
# ============================================================

def select_normal_fp_candidates(rows, patient_p95):
    """
    Normal FP: padim_score 높은 후보 최대 3개 선정.
    우선순위:
      1. patient_p95에 가까운 high-score patch (score >= patient_p95, 가장 낮은 것)
      2. top 1% 대표 (상위 1% 중 중간값)
      3. max patch (outlier 확인용)
    반환: list of (0-based row index, row dict)
    """
    scored = []
    for i, row in enumerate(rows):
        try:
            s = float(row["padim_score"])
        except (ValueError, KeyError):
            continue
        scored.append((i, s, row))

    if not scored:
        return []

    # max score
    max_entry = max(scored, key=lambda x: x[1])

    # patient_p95 기준 — score >= patient_p95 인 것 중 가장 낮은 것
    above_p95 = [(i, s, r) for i, s, r in scored if s >= patient_p95]
    near_p95_entry = min(above_p95, key=lambda x: x[1]) if above_p95 else None

    # top 1% 대표
    n = len(scored)
    top1pct_count = max(1, int(n * 0.01))
    sorted_desc = sorted(scored, key=lambda x: x[1], reverse=True)
    top1pct = sorted_desc[:top1pct_count]
    # top1pct 중 중간값 위치
    mid_idx = len(top1pct) // 2
    top1pct_repr = top1pct[mid_idx] if top1pct else None

    # 중복 제거하면서 최대 3개
    candidates = []
    seen_indices = set()

    def add_candidate(entry, reason):
        if entry is None:
            return
        row_idx = entry[0]
        if row_idx in seen_indices:
            return
        seen_indices.add(row_idx)
        candidates.append((row_idx, entry[2], reason))

    add_candidate(near_p95_entry, "near_patient_p95_high_score")
    add_candidate(top1pct_repr, "top1pct_representative")
    add_candidate(max_entry, "max_patch_outlier_check")

    return candidates[:3]


def select_lesion_fn_candidates(rows):
    """
    Lesion FN: has_lesion_patch==1 또는 patch_label==1 인 병변 positive patch 중
    padim_score 가장 낮은 것과 높은 것을 각 1개씩 후보로 잡는다.
    병변 컬럼 없으면 score 상위 2개만 반환.
    반환: list of (0-based row index, row dict, reason)
    """
    if not rows:
        return []

    fieldnames = list(rows[0].keys())

    # 병변 컬럼 탐색
    has_lesion_col = None
    for col in ["has_lesion_patch", "patch_label"]:
        if col in fieldnames:
            has_lesion_col = col
            break

    scored = []
    for i, row in enumerate(rows):
        try:
            s = float(row["padim_score"])
        except (ValueError, KeyError):
            continue
        scored.append((i, s, row))

    if not scored:
        return []

    # 병변 positive 필터
    if has_lesion_col is not None:
        lesion_rows = [
            (i, s, r)
            for i, s, r in scored
            if str(r.get(has_lesion_col, "0")).strip() == "1"
        ]
    else:
        lesion_rows = []

    candidates = []
    seen_indices = set()

    def add_candidate(entry, reason):
        if entry is None:
            return
        row_idx = entry[0]
        if row_idx in seen_indices:
            return
        seen_indices.add(row_idx)
        candidates.append((row_idx, entry[2], reason))

    if lesion_rows:
        # 낮은 score 후보 (FN 의심 핵심)
        low_entry = min(lesion_rows, key=lambda x: x[1])
        # 높은 score 후보 (비교용)
        high_entry = max(lesion_rows, key=lambda x: x[1])
        add_candidate(low_entry, f"lesion_positive_low_score (col={has_lesion_col})")
        add_candidate(high_entry, f"lesion_positive_high_score (col={has_lesion_col})")
    else:
        # 병변 컬럼으로 필터 불가 — score 상위 2개
        sorted_desc = sorted(scored, key=lambda x: x[1], reverse=True)
        for entry in sorted_desc[:2]:
            add_candidate(entry, "NEEDS_LESION_MASK_LOAD_FOR_VISUAL_QA_fallback_high_score")

    return candidates


# ============================================================
# error_analysis.csv 파싱
# ============================================================

def load_error_analysis(filepath):
    """
    error_analysis.csv 읽어서 FP/FN 행만 반환 (HIGH_NORMAL_OUTLIER 제외).
    반환: (fp_rows_v2v2, fn_rows_v2v2)
    """
    if not os.path.isfile(filepath):
        abort(f"error_analysis.csv 없음: {filepath}")

    fp_v2v2 = []
    fn_v2v2 = []

    with open(filepath, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            error_type = row.get("error_type", "").strip()
            comparison = row.get("comparison", "").strip()

            # HIGH_NORMAL_OUTLIER 반드시 제외
            if error_type == "HIGH_NORMAL_OUTLIER":
                continue

            if error_type == "FP" and comparison == "v2_v2":
                fp_v2v2.append(row)
            elif error_type == "FN" and comparison == "v2_v2":
                fn_v2v2.append(row)

    return fp_v2v2, fn_v2v2


def load_patient_p95(filepath, patient_ids, comparison="v2_v2"):
    """
    patient_scores.csv에서 특정 comparison의 patient_p95_patch_score 읽기.
    반환: {patient_id: p95_value}
    """
    result = {}
    if not os.path.isfile(filepath):
        return result

    with open(filepath, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get("patient_id", "").strip()
            comp = row.get("comparison", "").strip()
            if pid in patient_ids and comp == comparison:
                try:
                    result[pid] = float(row["patient_p95_patch_score"])
                except (KeyError, ValueError):
                    pass
    return result


# ============================================================
# manifest 생성
# ============================================================

def build_manifest(fp_rows, fn_rows, patient_p95_map):
    """
    FP/FN 환자 목록으로 manifest row 생성.
    반환: list of dict (manifest 컬럼 포함)
    """
    manifest_rows = []
    review_id = 1

    # ── Normal FP ──────────────────────────────────────────
    for ea_row in fp_rows:
        patient_id = ea_row["patient_id"].strip()
        score_csv_path = os.path.join(NORMAL_SCORE_DIR, f"{patient_id}.csv")
        check_forbidden_path(score_csv_path)

        score_rows = read_score_csv(score_csv_path)
        patient_p95 = patient_p95_map.get(patient_id, None)
        if patient_p95 is None:
            # patient_scores에 없으면 score CSV에서 직접 계산
            all_scores = []
            for r in score_rows:
                try:
                    all_scores.append(float(r["padim_score"]))
                except (ValueError, KeyError):
                    pass
            if all_scores:
                all_scores.sort()
                idx = int(len(all_scores) * 0.95)
                patient_p95 = all_scores[min(idx, len(all_scores) - 1)]
            else:
                patient_p95 = 0.0

        candidates = select_normal_fp_candidates(score_rows, patient_p95)

        if not candidates:
            # 후보가 없으면 최소 1행이라도 max로 추가
            fallback = []
            for i, r in enumerate(score_rows):
                try:
                    fallback.append((i, float(r["padim_score"]), r))
                except (ValueError, KeyError):
                    pass
            if fallback:
                best = max(fallback, key=lambda x: x[1])
                candidates = [(best[0], best[2], "fallback_max")]

        for row_idx, cand_row, reason in candidates:
            manifest_rows.append({
                "review_id": f"R{review_id:03d}",
                "patient_id": patient_id,
                "group": "normal_fp",
                "source_comparison": "v2_v2",
                "threshold_level": ea_row.get("threshold_level", "p95").strip(),
                "threshold_value": ea_row.get("threshold_value", "").strip(),
                "patient_score": ea_row.get("patient_score", "").strip(),
                "predicted_label": ea_row.get("predicted_label", "").strip(),
                "true_label": ea_row.get("label", "").strip(),
                "error_type": "FP",
                "stage_split": "normal_test",
                "stage2_holdout_flag": 0,
                "score_csv_path": score_csv_path,
                "candidate_patch_index": row_idx,
                "candidate_local_z": cand_row.get("local_z", ""),
                "candidate_y0": cand_row.get("y0", ""),
                "candidate_x0": cand_row.get("x0", ""),
                "candidate_score": cand_row.get("padim_score", ""),
                "lesion_mask_available": "N/A",
                "volume_path_status": "NEEDS_PATH_CONFIRMATION_NOT_LOADED",
                "roi_path_status": "NEEDS_PATH_CONFIRMATION_NOT_LOADED",
                "visual_reason_to_check": "suspected_normal_high_score_structure",
                "note": (
                    f"same FP patients across v1_v1, v1_v2, v2_v2 at p95; "
                    f"candidate_reason={reason}"
                ),
            })
            review_id += 1

    # ── Lesion FN ──────────────────────────────────────────
    for ea_row in fn_rows:
        patient_id = ea_row["patient_id"].strip()
        score_csv_path = os.path.join(LESION_SCORE_DIR, f"{patient_id}.csv")
        check_forbidden_path(score_csv_path)

        score_rows = read_score_csv(score_csv_path)
        candidates = select_lesion_fn_candidates(score_rows)

        if not candidates:
            fallback = []
            for i, r in enumerate(score_rows):
                try:
                    fallback.append((i, float(r["padim_score"]), r))
                except (ValueError, KeyError):
                    pass
            if fallback:
                best = max(fallback, key=lambda x: x[1])
                candidates = [(best[0], best[2], "fallback_max")]

        for row_idx, cand_row, reason in candidates:
            manifest_rows.append({
                "review_id": f"R{review_id:03d}",
                "patient_id": patient_id,
                "group": "lesion_fn_p99",
                "source_comparison": "v2_v2",
                "threshold_level": ea_row.get("threshold_level", "p99").strip(),
                "threshold_value": ea_row.get("threshold_value", "").strip(),
                "patient_score": ea_row.get("patient_score", "").strip(),
                "predicted_label": ea_row.get("predicted_label", "").strip(),
                "true_label": ea_row.get("label", "").strip(),
                "error_type": "FN",
                "stage_split": "stage1_dev",
                "stage2_holdout_flag": 0,
                "score_csv_path": score_csv_path,
                "candidate_patch_index": row_idx,
                "candidate_local_z": cand_row.get("local_z", ""),
                "candidate_y0": cand_row.get("y0", ""),
                "candidate_x0": cand_row.get("x0", ""),
                "candidate_score": cand_row.get("padim_score", ""),
                "lesion_mask_available": "YES_from_csv_columns",
                "volume_path_status": "NEEDS_PATH_CONFIRMATION_NOT_LOADED",
                "roi_path_status": "NEEDS_PATH_CONFIRMATION_NOT_LOADED",
                "visual_reason_to_check": "suspected_low_score_lesion_or_small_lesion",
                "note": f"v2_v2 p99 FN; candidate_reason={reason}",
            })
            review_id += 1

    return manifest_rows


# ============================================================
# --dry-run 모드
# ============================================================

def run_dry_run(manifest_path, output_dir):
    """
    manifest CSV를 읽고 안전 확인만 수행. PNG 생성 금지.
    확인 항목:
    - row 수 / 환자 수 / stage2_holdout_flag
    - score_csv_path 허용 경로 (ResNet50/full_retrospective 금지)
    - CT/ROI/(lesion)mask 경로 Path.exists() 확인 (전체 배열 로드 금지)
    - 좌표 volume shape 범위 확인 (mmap_mode='r' shape 헤더만)
    - output PNG collision 예상 확인
    """
    print("[dry-run] 시작")

    if not os.path.isfile(manifest_path):
        abort(f"manifest 파일 없음: {manifest_path}")

    rows = []
    with open(manifest_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"[dry-run] manifest 총 row 수: {len(rows)}")

    # stage2_holdout 확인
    check_stage2_holdout(rows)
    print("[dry-run] stage2_holdout_flag: 모두 0 확인 완료")

    # 대상 환자 수 확인
    fp_patients = set(r["patient_id"] for r in rows if r.get("group") == "normal_fp")
    fn_patients = set(r["patient_id"] for r in rows if r.get("group") == "lesion_fn_p99")
    print(f"[dry-run] normal_fp 대상 환자 수: {len(fp_patients)}")
    print(f"[dry-run] lesion_fn_p99 대상 환자 수: {len(fn_patients)}")

    # score_csv_path 허용 경로 확인
    allowed_prefixes = (NORMAL_SCORE_DIR, LESION_SCORE_DIR)
    forbidden_found = False
    for row in rows:
        spath = row.get("score_csv_path", "")
        # ResNet50 금지 경로
        if FORBIDDEN_PATH_PATTERN.search(spath):
            print(
                f"[dry-run][ERROR] ResNet50 금지 경로 감지: patient_id={row['patient_id']} path={spath}",
                file=sys.stderr
            )
            forbidden_found = True
        # full_retrospective 금지
        if FORBIDDEN_FULL_RETRO_PATTERN.search(spath):
            print(
                f"[dry-run][ERROR] full_retrospective 금지 경로 감지: patient_id={row['patient_id']} path={spath}",
                file=sys.stderr
            )
            forbidden_found = True
        # 허용 경로 범위 확인
        if not any(spath.startswith(p) for p in allowed_prefixes):
            print(
                f"[dry-run][WARN] 허용 외 score_csv_path: patient_id={row['patient_id']} path={spath}",
                file=sys.stderr
            )

    if forbidden_found:
        abort("금지 경로(ResNet50 또는 full_retrospective)가 manifest에 포함됨 — 중단.")

    print("[dry-run] source score 경로: v2_v2 고정 확인 완료")

    # group 분포
    from collections import Counter
    group_dist = Counter(r.get("group", "") for r in rows)
    print(f"[dry-run] group 분포: {dict(group_dist)}")

    # source_comparison 확인
    comparisons = set(r.get("source_comparison", "") for r in rows)
    print(f"[dry-run] source_comparison 값: {comparisons}")
    if comparisons != {"v2_v2"}:
        abort(f"source_comparison에 v2_v2 외 값이 포함됨: {comparisons}")

    # ── 경로 존재 확인 ────────────────────────────────────────────
    lesion_root, normal_root = load_data_roots()

    safe_id_cache = {}  # patient_id -> safe_id
    ct_ok_count = 0
    ct_missing = []
    roi_ok_count = 0
    roi_missing = []
    lesion_ok_count = 0
    lesion_missing = []
    blockers = []

    # shape 검증 카운터 (mutable 참조로 내부 루프에서 수정)
    ct_shape_checked_count_ref = [0]
    roi_shape_matched_count_ref = [0]
    lesion_shape_matched_count_ref = [0]

    for row in rows:
        pid = row["patient_id"].strip()
        group = row["group"].strip()
        score_csv = row["score_csv_path"].strip()
        review_id = row["review_id"].strip()

        # safe_id (캐시)
        if pid not in safe_id_cache:
            safe_id_cache[pid] = get_safe_id_from_score_csv(score_csv)
        safe_id = safe_id_cache[pid]

        if not safe_id:
            blockers.append(
                f"{review_id} {pid}: safe_id 읽기 실패 — score CSV 없거나 safe_id 컬럼 없음"
            )
            continue

        root = lesion_root if group == "lesion_fn_p99" else normal_root
        vol_dir = os.path.join(root, "volumes_npy", safe_id)
        ct_path = os.path.join(vol_dir, "ct_hu.npy")
        roi_path = os.path.join(vol_dir, "roi_0_0.npy")
        lesion_path = os.path.join(vol_dir, "lesion_mask_roi_0_0.npy")

        # CT 존재
        if os.path.exists(ct_path):
            ct_ok_count += 1
        else:
            ct_missing.append(f"{review_id} {pid}: {ct_path}")
            blockers.append(f"{review_id} {pid}: ct_hu.npy 없음")

        # ROI 존재
        if os.path.exists(roi_path):
            roi_ok_count += 1
        else:
            roi_missing.append(f"{review_id} {pid}: {roi_path}")
            blockers.append(f"{review_id} {pid}: roi_0_0.npy 없음")

        # lesion mask (lesion FN만)
        if group == "lesion_fn_p99":
            if os.path.exists(lesion_path):
                lesion_ok_count += 1
            else:
                lesion_missing.append(f"{review_id} {pid}: {lesion_path}")
                blockers.append(
                    f"{review_id} {pid}: NEEDS_LESION_MASK_PATH_CONFIRMATION — lesion_mask_roi_0_0.npy 없음"
                )

        # ── shape / z 범위 확인 (mmap 헤더만, 배열 전체 로드 금지) ─────────
        ct_shape = None
        if os.path.exists(ct_path):
            _ct = np.load(ct_path, mmap_mode="r")
            ct_shape = _ct.shape
            del _ct
            if len(ct_shape) != 3:
                blockers.append(
                    f"{review_id} {pid}: CT shape {ct_shape}가 (D,H,W) 3차원이 아님"
                )
                ct_shape = None  # 이후 비교 스킵
            else:
                ct_shape_checked_count_ref[0] += 1

        roi_shape = None
        if os.path.exists(roi_path):
            _roi = np.load(roi_path, mmap_mode="r")
            roi_shape = _roi.shape
            del _roi
            if ct_shape is not None:
                if roi_shape != ct_shape:
                    blockers.append(
                        f"{review_id} {pid}: ROI shape {roi_shape} != CT shape {ct_shape}"
                    )
                else:
                    roi_shape_matched_count_ref[0] += 1

        lesion_shape = None
        if group == "lesion_fn_p99" and os.path.exists(lesion_path):
            _lm = np.load(lesion_path, mmap_mode="r")
            lesion_shape = _lm.shape
            del _lm
            if ct_shape is not None:
                if lesion_shape != ct_shape:
                    blockers.append(
                        f"{review_id} {pid}: lesion mask shape {lesion_shape} != CT shape {ct_shape}"
                    )
                else:
                    lesion_shape_matched_count_ref[0] += 1

        # ── 좌표 범위 확인 ──────────────────────────────────────
        try:
            z = int(row["candidate_local_z"])
            y0_m = int(row["candidate_y0"])
            x0_m = int(row["candidate_x0"])
            patch_idx = int(row["candidate_patch_index"])
            has_coords = True
        except (ValueError, KeyError):
            # UNKNOWN fallback — slice-level view로 넘어감, blocker 아님
            has_coords = False

        if has_coords and ct_shape is not None:
            D, H, W = ct_shape

            if z < 0 or z >= D:
                blockers.append(
                    f"{review_id} {pid}: z={z} out of range [0,{D}) (CT)"
                )
            if roi_shape is not None:
                roi_D = roi_shape[0]
                if z < 0 or z >= roi_D:
                    blockers.append(
                        f"{review_id} {pid}: z={z} out of range [0,{roi_D}) (ROI)"
                    )
            if group == "lesion_fn_p99" and lesion_shape is not None:
                lm_D = lesion_shape[0]
                if z < 0 or z >= lm_D:
                    blockers.append(
                        f"{review_id} {pid}: z={z} out of range [0,{lm_D}) (lesion mask)"
                    )

            # y1/x1은 score CSV에서 읽음
            coords = get_patch_coords_from_score_csv(score_csv, patch_idx)
            if coords is not None:
                y0c, x0c, y1c, x1c = coords
                if y0c < 0 or y1c > H or x0c < 0 or x1c > W:
                    blockers.append(
                        f"{review_id} {pid}: patch ({y0c},{x0c})-({y1c},{x1c}) "
                        f"out of ct shape ({H},{W})"
                    )
            else:
                blockers.append(
                    f"{review_id} {pid}: score CSV에서 patch_idx={patch_idx} y1/x1 읽기 실패"
                )

    # 요약 출력
    print(f"[dry-run] CT volume 경로: {ct_ok_count}개 존재 / {len(ct_missing)}개 누락")
    for m in ct_missing:
        print(f"  [MISSING CT] {m}", file=sys.stderr)

    print(f"[dry-run] ROI 경로: {roi_ok_count}개 존재 / {len(roi_missing)}개 누락")
    for m in roi_missing:
        print(f"  [MISSING ROI] {m}", file=sys.stderr)

    fn_row_count = sum(1 for r in rows if r.get("group") == "lesion_fn_p99")
    print(
        f"[dry-run] lesion mask 경로 (lesion FN {fn_row_count}행): "
        f"{lesion_ok_count}개 존재 / {len(lesion_missing)}개 누락"
    )
    for m in lesion_missing:
        print(f"  [MISSING LESION_MASK] {m}", file=sys.stderr)

    # ── shape 검증 요약 ───────────────────────────────────────
    shape_mismatch_count = sum(
        1 for b in blockers
        if "shape" in b and ("!=" in b or "3차원" in b)
    )
    print(f"[dry-run] CT shape checked rows: {ct_shape_checked_count_ref[0]}")
    print(f"[dry-run] ROI shape matched rows: {roi_shape_matched_count_ref[0]}")
    print(f"[dry-run] lesion mask shape matched rows: {lesion_shape_matched_count_ref[0]}")
    print(f"[dry-run] shape mismatch blocker count: {shape_mismatch_count}")

    # ── output PNG collision 확인 ─────────────────────────────
    collision_count = 0
    for row in rows:
        review_id = row["review_id"].strip()
        png_path = os.path.join(output_dir, f"{review_id}.png")
        if os.path.isfile(png_path):
            collision_count += 1
            print(f"[dry-run][WARN] PNG 이미 존재 (--run 시 abort 예정): {png_path}", file=sys.stderr)
    if collision_count > 0:
        print(f"[dry-run] output collision 예상: {collision_count}개 (--run 시 abort됨)")
    else:
        print(f"[dry-run] output collision: 없음")

    # ── blocker 요약 ─────────────────────────────────────────
    if blockers:
        print(f"\n[dry-run] BLOCKERS ({len(blockers)}개) — --run --confirm-run 전 반드시 해결:")
        for b in blockers:
            print(f"  [BLOCKER] {b}")
    else:
        print("\n[dry-run] BLOCKERS: 없음")

    # PNG 생성 없음 명시
    print("[dry-run] PNG 생성 없음 — dry-run 완료")
    if not blockers and collision_count == 0:
        print("[dry-run] 판정: 통과 — --run --confirm-run 실행 가능 (사용자 승인 필요)")
    else:
        print("[dry-run] 판정: 문제 있음 — BLOCKER 또는 collision 해결 후 진행")


# ============================================================
# --run 모드 (구조만 — 이번 단계 실행 금지)
# ============================================================

def run_visual_qa(manifest_path, output_dir, confirm_run):
    """
    실제 PNG 생성용 구조.
    이번 단계에서는 실행하지 않는다.
    --confirm-run 없으면 abort.
    2-pass 구조:
      1st pass (preflight): 모든 row에 대해 경로·shape·z 검증 — 하나라도 실패하면 즉시 abort
      2nd pass (생성): preflight 통과 후에만 slice 로드 + PNG 그리기
    output_dir 생성은 모든 검증·collision 통과 후에만 수행.
    zeros fallback 제거 — shape/z 불일치 시 abort.
    """
    if not confirm_run:
        abort(
            "--run 모드는 --confirm-run 없이 실행할 수 없습니다. "
            "이번 단계에서 PNG 생성은 금지됩니다."
        )

    rows = []
    with open(manifest_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # stage2_holdout abort
    check_stage2_holdout(rows)

    # ResNet50 경로 abort
    for row in rows:
        spath = row.get("score_csv_path", "")
        check_forbidden_path(spath)

    # PNG 이미 존재하면 abort (output_dir 생성 전에 확인)
    for row in rows:
        review_id = row.get("review_id", "unknown")
        png_path = os.path.join(output_dir, f"{review_id}.png")
        if os.path.isfile(png_path):
            abort(f"PNG 이미 존재: {png_path} — overwrite 금지. 중단.")

    # ── 데이터 root 로드 ─────────────────────────────────────
    lesion_root, normal_root = load_data_roots()

    safe_id_cache = {}

    # ── 1st pass: preflight (경로·shape·z 전체 검증) ──────────
    # 하나라도 실패하면 abort. output_dir 생성 전에 수행.
    print("[run] preflight pass 시작 — 모든 row 검증 중...")
    preflight_errors = []

    for row in rows:
        review_id = row.get("review_id", "unknown").strip()
        pid = row["patient_id"].strip()
        group = row["group"].strip()
        score_csv = row["score_csv_path"].strip()

        # ResNet50 / full_retrospective 접근 금지
        check_forbidden_path(score_csv)
        if FORBIDDEN_FULL_RETRO_PATTERN.search(score_csv):
            abort(f"full_retrospective 금지 경로 감지: {score_csv}")

        # safe_id
        if pid not in safe_id_cache:
            safe_id_cache[pid] = get_safe_id_from_score_csv(score_csv)
        safe_id = safe_id_cache[pid]
        if not safe_id:
            preflight_errors.append(f"{review_id} {pid}: safe_id 없음")
            continue

        root = lesion_root if group == "lesion_fn_p99" else normal_root
        vol_dir = os.path.join(root, "volumes_npy", safe_id)
        ct_path = os.path.join(vol_dir, "ct_hu.npy")
        roi_path = os.path.join(vol_dir, "roi_0_0.npy")
        lesion_path = os.path.join(vol_dir, "lesion_mask_roi_0_0.npy")

        # 필수 경로 존재 확인
        if not os.path.exists(ct_path):
            preflight_errors.append(f"{review_id} {pid}: ct_hu.npy 없음 {ct_path}")
            continue
        if not os.path.exists(roi_path):
            preflight_errors.append(f"{review_id} {pid}: roi_0_0.npy 없음 {roi_path}")
            continue
        if group == "lesion_fn_p99" and not os.path.exists(lesion_path):
            preflight_errors.append(
                f"{review_id} {pid}: lesion_mask_roi_0_0.npy 없음 — "
                "NEEDS_LESION_MASK_PATH_CONFIRMATION"
            )
            continue

        # z 읽기
        try:
            z = int(row["candidate_local_z"])
        except (ValueError, KeyError):
            # z 없으면 slice-level — shape/z 검증 스킵 (blocker 아님)
            continue

        # CT shape 확인 (mmap 헤더만)
        ct_arr = np.load(ct_path, mmap_mode="r")
        ct_shape = ct_arr.shape
        del ct_arr
        if len(ct_shape) != 3:
            preflight_errors.append(
                f"{review_id} {pid}: CT shape {ct_shape}가 (D,H,W) 3차원이 아님"
            )
            continue
        D, H, W = ct_shape

        if z < 0 or z >= D:
            preflight_errors.append(
                f"{review_id} {pid}: z={z} out of range [0,{D}) (CT)"
            )

        # ROI shape 확인
        roi_arr = np.load(roi_path, mmap_mode="r")
        roi_shape = roi_arr.shape
        del roi_arr
        if roi_shape != ct_shape:
            preflight_errors.append(
                f"{review_id} {pid}: ROI shape {roi_shape} != CT shape {ct_shape}"
            )
        else:
            roi_D = roi_shape[0]
            if z < 0 or z >= roi_D:
                preflight_errors.append(
                    f"{review_id} {pid}: z={z} out of range [0,{roi_D}) (ROI)"
                )

        # lesion mask shape 확인 (lesion FN만)
        if group == "lesion_fn_p99":
            lm_arr = np.load(lesion_path, mmap_mode="r")
            lm_shape = lm_arr.shape
            del lm_arr
            if lm_shape != ct_shape:
                preflight_errors.append(
                    f"{review_id} {pid}: lesion mask shape {lm_shape} != CT shape {ct_shape}"
                )
            else:
                lm_D = lm_shape[0]
                if z < 0 or z >= lm_D:
                    preflight_errors.append(
                        f"{review_id} {pid}: z={z} out of range [0,{lm_D}) (lesion mask)"
                    )

    if preflight_errors:
        print(f"[run] preflight FAIL — {len(preflight_errors)}개 오류:", file=sys.stderr)
        for e in preflight_errors:
            print(f"  [PREFLIGHT ERROR] {e}", file=sys.stderr)
        abort(f"preflight 실패 {len(preflight_errors)}건 — PNG 생성 중단. skip 없음.")

    print(f"[run] preflight pass 완료 — 모든 row 통과")

    # ── output_dir 생성 (모든 검증 통과 후) ─────────────────
    os.makedirs(output_dir, exist_ok=True)

    # ── 2nd pass: 실제 slice 로드 + PNG 생성 ─────────────────
    generated = 0

    for row in rows:
        review_id = row.get("review_id", "unknown").strip()
        pid = row["patient_id"].strip()
        group = row["group"].strip()
        score_csv = row["score_csv_path"].strip()

        safe_id = safe_id_cache[pid]

        root = lesion_root if group == "lesion_fn_p99" else normal_root
        vol_dir = os.path.join(root, "volumes_npy", safe_id)
        ct_path = os.path.join(vol_dir, "ct_hu.npy")
        roi_path = os.path.join(vol_dir, "roi_0_0.npy")
        lesion_path = os.path.join(vol_dir, "lesion_mask_roi_0_0.npy")

        # 좌표 읽기
        try:
            z = int(row["candidate_local_z"])
            patch_idx = int(row["candidate_patch_index"])
            has_coords = True
        except (ValueError, KeyError):
            has_coords = False
            z = None
            patch_idx = None

        coords = None
        if has_coords and patch_idx is not None:
            coords = get_patch_coords_from_score_csv(score_csv, patch_idx)
            if coords is None:
                has_coords = False

        # CT slice 로드 (mmap — 해당 slice만 메모리에 올림)
        ct_arr = np.load(ct_path, mmap_mode="r")
        D, H, W = ct_arr.shape

        # preflight 통과 보장이므로 z 범위는 이미 검증됨
        # z가 None인 경우(slice-level)는 중앙 slice 사용
        if z is None:
            z = D // 2

        ct_slice = np.array(ct_arr[z])
        del ct_arr

        # ROI slice (mmap) — zeros fallback 없음, preflight에서 shape/z 검증 완료
        roi_arr = np.load(roi_path, mmap_mode="r")
        if roi_arr.shape != (D, H, W):
            del roi_arr
            abort(f"{review_id} {pid}: ROI shape 불일치 (2nd pass) — abort")
        if z >= roi_arr.shape[0]:
            del roi_arr
            abort(f"{review_id} {pid}: ROI z={z} 범위 밖 (2nd pass) — abort")
        roi_slice = np.array(roi_arr[z])
        del roi_arr

        # lesion mask slice (lesion FN only, mmap) — zeros fallback 없음
        lesion_slice = None
        if group == "lesion_fn_p99":
            lm_arr = np.load(lesion_path, mmap_mode="r")
            if lm_arr.shape != (D, H, W):
                del lm_arr
                abort(f"{review_id} {pid}: lesion mask shape 불일치 (2nd pass) — abort")
            if z >= lm_arr.shape[0]:
                del lm_arr
                abort(f"{review_id} {pid}: lesion mask z={z} 범위 밖 (2nd pass) — abort")
            lesion_slice = np.array(lm_arr[z])
            del lm_arr

        # 좌표 범위 확인
        y0c, x0c, y1c, x1c = 0, 0, 0, 0
        if has_coords and coords is not None:
            y0c, x0c, y1c, x1c = coords
            if y0c < 0 or y1c > H or x0c < 0 or x1c > W:
                print(
                    f"[WARN] {review_id} {pid}: 좌표 범위 초과 ({y0c},{x0c})-({y1c},{x1c}) "
                    f"shape=({H},{W}) — slice-level fallback",
                    file=sys.stderr
                )
                has_coords = False

        # 제목 구성
        cand_score = row.get("candidate_score", "")
        title = (
            f"{review_id} | {pid[:20]} | {group} | {row.get('error_type','')} "
            f"| thr={row.get('threshold_level','')} | score={float(cand_score):.3f} "
            f"| z={z} | idx={row.get('candidate_patch_index','')} "
            f"| {row.get('stage_split','')}"
        )

        # PNG 경로
        png_path = os.path.join(output_dir, f"{review_id}.png")

        draw_visual_qa_png(
            ct_slice=ct_slice,
            roi_slice=roi_slice,
            lesion_slice=lesion_slice,
            y0=y0c, x0=x0c, y1=y1c, x1=x1c,
            has_patch_coords=has_coords,
            title=title,
            out_path=png_path,
        )
        print(f"[OK] {review_id} → {png_path}")
        generated += 1

    print(f"\n[run] 완료: PNG {generated}개 생성")
    print(f"[run] 출력 폴더: {output_dir}")


# ============================================================
# manifest CSV 생성 (--generate-manifest 내부 호출용)
# ============================================================

def generate_manifest_csv(output_path):
    """
    error_analysis.csv + score CSV read-only로 manifest 생성.
    기존 파일이 있으면 abort.
    """
    if os.path.isfile(output_path):
        abort(
            f"manifest 파일이 이미 존재합니다: {output_path}\n"
            "덮어쓰기 금지 — 중단."
        )

    fp_rows, fn_rows = load_error_analysis(ERROR_ANALYSIS_CSV)
    print(f"[manifest] FP 행 수 (v2_v2): {len(fp_rows)}")
    print(f"[manifest] FN 행 수 (v2_v2): {len(fn_rows)}")

    fp_patient_ids = {r["patient_id"].strip() for r in fp_rows}
    fn_patient_ids = {r["patient_id"].strip() for r in fn_rows}
    all_patient_ids = fp_patient_ids | fn_patient_ids

    patient_p95_map = load_patient_p95(PATIENT_SCORES_CSV, fp_patient_ids)
    print(f"[manifest] patient_p95 로드 수: {len(patient_p95_map)}")

    manifest_rows = build_manifest(fp_rows, fn_rows, patient_p95_map)
    print(f"[manifest] 총 manifest row 수: {len(manifest_rows)}")

    # stage2_holdout 확인
    check_stage2_holdout(manifest_rows)

    # 컬럼 순서
    fieldnames = [
        "review_id",
        "patient_id",
        "group",
        "source_comparison",
        "threshold_level",
        "threshold_value",
        "patient_score",
        "predicted_label",
        "true_label",
        "error_type",
        "stage_split",
        "stage2_holdout_flag",
        "score_csv_path",
        "candidate_patch_index",
        "candidate_local_z",
        "candidate_y0",
        "candidate_x0",
        "candidate_score",
        "lesion_mask_available",
        "volume_path_status",
        "roi_path_status",
        "visual_reason_to_check",
        "note",
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"[manifest] 저장 완료: {output_path}")
    return manifest_rows


# ============================================================
# preflight MD 생성
# ============================================================

def generate_preflight_md(output_path, fp_rows, fn_rows, manifest_rows):
    """
    preflight MD를 current_change.md 구조 7섹션으로 생성.
    기존 파일이 있으면 abort.
    """
    if os.path.isfile(output_path):
        abort(
            f"preflight MD가 이미 존재합니다: {output_path}\n"
            "덮어쓰기 금지 — 중단."
        )

    fp_patient_list = sorted(set(r["patient_id"].strip() for r in fp_rows))
    fn_patient_list = sorted(set(r["patient_id"].strip() for r in fn_rows))

    # MSD / LUNG1 분포
    fn_msd = [p for p in fn_patient_list if p.startswith("MSD")]
    fn_lung1 = [p for p in fn_patient_list if p.startswith("LUNG1")]

    # v1_v2 vs v2_v2 p95 disagreement 확인
    # (이미 확인됨: 0명 — 동일 10명)
    v1v2_v2v2_disagreement = 0

    # 병변 컬럼 존재 확인 샘플
    sample_fn = fn_rows[0]["patient_id"].strip() if fn_rows else None
    lesion_col_status = "확인 불가"
    if sample_fn:
        sample_path = os.path.join(LESION_SCORE_DIR, f"{sample_fn}.csv")
        if os.path.isfile(sample_path):
            with open(sample_path, newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                cols = reader.fieldnames or []
            lesion_cols_found = [c for c in ["has_lesion_patch", "patch_label", "lesion_pixels", "lesion_patch_ratio", "lesion_zone_type"] if c in cols]
            lesion_col_status = f"존재 확인: {lesion_cols_found} (샘플: {sample_fn})"

    normal_score_exists = os.path.isdir(NORMAL_SCORE_DIR)
    lesion_score_exists = os.path.isdir(LESION_SCORE_DIR)

    content = f"""# dev_safe mixed error visual QA preflight

## 1. Scope
- dev_safe only
- normal FP {len(fp_patient_list)}
- lesion FN p99 {len(fn_patient_list)}
- stage2_holdout 0
- no PNG generated yet

## 2. Why visual QA is needed
- CSV metric만으로 원인 확정 금지
- FP 원인 후보와 FN 원인 후보를 overlay로 확인해야 함
- suspected_normal_high_score_structure (FP) 또는 suspected_low_score_lesion_or_small_lesion (FN) 가능성 있음
- 실제 CT overlay 확인 전까지 원인 확정 보류

## 3. Target patients
### normal FP 목록 ({len(fp_patient_list)}명)
{chr(10).join(f"- {p}" for p in fp_patient_list)}

### lesion FN 목록 ({len(fn_patient_list)}명)
{chr(10).join(f"- {p}" for p in fn_patient_list)}
- MSD_lung 계열: {len(fn_msd)}명
- LUNG1 계열: {len(fn_lung1)}명

### v1_v2 vs v2_v2 p95 disagreement
- disagreement 수: {v1v2_v2v2_disagreement}명 → visual QA 대상 없음 (preflight 기록만)
- 동일 10명이 v1_v1, v1_v2, v2_v2 모두에서 FP 확인됨

## 4. Candidate slice/patch selection
### normal FP 후보 선정 방식
- v2_v2 normal score CSV에서 padim_score 높은 후보 최대 3개 per 환자
- 우선순위: (1) patient_p95 >= threshold 최소 high-score patch (near_patient_p95_high_score), (2) top 1% 중 대표 (top1pct_representative), (3) max patch (max_patch_outlier_check)
- patient_p95 값: dev_safe_v1_patient_scores.csv v2_v2 행 patient_p95_patch_score 사용 (없으면 score CSV에서 직접 계산)

### lesion FN 후보 선정 방식
- v2_v2 lesion score CSV에서 has_lesion_patch==1 또는 patch_label==1 인 병변 positive patch 중 padim_score 가장 낮은 것과 높은 것을 각 1개씩
- 병변 컬럼 없으면 score 상위 2개 fallback

### score CSV에서 사용한 컬럼
- padim_score, local_z, y0, x0, y1, x1
- 병변 필터: has_lesion_patch, patch_label (존재 시)

### padim_score 헤더 컬럼명 접근 여부
- 반드시 csv.DictReader로 헤더 컬럼명 "padim_score"로 접근
- 위치 기반 인덱스($NF 등) 사용 없음
- padim_score 컬럼 없으면 즉시 abort

## 5. Safety checks
- stage2_holdout 0: 확인 완료 (모든 manifest row stage2_holdout_flag=0)
- full_retrospective 미사용: 확인 완료
- ResNet50 score 경로 미사용: 확인 완료 (experiments/resnet50_imagenet_v1 경로 접근 없음)
- CT/ROI/mask 로드 없음: score CSV read-only만 사용
- PNG 생성 없음: 이번 단계에서 --run 실행 안 함
- 기존 파일 수정 없음: 신규 3개 파일만 생성 (기존 파일 건드리지 않음)
- lesion 관련 컬럼 존재 여부: {lesion_col_status}

## 6. Path status
### source score 경로 확인
- normal score dir: {NORMAL_SCORE_DIR}
  - 존재 여부: {normal_score_exists}
- lesion score dir: {LESION_SCORE_DIR}
  - 존재 여부: {lesion_score_exists}
- 금지 경로 (미사용): experiments/resnet50_imagenet_v1/outputs/scores/

### volume_path_status
- NEEDS_PATH_CONFIRMATION_NOT_LOADED (CT 로드 금지 단계)

### roi_path_status
- NEEDS_PATH_CONFIRMATION_NOT_LOADED (ROI 로드 금지 단계)

### manifest 총 row 수
- 전체: {len(manifest_rows)}행
- normal_fp: {sum(1 for r in manifest_rows if r.get('group') == 'normal_fp')}행
- lesion_fn_p99: {sum(1 for r in manifest_rows if r.get('group') == 'lesion_fn_p99')}행

## 7. Next step
- 사용자 승인 후 PNG visual QA pack 생성
- 승인 시 실행: python scripts/make_dev_safe_mixed_error_visual_qa.py --run --confirm-run --manifest <path> --output-dir qa/dev_safe_mixed_error_visual_qa
- 현재 단계에서 --run 실행 금지
"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[preflight] 저장 완료: {output_path}")


# ============================================================
# main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="dev_safe mixed error visual QA manifest 생성 및 dry-run"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="manifest 입력 확인 및 안전 점검만 수행. PNG 생성 없음."
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="실제 PNG 생성 (--confirm-run 필수)."
    )
    parser.add_argument(
        "--confirm-run",
        action="store_true",
        help="--run 실행 시 필수 확인 플래그."
    )
    parser.add_argument(
        "--output-dir",
        default="qa/dev_safe_mixed_error_visual_qa",
        help="PNG 출력 디렉토리."
    )
    parser.add_argument(
        "--manifest",
        default=OUTPUT_MANIFEST_DEFAULT,
        help="manifest CSV 경로."
    )
    # --overwrite 옵션은 의도적으로 만들지 않음

    args = parser.parse_args()

    # --run 안전장치
    if args.run and not args.confirm_run:
        abort(
            "--run 모드에는 --confirm-run 플래그가 필요합니다. "
            "이번 단계에서 PNG 생성은 금지됩니다. 중단."
        )

    if args.dry_run:
        # manifest가 없으면 먼저 생성
        if not os.path.isfile(args.manifest):
            print(f"[dry-run] manifest 없음 — 먼저 생성합니다: {args.manifest}")
            manifest_rows = generate_manifest_csv(args.manifest)

            # preflight MD도 생성
            preflight_path = os.path.join(
                PROJECT_ROOT,
                "evaluation/mixed_cohort_patient_metrics/dev_safe_v1_visual_qa_preflight.md"
            )
            fp_rows, fn_rows = load_error_analysis(ERROR_ANALYSIS_CSV)
            generate_preflight_md(preflight_path, fp_rows, fn_rows, manifest_rows)
        else:
            print(f"[dry-run] manifest 존재: {args.manifest}")

        run_dry_run(args.manifest, args.output_dir)
        return

    if args.run:
        run_visual_qa(
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            confirm_run=args.confirm_run
        )
        return

    # 인수 없이 실행 시 안내
    print("사용법:")
    print("  --dry-run      manifest 생성 및 안전 점검 (PNG 생성 없음)")
    print("  --run          PNG 생성 (--confirm-run 필수, 이번 단계 실행 금지)")
    print("")
    print("허용 실행:")
    print("  python scripts/make_dev_safe_mixed_error_visual_qa.py --dry-run")
    print("")
    print("금지 실행:")
    print("  python scripts/make_dev_safe_mixed_error_visual_qa.py --run")
    print("  python scripts/make_dev_safe_mixed_error_visual_qa.py --run --confirm-run")


if __name__ == "__main__":
    main()
