#!/usr/bin/env python3
"""
Phase 5.41: Hard Negative Top-Score QA PNG Pack 생성 script

이 script는 QA manifest 150개 crop을 PNG QA pack으로 생성합니다.
crop npz의 image key만 사용하며 원본 volume에 접근하지 않습니다.

주의:
- threshold 확정 아님
- 병변 성능 결론 금지
- stage2_holdout 미사용
- v2 미사용
- 원본 volume 미접근
- crop 복사 없음
- score 재계산 없음
"""

import argparse
import json
import sys
import os
import re
import math
import numpy as np
import pandas as pd
import cv2
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

EXPECTED_ROW_COUNT = 150
EXPECTED_PATIENT_COUNT = 32

REQUIRED_COLUMNS = [
    "patient_id",
    "safe_id",
    "crop_id",
    "crop_path",
    "crop_score_l1_mean",
    "crop_score_l1_max",
    "crop_score_mse_mean",
    "threshold_exceed_val_p90",
    "threshold_exceed_val_p95",
    "threshold_exceed_val_p99",
    "padim_score_mean",
    "padim_score_max",
    "large_bbox_flag",
    "rd4ad_label",
    "binary_label",
    "original_candidate_role",
    "qa_group",
    "qa_priority",
    "suggested_review_reason",
    "manual_label_placeholder",
]

BLOCKED_PATH_KEYWORDS = ["stage2_holdout", "holdout", "v2"]
BLOCKED_EXTENSIONS = (".nii", ".nii.gz")

EXPECTED_IMAGE_KEY = "image"
EXPECTED_SHAPE = (6, 96, 96)

CELL_H = 96
CELL_W = 96
MARGIN_TOP = 64
MARGIN_BOTTOM = 52
CONTACT_CELL_H = 96
CONTACT_CELL_W = 192   # lung ch1(96) + medi ch4(96) 나란히
CONTACT_GAP = 1

RUNTIME_SUMMARY_KEY = "phase5_41_hard_negative_top_score_png_pack"

DEFAULT_OUTPUT_ROOT = (
    "outputs/second-stage-lesion-refiner-v1/"
    "visualizations/hard_negative_top_score_qa_v1"
)

PNG_INDEX_COLUMNS = [
    "patient_id",
    "crop_id",
    "crop_path",
    "png_path",
    "contact_sheet_path",
    "qa_group",
    "qa_priority",
    "crop_score_l1_mean",
    "threshold_exceed_val_p90",
    "threshold_exceed_val_p95",
    "threshold_exceed_val_p99",
    "manual_label_placeholder",
    "png_generated",
    "generation_error",
]


# ---------------------------------------------------------------------------
# 안전장치
# ---------------------------------------------------------------------------

def _assert_no_blocked_path(path_str: str, label: str):
    """경로에 금지 키워드 또는 금지 확장자가 있으면 즉시 종료."""
    path_lower = str(path_str).lower()
    for kw in BLOCKED_PATH_KEYWORDS:
        if kw in path_lower:
            print(f"[BLOCKED] {label} 경로에 '{kw}' 포함 — 사용 금지: {path_str}", file=sys.stderr)
            sys.exit(1)
    for ext in BLOCKED_EXTENSIONS:
        if path_lower.endswith(ext):
            print(f"[BLOCKED] {label} 경로에 '{ext}' 확장자 포함 — 원본 volume 접근 금지: {path_str}", file=sys.stderr)
            sys.exit(1)


def check_crop_paths_blocked(df: pd.DataFrame):
    """crop_path 전체에 금지 키워드/확장자가 없는지 확인. 1건이라도 있으면 종료."""
    if "crop_path" not in df.columns:
        return
    errors = []
    for path_str in df["crop_path"].dropna().astype(str):
        path_lower = path_str.lower()
        for kw in BLOCKED_PATH_KEYWORDS:
            if kw in path_lower:
                errors.append(f"[BLOCKED] crop_path에 '{kw}' 포함: {path_str}")
        for ext in BLOCKED_EXTENSIONS:
            if path_lower.endswith(ext):
                errors.append(f"[BLOCKED] crop_path에 '{ext}' 포함 — 원본 volume 접근 금지: {path_str}")
        if not path_str.endswith(".npz"):
            errors.append(f"[BLOCKED] crop_path가 .npz 아님: {path_str}")

    if errors:
        for e in errors[:5]:
            print(e, file=sys.stderr)
        if len(errors) > 5:
            print(f"[BLOCKED] ... 이하 {len(errors) - 5}건 더 있음", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# 입력 파일 로드 (read-only)
# ---------------------------------------------------------------------------

def load_manifest(manifest_path: str) -> pd.DataFrame:
    _assert_no_blocked_path(manifest_path, "qa-manifest")
    if not os.path.isfile(manifest_path):
        print(f"[ERROR] QA manifest 파일 없음: {manifest_path}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(manifest_path)
    print(f"[INFO] QA manifest 로드 완료 — rows={len(df)}, columns={len(df.columns)}")
    return df


def load_summary(summary_path: str) -> dict:
    _assert_no_blocked_path(summary_path, "qa-summary")
    if not os.path.isfile(summary_path):
        print(f"[ERROR] QA summary JSON 파일 없음: {summary_path}", file=sys.stderr)
        sys.exit(1)
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[INFO] QA summary JSON 로드 완료")
    return data


def check_review_plan_exists(review_plan_path: str):
    _assert_no_blocked_path(review_plan_path, "review-plan")
    if not os.path.isfile(review_plan_path):
        print(f"[ERROR] review plan 파일 없음: {review_plan_path}", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] review plan 존재 확인 완료: {review_plan_path}")


# ---------------------------------------------------------------------------
# manifest 검증
# ---------------------------------------------------------------------------

def validate_manifest(df: pd.DataFrame, strict_row_count: bool = True) -> list:
    """manifest 검증. 오류 목록 반환 (치명적이면 즉시 종료)."""
    errors = []
    warnings = []

    # 필수 컬럼
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"[ERROR] 필수 컬럼 누락: {missing_cols}", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] 필수 컬럼 확인 완료 ({len(REQUIRED_COLUMNS)}개)")

    # row 수
    if len(df) != EXPECTED_ROW_COUNT:
        msg = f"[WARNING] row 수 불일치. 기대={EXPECTED_ROW_COUNT}, 실제={len(df)}"
        warnings.append(msg)
        print(msg)
    else:
        print(f"[INFO] row 수 확인: {len(df)}행 (정상)")

    # 환자 수
    n_patients = df["patient_id"].nunique()
    if n_patients != EXPECTED_PATIENT_COUNT:
        msg = f"[WARNING] patient 수 불일치. 기대={EXPECTED_PATIENT_COUNT}, 실제={n_patients}"
        warnings.append(msg)
        print(msg)
    else:
        print(f"[INFO] patient 수 확인: {n_patients}명 (정상)")

    # crop_id 중복
    dup_count = df["crop_id"].duplicated().sum()
    if dup_count > 0:
        print(f"[ERROR] crop_id 중복 {dup_count}건 발견", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] crop_id 중복 확인: 0건 (정상)")

    # crop_path 존재 및 .npz 확인
    missing_paths = []
    non_npz_paths = []
    for path_str in df["crop_path"].dropna().astype(str):
        if not path_str.endswith(".npz"):
            non_npz_paths.append(path_str)
        elif not os.path.isfile(path_str):
            missing_paths.append(path_str)

    if non_npz_paths:
        print(f"[ERROR] crop_path 중 .npz 아닌 경로 {len(non_npz_paths)}건", file=sys.stderr)
        for p in non_npz_paths[:3]:
            print(f"  {p}", file=sys.stderr)
        sys.exit(1)

    if missing_paths:
        print(f"[ERROR] crop_path 파일 없음 {len(missing_paths)}건", file=sys.stderr)
        for p in missing_paths[:3]:
            print(f"  {p}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] crop_path 전체 존재 및 .npz 확인 완료 ({len(df)}건)")

    return warnings


# ---------------------------------------------------------------------------
# npz 검증
# ---------------------------------------------------------------------------

def validate_npz_samples(df: pd.DataFrame, image_key: str, n_sample: int = 3):
    """sample npz n_sample개에 대해 key/shape/dtype/range/NaN 검증."""
    sample_paths = df["crop_path"].dropna().astype(str).head(n_sample).tolist()
    print(f"[INFO] npz sample 검증 시작 ({len(sample_paths)}개)")

    for crop_path in sample_paths:
        try:
            data = np.load(crop_path)
        except Exception as e:
            print(f"[ERROR] npz 로드 실패: {crop_path} — {e}", file=sys.stderr)
            sys.exit(1)

        if image_key not in data:
            print(f"[ERROR] image key '{image_key}' 없음: {crop_path}", file=sys.stderr)
            print(f"        사용 가능한 key: {list(data.keys())}", file=sys.stderr)
            sys.exit(1)

        img = data[image_key]

        if img.shape != EXPECTED_SHAPE:
            print(f"[ERROR] image shape 불일치. 기대={EXPECTED_SHAPE}, 실제={img.shape}: {crop_path}", file=sys.stderr)
            sys.exit(1)

        if not np.issubdtype(img.dtype, np.floating):
            print(f"[WARNING] dtype이 float 계열 아님: dtype={img.dtype}, path={crop_path}")

        img_min = float(img.min())
        img_max = float(img.max())
        if img_min < -0.01 or img_max > 1.01:
            print(f"[WARNING] image 값 범위 이상: min={img_min:.4f}, max={img_max:.4f}, path={crop_path}")

        if np.any(np.isnan(img)) or np.any(np.isinf(img)):
            print(f"[ERROR] image에 NaN/Inf 포함: {crop_path}", file=sys.stderr)
            sys.exit(1)

        print(f"[INFO]   OK — shape={img.shape}, dtype={img.dtype}, min={img_min:.4f}, max={img_max:.4f}: {Path(crop_path).name}")

    print(f"[INFO] npz sample 검증 완료")


# ---------------------------------------------------------------------------
# 이미지 변환 유틸
# ---------------------------------------------------------------------------

def channel_to_uint8(channel: np.ndarray) -> np.ndarray:
    """[0,1] float 채널을 uint8(0~255)로 변환. 추가 windowing/contrast 보정 없음."""
    clipped = np.clip(channel, 0.0, 1.0)
    return (clipped * 255.0).astype(np.uint8)


def make_safe_filename(patient_id: str, crop_id, score: float, priority: str) -> str:
    """파일명에 사용할 안전 문자열 생성. slash/backslash/colon 등 위험 문자 제거."""
    raw = f"{priority}__{patient_id}__crop{crop_id}__l1mean{score:.4f}"
    safe = re.sub(r"[\\/:*?\"<>|;]", "_", raw)
    return safe


# ---------------------------------------------------------------------------
# annotation text 생성
# ---------------------------------------------------------------------------

def _bool_str(val) -> str:
    """True/False 또는 문자열 값을 'T'/'F'로 요약."""
    if pd.isna(val):
        return "?"
    if isinstance(val, bool):
        return "T" if val else "F"
    s = str(val).strip().lower()
    if s in ("true", "1", "yes"):
        return "T"
    if s in ("false", "0", "no"):
        return "F"
    return s[:3]


def _trunc(s: str, max_len: int = 40) -> str:
    s = str(s) if not pd.isna(s) else ""
    return s if len(s) <= max_len else s[:max_len - 2] + ".."


def draw_annotation(canvas: np.ndarray, row: pd.Series, img_width: int):
    """canvas의 상단/하단 margin 영역에 annotation text를 그린다."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.35
    thickness = 1
    color = (30, 30, 30)      # 어두운 색
    line_h = 14

    patient_id = str(row.get("patient_id", ""))
    crop_id = str(row.get("crop_id", ""))
    l1_mean = row.get("crop_score_l1_mean", float("nan"))
    l1_max = row.get("crop_score_l1_max", float("nan"))
    priority = str(row.get("qa_priority", ""))
    group = _trunc(str(row.get("qa_group", "")), 45)
    p90 = _bool_str(row.get("threshold_exceed_val_p90", ""))
    p95 = _bool_str(row.get("threshold_exceed_val_p95", ""))
    p99 = _bool_str(row.get("threshold_exceed_val_p99", ""))
    padim_mean = row.get("padim_score_mean", float("nan"))
    padim_max = row.get("padim_score_max", float("nan"))
    large_bbox = _bool_str(row.get("large_bbox_flag", ""))
    reason = _trunc(str(row.get("suggested_review_reason", "")), 45)

    l1_mean_s = f"{l1_mean:.4f}" if not pd.isna(l1_mean) else "?"
    l1_max_s = f"{l1_max:.4f}" if not pd.isna(l1_max) else "?"
    padim_mean_s = f"{padim_mean:.2f}" if not pd.isna(padim_mean) else "?"
    padim_max_s = f"{padim_max:.2f}" if not pd.isna(padim_max) else "?"

    # 상단 margin (y=10, 24, 38, 52)
    top_lines = [
        f"pt={patient_id}  crop={crop_id}  {priority}",
        f"l1_mean={l1_mean_s}  l1_max={l1_max_s}  p90={p90} p95={p95} p99={p99}",
        f"group={group}",
        f"large_bbox={large_bbox}  padim={padim_mean_s}/{padim_max_s}",
    ]
    for i, line in enumerate(top_lines):
        y = 10 + i * line_h
        cv2.putText(canvas, line, (4, y), font, font_scale, color, thickness, cv2.LINE_AA)

    # 하단 margin (img_height - MARGIN_BOTTOM + offset)
    img_height = canvas.shape[0]
    bottom_start = img_height - MARGIN_BOTTOM + 10
    bottom_lines = [
        f"reason: {reason}",
        "[manual_label: ___________]",
    ]
    for i, line in enumerate(bottom_lines):
        y = bottom_start + i * line_h
        cv2.putText(canvas, line, (4, y), font, font_scale, color, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# 개별 PNG 생성
# ---------------------------------------------------------------------------

def make_individual_png(
    row: pd.Series,
    image_key: str,
    output_dir: Path,
    dry_run: bool = False,
) -> tuple:
    """
    단일 crop의 개별 PNG를 생성한다.
    반환: (png_path_str, error_str)
    error_str이 빈 문자열이면 성공.
    """
    crop_path = str(row.get("crop_path", ""))
    patient_id = str(row.get("patient_id", ""))
    crop_id = row.get("crop_id", "")
    l1_mean = row.get("crop_score_l1_mean", 0.0)
    priority = str(row.get("qa_priority", "unknown"))

    safe_fn = make_safe_filename(patient_id, crop_id, float(l1_mean) if not pd.isna(l1_mean) else 0.0, priority)
    priority_dir = output_dir / "individual" / re.sub(r"[\\/:*?\"<>|;]", "_", priority)
    png_path = priority_dir / f"{safe_fn}.png"

    if dry_run:
        return str(png_path), ""

    try:
        data = np.load(crop_path)
        img = data[image_key]

        # shape 검증
        if img.shape != EXPECTED_SHAPE:
            return str(png_path), f"shape 불일치: {img.shape}"

        n_channels, h, w = img.shape  # (6, 96, 96)
        total_h = MARGIN_TOP + 2 * h + MARGIN_BOTTOM
        total_w = 3 * w
        canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 240  # 밝은 회색 배경

        # 1행: lung z-1(ch0), z(ch1), z+1(ch2)
        # 2행: mediastinal z-1(ch3), z(ch4), z+1(ch5)
        for col_idx, ch_idx in enumerate([0, 1, 2]):
            ch_img = channel_to_uint8(img[ch_idx])
            ch_bgr = cv2.cvtColor(ch_img, cv2.COLOR_GRAY2BGR)
            y0 = MARGIN_TOP
            x0 = col_idx * w
            canvas[y0:y0 + h, x0:x0 + w] = ch_bgr

        for col_idx, ch_idx in enumerate([3, 4, 5]):
            ch_img = channel_to_uint8(img[ch_idx])
            ch_bgr = cv2.cvtColor(ch_img, cv2.COLOR_GRAY2BGR)
            y0 = MARGIN_TOP + h
            x0 = col_idx * w
            canvas[y0:y0 + h, x0:x0 + w] = ch_bgr

        # 행 구분선
        cv2.line(canvas, (0, MARGIN_TOP + h), (total_w, MARGIN_TOP + h), (180, 180, 180), 1)
        # 열 구분선
        for col_idx in [1, 2]:
            x = col_idx * w
            cv2.line(canvas, (x, MARGIN_TOP), (x, MARGIN_TOP + 2 * h), (180, 180, 180), 1)

        # annotation text
        draw_annotation(canvas, row, total_w)

        priority_dir.mkdir(parents=True, exist_ok=True)
        success = cv2.imwrite(str(png_path), canvas)
        if not success:
            return str(png_path), f"cv2.imwrite 실패: {png_path}"

    except Exception as e:
        return str(png_path), str(e)

    return str(png_path), ""


# ---------------------------------------------------------------------------
# contact sheet 생성
# ---------------------------------------------------------------------------

def make_contact_sheet(
    rows: list,
    image_key: str,
    output_path: Path,
    max_cols: int = 5,
    max_rows: int = 5,
    dry_run: bool = False,
) -> tuple:
    """
    rows 목록에서 contact sheet을 생성한다.
    각 cell: lung ch1 + medi ch4 나란히 (192×96).
    반환: (output_path_str, error_str)
    """
    if dry_run:
        return str(output_path), ""

    max_per_sheet = max_cols * max_rows
    rows_chunk = rows[:max_per_sheet]

    cell_h = CONTACT_CELL_H
    cell_w = CONTACT_CELL_W
    gap = CONTACT_GAP

    n_cells = len(rows_chunk)
    n_cols = min(n_cells, max_cols)
    n_rows = math.ceil(n_cells / n_cols) if n_cols > 0 else 1

    sheet_h = n_rows * (cell_h + gap) + gap
    sheet_w = n_cols * (cell_w + gap) + gap
    sheet = np.ones((sheet_h, sheet_w, 3), dtype=np.uint8) * 200

    for cell_idx, row in enumerate(rows_chunk):
        row_i = cell_idx // n_cols
        col_i = cell_idx % n_cols
        y0 = gap + row_i * (cell_h + gap)
        x0 = gap + col_i * (cell_w + gap)

        try:
            crop_path = str(row.get("crop_path", ""))
            data = np.load(crop_path)
            img = data[image_key]

            if img.shape != EXPECTED_SHAPE:
                cell_img = np.ones((cell_h, cell_w, 3), dtype=np.uint8) * 128
            else:
                lung_center = channel_to_uint8(img[1])
                medi_center = channel_to_uint8(img[4])
                lung_bgr = cv2.cvtColor(lung_center, cv2.COLOR_GRAY2BGR)
                medi_bgr = cv2.cvtColor(medi_center, cv2.COLOR_GRAY2BGR)
                cell_img = np.concatenate([lung_bgr, medi_bgr], axis=1)

        except Exception:
            cell_img = np.ones((cell_h, cell_w, 3), dtype=np.uint8) * 100

        sheet[y0:y0 + cell_h, x0:x0 + cell_w] = cell_img

        # cell 내 간단 텍스트
        label = f"{str(row.get('crop_id',''))}"
        cv2.putText(sheet, label, (x0 + 2, y0 + cell_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (30, 30, 200), 1, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(output_path), sheet)
    if not success:
        return str(output_path), f"cv2.imwrite 실패: {output_path}"

    return str(output_path), ""


# ---------------------------------------------------------------------------
# PNG index CSV 구성
# ---------------------------------------------------------------------------

def build_png_index_df(
    df_manifest: pd.DataFrame,
    individual_results: dict,
    contact_sheet_map: dict,
) -> pd.DataFrame:
    """
    manifest 행별 PNG 경로 및 생성 결과를 담은 index DataFrame 생성.
    individual_results: {crop_id → (png_path, error)}
    contact_sheet_map: {crop_id → contact_sheet_path}
    """
    records = []
    for _, row in df_manifest.iterrows():
        crop_id = row.get("crop_id", "")
        result = individual_results.get(crop_id, ("", "not_generated"))
        png_path, gen_error = result
        contact_path = contact_sheet_map.get(crop_id, "")

        records.append({
            "patient_id": row.get("patient_id", ""),
            "crop_id": crop_id,
            "crop_path": row.get("crop_path", ""),
            "png_path": png_path,
            "contact_sheet_path": contact_path,
            "qa_group": row.get("qa_group", ""),
            "qa_priority": row.get("qa_priority", ""),
            "crop_score_l1_mean": row.get("crop_score_l1_mean", ""),
            "threshold_exceed_val_p90": row.get("threshold_exceed_val_p90", ""),
            "threshold_exceed_val_p95": row.get("threshold_exceed_val_p95", ""),
            "threshold_exceed_val_p99": row.get("threshold_exceed_val_p99", ""),
            "manual_label_placeholder": row.get("manual_label_placeholder", ""),
            "png_generated": gen_error == "",
            "generation_error": gen_error,
        })

    return pd.DataFrame(records, columns=PNG_INDEX_COLUMNS)


# ---------------------------------------------------------------------------
# summary JSON 구성
# ---------------------------------------------------------------------------

def build_summary(
    args,
    df_manifest: pd.DataFrame,
    n_individual: int,
    n_contact_sheets: int,
    n_error: int,
    n_by_priority: dict,
    n_by_group: dict,
) -> dict:
    return {
        "input_qa_manifest": str(args.qa_manifest),
        "input_qa_summary": str(args.qa_summary),
        "input_review_plan": str(args.review_plan),
        "output_root": str(args.output_root),
        "n_input_rows": len(df_manifest),
        "n_patients": int(df_manifest["patient_id"].nunique()) if "patient_id" in df_manifest.columns else -1,
        "n_individual_png": n_individual,
        "n_contact_sheets": n_contact_sheets,
        "n_generation_error": n_error,
        "n_by_priority": n_by_priority,
        "n_by_group": n_by_group,
        "image_key": args.image_key,
        "layout": "2row_3col_lung_medi",
        "note": {
            "crop_npz_only": "crop npz only — image key 사용",
            "original_volume_unused": "원본 volume 미사용",
            "v2_unused": "v2 volume source 미사용",
            "stage2_holdout_unused": "stage2_holdout 미사용",
            "threshold_not_finalized": "threshold 확정 아님",
            "lesion_conclusion_forbidden": "병변 성능 결론 금지",
            "no_crop_copy": "crop 복사 없음",
        },
        "timestamp": datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# runtime summary append
# ---------------------------------------------------------------------------

def append_runtime_summary(args, n_individual: int, output_root: str):
    runtime_path = (
        "outputs/second-stage-lesion-refiner-v1/evaluation/"
        "rd4ad_2p5d_normal_mw_fixed96_v1/runtime_summary_v1.json"
    )
    summary_path = Path(runtime_path)
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    else:
        data = {}

    data[RUNTIME_SUMMARY_KEY] = {
        "timestamp": datetime.now().isoformat(),
        "n_individual_png": n_individual,
        "output_root": output_root,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[INFO] runtime summary append 완료: {runtime_path}")


# ---------------------------------------------------------------------------
# preflight-only 모드
# ---------------------------------------------------------------------------

def run_preflight(args, df_manifest: pd.DataFrame, summary: dict):
    """입력 검증만 수행. 파일 생성 없음."""
    print("[INFO] === preflight-only 모드 ===")

    check_crop_paths_blocked(df_manifest)
    validate_manifest(df_manifest)
    validate_npz_samples(df_manifest, args.image_key, n_sample=3)

    output_root = Path(args.output_root)
    if output_root.exists():
        print(f"[WARNING] output_root 이미 존재: {output_root} (경고만, 중단 아님)")
    else:
        print(f"[INFO] output_root 없음 (정상, 생성 전)")

    print("[INFO] preflight-only 완료. PNG/CSV/JSON 생성 없음.")


# ---------------------------------------------------------------------------
# dry-run 모드
# ---------------------------------------------------------------------------

def run_dry_run(args, df_manifest: pd.DataFrame):
    """PNG layout 생성 가능성 검사. 실제 파일 저장 없음."""
    print("[INFO] === dry-run 모드 ===")

    limit = args.limit
    df_work = df_manifest.head(limit) if limit is not None else df_manifest
    print(f"[INFO] 처리 대상: {len(df_work)}행")

    # 예정 수 계산
    n_individual = len(df_work) if args.make_individual else 0
    max_per_sheet = args.max_contact_sheet_cols * args.max_contact_sheet_rows
    n_contact = 0
    if args.make_contact_sheets:
        # priority별
        for priority in df_work["qa_priority"].unique():
            n = len(df_work[df_work["qa_priority"] == priority])
            n_contact += math.ceil(n / max_per_sheet)
        # group별 (첫 번째 그룹만 계산 예시)
        group_counts: dict = {}
        for groups_str in df_work["qa_group"].dropna().astype(str):
            for g in groups_str.split(";"):
                g = g.strip()
                if g:
                    group_counts[g] = group_counts.get(g, 0) + 1
        for g, cnt in group_counts.items():
            n_contact += math.ceil(cnt / max_per_sheet)

    print(f"[DRY-RUN] 개별 PNG 생성 예정: {n_individual}개")
    print(f"[DRY-RUN] contact sheet 생성 예정: {n_contact}장")

    # 상위 5개 PNG 파일명 예시
    print("[DRY-RUN] 상위 5개 PNG 파일명 예시:")
    for _, row in df_work.head(5).iterrows():
        patient_id = str(row.get("patient_id", ""))
        crop_id = row.get("crop_id", "")
        l1_mean = row.get("crop_score_l1_mean", 0.0)
        priority = str(row.get("qa_priority", "unknown"))
        safe_fn = make_safe_filename(
            patient_id, crop_id,
            float(l1_mean) if not pd.isna(l1_mean) else 0.0,
            priority,
        )
        print(f"  individual/{priority}/{safe_fn}.png")

    print("[DRY-RUN] individual layout: 2row_3col_lung_medi")
    print(f"[DRY-RUN] individual canvas size: {3 * 96}w x {MARGIN_TOP + 2 * 96 + MARGIN_BOTTOM}h (px)")
    print("[DRY-RUN] contact sheet cell: lung ch1 + medi ch4 = 192x96 (2ch preview)")
    print("[DRY-RUN] 완료. CSV/JSON 저장 없음.")


# ---------------------------------------------------------------------------
# output conflict guard
# ---------------------------------------------------------------------------

def check_output_conflict(
    output_root: Path,
    make_individual: bool,
    make_contact_sheets: bool,
    force: bool,
) -> None:
    """full run 시작 전 output 충돌 검사. 충돌 시 sys.exit(1)."""
    if force:
        return

    if not output_root.exists():
        return

    children = list(output_root.iterdir())
    if not children:
        return

    conflict_paths = []
    for target in [
        output_root / "individual",
        output_root / "contact_sheets",
        output_root / "manifest_png_index_v1.csv",
        output_root / "png_generation_summary_v1.json",
    ]:
        if target.exists():
            conflict_paths.append(target)

    # 위 4개 외 다른 파일/폴더가 있을 때도 output_root 자체를 충돌로 처리
    if not conflict_paths and children:
        conflict_paths.append(output_root)

    if conflict_paths:
        print(
            "[ERROR] output conflict detected. Use --force only if overwrite is intended.",
            file=sys.stderr,
        )
        for p in conflict_paths:
            print(f"  existing: {p}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# full run 모드
# ---------------------------------------------------------------------------

def run_full(args, df_manifest: pd.DataFrame):
    """full run: PNG 생성 + index CSV + summary JSON 저장."""
    print("[INFO] === full run 모드 ===")

    output_root = Path(args.output_root)
    individual_dir = output_root / "individual"
    contact_dir = output_root / "contact_sheets"

    # output_root 충돌 확인
    check_output_conflict(output_root, args.make_individual, args.make_contact_sheets, args.force)

    if not args.make_individual and not args.make_contact_sheets:
        print(
            "[ERROR] --make-individual 또는 --make-contact-sheets 중 하나 이상 지정 필요",
            file=sys.stderr,
        )
        sys.exit(1)

    limit = args.limit
    df_work = df_manifest.head(limit) if limit is not None else df_manifest

    individual_results: dict = {}
    contact_sheet_map: dict = {}
    n_individual = 0
    n_contact_sheets = 0
    n_error = 0

    # --- 개별 PNG 생성 ---
    if args.make_individual:
        print(f"[INFO] 개별 PNG 생성 시작 ({len(df_work)}개)")
        for _, row in df_work.iterrows():
            png_path, error = make_individual_png(
                row=row,
                image_key=args.image_key,
                output_dir=output_root,
                dry_run=False,
            )
            crop_id = row.get("crop_id", "")
            individual_results[crop_id] = (png_path, error)
            if error:
                n_error += 1
                print(f"[WARNING] 개별 PNG 생성 오류 crop_id={crop_id}: {error}")
            else:
                n_individual += 1
        print(f"[INFO] 개별 PNG 생성 완료 — 성공={n_individual}, 오류={n_error}")

    # --- contact sheet 생성 ---
    if args.make_contact_sheets:
        max_per_sheet = args.max_contact_sheet_cols * args.max_contact_sheet_rows

        def _gen_contact_sheets(rows_list: list, prefix: str):
            nonlocal n_contact_sheets
            n_sheets = math.ceil(len(rows_list) / max_per_sheet) if rows_list else 0
            for sheet_idx in range(n_sheets):
                chunk = rows_list[sheet_idx * max_per_sheet:(sheet_idx + 1) * max_per_sheet]
                sheet_num = f"{sheet_idx + 1:03d}"
                sheet_name = f"{prefix}_{sheet_num}.png"
                out_path = contact_dir / sheet_name
                _, err = make_contact_sheet(
                    rows=chunk,
                    image_key=args.image_key,
                    output_path=out_path,
                    max_cols=args.max_contact_sheet_cols,
                    max_rows=args.max_contact_sheet_rows,
                    dry_run=False,
                )
                if err:
                    print(f"[WARNING] contact sheet 생성 오류 {sheet_name}: {err}")
                else:
                    n_contact_sheets += 1
                    print(f"[INFO] contact sheet 저장: {out_path}")
                # contact_sheet_map에 이 chunk의 crop_id 등록
                for row in chunk:
                    crop_id = row.get("crop_id", "")
                    if contact_sheet_map.get(crop_id, "") == "":
                        contact_sheet_map[crop_id] = str(out_path)

        print("[INFO] contact sheet 생성 시작")

        # priority별
        for priority in ["keep_high_priority", "review_group", "low_priority"]:
            rows_p = [
                row for _, row in df_work.iterrows()
                if str(row.get("qa_priority", "")) == priority
            ]
            if rows_p:
                safe_priority = re.sub(r"[\\/:*?\"<>|;]", "_", priority)
                _gen_contact_sheets(rows_p, f"contact_sheet_priority_{safe_priority}")

        # group별
        group_rows_map: dict = {}
        for _, row in df_work.iterrows():
            groups_str = str(row.get("qa_group", ""))
            for g in groups_str.split(";"):
                g = g.strip()
                if g:
                    group_rows_map.setdefault(g, []).append(row)

        for group_name, rows_g in sorted(group_rows_map.items()):
            safe_group = re.sub(r"[\\/:*?\"<>|;]", "_", group_name)
            _gen_contact_sheets(rows_g, f"contact_sheet_group_{safe_group}")

        print(f"[INFO] contact sheet 생성 완료 — {n_contact_sheets}장")

    # --- n_by_priority / n_by_group ---
    n_by_priority: dict = {}
    n_by_group: dict = {}
    for _, row in df_work.iterrows():
        p = str(row.get("qa_priority", ""))
        n_by_priority[p] = n_by_priority.get(p, 0) + 1
        for g in str(row.get("qa_group", "")).split(";"):
            g = g.strip()
            if g:
                n_by_group[g] = n_by_group.get(g, 0) + 1

    # --- PNG index CSV 저장 ---
    output_root.mkdir(parents=True, exist_ok=True)
    index_df = build_png_index_df(df_work, individual_results, contact_sheet_map)
    index_path = output_root / "manifest_png_index_v1.csv"
    index_df.to_csv(index_path, index=False, encoding="utf-8")
    print(f"[INFO] PNG index CSV 저장: {index_path}")

    # --- summary JSON 저장 ---
    summary_dict = build_summary(
        args=args,
        df_manifest=df_work,
        n_individual=n_individual,
        n_contact_sheets=n_contact_sheets,
        n_error=n_error,
        n_by_priority=n_by_priority,
        n_by_group=n_by_group,
    )
    summary_path = output_root / "png_generation_summary_v1.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, ensure_ascii=False, indent=2)
    print(f"[INFO] summary JSON 저장: {summary_path}")

    # --- runtime summary ---
    if not args.no_runtime_append:
        append_runtime_summary(args, n_individual, str(output_root))

    print(f"[INFO] full run 완료 — 개별={n_individual}, contact={n_contact_sheets}, 오류={n_error}")


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 5.41: Hard Negative Top-Score QA PNG Pack 생성"
    )
    parser.add_argument(
        "--qa-manifest",
        default=(
            "outputs/second-stage-lesion-refiner-v1/evaluation/"
            "rd4ad_2p5d_normal_mw_fixed96_v1/hard_negative_top_score_qa_v1/"
            "hard_negative_top_score_qa_manifest_v1.csv"
        ),
        help="입력 QA manifest CSV 경로",
    )
    parser.add_argument(
        "--qa-summary",
        default=(
            "outputs/second-stage-lesion-refiner-v1/evaluation/"
            "rd4ad_2p5d_normal_mw_fixed96_v1/hard_negative_top_score_qa_v1/"
            "hard_negative_top_score_qa_summary_v1.json"
        ),
        help="입력 QA summary JSON 경로",
    )
    parser.add_argument(
        "--review-plan",
        default=(
            "outputs/second-stage-lesion-refiner-v1/evaluation/"
            "rd4ad_2p5d_normal_mw_fixed96_v1/hard_negative_top_score_qa_v1/"
            "hard_negative_top_score_qa_review_plan_v1.md"
        ),
        help="입력 review plan MD 경로",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="출력 루트 디렉토리",
    )
    parser.add_argument(
        "--image-key",
        default=EXPECTED_IMAGE_KEY,
        help="crop npz에서 읽을 key (기본: image)",
    )
    parser.add_argument("--make-individual", action="store_true", help="개별 PNG 생성")
    parser.add_argument("--make-contact-sheets", action="store_true", help="contact sheet 생성")
    parser.add_argument("--max-contact-sheet-cols", type=int, default=5, help="contact sheet 열 수")
    parser.add_argument("--max-contact-sheet-rows", type=int, default=5, help="contact sheet 행 수")
    parser.add_argument("--limit", type=int, default=None, help="처리 행 수 제한 (기본: 전체)")
    parser.add_argument("--preflight-only", action="store_true", help="입력 검증만 수행 (파일 생성 없음)")
    parser.add_argument("--dry-run", action="store_true", help="실제 저장 없이 흐름만 테스트")
    parser.add_argument("--force", action="store_true", help="output_root 충돌 시 강제 덮어쓰기")
    parser.add_argument("--no-runtime-append", action="store_true", help="runtime summary append 비활성화")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # output_root 금지 경로 확인
    _assert_no_blocked_path(str(args.output_root), "output-root")

    # 입력 파일 로드
    df_manifest = load_manifest(str(args.qa_manifest))
    summary = load_summary(str(args.qa_summary))
    check_review_plan_exists(str(args.review_plan))

    # crop_path 금지 경로 사전 확인
    check_crop_paths_blocked(df_manifest)

    if args.preflight_only:
        run_preflight(args, df_manifest, summary)
        return

    # preflight가 아닌 경우 manifest 검증
    validate_manifest(df_manifest)
    validate_npz_samples(df_manifest, args.image_key, n_sample=3)

    if args.dry_run:
        run_dry_run(args, df_manifest)
        return

    run_full(args, df_manifest)


if __name__ == "__main__":
    main()
