#!/usr/bin/env python3
"""
B0 CT-context panel 생성 스크립트

dev_safe_mixed_error_visual_qa 대상 33건에 대해 z-context 5 slice 패널 PNG를 생성한다.

주의:
- stage2_holdout 미사용
- score CSV 로드 없음
- suppression/score adjustment 없음
- --overwrite 옵션 없음
- --run --confirm-run 없이는 실제 PNG 생성 불가
- dry-run에서 np.load / slice indexing / matplotlib figure / output dir 생성 금지
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 경로 상수
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 지시서 내 추가 metadata join 대상 (b0_visual_label / candidate_score / planned_z_context)
# 실제 run 시 review_id로 join 시도. 없으면 N/A 처리.
B0_PANEL_TARGETS_BLOCKED_CSV = (
    PROJECT_ROOT
    / "qa/dev_safe_mixed_error_visual_qa"
    / "b0_ct_context_panel_targets_blocked.csv"
)

# ---------------------------------------------------------------------------
# 금지 키워드
# ---------------------------------------------------------------------------

BLOCKED_PATH_KEYWORDS = ["full_retrospective", "resnet50"]

# ---------------------------------------------------------------------------
# HU windowing (lung window: -1000 ~ 400, 지시서 지정값)
# ---------------------------------------------------------------------------

LUNG_HU_MIN = -1000
LUNG_HU_MAX = 400

# ---------------------------------------------------------------------------
# 필수 카운트 (안전장치)
# ---------------------------------------------------------------------------

REQUIRED_TOTAL_ROWS = 33
REQUIRED_LESION_PROTECT_ROWS = 24
REQUIRED_FP_CANDIDATE_ROWS = 9

# ---------------------------------------------------------------------------
# path guard
# ---------------------------------------------------------------------------


def _guard_keyword(path_str: str, label: str):
    """경로 문자열에 금지 키워드가 있으면 즉시 abort."""
    s = str(path_str).lower()
    for kw in BLOCKED_PATH_KEYWORDS:
        if kw.lower() in s:
            print(
                f"[GUARD] {label} 경로에 금지 키워드 '{kw}' 발견 — 즉시 중단: {path_str}",
                file=sys.stderr,
            )
            sys.exit(2)


def guard_all_paths(args_targets: str, args_output_dir: str, df: pd.DataFrame):
    """targets CSV, output-dir, CSV 내 path 컬럼 전체 금지 키워드 검사."""
    _guard_keyword(args_targets, "targets CSV")
    _guard_keyword(args_output_dir, "output-dir")

    path_cols = [c for c in df.columns if "path" in c.lower()]
    for col in path_cols:
        for val in df[col].dropna().astype(str):
            if val in ("NOT_REQUIRED_FOR_NORMAL",):
                continue
            _guard_keyword(val, f"CSV 컬럼 {col}")

    print("[GUARD] 경로 금지 키워드 검사 통과")


# ---------------------------------------------------------------------------
# 안전 카운트 검증
# ---------------------------------------------------------------------------


def guard_target_counts(df: pd.DataFrame):
    """target row 수 / lesion_protect / fp_candidate 수 불일치 시 abort."""
    total = len(df)
    if total != REQUIRED_TOTAL_ROWS:
        print(
            f"[GUARD] target row count={total}, expected={REQUIRED_TOTAL_ROWS} — 즉시 중단",
            file=sys.stderr,
        )
        sys.exit(2)

    lesion_count = (df["safety_role"] == "lesion_protect").sum()
    if lesion_count != REQUIRED_LESION_PROTECT_ROWS:
        print(
            f"[GUARD] lesion_protect count={lesion_count}, expected={REQUIRED_LESION_PROTECT_ROWS} — 즉시 중단",
            file=sys.stderr,
        )
        sys.exit(2)

    fp_count = (df["safety_role"] == "fp_candidate").sum()
    if fp_count != REQUIRED_FP_CANDIDATE_ROWS:
        print(
            f"[GUARD] fp_candidate count={fp_count}, expected={REQUIRED_FP_CANDIDATE_ROWS} — 즉시 중단",
            file=sys.stderr,
        )
        sys.exit(2)

    print(
        f"[GUARD] target count OK: total={total}, lesion_protect={lesion_count}, fp_candidate={fp_count}"
    )


def guard_stage2_holdout(df: pd.DataFrame):
    """stage2_holdout_flag==1 row 있으면 abort."""
    bad = df[df["stage2_holdout_flag"] != 0]
    if len(bad) > 0:
        ids = bad["review_id"].tolist()[:5]
        print(
            f"[GUARD] stage2_holdout_flag != 0 발견 ({len(bad)}행): {ids} — 즉시 중단",
            file=sys.stderr,
        )
        sys.exit(2)
    print("[GUARD] stage2_holdout_flag 전부 0 확인")


# ---------------------------------------------------------------------------
# HU windowing / contour (기존 스크립트 관례 재사용)
# ---------------------------------------------------------------------------


def apply_window(slice_hu: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    """HU 값을 [0, 1] float으로 클리핑 후 정규화."""
    clipped = np.clip(slice_hu.astype(np.float32), hu_min, hu_max)
    normed = (clipped - hu_min) / (hu_max - hu_min + 1e-8)
    return normed


def draw_contour_overlay(ax, mask_2d: np.ndarray, color: str, linewidth: float = 0.8):
    """mask_2d 2D의 contour를 ax에 그린다."""
    if mask_2d.max() == 0:
        return
    ax.contour(mask_2d, levels=[0.5], colors=[color], linewidths=[linewidth])


# ---------------------------------------------------------------------------
# output 파일명 생성
# ---------------------------------------------------------------------------


def make_output_filename(row: pd.Series) -> str:
    """B0CTX_{ct_context_id}_{review_id}_{group}.png"""
    ct_ctx_id = row["ct_context_id"]
    review_id = row["review_id"]
    group = str(row["group"])
    return f"B0CTX_{ct_ctx_id}_{review_id}_{group}.png"


# ---------------------------------------------------------------------------
# dry-run
# ---------------------------------------------------------------------------


def run_dry_run(df: pd.DataFrame, output_dir: Path) -> dict:
    """
    경로/shape/status/output collision/대상수만 확인.
    np.load / slice / matplotlib figure / output dir 생성 전부 금지.
    """
    print("\n" + "=" * 60)
    print("DRY-RUN")
    print("=" * 60)

    result = {
        "target_row_count": len(df),
        "group_counts": {},
        "stage2_holdout_all_zero": True,
        "shape_check_status_summary": {},
        "z_range_ok_summary": {},
        "xy_range_ok_summary": {},
        "path_status": {
            "ct_exists_ok": 0,
            "ct_missing": [],
            "roi_exists_ok": 0,
            "roi_missing": [],
            "lesion_exists_ok": 0,
            "lesion_missing": [],
            "lesion_not_required": 0,
        },
        "patch_extent_source_summary": {
            "rows_using_y1_x1_extent": 0,
            "rows_using_patch_size_column": 0,
            "rows_using_center_only_marker_extent_unknown": len(df),
            "rows_using_default_approximate_64": 0,
            "patch_size_source_status": "CENTER_ONLY_MARKER_EXTENT_UNKNOWN",
        },
        "metadata_join_result": {
            "blocked_csv_exists": False,
            "rows_matched": 0,
            "missing": [],
            "duplicates": [],
            "missing_count": 0,
            "duplicate_count": 0,
            "b0_visual_label_missing": 0,
            "candidate_score_missing": 0,
            "planned_z_context_missing": 0,
            "existing_png_path_missing": 0,
            "row_count_after_join": 0,
        },
        "output_filenames": [],
        "output_collision": [],
        "collision_count": 0,
        "output_dir_exists": output_dir.exists(),
        "output_dir_created_by_dry_run": False,  # dry-run에서 mkdir 절대 금지
        "expected_png_count": len(df),
        "blockers": [],
    }

    # group count
    result["group_counts"] = df["group"].value_counts().to_dict()

    # stage2_holdout 확인
    bad_holdout = df[df["stage2_holdout_flag"] != 0]
    if len(bad_holdout) > 0:
        result["stage2_holdout_all_zero"] = False
        result["blockers"].append(f"stage2_holdout_flag != 0: {len(bad_holdout)}행")

    # shape_check_status
    result["shape_check_status_summary"] = df["shape_check_status"].value_counts().to_dict()
    non_ok = df[df["shape_check_status"] != "SHAPE_OK"]
    if len(non_ok) > 0:
        result["blockers"].append(f"shape_check_status != SHAPE_OK: {len(non_ok)}행")

    # z_range_ok / xy_range_ok (CSV 값 기준)
    result["z_range_ok_summary"] = df["z_range_ok"].value_counts().to_dict()
    result["xy_range_ok_summary"] = df["xy_range_ok"].value_counts().to_dict()
    z_bad = df[df["z_range_ok"] != True]
    xy_bad = df[df["xy_range_ok"] != True]
    if len(z_bad) > 0:
        result["blockers"].append(f"z_range_ok != True: {len(z_bad)}행")
    if len(xy_bad) > 0:
        result["blockers"].append(f"xy_range_ok != True: {len(xy_bad)}행")

    # path 존재 확인 (Path.exists 만, np.load 금지)
    ps = result["path_status"]
    for _, row in df.iterrows():
        review_id = row["review_id"]
        role = row["safety_role"]

        ct_p = row["candidate_ct_path"]
        roi_p = row["candidate_roi_path"]
        lesion_p = row["candidate_lesion_mask_path"]

        # CT path
        if pd.isna(ct_p) or ct_p in ("", "NOT_REQUIRED_FOR_NORMAL"):
            ps["ct_missing"].append(f"{review_id}:MISSING_CT_PATH")
        elif Path(ct_p).exists():
            ps["ct_exists_ok"] += 1
        else:
            ps["ct_missing"].append(f"{review_id}:{ct_p}")

        # ROI path
        if pd.isna(roi_p) or roi_p in ("", "NOT_REQUIRED_FOR_NORMAL"):
            ps["roi_missing"].append(f"{review_id}:MISSING_ROI_PATH")
        elif Path(roi_p).exists():
            ps["roi_exists_ok"] += 1
        else:
            ps["roi_missing"].append(f"{review_id}:{roi_p}")

        # Lesion mask path (normal/fp_candidate는 NOT_REQUIRED_FOR_NORMAL)
        if lesion_p == "NOT_REQUIRED_FOR_NORMAL":
            ps["lesion_not_required"] += 1
        elif pd.isna(lesion_p) or lesion_p == "":
            if role == "lesion_protect":
                ps["lesion_missing"].append(f"{review_id}:MISSING_LESION_PATH")
        elif Path(lesion_p).exists():
            ps["lesion_exists_ok"] += 1
        else:
            ps["lesion_missing"].append(f"{review_id}:{lesion_p}")

    # lesion_protect 행의 lesion mask 누락 체크
    if ps["lesion_missing"]:
        result["blockers"].append(
            f"lesion mask 누락: {len(ps['lesion_missing'])}건 — {ps['lesion_missing'][:3]}"
        )

    # ct/roi 누락 체크
    if ps["ct_missing"]:
        result["blockers"].append(f"CT path 누락: {len(ps['ct_missing'])}건")
    if ps["roi_missing"]:
        result["blockers"].append(f"ROI path 누락: {len(ps['roi_missing'])}건")

    # output 파일명 생성 및 collision 확인
    # output_dir 존재하면 collision 확인, 없으면 collision=0 (mkdir 금지)
    for _, row in df.iterrows():
        fname = make_output_filename(row)
        result["output_filenames"].append(fname)
        if output_dir.exists():
            candidate_path = output_dir / fname
            if candidate_path.exists():
                result["output_collision"].append(str(candidate_path))

    result["collision_count"] = len(result["output_collision"])
    if result["collision_count"] > 0:
        result["blockers"].append(
            f"output collision {result['collision_count']}건: {result['output_collision'][:3]}"
        )

    # ---------------------------------------------------------------------------
    # blocked.csv CSV-level join 검증 (dry-run에서 CSV 읽기만 허용, np.load/slice/figure 금지)
    # ---------------------------------------------------------------------------
    mjr = result["metadata_join_result"]
    mjr["blocked_csv_exists"] = B0_PANEL_TARGETS_BLOCKED_CSV.is_file()

    if mjr["blocked_csv_exists"]:
        try:
            blocked_df = pd.read_csv(str(B0_PANEL_TARGETS_BLOCKED_CSV))
            target_ids = df["review_id"].tolist()

            # 중복 확인
            dup_ids = blocked_df[blocked_df["review_id"].isin(target_ids)]["review_id"]
            dup_ids = dup_ids[dup_ids.duplicated()].tolist()
            mjr["duplicates"] = dup_ids
            mjr["duplicate_count"] = len(dup_ids)

            # matched / missing 확인
            matched_ids = set(blocked_df["review_id"].tolist()) & set(target_ids)
            missing_ids = [rid for rid in target_ids if rid not in matched_ids]
            mjr["rows_matched"] = len(matched_ids)
            mjr["missing"] = missing_ids
            mjr["missing_count"] = len(missing_ids)

            # join 후 필수 컬럼 missing 확인 (CSV 수준, row count 유지 확인)
            joined = df[["review_id"]].merge(
                blocked_df[["review_id", "b0_visual_label", "candidate_score",
                             "planned_z_context", "existing_png_path"]],
                on="review_id",
                how="left",
            )
            mjr["row_count_after_join"] = len(joined)
            mjr["b0_visual_label_missing"] = int(joined["b0_visual_label"].isna().sum())
            mjr["candidate_score_missing"] = int(joined["candidate_score"].isna().sum())
            mjr["planned_z_context_missing"] = int(joined["planned_z_context"].isna().sum())
            mjr["existing_png_path_missing"] = int(joined["existing_png_path"].isna().sum())

            if mjr["row_count_after_join"] != len(df):
                result["blockers"].append(
                    f"metadata join 후 row count 불일치: {mjr['row_count_after_join']} (기대 {len(df)})"
                )
            if mjr["duplicate_count"] > 0:
                result["blockers"].append(
                    f"blocked.csv review_id 중복: {mjr['duplicate_count']}건 — {dup_ids[:3]}"
                )
            if mjr["missing_count"] > 0:
                result["blockers"].append(
                    f"blocked.csv review_id 누락: {mjr['missing_count']}건 — {missing_ids[:3]}"
                )

        except Exception as e:
            result["blockers"].append(f"blocked.csv join 검증 실패: {e}")
            mjr["rows_matched"] = -1
    else:
        # 파일 없으면 blocker는 아님 (실제 run 시 N/A 처리됨). 단 명시적으로 기록.
        mjr["rows_matched"] = 0
        mjr["missing"] = []
        mjr["row_count_after_join"] = 0

    # 결과 출력
    print(f"  target row count       : {result['target_row_count']}")
    print(f"  group counts           : {result['group_counts']}")
    print(f"  stage2_holdout_all_zero: {result['stage2_holdout_all_zero']}")
    print(f"  shape_check_status     : {result['shape_check_status_summary']}")
    print(f"  z_range_ok             : {result['z_range_ok_summary']}")
    print(f"  xy_range_ok            : {result['xy_range_ok_summary']}")
    print(f"  ct_exists_ok           : {ps['ct_exists_ok']}")
    print(f"  ct_missing             : {len(ps['ct_missing'])}")
    print(f"  roi_exists_ok          : {ps['roi_exists_ok']}")
    print(f"  roi_missing            : {len(ps['roi_missing'])}")
    print(f"  lesion_exists_ok       : {ps['lesion_exists_ok']}")
    print(f"  lesion_not_required    : {ps['lesion_not_required']}")
    print(f"  lesion_missing         : {len(ps['lesion_missing'])}")
    print(f"  expected PNG count     : {result['expected_png_count']}")
    print(f"  output_dir exists      : {result['output_dir_exists']}")
    print(f"  output_dir created     : {result['output_dir_created_by_dry_run']}  (dry-run에서 mkdir 금지)")
    print(f"  collision_count        : {result['collision_count']}")

    pes = result["patch_extent_source_summary"]
    print(f"\n  [patch extent source]")
    print(f"  rows using y1/x1 extent              : {pes['rows_using_y1_x1_extent']}")
    print(f"  rows using patch_size column         : {pes['rows_using_patch_size_column']}")
    print(f"  rows using center-only marker        : {pes['rows_using_center_only_marker_extent_unknown']}")
    print(f"  rows using default approximate 64    : {pes['rows_using_default_approximate_64']}")
    print(f"  patch_size_source_status             : {pes['patch_size_source_status']}")

    mjr = result["metadata_join_result"]
    print(f"\n  [metadata join result]")
    print(f"  blocked_csv_exists     : {mjr['blocked_csv_exists']}")
    print(f"  rows_matched           : {mjr['rows_matched']}")
    print(f"  missing_count          : {mjr['missing_count']}")
    print(f"  duplicate_count        : {mjr['duplicate_count']}")
    print(f"  row_count_after_join   : {mjr['row_count_after_join']}")
    print(f"  b0_visual_label_missing: {mjr['b0_visual_label_missing']}")
    print(f"  candidate_score_missing: {mjr['candidate_score_missing']}")
    print(f"  planned_z_context_miss : {mjr['planned_z_context_missing']}")
    print(f"  existing_png_path_miss : {mjr['existing_png_path_missing']}")
    print(f"\n  [deferred items]")
    print(f"  local zoom             : deferred (미구현, TODO 주석 처리)")
    print(f"  MIP                    : deferred (이번 단계 범위 외)")
    print(f"  vessel mask            : no vessel mask (이번 단계 범위 외)")

    if result["blockers"]:
        print(f"\n  [BLOCKERS] {result['blockers']}")
        print(f"\n  DRY-RUN RESULT: FAIL")
    else:
        print(f"\n  DRY-RUN RESULT: PASS")

    print("=" * 60)
    print("DRY-RUN 완료. PNG 생성 없음. output_dir 미생성.\n")
    return result


# ---------------------------------------------------------------------------
# 실제 run — z-context panel 생성 (구현만, 이번 단계에서 실행 금지)
# ---------------------------------------------------------------------------


def _load_extra_metadata(review_id: str) -> dict:
    """
    b0_ct_context_panel_targets_blocked.csv 에서 review_id로 join하여
    b0_visual_label / candidate_score / planned_z_context 를 가져온다.
    파일이 없거나 review_id가 없으면 N/A 반환.
    """
    extra = {
        "b0_visual_label": "N/A",
        "candidate_score": "N/A",
        "planned_z_context": "N/A",
    }
    if not B0_PANEL_TARGETS_BLOCKED_CSV.is_file():
        # 파일 없음 — N/A 유지
        return extra

    try:
        blocked_df = pd.read_csv(str(B0_PANEL_TARGETS_BLOCKED_CSV))
        matched = blocked_df[blocked_df["review_id"] == review_id]
        if len(matched) == 0:
            return extra
        row = matched.iloc[0]
        for key in extra:
            if key in row.index and not pd.isna(row[key]):
                extra[key] = str(row[key])
    except Exception as e:
        print(f"[WARNING] extra metadata join 실패 (review_id={review_id}): {e}")
    return extra


def _make_panel_figure(
    z_slices: list,
    z_indices: list,
    z_center: int,
    roi_vol: np.ndarray,
    lesion_vol,  # None for fp_candidate
    candidate_y0: int,
    candidate_x0: int,
    row_meta: dict,
) -> plt.Figure:
    """
    z-context 5 slice 패널 figure 생성 (1행 5열).

    z_slices: list of 2D CT HU array (len 1~5; 범위 벗어나면 빈 패널)
    z_indices: list of 실제 z 인덱스 (z_slices와 동일 길이)
    z_center: 후보 z
    roi_vol: 3D roi array (mmap)
    lesion_vol: 3D lesion mask array 또는 None
    candidate_y0, candidate_x0: 후보 patch 좌상단 좌표
    row_meta: review_id, patient_id, group, safety_role, b0_visual_label,
              candidate_score, planned_z_context, ct_context_id
    """
    n_panels = 5  # z-2, z-1, z, z+1, z+2
    fig, axes = plt.subplots(1, n_panels, figsize=(n_panels * 4, 4.5))
    if n_panels == 1:
        axes = [axes]

    z_offsets = [-2, -1, 0, 1, 2]

    for panel_idx, offset in enumerate(z_offsets):
        ax = axes[panel_idx]
        target_z = z_center + offset

        # z가 z_indices 범위 밖이면 빈 패널
        if target_z not in z_indices:
            ax.set_facecolor("black")
            ax.text(
                0.5, 0.5, f"z={target_z}\n(out of range)",
                ha="center", va="center", color="gray", fontsize=7,
                transform=ax.transAxes,
            )
            ax.axis("off")
            # 주석: z 범위 초과 시 clip 또는 빈 패널 중 빈 패널로 처리
            continue

        slice_idx = z_indices.index(target_z)
        ct_slice_hu = z_slices[slice_idx]
        ct_windowed = apply_window(ct_slice_hu, LUNG_HU_MIN, LUNG_HU_MAX)

        ax.imshow(ct_windowed, cmap="gray", vmin=0, vmax=1, aspect="equal")

        # ROI contour
        roi_2d = roi_vol[target_z].astype(np.uint8)
        draw_contour_overlay(ax, roi_2d, color="lime", linewidth=0.8)

        # Lesion mask contour (lesion_protect만)
        if lesion_vol is not None:
            lesion_2d = lesion_vol[target_z].astype(np.uint8)
            draw_contour_overlay(ax, lesion_2d, color="red", linewidth=0.8)

        # candidate marker — patch_extent 정보(y1/x1/patch_size) 없음.
        # center-only marker만 표시. default approximate 64 절대 사용 금지.
        marker_size = 10 if offset == 0 else 6
        marker_color = "yellow" if offset == 0 else "orange"
        ax.plot(
            candidate_x0, candidate_y0,
            marker="+",
            markersize=marker_size,
            color=marker_color,
            markeredgewidth=1.5 if offset == 0 else 0.8,
        )

        label = f"z={target_z}"
        if offset == 0:
            label += " [center]\npatch_extent=unknown; marker=center-only"
        ax.set_title(label, fontsize=6)
        ax.axis("off")

    # suptitle: 메타데이터
    review_id = row_meta.get("review_id", "")
    patient_id = row_meta.get("patient_id", "")
    group = row_meta.get("group", "")
    safety_role = row_meta.get("safety_role", "")
    b0_visual_label = row_meta.get("b0_visual_label", "N/A")
    candidate_score = row_meta.get("candidate_score", "N/A")
    planned_z_context = row_meta.get("planned_z_context", "N/A")
    ct_ctx_id = row_meta.get("ct_context_id", "")

    suptitle = (
        f"review_id={review_id} | ct_context_id={ct_ctx_id} | patient_id={patient_id}\n"
        f"group={group} | safety_role={safety_role}\n"
        f"b0_visual_label={b0_visual_label} | candidate_score={candidate_score} | "
        f"z_center={z_center} | planned_z_context={planned_z_context}"
    )
    fig.suptitle(suptitle, fontsize=7, y=1.02)
    fig.tight_layout()
    return fig


def run_actual_panels(df: pd.DataFrame, output_dir: Path):
    """
    실제 panel PNG 생성 로직.
    --run --confirm-run 으로만 진입 가능.

    안전 구조:
    - 생성 루프 전 shape preflight 전수 확인 → 한 row라도 blocker면 전체 abort
    - np.load(mmap_mode="r") 사용
    - 환자별 volume 1개씩 캐시 (교체 시 del + gc 명시)
    """
    import gc

    # actual run 전수 preflight — 하나라도 실패하면 output_dir mkdir 전에 전체 abort
    preflight_blockers = []
    for _, row in df.iterrows():
        rid = row["review_id"]
        if row["shape_check_status"] != "SHAPE_OK":
            preflight_blockers.append(f"{rid}: shape_check_status={row['shape_check_status']}")
        if row["stage2_holdout_flag"] != 0:
            preflight_blockers.append(f"{rid}: stage2_holdout_flag={row['stage2_holdout_flag']}")
        if row.get("z_range_ok") != True:
            preflight_blockers.append(f"{rid}: z_range_ok={row.get('z_range_ok')}")
        if row.get("xy_range_ok") != True:
            preflight_blockers.append(f"{rid}: xy_range_ok={row.get('xy_range_ok')}")
        # CT path exists
        ct_p_str = row.get("candidate_ct_path", "")
        if pd.isna(ct_p_str) or str(ct_p_str) in ("", "NOT_REQUIRED_FOR_NORMAL"):
            preflight_blockers.append(f"{rid}: candidate_ct_path missing")
        elif not Path(ct_p_str).exists():
            preflight_blockers.append(f"{rid}: CT not found: {ct_p_str}")
        # ROI path exists
        roi_p_str = row.get("candidate_roi_path", "")
        if pd.isna(roi_p_str) or str(roi_p_str) in ("", "NOT_REQUIRED_FOR_NORMAL"):
            preflight_blockers.append(f"{rid}: candidate_roi_path missing")
        elif not Path(roi_p_str).exists():
            preflight_blockers.append(f"{rid}: ROI not found: {roi_p_str}")
        # lesion mask path exists (lesion_protect만)
        if row.get("safety_role") == "lesion_protect":
            lp_str = row.get("candidate_lesion_mask_path", "")
            if pd.isna(lp_str) or str(lp_str) in ("", "NOT_REQUIRED_FOR_NORMAL"):
                preflight_blockers.append(f"{rid}: lesion_mask_path missing for lesion_protect")
            elif not Path(lp_str).exists():
                preflight_blockers.append(f"{rid}: lesion mask not found: {lp_str}")
        # output filename 생성 가능 여부
        try:
            _ = make_output_filename(row)
        except Exception as e_fn:
            preflight_blockers.append(f"{rid}: make_output_filename 실패: {e_fn}")

    # output collision 전수 확인 (output_dir가 이미 존재할 때만)
    if output_dir.exists():
        for _, row in df.iterrows():
            rid = row["review_id"]
            try:
                fname = make_output_filename(row)
                candidate_path = output_dir / fname
                if candidate_path.exists():
                    preflight_blockers.append(f"{rid}: output collision: {candidate_path}")
            except Exception:
                pass

    if preflight_blockers:
        print(
            f"[GUARD] run_actual_panels preflight blocker {len(preflight_blockers)}건 — output_dir mkdir 전 전체 abort:",
            file=sys.stderr,
        )
        for b in preflight_blockers[:20]:
            print(f"  {b}", file=sys.stderr)
        sys.exit(2)

    print(f"[GUARD] actual run preflight 전수 통과 ({len(df)}행)")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] output_dir 생성: {output_dir}")

    generated_ok = 0
    generated_error = 0
    failed_rows = []
    _vol_cache = {}

    for _, row in df.iterrows():
        review_id = row["review_id"]
        patient_id = str(row["patient_id"])
        z_center = int(row["candidate_local_z"])
        candidate_y0 = int(row["candidate_y0"])
        candidate_x0 = int(row["candidate_x0"])
        safety_role = row["safety_role"]

        ct_path_str = row["candidate_ct_path"]
        roi_path_str = row["candidate_roi_path"]
        lesion_path_str = row["candidate_lesion_mask_path"]

        output_fname = make_output_filename(row)
        output_path = output_dir / output_fname

        # output collision abort (--overwrite 없음)
        if output_path.exists():
            print(
                f"[GUARD] output collision — {output_path} 이미 존재. --overwrite 없음. 전체 abort.",
                file=sys.stderr,
            )
            sys.exit(2)

        # volume 캐시 (patient_id 기준; 교체 시 del + gc)
        if patient_id not in _vol_cache:
            if _vol_cache:
                old_pid = next(iter(_vol_cache))
                del _vol_cache[old_pid]
                gc.collect()
                print(f"[INFO] volume cache 교체: released {old_pid}")

            ct_p = Path(ct_path_str)
            roi_p = Path(roi_path_str)

            if not ct_p.exists():
                print(f"[ERROR] CT 없음: {ct_p}", file=sys.stderr)
                failed_rows.append({"review_id": review_id, "output_path": str(output_path), "error": f"CT not found: {ct_p}"})
                generated_error += 1
                continue
            if not roi_p.exists():
                print(f"[ERROR] ROI 없음: {roi_p}", file=sys.stderr)
                failed_rows.append({"review_id": review_id, "output_path": str(output_path), "error": f"ROI not found: {roi_p}"})
                generated_error += 1
                continue

            ct_vol = np.load(str(ct_p), mmap_mode="r")
            roi_vol = np.load(str(roi_p), mmap_mode="r")

            lesion_vol = None
            if safety_role == "lesion_protect" and lesion_path_str not in (
                "NOT_REQUIRED_FOR_NORMAL", "", None
            ) and not pd.isna(lesion_path_str):
                lesion_p = Path(lesion_path_str)
                if not lesion_p.exists():
                    print(f"[ERROR] lesion mask 없음: {lesion_p}", file=sys.stderr)
                    failed_rows.append({"review_id": review_id, "output_path": str(output_path), "error": f"lesion mask not found: {lesion_p}"})
                    generated_error += 1
                    continue
                lesion_vol = np.load(str(lesion_p), mmap_mode="r")

            _vol_cache[patient_id] = (ct_vol, roi_vol, lesion_vol)
            print(f"[INFO] volume 로드 완료: patient_id={patient_id}")

        ct_vol, roi_vol, lesion_vol = _vol_cache[patient_id]
        Z = ct_vol.shape[0]

        # z-context: z-2, z-1, z, z+1, z+2 (범위 clip; 범위 벗어나면 빈 패널 처리)
        z_offsets = [-2, -1, 0, 1, 2]
        z_indices = []
        z_slices = []
        for offset in z_offsets:
            zi = z_center + offset
            if 0 <= zi < Z:
                z_indices.append(zi)
                z_slices.append(np.array(ct_vol[zi]))  # mmap slice → 실제 array 복사
            # 범위 밖이면 z_indices/z_slices에 추가 안 함 → _make_panel_figure에서 빈 패널

        # extra metadata join
        extra = _load_extra_metadata(review_id)

        row_meta = {
            "review_id": review_id,
            "patient_id": patient_id,
            "group": row["group"],
            "safety_role": safety_role,
            "ct_context_id": row["ct_context_id"],
            "b0_visual_label": extra["b0_visual_label"],
            "candidate_score": extra["candidate_score"],
            "planned_z_context": extra["planned_z_context"],
        }

        try:
            # TODO: local zoom panel 미구현. full-slice context panel만 생성.
            # local zoom은 candidate 주변 crop → 추후 구현 예정.
            fig = _make_panel_figure(
                z_slices=z_slices,
                z_indices=z_indices,
                z_center=z_center,
                roi_vol=roi_vol,
                lesion_vol=lesion_vol,
                candidate_y0=candidate_y0,
                candidate_x0=candidate_x0,
                row_meta=row_meta,
            )
            fig.savefig(str(output_path), dpi=100, bbox_inches="tight")
            plt.close(fig)
            print(f"[OK] {output_fname}")
            generated_ok += 1

        except Exception as e:
            print(f"[ERROR] review_id={review_id} PNG 생성 실패: {e}", file=sys.stderr)
            failed_rows.append({"review_id": review_id, "output_path": str(output_path), "error": str(e)})
            generated_error += 1

    print(
        f"\n[SUMMARY] generated_ok={generated_ok}, generated_error={generated_error}"
    )
    print(f"[INFO] output_dir: {output_dir}")

    if generated_error > 0 or generated_ok != REQUIRED_TOTAL_ROWS:
        print(
            f"[ERROR] 생성 실패 {generated_error}건 / 성공 {generated_ok}/{REQUIRED_TOTAL_ROWS}. abort.",
            file=sys.stderr,
        )
        for fr in failed_rows:
            print(
                f"  review_id={fr['review_id']} | path={fr['output_path']} | error={fr['error']}",
                file=sys.stderr,
            )
        sys.exit(2)


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="B0 CT-context panel 생성 스크립트"
    )
    parser.add_argument(
        "--targets",
        type=str,
        default=None,
        help="target CSV 경로 (예: qa/dev_safe_mixed_error_visual_qa/b0_ct_context_shape_checked_targets.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="출력 디렉토리 (예: qa/dev_safe_mixed_error_visual_qa/b0_ct_context_panels)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="경로/shape/collision만 확인. PNG 생성, np.load, slice, figure, output_dir mkdir 전부 금지.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="실제 panel 생성 실행 (--confirm-run 필수).",
    )
    parser.add_argument(
        "--confirm-run",
        action="store_true",
        help="--run 활성화 확인 플래그. --run 없이는 무시됨.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    # 아무 모드도 없으면 사용법 출력 후 종료
    if not args.dry_run and not args.run:
        print(
            "사용법:\n"
            "  --dry-run  : 경로/shape/collision만 확인 (PNG 생성 없음)\n"
            "  --run --confirm-run : 실제 panel 생성\n"
            "\n"
            "예시:\n"
            "  python scripts/make_b0_ct_context_panels.py \\\n"
            "    --targets qa/dev_safe_mixed_error_visual_qa/b0_ct_context_shape_checked_targets.csv \\\n"
            "    --output-dir qa/dev_safe_mixed_error_visual_qa/b0_ct_context_panels \\\n"
            "    --dry-run\n"
        )
        sys.exit(0)

    # --run 단독: --confirm-run 없으면 abort
    if args.run and not args.confirm_run:
        print(
            "[ABORT] --run 단독 실행 거부. 실제 PNG 생성은 --run --confirm-run 함께 필요.",
            file=sys.stderr,
        )
        sys.exit(2)

    # 필수 인자 확인
    if args.targets is None:
        print("[ERROR] --targets 인자 필요", file=sys.stderr)
        sys.exit(2)
    if args.output_dir is None:
        print("[ERROR] --output-dir 인자 필요", file=sys.stderr)
        sys.exit(2)

    targets_path = Path(args.targets)
    output_dir = Path(args.output_dir)

    # targets CSV 존재 확인
    if not targets_path.is_file():
        print(f"[ERROR] targets CSV 없음: {targets_path}", file=sys.stderr)
        sys.exit(2)

    # CSV 로드
    df = pd.read_csv(str(targets_path))

    # 금지 키워드 검사 (output-dir / targets / CSV path 컬럼 전체)
    guard_all_paths(str(targets_path), str(output_dir), df)

    # 안전 카운트 검증
    guard_target_counts(df)

    # stage2_holdout 검증
    guard_stage2_holdout(df)

    # ---------------------------------------------------------------------------
    # dry-run 모드
    # ---------------------------------------------------------------------------
    if args.dry_run:
        dry_result = run_dry_run(df, output_dir)

        # dry-run 이후 output_dir 미생성 확인 주석:
        # run_dry_run 내에서 mkdir 호출 없음 — output_dir_created_by_dry_run은 항상 False

        dry_pass = len(dry_result["blockers"]) == 0
        return dry_result, dry_pass

    # ---------------------------------------------------------------------------
    # 실제 run 모드 (--run --confirm-run)
    # ---------------------------------------------------------------------------
    if args.run and args.confirm_run:
        # 실제 run에서도 output collision abort (run_actual_panels 내부에서 처리)
        run_actual_panels(df, output_dir)
        return

    # 이 줄에 도달하면 안 됨 (argparse 로직상)
    print("[ERROR] 예상하지 못한 실행 경로 — abort", file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
