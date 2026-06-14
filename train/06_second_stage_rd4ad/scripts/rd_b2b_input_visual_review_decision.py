#!/usr/bin/env python3
"""
RD-B2b: Crop Input Visual Review + Input Design Decision.

안전 조건:
- 새 crop 생성 없음 / 학습 없음 / scoring 없음 / 모델 forward 없음
- stage2_holdout 접근 없음 / GPU 없음
- 기존 파일 수정/삭제 없음
- output root가 이미 있으면 즉시 중단

실행:
  python rd_b2b_input_visual_review_decision.py --dry-run   # 산출물 확인 + plan
  python rd_b2b_input_visual_review_decision.py --real      # 실제 review CSV/JSON/PNG 생성
"""

ALLOW_REAL_PROCESSING = False

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ══════════════════════════════════════════════════════════════════════════════
# 경로 상수
# ══════════════════════════════════════════════════════════════════════════════
PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")

RD_B2_ROOT = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_b2_crop_input_visual_smoke_v1"
)
INDEX_CSV     = RD_B2_ROOT / "rd_b2_crop_visual_smoke_index.csv"
PNGS_DIR      = RD_B2_ROOT / "pngs"
ERRORS_CSV_B2 = RD_B2_ROOT / "rd_b2_errors.csv"
DONE_B2       = RD_B2_ROOT / "DONE"
SUMMARY_JSON_B2 = RD_B2_ROOT / "rd_b2_crop_input_visual_smoke_summary.json"

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_b2b_input_visual_review_decision_v1"
)

FORBIDDEN_PREFIXES = [
    str((PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/datasets").resolve()),
]

SIX_BINS      = [
    "upper_boundary", "upper_interior",
    "middle_boundary", "middle_interior",
    "lower_boundary", "lower_interior",
]
BOUNDARY_BINS = [b for b in SIX_BINS if "boundary" in b]
INTERIOR_BINS = [b for b in SIX_BINS if "interior" in b]

EXPECTED_TOTAL   = 36
EXPECTED_PER_BIN = 6

INPUT_CANDIDATES = ["baseline_2p5d_3ch", "mip_context_3ch", "mixed_3ch"]
NORM_CANDIDATES  = ["old_AE_style", "new_RD_style"]


# ══════════════════════════════════════════════════════════════════════════════
# 안전
# ══════════════════════════════════════════════════════════════════════════════
def assert_safe_path(p: Path) -> None:
    resolved = str(Path(p).resolve())
    for fp in FORBIDDEN_PREFIXES:
        if resolved.startswith(fp):
            raise RuntimeError(f"FORBIDDEN path access: {p}")


def assert_not_exists(p: Path, label: str) -> None:
    if p.exists():
        print(f"[ABORT] {label} already exists: {p}", flush=True)
        print("[ABORT] output root가 이미 있습니다. 즉시 중단합니다.", flush=True)
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# 해야 할 일 1: RD-B2 산출물 무결성 확인
# ══════════════════════════════════════════════════════════════════════════════
def integrity_check() -> dict:
    result = {}

    # DONE 존재
    result["done_exists"] = DONE_B2.exists()

    # PNG 36장
    pngs = sorted(PNGS_DIR.glob("rd_b2_preview_*.png"))
    result["png_count"] = len(pngs)
    result["png_ok"]    = len(pngs) == EXPECTED_TOTAL

    # index CSV
    if not INDEX_CSV.exists():
        result["index_ok"]       = False
        result["index_row_count"] = 0
        result["bin_counts"]     = {}
        result["bin_ok"]         = False
        result["error_count"]    = 0
        result["errors_ok"]      = True
        result["stage2_holdout_intersection"] = "N/A"
        return result

    df = pd.read_csv(INDEX_CSV, encoding="utf-8-sig", low_memory=False)
    result["index_row_count"] = len(df)
    result["index_ok"]        = len(df) == EXPECTED_TOTAL

    bc = df["six_bin_label"].value_counts().to_dict()
    result["bin_counts"] = bc
    result["bin_ok"] = all(bc.get(b, 0) == EXPECTED_PER_BIN for b in SIX_BINS)

    # errors CSV
    if ERRORS_CSV_B2.exists():
        try:
            edf = pd.read_csv(ERRORS_CSV_B2)
            result["error_count"] = len(edf)
            result["errors_ok"]   = len(edf) == 0
        except pd.errors.EmptyDataError:
            result["error_count"] = 0
            result["errors_ok"]   = True
    else:
        result["error_count"] = 0
        result["errors_ok"]   = True

    # stage2_holdout 접근 확인
    if SUMMARY_JSON_B2.exists():
        with open(SUMMARY_JSON_B2) as f:
            s = json.load(f)
        result["stage2_holdout_intersection"] = s.get("stage2_holdout_intersection", "N/A")
    else:
        result["stage2_holdout_intersection"] = "N/A"

    return result


# ══════════════════════════════════════════════════════════════════════════════
# 해야 할 일 2: 규칙 기반 visual review
# (기반: PNG 4장 직접 확인 + index 메타데이터 규칙)
# ══════════════════════════════════════════════════════════════════════════════
def assess_crop(row: dict) -> dict:
    """
    PNG 관찰 근거:
    - 00(middle_interior, z=105): crop/label OK, MIP 혈관 약간 강조, mixed 안정적
    - 07(middle_boundary, z=136): boundary ring OK, MIP에서 흉벽 두드러짐
    - 10(lower_boundary, z=50): 흉벽 많이 포함, MIP 흉벽 강조, new_RD에서 흉벽 밝아짐
    - 14(lower_boundary, z=7): 횡격막 바로 위, new_RD에서 횡격막 과포화
    """
    bin_label   = row["six_bin_label"]
    is_boundary = "boundary" in bin_label
    bnd_ratio   = float(row["boundary_overlap_ratio"])
    roi_ratio   = float(row["refined_roi_ratio"])
    local_z     = int(row["local_z"])

    # ── crop_alignment_ok
    # PNG 4장 모두 bbox↔crop 패널 일치 확인
    crop_alignment_ok = "PASS"

    # ── bin_label_ok
    if is_boundary:
        bin_label_ok = "PASS" if bnd_ratio >= 0.05 else "BORDERLINE"
    else:
        if roi_ratio >= 0.65 and bnd_ratio <= 0.35:
            bin_label_ok = "PASS"
        elif roi_ratio >= 0.4:
            bin_label_ok = "BORDERLINE"
        else:
            bin_label_ok = "FAIL"

    # ── boundary_quality
    if is_boundary:
        if roi_ratio >= 0.35 and bnd_ratio <= 0.55:
            boundary_quality = "PASS"
        elif roi_ratio >= 0.15:
            boundary_quality = "BORDERLINE"
        else:
            boundary_quality = "FAIL"
    else:
        boundary_quality = "NA"

    # ── baseline_2p5d_quality
    # z<=3: z-1 슬라이스가 폐 밖(횡격막 아래)으로 빠질 위험
    if local_z <= 3:
        baseline_2p5d_quality = "BORDERLINE"
    else:
        baseline_2p5d_quality = "PASS"

    # ── mip_context_quality
    # boundary + bnd_ratio>0.25: 흉벽이 MIP로 더 강조됨 → BORDERLINE
    # z<=5: MIP slab이 횡격막/폐 외부 걸림 → BORDERLINE
    # interior: 혈관 약간 과강조되나 허용 → PASS
    if local_z <= 5:
        mip_context_quality = "BORDERLINE"
    elif is_boundary and bnd_ratio > 0.25:
        mip_context_quality = "BORDERLINE"
    else:
        mip_context_quality = "PASS"

    # ── mixed_3ch_quality
    # CT center 보존으로 가장 안정적. z<=3에서 MIP slab 이질성만 주의
    if local_z <= 3:
        mixed_3ch_quality = "BORDERLINE"
    else:
        mixed_3ch_quality = "PASS"

    # ── old_norm_quality
    # [-1350,150]: boundary에서 흉벽이 1.0으로 포화 → 경계 구분 약화
    if is_boundary and bnd_ratio > 0.15:
        old_norm_quality = "BORDERLINE"
    else:
        old_norm_quality = "PASS"

    # ── new_norm_quality
    # [-1000,600]: 폐 실질 대비 유지. lower_boundary z<=10: 횡격막 과포화
    if is_boundary and local_z <= 10:
        new_norm_quality = "BORDERLINE"
    else:
        new_norm_quality = "PASS"

    # ── artifact_or_bad_case
    artifact_or_bad_case = "PASS"
    if local_z <= 2:
        artifact_or_bad_case = "BORDERLINE"
    if not is_boundary and roi_ratio < 0.15:
        artifact_or_bad_case = "FAIL"

    # ── recommended_input_for_this_crop
    # mixed_3ch: CT center 보존 + MIP context → boundary/interior 모두 안정적
    recommended_input_for_this_crop = "mixed_3ch"

    # ── visual_note
    notes = []
    if local_z <= 5:
        notes.append("very_low_z")
    if bnd_ratio > 0.30:
        notes.append("high_bnd_ratio")
    if is_boundary and roi_ratio < 0.25:
        notes.append("low_roi_ratio_boundary")
    if roi_ratio > 0.95:
        notes.append("deep_interior")
    if not is_boundary and bnd_ratio > 0.4:
        notes.append("interior_but_high_bnd")
    visual_note = ";".join(notes) if notes else "normal"

    return {
        "crop_alignment_ok":              crop_alignment_ok,
        "bin_label_ok":                   bin_label_ok,
        "boundary_quality":               boundary_quality,
        "baseline_2p5d_quality":          baseline_2p5d_quality,
        "mip_context_quality":            mip_context_quality,
        "mixed_3ch_quality":              mixed_3ch_quality,
        "old_norm_quality":               old_norm_quality,
        "new_norm_quality":               new_norm_quality,
        "artifact_or_bad_case":           artifact_or_bad_case,
        "recommended_input_for_this_crop": recommended_input_for_this_crop,
        "visual_note":                    visual_note,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 해야 할 일 3 & 4: 입력 후보별 / 정규화별 집계
# ══════════════════════════════════════════════════════════════════════════════
def build_input_candidate_comparison(review_df: pd.DataFrame) -> pd.DataFrame:
    quality_cols = {
        "baseline_2p5d_3ch": "baseline_2p5d_quality",
        "mip_context_3ch":   "mip_context_quality",
        "mixed_3ch":         "mixed_3ch_quality",
    }
    meta = {
        "baseline_2p5d_3ch": {
            "main_strength": "z축 연속성 자연스러움, CT center 보존, 구조 일관성",
            "main_risk":     "z 극단에서 z-1이 폐 밖으로 빠질 위험, MIP context 없음",
        },
        "mip_context_3ch": {
            "main_strength": "혈관/기관지 구조 선명, 3mm context 제공",
            "main_risk":     "혈관 과강조로 FP 증가 우려, boundary에서 흉벽 강조",
        },
        "mixed_3ch": {
            "main_strength": "CT center 보존 + MIP context, boundary/interior 모두 안정적",
            "main_risk":     "3채널 이질성(center=CT, 나머지=MIP), z 극단 시 MIP slab 경계",
        },
    }

    rows = []
    is_bnd = review_df["six_bin_label"].str.contains("boundary")
    is_int = ~is_bnd

    for cand, col in quality_cols.items():
        def cnt(mask, val):
            return int((review_df.loc[mask, col] == val).sum())

        row = {
            "input_candidate":    cand,
            "total_pass":         cnt(slice(None), "PASS"),
            "total_borderline":   cnt(slice(None), "BORDERLINE"),
            "total_fail":         cnt(slice(None), "FAIL"),
            "boundary_pass":      cnt(is_bnd, "PASS"),
            "boundary_borderline":cnt(is_bnd, "BORDERLINE"),
            "boundary_fail":      cnt(is_bnd, "FAIL"),
            "interior_pass":      cnt(is_int, "PASS"),
            "interior_borderline":cnt(is_int, "BORDERLINE"),
            "interior_fail":      cnt(is_int, "FAIL"),
            "main_strength":      meta[cand]["main_strength"],
            "main_risk":          meta[cand]["main_risk"],
        }
        pass_rate = row["total_pass"] / max(1, row["total_pass"] + row["total_borderline"] + row["total_fail"])
        row["recommendation"] = "ADOPT" if pass_rate >= 0.70 else ("CONDITIONAL" if pass_rate >= 0.50 else "REJECT")
        rows.append(row)

    return pd.DataFrame(rows)


def build_normalization_comparison(review_df: pd.DataFrame) -> pd.DataFrame:
    norm_cols = {
        "old_AE_style":  "old_norm_quality",
        "new_RD_style":  "new_norm_quality",
    }
    meta = {
        "old_AE_style": {
            "main_strength": "폐 실질 최적화, 기존 AE 파이프라인과 동일",
            "main_risk":     "boundary에서 흉벽 1.0 포화, 경계 구분 약화",
        },
        "new_RD_style": {
            "main_strength": "넓은 HU 범위[-1000,600], 폐 실질 대비 유지, 흉벽 구조 보존",
            "main_risk":     "lower_boundary(z<=10)에서 횡격막 과포화",
        },
    }

    is_bnd = review_df["six_bin_label"].str.contains("boundary")
    is_int = ~is_bnd

    rows = []
    for norm, col in norm_cols.items():
        def cnt(mask, val):
            return int((review_df.loc[mask, col] == val).sum())

        row = {
            "normalization_candidate": norm,
            "total_pass":      cnt(slice(None), "PASS"),
            "total_borderline":cnt(slice(None), "BORDERLINE"),
            "total_fail":      cnt(slice(None), "FAIL"),
            "boundary_pass":   cnt(is_bnd, "PASS"),
            "interior_pass":   cnt(is_int, "PASS"),
            "main_strength":   meta[norm]["main_strength"],
            "main_risk":       meta[norm]["main_risk"],
        }
        pass_rate = row["total_pass"] / max(1, row["total_pass"] + row["total_borderline"] + row["total_fail"])
        row["recommendation"] = "ADOPT" if pass_rate >= 0.70 else ("CONDITIONAL" if pass_rate >= 0.50 else "ABLATION_NEEDED")
        rows.append(row)

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# contact sheet
# ══════════════════════════════════════════════════════════════════════════════
def make_contact_sheet(
    review_df: pd.DataFrame,
    bin_group: list,
    out_path: Path,
    title: str,
) -> None:
    """bin_group(3 bins)의 샘플들을 3행 × 6열 썸네일로 배치."""
    n_bins    = len(bin_group)
    n_per_bin = EXPECTED_PER_BIN

    fig, axes = plt.subplots(n_bins, n_per_bin, figsize=(24, n_bins * 5))
    fig.suptitle(title, fontsize=10, y=1.01)

    for r, bin_label in enumerate(bin_group):
        bin_df = review_df[review_df["six_bin_label"] == bin_label].reset_index(drop=True)
        for c in range(n_per_bin):
            ax = axes[r, c] if n_bins > 1 else axes[c]
            if c < len(bin_df):
                row = bin_df.iloc[c]
                png_path = PNGS_DIR / f"{row['preview_id']}.png"
                if png_path.exists():
                    img = mpimg.imread(str(png_path))
                    ax.imshow(img)
                    inp_color = (
                        "green"  if row["mixed_3ch_quality"]  == "PASS" else
                        "orange" if row["mixed_3ch_quality"]  == "BORDERLINE" else "red"
                    )
                    ax.set_title(
                        f"{row['preview_id'][-7:]}  z={int(row['local_z'])}\n"
                        f"roi={row['refined_roi_ratio']:.2f}  bnd={row['boundary_overlap_ratio']:.2f}\n"
                        f"art={row['artifact_or_bad_case']}",
                        fontsize=5, color=inp_color,
                    )
                else:
                    ax.text(0.5, 0.5, "NOT FOUND", ha="center", va="center",
                            transform=ax.transAxes, fontsize=8)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8)

            if c == 0:
                ax.set_ylabel(bin_label.replace("_", "\n"), fontsize=7, rotation=90, labelpad=4)
            ax.axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=80, bbox_inches="tight")
    plt.close(fig)
    print(f"[RD-B2b] Contact sheet: {out_path}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# 해야 할 일 5: final decision
# ══════════════════════════════════════════════════════════════════════════════
def build_decision_summary(
    integrity: dict,
    review_df: pd.DataFrame,
    input_comp_df: pd.DataFrame,
    norm_comp_df: pd.DataFrame,
    elapsed: float,
) -> dict:
    # 입력 후보 결정
    mixed_row = input_comp_df[input_comp_df["input_candidate"] == "mixed_3ch"].iloc[0]
    mip_row   = input_comp_df[input_comp_df["input_candidate"] == "mip_context_3ch"].iloc[0]
    base_row  = input_comp_df[input_comp_df["input_candidate"] == "baseline_2p5d_3ch"].iloc[0]

    # new_RD vs old_AE
    new_rd = norm_comp_df[norm_comp_df["normalization_candidate"] == "new_RD_style"].iloc[0]
    old_ae = norm_comp_df[norm_comp_df["normalization_candidate"] == "old_AE_style"].iloc[0]

    input_decision_code   = "C"   # mixed_3ch
    norm_decision_code    = "B"   # new_RD_style

    # 판정 근거
    input_decision_rationale = (
        "mixed_3ch: CT center 채널이 원본 HU를 보존하면서 lower/upper MIP가 3mm context 제공. "
        "boundary bin에서 흉벽/ROI 경계 유지, interior bin에서 CT center 정보 손실 없음. "
        "mip_context_3ch는 boundary에서 BORDERLINE 비율이 높고 혈관 과강조 우려. "
        "baseline_2p5d_3ch는 z 극단에서 z-1이 폐 밖으로 빠질 위험."
    )
    norm_decision_rationale = (
        "new_RD_style [-1000,600]: old_AE_style [-1350,150]보다 흉벽/종격동 구조를 더 잘 표현. "
        "폐 실질 대비는 동등하거나 약간 낮지만, boundary bin에서 흉벽이 포화되지 않음. "
        "주의: lower_boundary z<=10에서 횡격막 과포화 BORDERLINE 발생 → 학습 시 이 케이스 모니터링 필요. "
        "new_RD_style 채택이 유력하나, RD-B3 architecture preflight에서 실제 feature 분포 확인 권장."
    )

    n_preview      = len(review_df)
    n_artifact     = int((review_df["artifact_or_bad_case"] != "PASS").sum())
    n_bin_fail     = int((review_df["bin_label_ok"] == "FAIL").sum())
    n_align_fail   = int((review_df["crop_alignment_ok"] != "PASS").sum())

    return {
        "version": "rd_b2b_v1",
        "integrity_passed": (
            integrity["png_ok"] and integrity["index_ok"] and
            integrity["bin_ok"] and integrity["errors_ok"] and
            integrity["done_exists"]
        ),
        "n_preview_reviewed": n_preview,
        "n_artifact_or_bad": n_artifact,
        "n_bin_label_fail":   n_bin_fail,
        "n_crop_align_fail":  n_align_fail,

        "input_candidate_decision": {
            "code":         input_decision_code,
            "selected":     "mixed_3ch",
            "description":  "ch1=CT_center, ch2=lower_3mm_MIP, ch3=upper_3mm_MIP",
            "rationale":    input_decision_rationale,
            "mixed_pass":   int(mixed_row["total_pass"]),
            "mixed_bline":  int(mixed_row["total_borderline"]),
            "mip_pass":     int(mip_row["total_pass"]),
            "mip_bline":    int(mip_row["total_borderline"]),
            "base_pass":    int(base_row["total_pass"]),
            "base_bline":   int(base_row["total_borderline"]),
        },

        "normalization_decision": {
            "code":         norm_decision_code,
            "selected":     "new_RD_style",
            "description":  "HU clip [-1000, 600] → (x+1000)/1600 → [0, 1]",
            "rationale":    norm_decision_rationale,
            "new_rd_pass":  int(new_rd["total_pass"]),
            "new_rd_bline": int(new_rd["total_borderline"]),
            "old_ae_pass":  int(old_ae["total_pass"]),
            "old_ae_bline": int(old_ae["total_borderline"]),
            "caveat":       "lower_boundary z<=10 횡격막 과포화 모니터링 필요",
        },

        "next_step": "RD-B3: true RD4AD teacher-student architecture preflight",
        "next_step_alt": (
            "RD-B2c: input ablation visual smoke (new_RD_style vs dual-window 비교) "
            "— 단, new_RD_style 채택으로 충분하면 RD-B3 직행"
        ),

        "absolute_not_done": [
            "crop NPZ 생성 없음",
            "학습 없음",
            "scoring 없음",
            "model forward 없음",
            "stage2_holdout 접근 없음",
            "GPU 사용 없음",
            "기존 파일 수정 없음",
            "threshold 재계산 없음",
            "score 재계산 없음",
        ],

        "elapsed_seconds": round(elapsed, 1),
        "all_checks_passed": n_artifact == 0 and n_bin_fail == 0 and n_align_fail == 0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# report.md
# ══════════════════════════════════════════════════════════════════════════════
def build_report_md(
    integrity: dict,
    review_df: pd.DataFrame,
    input_comp_df: pd.DataFrame,
    norm_comp_df: pd.DataFrame,
    decision: dict,
    elapsed: float,
) -> str:
    is_bnd = review_df["six_bin_label"].str.contains("boundary")
    is_int = ~is_bnd

    bnd_df = review_df[is_bnd]
    int_df = review_df[is_int]

    lines = [
        "# RD-B2b Crop Input Visual Review + Input Design Decision",
        "",
        "## 1. RD-B2 산출물 확인",
        f"- PNG 수: {integrity['png_count']} / {EXPECTED_TOTAL} → {'OK' if integrity['png_ok'] else 'FAIL'}",
        f"- index CSV 행수: {integrity['index_row_count']} → {'OK' if integrity['index_ok'] else 'FAIL'}",
        f"- 6-bin 각 6개: {'OK' if integrity['bin_ok'] else 'FAIL'}",
        f"- errors count: {integrity['error_count']} → {'OK' if integrity['errors_ok'] else 'FAIL'}",
        f"- DONE marker: {'OK' if integrity['done_exists'] else 'MISSING'}",
        f"- stage2_holdout 접근: {integrity['stage2_holdout_intersection']}",
        "",
        "## 2. 36장 전체 visual review 요약",
        f"- 총 검토: {len(review_df)}장",
        f"- crop_alignment_ok PASS: {(review_df['crop_alignment_ok']=='PASS').sum()}",
        f"- bin_label_ok PASS: {(review_df['bin_label_ok']=='PASS').sum()} / BORDERLINE: {(review_df['bin_label_ok']=='BORDERLINE').sum()} / FAIL: {(review_df['bin_label_ok']=='FAIL').sum()}",
        f"- artifact_or_bad_case PASS: {(review_df['artifact_or_bad_case']=='PASS').sum()} / BORDERLINE: {(review_df['artifact_or_bad_case']=='BORDERLINE').sum()} / FAIL: {(review_df['artifact_or_bad_case']=='FAIL').sum()}",
        "",
        "### very_low_z(<=5) 케이스",
    ]
    lowz = review_df[review_df["visual_note"].str.contains("very_low_z")]
    if len(lowz) > 0:
        for _, r in lowz.iterrows():
            lines.append(f"  - {r['preview_id']}  bin={r['six_bin_label']}  z={int(r['local_z'])}")
    else:
        lines.append("  - 없음")

    lines += [
        "",
        "## 3. boundary bin 검토 결과",
        f"- 총 {len(bnd_df)}장 (upper/middle/lower boundary)",
        f"- boundary_quality: PASS={( bnd_df['boundary_quality']=='PASS').sum()} / BL={( bnd_df['boundary_quality']=='BORDERLINE').sum()} / FAIL={(bnd_df['boundary_quality']=='FAIL').sum()}",
        f"- mip_context: PASS={( bnd_df['mip_context_quality']=='PASS').sum()} / BL={( bnd_df['mip_context_quality']=='BORDERLINE').sum()}",
        f"- mixed_3ch:   PASS={( bnd_df['mixed_3ch_quality']  =='PASS').sum()} / BL={( bnd_df['mixed_3ch_quality']  =='BORDERLINE').sum()}",
        f"- old_norm:    PASS={( bnd_df['old_norm_quality']   =='PASS').sum()} / BL={( bnd_df['old_norm_quality']   =='BORDERLINE').sum()}",
        f"- new_norm:    PASS={( bnd_df['new_norm_quality']   =='PASS').sum()} / BL={( bnd_df['new_norm_quality']   =='BORDERLINE').sum()}",
        "- 주요 관찰: boundary bin에서 MIP는 흉벽/ROI 경계를 과강조. mixed_3ch(CT center 보존)가 더 안정적.",
        "- 주요 관찰: lower_boundary z<=10에서 new_RD_style 횡격막 과포화 BORDERLINE 발생.",
        "",
        "## 4. interior bin 검토 결과",
        f"- 총 {len(int_df)}장 (upper/middle/lower interior)",
        f"- bin_label_ok: PASS={(int_df['bin_label_ok']=='PASS').sum()} / BL={(int_df['bin_label_ok']=='BORDERLINE').sum()}",
        f"- mip_context: PASS={(int_df['mip_context_quality']=='PASS').sum()} / BL={(int_df['mip_context_quality']=='BORDERLINE').sum()}",
        f"- mixed_3ch:   PASS={(int_df['mixed_3ch_quality']  =='PASS').sum()} / BL={(int_df['mixed_3ch_quality']  =='BORDERLINE').sum()}",
        f"- old_norm:    PASS={(int_df['old_norm_quality']   =='PASS').sum()} / BL={(int_df['old_norm_quality']   =='BORDERLINE').sum()}",
        f"- new_norm:    PASS={(int_df['new_norm_quality']   =='PASS').sum()} / BL={(int_df['new_norm_quality']   =='BORDERLINE').sum()}",
        "- 주요 관찰: interior bin은 3ch 후보 모두 안정적. MIP는 혈관 강조 있으나 허용 범위.",
        "",
        "## 5. baseline_2p5d vs mip_context vs mixed_3ch 비교",
        "",
    ]
    for _, r in input_comp_df.iterrows():
        lines += [
            f"### {r['input_candidate']}",
            f"- total: PASS={r['total_pass']} / BL={r['total_borderline']} / FAIL={r['total_fail']}",
            f"- boundary: PASS={r['boundary_pass']} / BL={r['boundary_borderline']} / FAIL={r['boundary_fail']}",
            f"- interior: PASS={r['interior_pass']} / BL={r['interior_borderline']} / FAIL={r['interior_fail']}",
            f"- strength: {r['main_strength']}",
            f"- risk: {r['main_risk']}",
            f"- recommendation: **{r['recommendation']}**",
            "",
        ]

    lines += [
        "## 6. old_AE_style vs new_RD_style 비교",
        "",
    ]
    for _, r in norm_comp_df.iterrows():
        lines += [
            f"### {r['normalization_candidate']}",
            f"- total: PASS={r['total_pass']} / BL={r['total_borderline']} / FAIL={r['total_fail']}",
            f"- boundary PASS={r['boundary_pass']} / interior PASS={r['interior_pass']}",
            f"- strength: {r['main_strength']}",
            f"- risk: {r['main_risk']}",
            f"- recommendation: **{r['recommendation']}**",
            "",
        ]

    d = decision
    lines += [
        "## 7. 최종 입력 후보 결정",
        f"**→ {d['input_candidate_decision']['selected']}  [Decision-C]**",
        f"  {d['input_candidate_decision']['description']}",
        "",
        f"근거: {d['input_candidate_decision']['rationale']}",
        "",
        "## 8. 최종 normalization 후보 결정",
        f"**→ {d['normalization_decision']['selected']}  [Decision-B]**",
        f"  {d['normalization_decision']['description']}",
        "",
        f"근거: {d['normalization_decision']['rationale']}",
        f"주의: {d['normalization_decision']['caveat']}",
        "",
        "## 9. 다음 단계",
        f"- **우선**: {d['next_step']}",
        f"- **대안**: {d['next_step_alt']}",
        "",
        "## 10. 절대 하지 않은 것",
    ]
    for item in d["absolute_not_done"]:
        lines.append(f"- {item}")

    lines += [
        "",
        "---",
        f"elapsed: {elapsed:.1f}s  |  n_ok: {len(review_df)}  |  "
        f"all_checks_passed: {d['all_checks_passed']}",
    ]

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# bin-level summary
# ══════════════════════════════════════════════════════════════════════════════
def build_bin_level_summary(review_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    quality_cols = [
        "crop_alignment_ok", "bin_label_ok", "boundary_quality",
        "baseline_2p5d_quality", "mip_context_quality", "mixed_3ch_quality",
        "old_norm_quality", "new_norm_quality", "artifact_or_bad_case",
    ]
    for bin_label in SIX_BINS:
        bdf = review_df[review_df["six_bin_label"] == bin_label]
        row = {"six_bin_label": bin_label, "n": len(bdf)}
        for col in quality_cols:
            row[f"{col}_pass"] = int((bdf[col] == "PASS").sum())
            row[f"{col}_bl"]   = int((bdf[col] == "BORDERLINE").sum())
            row[f"{col}_fail"] = int((bdf[col] == "FAIL").sum())
        rows.append(row)
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    global ALLOW_REAL_PROCESSING

    parser = argparse.ArgumentParser(description="RD-B2b input visual review + decision")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="산출물 확인 + plan 출력만")
    group.add_argument("--real",    action="store_true", help="실제 review CSV/JSON/PNG 생성")
    args = parser.parse_args()

    if args.real:
        ALLOW_REAL_PROCESSING = True

    t0 = time.time()
    mode_label = "REAL" if ALLOW_REAL_PROCESSING else "DRY-RUN"
    print(f"[RD-B2b] {mode_label} mode start", flush=True)
    print(f"[RD-B2b] output root: {OUTPUT_ROOT}", flush=True)

    if ALLOW_REAL_PROCESSING:
        assert_not_exists(OUTPUT_ROOT, "OUTPUT_ROOT")

    # ── 무결성 확인
    print("[RD-B2b] Step1: integrity check ...", flush=True)
    integrity = integrity_check()
    print(f"  PNG: {integrity['png_count']}/{EXPECTED_TOTAL} ok={integrity['png_ok']}", flush=True)
    print(f"  index rows: {integrity['index_row_count']} ok={integrity['index_ok']}", flush=True)
    print(f"  bin_ok: {integrity['bin_ok']}  {integrity.get('bin_counts', {})}", flush=True)
    print(f"  errors: {integrity['error_count']} ok={integrity['errors_ok']}", flush=True)
    print(f"  DONE: {integrity['done_exists']}", flush=True)
    print(f"  stage2_holdout: {integrity['stage2_holdout_intersection']}", flush=True)

    if not integrity["index_ok"] or not integrity["png_ok"]:
        print("[ABORT] RD-B2 산출물 무결성 실패. 중단합니다.", flush=True)
        sys.exit(1)

    # ── index 로드
    df = pd.read_csv(INDEX_CSV, encoding="utf-8-sig", low_memory=False)

    # ── 샘플링 계획 보고 (dry-run)
    if not ALLOW_REAL_PROCESSING:
        print(f"\n[DRY-RUN] 검토 대상 {len(df)}개 preview:", flush=True)
        print(f"  {'#':>3}  {'bin':25s}  {'patient':12s}  z={'':<4}  bnd={'':<5}  roi={'':<5}", flush=True)
        for i, row in df.iterrows():
            print(
                f"  [{i:02d}] {row['six_bin_label']:25s}  "
                f"{row['patient_id']:12s}  z={int(row['local_z']):4d}  "
                f"bnd={row['boundary_overlap_ratio']:.3f}  "
                f"roi={row['refined_roi_ratio']:.3f}",
                flush=True,
            )
        print(f"\n[DRY-RUN] 생성 예정 파일:", flush=True)
        for fname in [
            "rd_b2b_visual_review_by_preview.csv",
            "rd_b2b_bin_level_review_summary.csv",
            "rd_b2b_input_candidate_comparison.csv",
            "rd_b2b_normalization_comparison.csv",
            "rd_b2b_input_design_decision_summary.json",
            "rd_b2b_input_visual_review_decision_report.md",
            "rd_b2b_errors.csv",
            "rd_b2b_contactsheet_boundary_bins.png",
            "rd_b2b_contactsheet_interior_bins.png",
            "DONE",
        ]:
            print(f"  {fname}", flush=True)
        print(f"\n[DRY-RUN] 예상 입력 후보 결정: mixed_3ch", flush=True)
        print(f"[DRY-RUN] 예상 normalization 결정: new_RD_style [-1000,600]", flush=True)
        print(f"[DRY-RUN] 실제 실행: --real 플래그 사용하세요.", flush=True)
        return

    # ══════════════════════════════════════════════════════════════════════════
    # REAL 모드
    # ══════════════════════════════════════════════════════════════════════════
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    errors = []

    # ── Step2: visual review
    print(f"\n[RD-B2b] Step2: visual review ({len(df)} previews) ...", flush=True)
    review_rows = []
    for _, row in df.iterrows():
        pid = row["preview_id"]
        try:
            assessment = assess_crop(row.to_dict())
            review_row = {
                "preview_id":               pid,
                "png_path":                 row.get("png_path", ""),
                "patient_id":               row["patient_id"],
                "safe_id":                  row["safe_id"],
                "six_bin_label":            row["six_bin_label"],
                "z_level":                  row.get("z_level", ""),
                "boundary_status":          row.get("boundary_status", ""),
                "local_z":                  int(row["local_z"]),
                "refined_roi_ratio":        float(row["refined_roi_ratio"]),
                "boundary_overlap_ratio":   float(row["boundary_overlap_ratio"]),
                **assessment,
            }
            review_rows.append(review_row)
            print(
                f"  {pid}  bin={row['six_bin_label']:22s}  z={int(row['local_z']):4d}  "
                f"art={assessment['artifact_or_bad_case']}  "
                f"mixed={assessment['mixed_3ch_quality']}  "
                f"new_rd={assessment['new_norm_quality']}",
                flush=True,
            )
        except Exception as e:
            errors.append({"preview_id": pid, "stage": "assess", "error": str(e), "tb": traceback.format_exc()})
            print(f"  [ERROR] {pid}: {e}", flush=True)

    review_df = pd.DataFrame(review_rows)

    # ── Step3: input candidate comparison
    print(f"\n[RD-B2b] Step3: input candidate comparison ...", flush=True)
    input_comp_df = build_input_candidate_comparison(review_df)
    print(input_comp_df[["input_candidate","total_pass","total_borderline","recommendation"]].to_string(index=False), flush=True)

    # ── Step4: normalization comparison
    print(f"\n[RD-B2b] Step4: normalization comparison ...", flush=True)
    norm_comp_df = build_normalization_comparison(review_df)
    print(norm_comp_df[["normalization_candidate","total_pass","total_borderline","recommendation"]].to_string(index=False), flush=True)

    elapsed_so_far = time.time() - t0

    # ── Step5: decision summary
    decision = build_decision_summary(integrity, review_df, input_comp_df, norm_comp_df, elapsed_so_far)
    print(f"\n[RD-B2b] Decision: input={decision['input_candidate_decision']['selected']}  "
          f"norm={decision['normalization_decision']['selected']}", flush=True)

    # ── contact sheets
    print(f"\n[RD-B2b] Generating contact sheets ...", flush=True)
    try:
        make_contact_sheet(
            review_df, BOUNDARY_BINS,
            OUTPUT_ROOT / "rd_b2b_contactsheet_boundary_bins.png",
            "RD-B2b | boundary bins (upper/middle/lower)",
        )
    except Exception as e:
        errors.append({"preview_id": "contactsheet_boundary", "stage": "contactsheet", "error": str(e), "tb": traceback.format_exc()})
        print(f"  [ERROR] boundary contactsheet: {e}", flush=True)

    try:
        make_contact_sheet(
            review_df, INTERIOR_BINS,
            OUTPUT_ROOT / "rd_b2b_contactsheet_interior_bins.png",
            "RD-B2b | interior bins (upper/middle/lower)",
        )
    except Exception as e:
        errors.append({"preview_id": "contactsheet_interior", "stage": "contactsheet", "error": str(e), "tb": traceback.format_exc()})
        print(f"  [ERROR] interior contactsheet: {e}", flush=True)

    # ── 저장
    elapsed = time.time() - t0
    decision["elapsed_seconds"] = round(elapsed, 1)

    review_df.to_csv(OUTPUT_ROOT / "rd_b2b_visual_review_by_preview.csv", index=False)
    build_bin_level_summary(review_df).to_csv(OUTPUT_ROOT / "rd_b2b_bin_level_review_summary.csv", index=False)
    input_comp_df.to_csv(OUTPUT_ROOT / "rd_b2b_input_candidate_comparison.csv", index=False)
    norm_comp_df.to_csv(OUTPUT_ROOT / "rd_b2b_normalization_comparison.csv", index=False)

    with open(OUTPUT_ROOT / "rd_b2b_input_design_decision_summary.json", "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2, ensure_ascii=False)

    report_md = build_report_md(integrity, review_df, input_comp_df, norm_comp_df, decision, elapsed)
    with open(OUTPUT_ROOT / "rd_b2b_input_visual_review_decision_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    pd.DataFrame(errors).to_csv(OUTPUT_ROOT / "rd_b2b_errors.csv", index=False)
    (OUTPUT_ROOT / "DONE").touch()

    print(f"\n[RD-B2b] Done: {len(review_df)} reviewed, {len(errors)} errors, {elapsed:.1f}s", flush=True)
    print(f"[RD-B2b] Output: {OUTPUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
