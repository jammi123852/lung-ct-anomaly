"""
Phase 5.52 - ROI/patch score 기반 candidate re-ranking 분석

read-only 분석만 수행. model forward 없음. score 재계산 없음.
기존 파일 수정 없음. 신규 output root에만 저장.
threshold 확정 금지. 병변 성능 결론 금지.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────
# 입력 경로
# ──────────────────────────────────────────────────────────
BASE_EVAL = (
    "outputs/second-stage-lesion-refiner-v1/evaluation/"
    "rd4ad_2p5d_normal_mw_fixed96_v1"
)
QA_DIR = f"{BASE_EVAL}/hard_negative_top_score_qa_v1"

DIAGNOSTIC_CSV_PATH = (
    f"{QA_DIR}/roi_patch_masked_score_diagnostic_v1/"
    "rd4ad_roi_patch_masked_scores_v1.csv"
)
DIAGNOSTIC_JSON_PATH = (
    f"{QA_DIR}/roi_patch_masked_score_diagnostic_v1/"
    "rd4ad_roi_patch_masked_scores_summary_v1.json"
)
QA_MANIFEST_PATH = f"{QA_DIR}/hard_negative_top_score_qa_manifest_v1.csv"
OVERLAP_ANALYSIS_PATH = (
    f"{QA_DIR}/hard_negative_top_score_lesion_overlap_analysis_v1.csv"
)
CONTEXT_OVERLAY_INDEX_PATH = (
    "outputs/second-stage-lesion-refiner-v1/visualizations/"
    "hard_negative_top_score_context_overlay_qa_v1/"
    "manifest_png_index_context_overlay_v1.csv"
)

OUTPUT_ROOT = (
    "outputs/second-stage-lesion-refiner-v1/review_annotations/"
    "hard_negative_top_score_qa_v1/roi_patch_reranking_review_v1"
)
OUTPUT_CSV_NAME = "roi_patch_reranking_review_sheet_v1.csv"
OUTPUT_JSON_NAME = "roi_patch_reranking_review_summary_v1.json"
OUTPUT_MD_NAME = "phase5_52_roi_patch_reranking_analysis_v1.md"
OUTPUT_GUIDE_MD_NAME = "roi_patch_reranking_review_guide_v1.md"

FORBIDDEN_PATH_KEYWORDS = ["stage2_holdout", "holdout", "v2"]

TOP_N = 20
ROI_PATCH_HIGH_TOP_N = 20

REQUIRED_SCORE_COLS = [
    "crop_id", "patient_id", "qa_group", "qa_priority", "lesion_overlap_class",
    "rd4ad_l1_whole_crop_stored", "rd4ad_l1_roi_mean", "rd4ad_l1_outside_roi_mean",
    "rd4ad_l1_patch_mean", "rd4ad_l1_roi_patch_mean",
    "rd4ad_l1_lung_channel_roi_mean", "rd4ad_l1_med_channel_roi_mean",
    "roi_ratio_fixed96", "patch_roi_ratio", "patch_bbox_clipped",
    "lesion_pixels_patch", "lesion_pixels_context192",
]


# ──────────────────────────────────────────────────────────
# Guard
# ──────────────────────────────────────────────────────────
def check_forbidden_path(path_str: str) -> None:
    p = str(path_str).lower()
    for kw in FORBIDDEN_PATH_KEYWORDS:
        if kw in p:
            raise RuntimeError(
                f"[GUARD] 금지된 경로 키워드 '{kw}' 감지: {path_str}"
            )


# ──────────────────────────────────────────────────────────
# 입력 로드
# ──────────────────────────────────────────────────────────
def load_inputs(project_root: Path) -> tuple:
    paths = {
        "diagnostic_csv": project_root / DIAGNOSTIC_CSV_PATH,
        "diagnostic_json": project_root / DIAGNOSTIC_JSON_PATH,
        "qa_manifest": project_root / QA_MANIFEST_PATH,
        "overlap_analysis": project_root / OVERLAP_ANALYSIS_PATH,
        "context_overlay_index": project_root / CONTEXT_OVERLAY_INDEX_PATH,
    }
    for label, p in paths.items():
        check_forbidden_path(str(p))
        if not p.exists():
            raise FileNotFoundError(f"[ERROR] {label} 없음: {p}")

    df_diag = pd.read_csv(paths["diagnostic_csv"])
    with open(paths["diagnostic_json"], "r", encoding="utf-8") as f:
        diag_json = json.load(f)
    df_qa = pd.read_csv(paths["qa_manifest"])
    df_overlap = pd.read_csv(paths["overlap_analysis"])
    df_ctx = pd.read_csv(paths["context_overlay_index"])

    return df_diag, diag_json, df_qa, df_overlap, df_ctx


# ──────────────────────────────────────────────────────────
# rank 계산
# ──────────────────────────────────────────────────────────
def add_ranks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rank_whole"] = df["rd4ad_l1_whole_crop_stored"].rank(
        ascending=False, method="min"
    ).astype(int)
    df["rank_roi"] = df["rd4ad_l1_roi_mean"].rank(
        ascending=False, method="min"
    ).astype(int)
    df["rank_roi_patch"] = df["rd4ad_l1_roi_patch_mean"].rank(
        ascending=False, method="min", na_option="bottom"
    ).astype(int)
    df["rank_outside_roi"] = df["rd4ad_l1_outside_roi_mean"].rank(
        ascending=False, method="min"
    ).astype(int)
    df["rank_shift_whole_to_roi_patch"] = df["rank_whole"] - df["rank_roi_patch"]
    return df


# ──────────────────────────────────────────────────────────
# 파생 비율 컬럼
# ──────────────────────────────────────────────────────────
def add_ratio_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["outside_to_roi_ratio"] = df["rd4ad_l1_outside_roi_mean"] / df[
        "rd4ad_l1_roi_mean"
    ].replace(0, np.nan)
    df["whole_to_roi_patch_ratio"] = df["rd4ad_l1_whole_crop_stored"] / df[
        "rd4ad_l1_roi_patch_mean"
    ].replace(0, np.nan)
    return df


# ──────────────────────────────────────────────────────────
# diagnostic_class 부여
# ──────────────────────────────────────────────────────────
def assign_diagnostic_class(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    n = len(df)
    roi_patch_high_cutoff = df["rd4ad_l1_roi_patch_mean"].quantile(
        1 - ROI_PATCH_HIGH_TOP_N / n
    )
    whole_high_cutoff = df["rd4ad_l1_whole_crop_stored"].quantile(
        1 - TOP_N / n
    )

    def _classify(row):
        classes = []

        if row.get("lesion_overlap_class") == "patch_overlap":
            classes.append("lesion_overlap_exclude")

        if row.get("lesion_overlap_class") == "context192_overlap_only":
            classes.append("context_overlap_caution")

        if (
            row.get("rank_roi_patch", 999) <= ROI_PATCH_HIGH_TOP_N
            or row.get("rd4ad_l1_roi_patch_mean", 0) >= roi_patch_high_cutoff
        ):
            classes.append("roi_patch_high_candidate")

        if (
            row.get("rd4ad_l1_outside_roi_mean", 0) > row.get("rd4ad_l1_roi_mean", 0)
            and row.get("outside_to_roi_ratio", 1.0) > 2.0
            and row.get("rd4ad_l1_whole_crop_stored", 0) >= whole_high_cutoff
            and row.get("rank_roi_patch", 999) > TOP_N
        ):
            classes.append("outside_roi_driven_candidate")

        if row.get("patch_bbox_clipped") is True or row.get("patch_bbox_clipped") == 1:
            classes.append("clipped_patch_uncertain")

        if not classes:
            classes.append("stable_low_candidate")

        return ";".join(classes)

    df["diagnostic_class"] = df.apply(_classify, axis=1)
    return df


# ──────────────────────────────────────────────────────────
# suggested_review_action
# ──────────────────────────────────────────────────────────
def assign_review_action(row) -> str:
    cls = str(row.get("diagnostic_class", ""))
    overlap_cls = str(row.get("lesion_overlap_class", ""))

    if overlap_cls == "patch_overlap":
        return "병변 mask와 patch가 겹침. hard negative 제외 후보로 유지. 성능 결론에는 사용 금지."
    if overlap_cls == "context192_overlap_only":
        return "patch는 병변과 안 겹치지만 context에 병변 있음. 병변 주변 구조 여부 확인."
    if "outside_roi_driven_candidate" in cls:
        return "whole-crop 점수는 높지만 ROI 밖 error 영향 가능성 큼. 폐 안쪽 이상으로 해석 금지."
    if "roi_patch_high_candidate" in cls:
        return "ROI∩patch 내부 error가 높은 후보. 폐 내부 false positive 원인 우선 검토."
    if "clipped_patch_uncertain" in cls:
        return "patch bbox가 fixed96 경계에서 잘림. patch score 해석 주의. 시각 확인 필요."
    return "ROI/patch 기준 낮은 후보. 후순위 검토."


# ──────────────────────────────────────────────────────────
# summary JSON
# ──────────────────────────────────────────────────────────
def _stat_dict(arr_like) -> dict:
    arr = np.array(
        [v for v in arr_like if v is not None and not np.isnan(v) and not np.isinf(v)],
        dtype=float,
    )
    if len(arr) == 0:
        return {k: None for k in ["mean", "std", "min", "max", "p25", "p50", "p75", "p90", "p95", "p99"]}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def generate_summary(df: pd.DataFrame, run_ts: str) -> dict:
    top20_whole = (
        df.nlargest(TOP_N, "rd4ad_l1_whole_crop_stored")["crop_id"].astype(str).tolist()
    )
    df_valid_roi_patch = df[df["rd4ad_l1_roi_patch_mean"].notna()]
    top20_roi_patch = (
        df_valid_roi_patch.nlargest(TOP_N, "rd4ad_l1_roi_patch_mean")["crop_id"]
        .astype(str)
        .tolist()
    )

    top20_whole_set = set(top20_whole)
    top20_roi_patch_set = set(top20_roi_patch)
    overlap_set = top20_whole_set & top20_roi_patch_set

    n_by_diag = {}
    for cls_str in df["diagnostic_class"].tolist():
        for c in str(cls_str).split(";"):
            n_by_diag[c] = n_by_diag.get(c, 0) + 1

    mean_by_diag = {}
    for cls_str in df["diagnostic_class"].unique():
        mask = df["diagnostic_class"].str.contains(cls_str.split(";")[0], regex=False)
        sub = df[mask]
        mean_by_diag[cls_str] = {
            "n": len(sub),
            "whole_crop_mean": float(sub["rd4ad_l1_whole_crop_stored"].mean()),
            "roi_patch_mean": float(sub["rd4ad_l1_roi_patch_mean"].mean()) if sub["rd4ad_l1_roi_patch_mean"].notna().any() else None,
        }

    mean_by_qa_group = {}
    if "qa_group" in df.columns:
        for grp, sub in df.groupby("qa_group"):
            mean_by_qa_group[str(grp)] = {
                "n": len(sub),
                "whole_crop_mean": float(sub["rd4ad_l1_whole_crop_stored"].mean()),
                "roi_patch_mean": float(sub["rd4ad_l1_roi_patch_mean"].mean()) if sub["rd4ad_l1_roi_patch_mean"].notna().any() else None,
            }

    mean_by_overlap = {}
    for cls, sub in df.groupby("lesion_overlap_class"):
        mean_by_overlap[str(cls)] = {
            "n": len(sub),
            "whole_crop_mean": float(sub["rd4ad_l1_whole_crop_stored"].mean()),
            "roi_patch_mean": float(sub["rd4ad_l1_roi_patch_mean"].mean()) if sub["rd4ad_l1_roi_patch_mean"].notna().any() else None,
            "outside_roi_mean": float(sub["rd4ad_l1_outside_roi_mean"].mean()),
        }

    n_by_lesion_class = df["lesion_overlap_class"].value_counts().to_dict()

    return {
        "run_timestamp": run_ts,
        "n_rows": len(df),
        "n_patients": df["patient_id"].nunique() if "patient_id" in df.columns else None,
        "n_by_lesion_overlap_class": {str(k): int(v) for k, v in n_by_lesion_class.items()},
        "n_patch_bbox_clipped": int(df["patch_bbox_clipped"].sum()),
        "n_outside_roi_greater_than_roi": int(
            (df["rd4ad_l1_outside_roi_mean"] > df["rd4ad_l1_roi_mean"]).sum()
        ),
        "top20_whole_crop_ids": top20_whole,
        "top20_roi_patch_ids": top20_roi_patch,
        "top20_overlap_count": len(overlap_set),
        "top20_overlap_ids": sorted(overlap_set),
        "top20_only_whole": sorted(top20_whole_set - top20_roi_patch_set),
        "top20_only_roi_patch": sorted(top20_roi_patch_set - top20_whole_set),
        "n_by_diagnostic_class": n_by_diag,
        "mean_scores_by_diagnostic_class": mean_by_diag,
        "mean_scores_by_qa_group": mean_by_qa_group,
        "mean_scores_by_lesion_overlap_class": mean_by_overlap,
        "outside_to_roi_ratio": _stat_dict(df["outside_to_roi_ratio"].tolist()),
        "whole_to_roi_patch_ratio": _stat_dict(df["whole_to_roi_patch_ratio"].tolist()),
        "caution": {
            "threshold_not_finalized": True,
            "lesion_conclusion_forbidden": True,
            "whole_crop_score_not_lung_specific": True,
            "roi_patch_score_is_diagnostic_not_retrained_model_score": True,
            "stage2_holdout_unused": True,
            "v2_unused": True,
            "volume_source_not_accessed": True,
        },
    }


# ──────────────────────────────────────────────────────────
# MD report
# ──────────────────────────────────────────────────────────
def generate_md_report(df: pd.DataFrame, summary: dict, run_ts: str) -> str:
    lines = []
    lines.append("# Phase 5.52 ROI/Patch Score 기반 Candidate Re-ranking 분석")
    lines.append("")
    lines.append(f"- run_timestamp: {run_ts}")
    lines.append(f"- n_rows: {summary['n_rows']}")
    lines.append(f"- n_patients: {summary['n_patients']}")
    lines.append("")

    lines.append("## 1. 왜 Phase 5.52가 필요한가")
    lines.append("")
    lines.append(
        "Phase 5.51에서 생성된 ROI/patch-masked diagnostic score를 통해 "
        "기존 whole-crop L1 score가 폐 실질(ROI) 내부 오차를 과대/과소 반영하고 있음을 확인했다. "
        "ROI 밖 error가 149/150개에서 ROI 안보다 컸으므로, "
        "whole-crop 점수만으로 '폐 안쪽 이상 후보'를 결론짓는 것은 위험하다. "
        "Phase 5.52에서는 이 정보를 바탕으로 후보 순위를 재정렬하고 검토 우선순위를 제시한다."
    )
    lines.append("")

    lines.append("## 2. 기존 whole-crop score의 문제")
    lines.append("")
    lines.append(
        f"- outside_roi_mean > roi_mean: {summary['n_outside_roi_greater_than_roi']}/150개 (99.3%)"
    )
    lines.append(
        "- whole-crop score가 높은 케이스 대부분이 ROI 밖 오차로 인한 것일 수 있음."
    )
    lines.append(
        "- whole_to_roi_patch_ratio 평균: "
        f"{summary['whole_to_roi_patch_ratio'].get('mean', 'N/A'):.3f}"
        if summary["whole_to_roi_patch_ratio"].get("mean") else "- whole_to_roi_patch_ratio: 데이터 없음"
    )
    lines.append("")

    lines.append("## 3. ROI/patch score로 본 순위 변화")
    lines.append("")
    lines.append(
        f"- whole top20 ∩ roi_patch top20 overlap: {summary['top20_overlap_count']}개"
    )
    lines.append(f"- overlap crop_id: {', '.join(summary['top20_overlap_ids'])}")
    lines.append(
        f"- only whole top20 (roi_patch에서 제외): {len(summary['top20_only_whole'])}개"
    )
    lines.append(
        f"- only roi_patch top20 (whole에서 없던 후보): {len(summary['top20_only_roi_patch'])}개"
    )
    lines.append("")

    lines.append("## 4. whole top20 vs roi_patch top20 비교")
    lines.append("")
    lines.append("### whole top20 crop_id")
    lines.append(", ".join(summary["top20_whole_crop_ids"]))
    lines.append("### roi_patch top20 crop_id")
    lines.append(", ".join(summary["top20_roi_patch_ids"]))
    lines.append("")

    lines.append("## 5. outside_roi 영향 큰 후보")
    lines.append("")
    outside_driven = df[df["diagnostic_class"].str.contains("outside_roi_driven_candidate", na=False)]
    lines.append(f"outside_roi_driven_candidate: {len(outside_driven)}개")
    if len(outside_driven) > 0:
        lines.append(
            "crop_id: "
            + ", ".join(outside_driven.nlargest(10, "outside_to_roi_ratio")["crop_id"].astype(str).tolist())
        )
    lines.append(
        "→ 이 후보들은 whole-crop score는 높지만 ROI 안쪽 error는 낮음. 폐 실질 이상 후보로 해석 금지."
    )
    lines.append("")

    lines.append("## 6. patch_bbox_clipped 케이스 해석 주의")
    lines.append("")
    lines.append(
        f"- patch_bbox_clipped=True: {summary['n_patch_bbox_clipped']}개/150개"
    )
    lines.append(
        "- 이 케이스들의 rd4ad_l1_patch_mean과 rd4ad_l1_roi_patch_mean은 whole-crop 범위와 일부 중복됨."
    )
    lines.append(
        "- patch_mean이 whole-crop mean과 유사하게 나오면 patch-local 해석이 아닌 crop-level 해석에 가까움."
    )
    lines.append("")

    lines.append("## 7. P1/P2/no_overlap별 차이")
    lines.append("")
    for cls, vals in summary["mean_scores_by_lesion_overlap_class"].items():
        roi_p = f"{vals['roi_patch_mean']:.6f}" if vals["roi_patch_mean"] else "N/A"
        lines.append(
            f"- {cls} (n={vals['n']}): whole={vals['whole_crop_mean']:.6f}, "
            f"roi_patch={roi_p}, outside_roi={vals['outside_roi_mean']:.6f}"
        )
    lines.append("")

    lines.append("## 8. 다음 시각 검토 우선순위")
    lines.append("")
    lines.append("1. roi_patch_high_candidate 상위 후보 시각 검토")
    lines.append("2. outside_roi_driven_candidate 후보 context overlay 재확인")
    lines.append("3. patch_bbox_clipped=True 중 roi_patch_high 후보")
    lines.append("4. lesion_overlap_exclude (P1) 후보 9개 hard negative 제외 확인")
    lines.append("")

    lines.append("## 9. threshold 확정 금지")
    lines.append("")
    lines.append(
        "이 분석은 diagnostic 전용이다. "
        "ROI/patch score 기반 threshold 재확정은 별도 단계에서만 수행한다."
    )
    lines.append("")

    lines.append("## 10. 병변 성능 결론 금지")
    lines.append("")
    lines.append(
        "이 결과로 병변 탐지 성능(sensitivity/specificity)을 결론짓지 않는다. "
        "hard negative 제거 후보 구분 목적으로만 사용한다."
    )
    lines.append("")

    lines.append("## 11. 다음 단계 제안")
    lines.append("")
    lines.append("- roi_patch top 후보 visual review pack 생성")
    lines.append("- outside_roi-driven 후보 분리 및 별도 context overlay 시각 검토")
    lines.append("- normal val/test ROI score threshold 검토는 별도 단계에서만 수행")
    lines.append("")
    lines.append("---")
    lines.append("*이 보고서는 자동 생성된 diagnostic 문서임. 임상 결론에 사용 금지.*")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────
# guide MD
# ──────────────────────────────────────────────────────────
def generate_guide_md() -> str:
    lines = []
    lines.append("# ROI/Patch Re-ranking Review Sheet 사용 가이드")
    lines.append("")
    lines.append("## 컬럼 설명")
    lines.append("")
    lines.append("| 컬럼 | 설명 |")
    lines.append("|------|------|")
    lines.append("| rank_whole | whole-crop L1 내림차순 rank (1=최고) |")
    lines.append("| rank_roi | ROI mean L1 내림차순 rank |")
    lines.append("| rank_roi_patch | ROI∩patch mean L1 내림차순 rank |")
    lines.append("| rank_outside_roi | outside ROI mean L1 내림차순 rank |")
    lines.append("| rank_shift_whole_to_roi_patch | rank_whole - rank_roi_patch (양수=whole 기준이 더 높음) |")
    lines.append("| outside_to_roi_ratio | outside ROI mean / ROI mean (클수록 ROI 밖이 더 비정상) |")
    lines.append("| whole_to_roi_patch_ratio | whole-crop / roi_patch (클수록 outside_roi 영향 큼) |")
    lines.append("| diagnostic_class | 자동 분류. 복수 가능 (;로 구분) |")
    lines.append("| suggested_review_action | 검토 가이드 문장 |")
    lines.append("| png_path | context overlay 개별 PNG 경로 |")
    lines.append("| contact_sheet_path | contact sheet PNG 경로 |")
    lines.append("| manual_label | 수동 검토 라벨 (빈칸→검토 후 채움) |")
    lines.append("| manual_confidence | 신뢰도 (1~5) |")
    lines.append("| manual_note | 자유 메모 |")
    lines.append("| reviewer | 검토자 이름 |")
    lines.append("| reviewed_at | 검토 일시 |")
    lines.append("")
    lines.append("## diagnostic_class 분류 기준")
    lines.append("")
    lines.append("- `lesion_overlap_exclude`: lesion_overlap_class == patch_overlap. 성능 결론 사용 금지.")
    lines.append("- `context_overlap_caution`: lesion_overlap_class == context192_overlap_only. context 확인 필요.")
    lines.append("- `roi_patch_high_candidate`: roi_patch rank 상위 20 또는 상위 분위. 폐 내부 FP 우선 검토.")
    lines.append("- `outside_roi_driven_candidate`: whole score 높지만 ROI/patch 낮고 outside_to_roi_ratio > 2. 해석 주의.")
    lines.append("- `clipped_patch_uncertain`: patch bbox가 fixed96 경계에서 잘림. score 해석 주의.")
    lines.append("- `stable_low_candidate`: 위 조건 해당 없음. 후순위 검토.")
    lines.append("")
    lines.append("## 주의사항")
    lines.append("")
    lines.append("- threshold 확정 금지.")
    lines.append("- 병변 성능 결론 금지.")
    lines.append("- model forward, score 재계산 금지.")
    lines.append("- stage2_holdout/v2 접근 금지.")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────
def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    print(f"[INFO] project_root: {project_root}")
    run_ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    print(f"[INFO] run_timestamp: {run_ts}")

    check_forbidden_path(str(project_root / OUTPUT_ROOT))

    out_root = project_root / OUTPUT_ROOT
    if out_root.exists():
        sys.exit(f"[ABORT] output root 이미 존재함. 덮어쓰기 방지로 중단: {out_root}")
    for _chk_f in [
        out_root / OUTPUT_CSV_NAME,
        out_root / OUTPUT_JSON_NAME,
        out_root / OUTPUT_MD_NAME,
        out_root / OUTPUT_GUIDE_MD_NAME,
    ]:
        if _chk_f.exists():
            sys.exit(f"[ABORT] 출력 파일 이미 존재함. 덮어쓰기 방지로 중단: {_chk_f}")

    print("[INFO] 입력 파일 로드 중...")
    df_diag, diag_json, df_qa, df_overlap, df_ctx = load_inputs(project_root)

    print(f"[INFO] diagnostic CSV rows: {len(df_diag)}")
    if len(df_diag) != 150:
        sys.exit(f"[ABORT] diagnostic CSV row 수 불일치: {len(df_diag)} (expected 150)")
    if df_diag["crop_id"].duplicated().any():
        dup = df_diag["crop_id"][df_diag["crop_id"].duplicated()].tolist()
        sys.exit(f"[ABORT] crop_id 중복 감지: {dup}")
    missing_cols = [c for c in REQUIRED_SCORE_COLS if c not in df_diag.columns]
    if missing_cols:
        sys.exit(f"[ABORT] required score columns 누락: {missing_cols}")

    # 1. rank / ratio 계산
    df = add_ranks(df_diag)
    df = add_ratio_columns(df)

    # 2. diagnostic_class
    df = assign_diagnostic_class(df)

    # 3. suggested_review_action
    df["suggested_review_action"] = df.apply(assign_review_action, axis=1)

    # 4. png_path, contact_sheet_path join (crop_id 기준)
    print(f"[INFO] context overlay index rows: {len(df_ctx)}")
    df_ctx_sub = df_ctx[["crop_id", "png_path", "contact_sheet_path"]].copy()
    df_ctx_sub["crop_id"] = df_ctx_sub["crop_id"].astype(str)
    df["crop_id"] = df["crop_id"].astype(str)
    df = df.merge(df_ctx_sub, on="crop_id", how="left")
    if len(df) != 150:
        sys.exit(f"[ABORT] context overlay join 후 row 수 불일치: {len(df)} (expected 150)")
    png_missing = df["png_path"].isna().sum()
    if png_missing > 0:
        sys.exit(f"[ABORT] png_path 누락: {png_missing}건")
    cs_missing = df["contact_sheet_path"].isna().sum()
    if cs_missing > 0:
        sys.exit(f"[ABORT] contact_sheet_path 누락: {cs_missing}건")
    png_not_exist = [
        str(p) for p in df["png_path"]
        if not Path(str(p)).exists() and not (project_root / str(p)).exists()
    ]
    if png_not_exist:
        sys.exit(
            f"[ABORT] png_path 실제 파일 없음: {len(png_not_exist)}건 (예: {png_not_exist[:3]})"
        )
    cs_not_exist = [
        str(p) for p in df["contact_sheet_path"]
        if not Path(str(p)).exists() and not (project_root / str(p)).exists()
    ]
    if cs_not_exist:
        sys.exit(
            f"[ABORT] contact_sheet_path 실제 파일 없음: {len(cs_not_exist)}건 (예: {cs_not_exist[:3]})"
        )
    print("[CHECK] png_path/contact_sheet_path 존재 150/150 확인 완료")

    # 5. manual 라벨 빈 컬럼
    for col in ["manual_label", "manual_confidence", "manual_note", "reviewer", "reviewed_at"]:
        if col not in df.columns:
            df[col] = ""

    # 6. 컬럼 순서 정리
    required_cols = [
        "crop_id", "patient_id", "qa_group", "qa_priority", "lesion_overlap_class",
        "rd4ad_l1_whole_crop_stored", "rd4ad_l1_roi_mean", "rd4ad_l1_outside_roi_mean",
        "rd4ad_l1_patch_mean", "rd4ad_l1_roi_patch_mean",
        "rd4ad_l1_lung_channel_roi_mean", "rd4ad_l1_med_channel_roi_mean",
        "roi_ratio_fixed96", "patch_roi_ratio", "patch_bbox_clipped",
        "lesion_pixels_patch", "lesion_pixels_context192",
        "rank_whole", "rank_roi", "rank_roi_patch", "rank_outside_roi",
        "rank_shift_whole_to_roi_patch",
        "outside_to_roi_ratio", "whole_to_roi_patch_ratio",
        "diagnostic_class", "suggested_review_action",
        "png_path", "contact_sheet_path",
        "manual_label", "manual_confidence", "manual_note", "reviewer", "reviewed_at",
    ]
    existing_cols = [c for c in required_cols if c in df.columns]
    extra_cols = [c for c in df.columns if c not in required_cols]
    df = df[existing_cols + extra_cols]

    # 7. summary 생성 및 검증 (저장 전 모든 검증 완료)
    summary = generate_summary(df, run_ts)
    if len(summary["top20_whole_crop_ids"]) != TOP_N:
        sys.exit(f"[ABORT] top20_whole_crop_ids 수 불일치: {len(summary['top20_whole_crop_ids'])}")
    if len(summary["top20_roi_patch_ids"]) != TOP_N:
        sys.exit(f"[ABORT] top20_roi_patch_ids 수 불일치: {len(summary['top20_roi_patch_ids'])}")
    if not (0 <= summary["top20_overlap_count"] <= TOP_N):
        sys.exit(f"[ABORT] top20_overlap_count 범위 초과: {summary['top20_overlap_count']}")
    empty_diag = df["diagnostic_class"].isna() | (df["diagnostic_class"].str.strip() == "")
    if empty_diag.sum() > 0:
        sys.exit(f"[ABORT] diagnostic_class 비어있는 row: {empty_diag.sum()}건")

    # 8. 모든 검증 통과 후 output root 생성
    out_root.mkdir(parents=True, exist_ok=True)
    for _save_f in [
        out_root / OUTPUT_CSV_NAME,
        out_root / OUTPUT_JSON_NAME,
        out_root / OUTPUT_MD_NAME,
        out_root / OUTPUT_GUIDE_MD_NAME,
    ]:
        if _save_f.exists():
            sys.exit(f"[ABORT] 저장 직전 출력 파일 충돌 감지: {_save_f}")

    # 9. output CSV
    csv_path = out_root / OUTPUT_CSV_NAME
    df.to_csv(str(csv_path), index=False)
    print(f"[SAVE] CSV: {csv_path} ({len(df)} rows)")

    # 10. summary JSON
    json_path = out_root / OUTPUT_JSON_NAME
    with open(str(json_path), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] JSON: {json_path}")

    # 11. MD report
    md_text = generate_md_report(df, summary, run_ts)
    md_path = out_root / OUTPUT_MD_NAME
    with open(str(md_path), "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"[SAVE] MD report: {md_path}")

    # 12. guide MD
    guide_md_text = generate_guide_md()
    guide_md_path = out_root / OUTPUT_GUIDE_MD_NAME
    with open(str(guide_md_path), "w", encoding="utf-8") as f:
        f.write(guide_md_text)
    print(f"[SAVE] Guide MD: {guide_md_path}")

    print(f"\n[DONE] Phase 5.52 완료. output root: {out_root}")

    # 간단 요약 출력
    print(f"\n=== 요약 ===")
    print(f"rows: {len(df)}")
    print(f"patch_bbox_clipped: {int(df['patch_bbox_clipped'].sum())}")
    print(
        f"outside_roi > roi: {int((df['rd4ad_l1_outside_roi_mean'] > df['rd4ad_l1_roi_mean']).sum())}"
    )
    print(f"top20_overlap: {summary['top20_overlap_count']}")
    print(f"diagnostic_class counts: {summary['n_by_diagnostic_class']}")


if __name__ == "__main__":
    main()
