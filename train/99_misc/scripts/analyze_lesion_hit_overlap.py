"""
analyze_lesion_hit_overlap.py
1차 PaDiM 스크리닝에서 병변 patch와 모델 positive patch가 얼마나 안정적으로 겹쳤는지 분석한다.
(ChatGPT 검토/승인 후에만 실행)

- score CSV는 read-only. score 재계산 / 모델 실행 없음.
- positive = padim_score >= p95 threshold (lesion_eval_p95_fast_summary.json에서 읽음).
- lesion patch = patch_label == 1 (저장된 any_pixel 기준).
- patch_dice/iou는 patch 집합(개수) 기준이며 pixel-level Dice/IoU가 아니다.
- 출력 파일이 이미 있으면 덮어쓰기하지 않고 중단한다.
"""
from __future__ import annotations
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

# v1 경로 (기존 동작 유지)
LESION_SCORE_DIR = REPO_ROOT / "outputs/position-aware-padim-v1/scores/padim_v1/lesion_by_patient"
EVAL_DIR = REPO_ROOT / "outputs/position-aware-padim-v1/evaluation/lesion_subset"
REPORTS_DIR = REPO_ROOT / "outputs/position-aware-padim-v1/reports"

OUT_PATIENT_CSV = REPORTS_DIR / "lesion_hit_overlap_by_patient.csv"
OUT_SLICE_CSV = REPORTS_DIR / "lesion_hit_overlap_by_slice.csv"
OUT_SUMMARY_JSON = REPORTS_DIR / "lesion_hit_overlap_summary.json"

# v2 전용 경로
LESION_SCORE_DIR_V2 = REPO_ROOT / "outputs/position-aware-padim-v1/scores/padim_v1/lesion_v2_by_patient"
EVAL_DIR_V2 = REPO_ROOT / "outputs/position-aware-padim-v1/evaluation/lesion_subset_v2"
OUT_PATIENT_CSV_V2 = REPORTS_DIR / "lesion_hit_overlap_by_patient_v2.csv"
OUT_SLICE_CSV_V2 = REPORTS_DIR / "lesion_hit_overlap_by_slice_v2.csv"
OUT_SUMMARY_JSON_V2 = REPORTS_DIR / "lesion_hit_overlap_summary_v2.json"

# stage split 경로 (선택적, 존재하면 로딩)
STAGE_SPLIT_CSV = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1_balanced.csv"

USECOLS = ["patient_id", "safe_id", "group", "local_z", "padim_score", "patch_label"]
EXPECTED_N_CSV = 308  # lesion subset 전체 환자 수. 일부 환자 기준 summary 생성 방지용 guard.


def load_p95(eval_dir: Path = None, v2: bool = False, threshold_json: str | None = None) -> float:
    """p95 threshold를 읽는다.

    threshold_json이 있으면 해당 JSON의 threshold_p95를 직접 읽고,
    없으면 eval_dir의 summary JSON에서 읽는다.
    """
    if threshold_json is not None:
        with open(threshold_json, encoding="utf-8") as f:
            return float(json.load(f)["threshold_p95"])
    if eval_dir is None:
        eval_dir = EVAL_DIR
    fname = "lesion_eval_v2_p95_fast_summary.json" if v2 else "lesion_eval_p95_fast_summary.json"
    with open(eval_dir / fname, encoding="utf-8") as f:
        return float(json.load(f)["threshold_value"])


def longest_consecutive(sorted_z) -> int:
    """정렬된 정수 리스트에서 연속 구간(차이 1) 최대 길이."""
    if not sorted_z:
        return 0
    best = run = 1
    for i in range(1, len(sorted_z)):
        if sorted_z[i] == sorted_z[i - 1] + 1:
            run += 1
            best = max(best, run)
        else:
            run = 1
    return best


def gap_count(sorted_z) -> int:
    """정렬된 hit z 사이 끊김(차이>1) 개수."""
    if len(sorted_z) <= 1:
        return 0
    return sum(1 for i in range(1, len(sorted_z)) if sorted_z[i] - sorted_z[i - 1] > 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="병변 hit overlap 분석")
    parser.add_argument(
        "--dataset-profile",
        type=str,
        default="v1_model_roi",
        choices=["v1_model_roi", "v2_roi_0_0"],
        help=(
            "분석할 데이터셋 profile. "
            "v1_model_roi: 기존 lesion_by_patient score 사용 (기본값). "
            "v2_roi_0_0: lesion_v2_by_patient score 사용 (출력도 v2 전용 파일명)."
        ),
    )
    parser.add_argument(
        "--score-dir",
        type=str,
        default=None,
        help="lesion score 경로 오버라이드 (기본: dataset-profile 기반 자동 결정)",
    )
    parser.add_argument(
        "--evaluation-dir",
        type=str,
        default=None,
        help="evaluation 경로 오버라이드 (threshold summary JSON도 이 경로에서 읽음)",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default=None,
        help="reports 출력 경로 오버라이드 (기본: reports)",
    )
    parser.add_argument(
        "--threshold-json",
        type=str,
        default=None,
        help="normal threshold JSON 경로 (이 옵션이 있으면 eval_dir summary JSON 대신 사용)",
    )
    parser.add_argument(
        "--expected-n",
        type=int,
        default=EXPECTED_N_CSV,
        help=f"lesion score CSV 예상 개수 (기본: {EXPECTED_N_CSV}). v2/v2 subset 크기가 다르면 변경 필요.",
    )
    args = parser.parse_args()
    is_v2 = (args.dataset_profile == "v2_roi_0_0")

    # profile에 따라 경로/파일 결정
    lesion_score_dir = LESION_SCORE_DIR_V2 if is_v2 else LESION_SCORE_DIR
    eval_dir = EVAL_DIR_V2 if is_v2 else EVAL_DIR
    out_patient_csv = OUT_PATIENT_CSV_V2 if is_v2 else OUT_PATIENT_CSV
    out_slice_csv = OUT_SLICE_CSV_V2 if is_v2 else OUT_SLICE_CSV
    out_summary_json = OUT_SUMMARY_JSON_V2 if is_v2 else OUT_SUMMARY_JSON

    # CLI 오버라이드
    if args.score_dir is not None:
        lesion_score_dir = REPO_ROOT / args.score_dir
    if args.evaluation_dir is not None:
        eval_dir = REPO_ROOT / args.evaluation_dir
    if args.reports_dir is not None:
        _reports = REPO_ROOT / args.reports_dir
        out_patient_csv = _reports / out_patient_csv.name
        out_slice_csv = _reports / out_slice_csv.name
        out_summary_json = _reports / out_summary_json.name
        _reports.mkdir(parents=True, exist_ok=True)

    existing = [str(p) for p in (out_patient_csv, out_slice_csv, out_summary_json) if p.exists()]
    if existing:
        print("[ERROR] 출력 파일이 이미 존재합니다. 덮어쓰기 금지 — 중단합니다:")
        for e in existing:
            print(f"  - {e}")
        sys.exit(1)

    thr = load_p95(eval_dir, v2=is_v2, threshold_json=args.threshold_json)
    print(f"[hit_overlap] p95 threshold = {thr:.4f}")

    csv_paths = sorted(glob.glob(str(lesion_score_dir / "*.csv")))
    expected_n = args.expected_n
    if len(csv_paths) != expected_n:
        print(f"[ERROR] lesion score CSV 개수가 {expected_n}개가 아닙니다 "
              f"(현재 {len(csv_paths)}개). 일부 환자 기준 summary 방지를 위해 중단합니다.")
        sys.exit(1)

    # stage_split 로딩 (선택적)
    stage_split_map: dict = {}
    if STAGE_SPLIT_CSV.exists():
        try:
            ss_df = pd.read_csv(STAGE_SPLIT_CSV, encoding="utf-8-sig",
                                usecols=["patient_id", "stage_split"])
            stage_split_map = dict(zip(
                ss_df["patient_id"].astype(str).str.strip(),
                ss_df["stage_split"].astype(str).str.strip(),
            ))
            print(f"[hit_overlap] stage_split 로드: {len(stage_split_map)}명")
        except Exception as exc:
            print(f"[hit_overlap] stage_split 로드 실패 (무시): {exc}")

    patient_rows = []
    slice_rows = []

    agg = {"n_patients": 0, "n_patient_hit": 0,
           "lesion_patch_total": 0, "hit_lesion_patch_total": 0,
           "lesion_slice_total": 0, "hit_lesion_slice_total": 0}

    for p in csv_paths:
        df = pd.read_csv(p, encoding="utf-8-sig", usecols=USECOLS)
        df = df[~df["padim_score"].isna()]
        if len(df) == 0:
            continue
        agg["n_patients"] += 1
        pid = str(df["patient_id"].iloc[0])
        safe = str(df["safe_id"].iloc[0])
        grp = str(df["group"].iloc[0])

        is_lesion = (df["patch_label"].values == 1)
        is_pos = (df["padim_score"].values >= thr)
        lz = df["local_z"].values
        is_hit = is_lesion & is_pos  # 병변 patch가 positive로 잡힘

        n_lesion = int(is_lesion.sum())
        n_pos = int(is_pos.sum())
        n_hit = int(is_hit.sum())

        # patch-level dice/iou (patch 집합 개수 기준)
        union = n_lesion + n_pos - n_hit
        patch_dice = (2 * n_hit / (n_lesion + n_pos)) if (n_lesion + n_pos) > 0 else float("nan")
        patch_iou = (n_hit / union) if union > 0 else float("nan")
        patch_recall = (n_hit / n_lesion) if n_lesion > 0 else float("nan")

        # slice-level
        lesion_slices = set(lz[is_lesion].tolist())
        hit_slices = set(lz[is_hit].tolist())
        lesion_slice_count = len(lesion_slices)
        hit_slice_count = len(hit_slices & lesion_slices)
        missed_slice_count = lesion_slice_count - hit_slice_count

        # continuous hit (병변 z 범위 기준)
        lesion_z_sorted = sorted(lesion_slices)
        hit_z_sorted = sorted(hit_slices & lesion_slices)
        lesion_z_min = lesion_z_sorted[0] if lesion_z_sorted else None
        lesion_z_max = lesion_z_sorted[-1] if lesion_z_sorted else None
        lesion_z_span = (lesion_z_max - lesion_z_min + 1) if lesion_z_sorted else 0
        hit_z_min = hit_z_sorted[0] if hit_z_sorted else None
        hit_z_max = hit_z_sorted[-1] if hit_z_sorted else None
        longest_run = longest_consecutive(hit_z_sorted)
        gaps = gap_count(hit_z_sorted)
        cont_ratio = (longest_run / lesion_slice_count) if lesion_slice_count > 0 else float("nan")

        patient_hit = 1 if n_hit >= 1 else 0
        agg["n_patient_hit"] += patient_hit
        agg["lesion_patch_total"] += n_lesion
        agg["hit_lesion_patch_total"] += n_hit
        agg["lesion_slice_total"] += lesion_slice_count
        agg["hit_lesion_slice_total"] += hit_slice_count

        patient_rows.append({
            "patient_id": pid, "safe_id": safe, "group": grp,
            "stage_split": stage_split_map.get(pid, None),
            "patient_hit": patient_hit,
            "lesion_patch_count": n_lesion,
            "positive_patch_count": n_pos,
            "hit_lesion_patch_count": n_hit,
            "patient_patch_recall": patch_recall,
            "patient_patch_dice": patch_dice,
            "patient_patch_iou": patch_iou,
            "lesion_slice_count": lesion_slice_count,
            "hit_lesion_slice_count": hit_slice_count,
            "missed_lesion_slice_count": missed_slice_count,
            "lesion_slice_recall": (hit_slice_count / lesion_slice_count) if lesion_slice_count > 0 else float("nan"),
            "lesion_z_min": lesion_z_min, "lesion_z_max": lesion_z_max, "lesion_z_span": lesion_z_span,
            "hit_z_min": hit_z_min, "hit_z_max": hit_z_max,
            "longest_consecutive_hit_slices": longest_run,
            "hit_gap_count": gaps,
            "continuous_hit_ratio": cont_ratio,
            "largest_connected_hit_component_size": None,  # 확인 필요(공간 연결성, 다음 단계)
        })

        # slice별 행 (병변 slice만)
        for z in lesion_z_sorted:
            zmask = (lz == z)
            zl = int((is_lesion & zmask).sum())
            zp = int((is_pos & zmask).sum())
            zh = int((is_hit & zmask).sum())
            slice_rows.append({
                "patient_id": pid, "group": grp, "local_z": int(z),
                "lesion_patch_in_slice": zl,
                "positive_patch_in_slice": zp,
                "hit_patch_in_slice": zh,
                "slice_hit": 1 if zh >= 1 else 0,
            })

    # 환자별 통계/약한 환자 목록 계산
    ppdf = pd.DataFrame(patient_rows)

    def _round_records(sub_df, cols, ndigits=4):
        out = []
        for _, r in sub_df.iterrows():
            rec = {}
            for c in cols:
                v = r[c]
                rec[c] = round(float(v), ndigits) if isinstance(v, (int, float, np.floating)) and pd.notna(v) else (None if pd.isna(v) else v)
            out.append(rec)
        return out

    lowest_recall = _round_records(
        ppdf.nsmallest(10, "patient_patch_recall"),
        ["patient_id", "group", "stage_split", "patient_patch_recall"])
    lowest_cont = _round_records(
        ppdf.nsmallest(10, "continuous_hit_ratio"),
        ["patient_id", "group", "stage_split", "continuous_hit_ratio"])
    most_missed = _round_records(
        ppdf.nlargest(10, "missed_lesion_slice_count"),
        ["patient_id", "group", "stage_split", "missed_lesion_slice_count"])

    # no-hit 환자 목록
    no_hit_df = ppdf[ppdf["patient_hit"] == 0]
    no_hit_list = _round_records(
        no_hit_df,
        ["patient_id", "group", "stage_split", "patient_patch_recall", "lesion_patch_count"])

    # summary
    np_ = agg["n_patients"]
    summary = {
        "dataset_profile": args.dataset_profile,
        "p95_threshold": thr,
        "n_patients": np_,
        "patient_hit_rate": (agg["n_patient_hit"] / np_) if np_ else None,
        "micro_lesion_patch_recall": (agg["hit_lesion_patch_total"] / agg["lesion_patch_total"]) if agg["lesion_patch_total"] else None,
        "micro_lesion_slice_recall": (agg["hit_lesion_slice_total"] / agg["lesion_slice_total"]) if agg["lesion_slice_total"] else None,
        "lesion_patch_total": agg["lesion_patch_total"],
        "hit_lesion_patch_total": agg["hit_lesion_patch_total"],
        "lesion_slice_total": agg["lesion_slice_total"],
        "hit_lesion_slice_total": agg["hit_lesion_slice_total"],
        # 환자별 평균/중앙값
        "patient_patch_recall_mean": float(ppdf["patient_patch_recall"].mean()),
        "patient_patch_recall_median": float(ppdf["patient_patch_recall"].median()),
        "patient_patch_dice_mean": float(ppdf["patient_patch_dice"].mean()),
        "patient_patch_iou_mean": float(ppdf["patient_patch_iou"].mean()),
        "lesion_slice_recall_mean": float(ppdf["lesion_slice_recall"].mean()),
        "continuous_hit_ratio_mean": float(ppdf["continuous_hit_ratio"].mean()),
        "continuous_hit_ratio_median": float(ppdf["continuous_hit_ratio"].median()),
        # 약한 환자 목록
        "n_patient_no_hit": int((ppdf["patient_hit"] == 0).sum()),
        "no_hit_patients": no_hit_list,
        "lowest_patient_patch_recall_top10": lowest_recall,
        "lowest_continuous_hit_ratio_top10": lowest_cont,
        "most_missed_lesion_slice_top10": most_missed,
        "patch_connection_note": "largest_connected_hit_component_size는 공간 연결성 구현 복잡 → 확인 필요(다음 단계 분리).",
        "note": "read-only 분석. score 재계산 없음. patch_dice/iou는 patch 집합(개수) 기준이며 pixel-level 아님. 1차 스크리닝이 2차 후보를 충분히 만드는지 판단 근거. 성능 결론 아님.",
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ppdf.to_csv(out_patient_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(slice_rows).to_csv(out_slice_csv, index=False, encoding="utf-8-sig")
    with open(out_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
