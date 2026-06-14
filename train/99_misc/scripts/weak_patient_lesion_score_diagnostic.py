#!/usr/bin/env python3
"""
weak_patient_lesion_score_diagnostic.py

weak/no-hit 환자에서 병변 부위 patch score 진단 분석
- 기존 score CSV read-only 사용
- scoring/metric 재실행 금지
- 기존 파일 수정/삭제 금지
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

V1V2_DIR = REPO_ROOT / "outputs/position-aware-padim-v1/scores/padim_v1/lesion_v2_by_patient"
V2V2_DIR = REPO_ROOT / "outputs/position-aware-padim-v1/scores/padim_v2_roi0_0/lesion_v2_by_patient"

THRESHOLDS = {
    "v1v2": {"p95": 14.377,     "p99": 18.673},
    "v2v2": {"p95": 14.092058,  "p99": 17.763281},
}

OUT_DIR  = REPO_ROOT / "outputs/position-aware-padim-v1/reports_v2_roi0_0_lesion"
OUT_CSV  = OUT_DIR / "weak_patient_lesion_score_diagnostic.csv"
OUT_JSON = OUT_DIR / "weak_patient_lesion_score_diagnostic.json"
OUT_MD   = OUT_DIR / "weak_patient_lesion_score_diagnostic.md"

TARGET_PATIENTS = [
    "LUNG1-156",
    "LUNG1-415",
    "MSD_lung_071",
    "MSD_lung_096",
    "MSD_lung_079",
]


def abort(msg: str) -> None:
    print(f"[ABORT] {msg}", file=sys.stderr)
    sys.exit(1)


def guard_no_overwrite() -> None:
    for p in [OUT_CSV, OUT_JSON, OUT_MD]:
        if p.exists():
            abort(f"출력 파일 이미 존재 (덮어쓰기 금지): {p}")


def load_score_csv(csv_dir: Path, patient_id: str) -> pd.DataFrame:
    p = csv_dir / f"{patient_id}.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def analyze_patient(df: pd.DataFrame, model_key: str) -> dict:
    """환자 1명에 대한 분석 결과 dict 반환"""
    thr = THRESHOLDS[model_key]
    p95 = thr["p95"]
    p99 = thr["p99"]

    total_patches = len(df)
    if total_patches == 0:
        return {"error": "CSV 없음 또는 빈 파일"}

    # 병변 overlap patch: patch_label == 1
    df_lesion = df[df["patch_label"] == 1].copy()
    lesion_count = len(df_lesion)

    if lesion_count == 0:
        # 병변 overlap patch 없음 → grid coverage issue
        return {
            "total_patches": total_patches,
            "lesion_overlap_patch_count": 0,
            "lesion_score_min": None,
            "lesion_score_mean": None,
            "lesion_score_median": None,
            "lesion_score_max": None,
            "lesion_score_p95": None,
            "lesion_score_p99": None,
            "above_p95_count": 0,
            "above_p99_count": 0,
            "above_p95_ratio": 0.0,
            "above_p99_ratio": 0.0,
            "lesion_max_score_global_percentile": None,
            "lesion_max_score_global_rank": None,
            "lesion_in_top10": False,
            "lesion_in_top50": False,
            "lesion_in_top100": False,
            "lesion_in_top500": False,
            "lesion_in_top1000": False,
            "slice_analysis": [],
            "verdict": "D_grid_coverage_issue",
            "verdict_reason": "patch_label==1인 병변 overlap patch 없음",
        }

    scores = df_lesion["padim_score"]

    # 1. 기본 통계
    lesion_score_min    = float(scores.min())
    lesion_score_mean   = float(scores.mean())
    lesion_score_median = float(scores.median())
    lesion_score_max    = float(scores.max())
    lesion_score_p95_val = float(scores.quantile(0.95)) if lesion_count >= 20 else None
    lesion_score_p99_val = float(scores.quantile(0.99)) if lesion_count >= 100 else None

    # 2. threshold 통과 여부
    above_p95 = int((scores >= p95).sum())
    above_p99 = int((scores >= p99).sum())

    # 3. rank 분석 (전체 patch 기준, 높을수록 rank 낮음=1등)
    df_sorted = df.sort_values("padim_score", ascending=False).reset_index(drop=True)
    df_sorted["rank"] = df_sorted.index + 1
    all_scores_sorted = df_sorted["padim_score"].values
    lesion_max_score   = lesion_score_max

    # percentile (전체 기준)
    percentile_val = float(np.mean(df["padim_score"] <= lesion_max_score) * 100)

    # rank (몇 번째로 높은가)
    rank_val = int((df["padim_score"] > lesion_max_score).sum()) + 1

    in_top10   = rank_val <= 10
    in_top50   = rank_val <= 50
    in_top100  = rank_val <= 100
    in_top500  = rank_val <= 500
    in_top1000 = rank_val <= 1000

    # 4. slice 내부 rank 분석
    slice_results = []
    if "local_z" in df.columns:
        lesion_slices = df_lesion["local_z"].unique()
        for z in sorted(lesion_slices):
            df_slice = df[df["local_z"] == z].copy()
            df_slice_sorted = df_slice.sort_values("padim_score", ascending=False).reset_index(drop=True)
            df_slice_lesion = df_slice[df_slice["patch_label"] == 1]

            slice_lesion_max = float(df_slice_lesion["padim_score"].max()) if len(df_slice_lesion) > 0 else None
            if slice_lesion_max is not None:
                slice_rank = int((df_slice["padim_score"] > slice_lesion_max).sum()) + 1
            else:
                slice_rank = None

            n_slice = len(df_slice)
            slice_results.append({
                "local_z":             int(z),
                "slice_total_patches": n_slice,
                "slice_lesion_patches": len(df_slice_lesion),
                "slice_lesion_max_score": slice_lesion_max,
                "slice_rank":          slice_rank,
                "in_slice_top10":      slice_rank <= 10  if slice_rank is not None else False,
                "in_slice_top50":      slice_rank <= 50  if slice_rank is not None else False,
                "in_slice_top100":     slice_rank <= 100 if slice_rank is not None else False,
            })

    # 5. 판정
    if lesion_score_max < p95:
        verdict = "A_lesion_score_low"
        verdict_reason = f"병변 max score {lesion_score_max:.3f} < p95 threshold {p95}"
    elif not in_top100:
        verdict = "B_lesion_score_exists_but_ranked_low"
        verdict_reason = f"병변 max score {lesion_score_max:.3f} >= p95 {p95} 이지만 전체 rank={rank_val} (top100 밖)"
    else:
        # top100 안에 들지만 실패: slice 내부 위치 문제
        # slice별로 병변 patch가 slice top10 밖인지 확인
        slice_missed = [s for s in slice_results if not s["in_slice_top10"]]
        if len(slice_missed) > 0 and len(slice_missed) == len(slice_results):
            verdict = "C_slice_hit_location_miss"
            verdict_reason = "전체 rank는 top100이나 slice 내부 top10에 병변 patch 없음"
        else:
            verdict = "B_lesion_score_exists_but_ranked_low"
            verdict_reason = f"병변 max score {lesion_score_max:.3f}, rank={rank_val}, 일부 slice에서 위치 miss"

    return {
        "total_patches":                  total_patches,
        "lesion_overlap_patch_count":     lesion_count,
        "lesion_score_min":               round(lesion_score_min, 4),
        "lesion_score_mean":              round(lesion_score_mean, 4),
        "lesion_score_median":            round(lesion_score_median, 4),
        "lesion_score_max":               round(lesion_score_max, 4),
        "lesion_score_p95":               round(lesion_score_p95_val, 4) if lesion_score_p95_val is not None else None,
        "lesion_score_p99":               round(lesion_score_p99_val, 4) if lesion_score_p99_val is not None else None,
        "above_p95_count":                above_p95,
        "above_p99_count":                above_p99,
        "above_p95_ratio":                round(above_p95 / lesion_count, 4),
        "above_p99_ratio":                round(above_p99 / lesion_count, 4),
        "lesion_max_score_global_percentile": round(percentile_val, 2),
        "lesion_max_score_global_rank":       rank_val,
        "lesion_in_top10":                in_top10,
        "lesion_in_top50":                in_top50,
        "lesion_in_top100":               in_top100,
        "lesion_in_top500":               in_top500,
        "lesion_in_top1000":              in_top1000,
        "slice_analysis":                 slice_results,
        "verdict":                        verdict,
        "verdict_reason":                 verdict_reason,
    }


def main() -> None:
    guard_no_overwrite()

    for d in [V1V2_DIR, V2V2_DIR]:
        if not d.exists():
            abort(f"score 디렉토리 없음: {d}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows  = []
    json_data = {}

    for patient_id in TARGET_PATIENTS:
        json_data[patient_id] = {}
        for model_key, csv_dir in [("v1v2", V1V2_DIR), ("v2v2", V2V2_DIR)]:
            df = load_score_csv(csv_dir, patient_id)
            if df.empty:
                print(f"[WARN] {patient_id} / {model_key}: CSV 없음")
                result = {"error": "CSV 없음"}
            else:
                result = analyze_patient(df, model_key)

            json_data[patient_id][model_key] = result

            # flat row for CSV (slice_analysis 제외)
            row = {
                "patient_id": patient_id,
                "model":      model_key,
            }
            for k, v in result.items():
                if k != "slice_analysis":
                    row[k] = v
            all_rows.append(row)

            verdict = result.get("verdict", "N/A")
            reason  = result.get("verdict_reason", "")
            cnt     = result.get("lesion_overlap_patch_count", "N/A")
            lmax    = result.get("lesion_score_max", "N/A")
            rank    = result.get("lesion_max_score_global_rank", "N/A")
            print(f"  [{model_key}] {patient_id}: lesion_patches={cnt}, max={lmax}, rank={rank} → {verdict}")

    # ── CSV 저장 ──────────────────────────────────────────
    df_out = pd.DataFrame(all_rows)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"\n[저장] {OUT_CSV}")

    # ── JSON 저장 ─────────────────────────────────────────
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2, default=str)
    print(f"[저장] {OUT_JSON}")

    # ── MD 생성 ───────────────────────────────────────────
    write_md(json_data)
    print(f"[저장] {OUT_MD}")


def write_md(data: dict) -> None:
    thr = THRESHOLDS
    lines = []
    lines.append("# Weak Patient Lesion Score Diagnostic\n")
    lines.append("| patient_id | model | lesion_patches | max_score | rank | top10 | top100 | verdict |")
    lines.append("|---|---|---|---|---|---|---|---|")

    for patient_id, models in data.items():
        for model_key, r in models.items():
            if "error" in r:
                lines.append(f"| {patient_id} | {model_key} | - | - | - | - | - | {r['error']} |")
                continue
            cnt   = r.get("lesion_overlap_patch_count", "-")
            lmax  = r.get("lesion_score_max", "-")
            rank  = r.get("lesion_max_score_global_rank", "-")
            t10   = "✅" if r.get("lesion_in_top10")  else "❌"
            t100  = "✅" if r.get("lesion_in_top100") else "❌"
            verd  = r.get("verdict", "-")
            lmax_str = f"{lmax:.3f}" if isinstance(lmax, float) else str(lmax)
            lines.append(f"| {patient_id} | {model_key} | {cnt} | {lmax_str} | {rank} | {t10} | {t100} | {verd} |")

    lines.append("")
    lines.append("## Threshold 기준")
    lines.append(f"- v1/v2: p95={thr['v1v2']['p95']}, p99={thr['v1v2']['p99']}")
    lines.append(f"- v2/v2: p95={thr['v2v2']['p95']}, p99={thr['v2v2']['p99']}")
    lines.append("")
    lines.append("## 판정 기준")
    lines.append("- **A_lesion_score_low**: 병변 max score < p95 threshold")
    lines.append("- **B_lesion_score_exists_but_ranked_low**: score는 p95 이상이나 전체 rank top100 밖")
    lines.append("- **C_slice_hit_location_miss**: 전체 rank top100이나 slice 내부 위치 miss")
    lines.append("- **D_grid_coverage_issue**: 병변 overlap patch 자체 없음")
    lines.append("")

    lines.append("## 환자별 slice 분석")
    for patient_id, models in data.items():
        lines.append(f"\n### {patient_id}")
        for model_key, r in models.items():
            if "error" in r or "slice_analysis" not in r:
                continue
            lines.append(f"\n**{model_key}** | verdict: `{r.get('verdict', '-')}` | {r.get('verdict_reason', '')}")
            sa = r["slice_analysis"]
            if sa:
                lines.append("")
                lines.append("| local_z | slice_patches | lesion_patches | lesion_max_score | slice_rank | top10 | top50 |")
                lines.append("|---|---|---|---|---|---|---|")
                for s in sa:
                    lms = f"{s['slice_lesion_max_score']:.3f}" if s["slice_lesion_max_score"] is not None else "-"
                    t10 = "✅" if s["in_slice_top10"] else "❌"
                    t50 = "✅" if s["in_slice_top50"] else "❌"
                    lines.append(
                        f"| {s['local_z']} | {s['slice_total_patches']} | {s['slice_lesion_patches']} "
                        f"| {lms} | {s['slice_rank']} | {t10} | {t50} |"
                    )
            else:
                lines.append("  slice 분석 없음")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
