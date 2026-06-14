"""
create_lesion_stage_split.py
308명 병변 환자를 stage1_dev / stage2_holdout으로 환자 단위 stratified split한다.

안전 원칙:
- lesion_hit_overlap_by_patient.csv read-only. score/metrics 재계산 없음.
- 기존 결과 수정·삭제 없음. 신규 split CSV / summary JSON만 생성.
- 출력 파일이 이미 있으면 실행 중단 (덮어쓰기 금지).
- 실행 전 ChatGPT 검토 필수.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

HIT_CSV = (
    REPO_ROOT
    / "outputs/position-aware-padim-v1/reports/lesion_hit_overlap_by_patient.csv"
)
OUT_DIR = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/splits"
REPORT_DIR = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/reports"
OUT_CSV = OUT_DIR / "lesion_stage_split_v1_balanced.csv"
OUT_JSON = REPORT_DIR / "lesion_stage_split_v1_balanced_summary.json"

SPLIT_SEED = 42
STAGE1_LABEL = "stage1_dev"
STAGE2_LABEL = "stage2_holdout"

WEAK_CASES = {
    "LUNG1-089",
    "LUNG1-231",
    "LUNG1-372",
    "MSD_lung_043",
    "MSD_lung_079",
    "MSD_lung_096",
}

SIM_COLS = [
    "patient_patch_recall",
    "continuous_hit_ratio",
    "lesion_patch_count",
    "lesion_slice_count",
    "missed_lesion_slice_count",
]

INPUT_COLS = [
    "patient_id",
    "safe_id",
    "group",
    "lesion_patch_count",
    "lesion_slice_count",
    "patient_patch_recall",
    "lesion_slice_recall",
    "continuous_hit_ratio",
    "missed_lesion_slice_count",
]

EXPECTED_N = 308
EXPECTED_NSCLC = 249
EXPECTED_MSD = 59
GROUP_STAGE1_TARGETS: dict = {"NSCLC": 125, "MSD_Lung": 29}
TOTAL_STAGE1_TARGET = 154


def _qbin(s: pd.Series, q: int) -> pd.Series:
    """분위수 구간화. 동점 edge는 duplicates='drop'으로 처리."""
    labels = list(range(q))
    try:
        return pd.qcut(s, q=q, labels=labels, duplicates="drop")
    except Exception:
        return pd.cut(s, bins=max(q, 2), labels=labels[:q], duplicates="drop")


def stratified_half_split(df: pd.DataFrame, seed: int) -> pd.Series:
    """
    group 내 stratified 50:50 split.
    층화 키: patient_patch_recall (3분위) × continuous_hit_ratio (2분위).
    반환: patient_id 기준 매핑 Series (STAGE1_LABEL / STAGE2_LABEL).
    """
    rng = np.random.default_rng(seed)
    label_map = {}

    for grp, sub in df.groupby("group", sort=True):
        sub = sub.copy().reset_index(drop=True)
        n_total = len(sub)
        n1_target = GROUP_STAGE1_TARGETS.get(str(grp), n_total // 2)

        sub["_r_bin"] = _qbin(sub["patient_patch_recall"], q=3)
        sub["_h_bin"] = _qbin(sub["continuous_hit_ratio"], q=2)
        sub["_stratum"] = sub["_r_bin"].astype(str) + "_" + sub["_h_bin"].astype(str)

        stage1_pids = []

        for _, stratum_sub in sub.groupby("_stratum", sort=True):
            pids = stratum_sub["patient_id"].tolist()
            rng.shuffle(pids)
            take = max(1, round(len(pids) * n1_target / n_total))
            take = min(take, len(pids))
            stage1_pids.extend(pids[:take])

        # 중복 제거 (순서 유지)
        seen = set()
        stage1_pids_dedup = []
        for p in stage1_pids:
            if p not in seen:
                seen.add(p)
                stage1_pids_dedup.append(p)
        stage1_pids = stage1_pids_dedup

        all_pids = sub["patient_id"].tolist()
        stage1_set = set(stage1_pids)
        remaining = [p for p in all_pids if p not in stage1_set]

        # stage1 수를 n1_target에 정확히 맞춤
        if len(stage1_pids) < n1_target:
            extra = n1_target - len(stage1_pids)
            rng.shuffle(remaining)
            stage1_pids.extend(remaining[:extra])
            remaining = remaining[extra:]
        elif len(stage1_pids) > n1_target:
            excess = len(stage1_pids) - n1_target
            to_move = stage1_pids[-excess:]
            stage1_pids = stage1_pids[:-excess]
            remaining.extend(to_move)

        for pid in stage1_pids:
            label_map[pid] = STAGE1_LABEL
        for pid in remaining:
            label_map[pid] = STAGE2_LABEL

    return df["patient_id"].map(label_map)


def _find_swap_partner(df: pd.DataFrame, weak_pid: str, partner_stage: str) -> str:
    """partner_stage의 non-weak 중 weak_pid와 같은 group, 가장 유사한 환자 반환. 없으면 빈 문자열."""
    weak_row = df[df["patient_id"] == weak_pid].iloc[0]
    grp = weak_row["group"]
    candidates = df[
        (df["stage_split"] == partner_stage)
        & (df["weak_case_flag"] == 0)
        & (df["group"] == grp)
    ].copy()
    if len(candidates) == 0:
        return ""
    col_max = df[SIM_COLS].max()
    col_min = df[SIM_COLS].min()
    col_range = (col_max - col_min).replace(0, 1.0)
    for col in SIM_COLS:
        candidates[f"_d_{col}"] = (candidates[col] - float(weak_row[col])).abs() / float(col_range[col])
    dist_cols = [f"_d_{col}" for col in SIM_COLS]
    candidates["_dist"] = candidates[dist_cols].sum(axis=1)
    return str(candidates.sort_values("_dist").iloc[0]["patient_id"])


def balance_weak_cases(df: pd.DataFrame) -> tuple:
    """
    Weak case 6명이 3:3이 되도록 같은 group non-weak과 swap 조정.
    - 목표: stage1_weak=3, stage2_weak=3 (최대 4:2까지 허용)
    - 5:1 또는 6:0이면 None 반환 (실패)
    - 반환: (수정된 df, swap_log). 실패 시 (df, None).
    """
    df = df.copy()
    swap_log = []

    for _ in range(10):
        s1_w = df[(df["weak_case_flag"] == 1) & (df["stage_split"] == STAGE1_LABEL)]["patient_id"].tolist()
        s2_w = df[(df["weak_case_flag"] == 1) & (df["stage_split"] == STAGE2_LABEL)]["patient_id"].tolist()
        n1, n2 = len(s1_w), len(s2_w)

        if abs(n1 - n2) <= 1:
            break

        if n1 > n2:
            weak_pid = sorted(s1_w)[0]
            partner_pid = _find_swap_partner(df, weak_pid, STAGE2_LABEL)
            if not partner_pid:
                return df, None
            df.loc[df["patient_id"] == weak_pid, "stage_split"] = STAGE2_LABEL
            df.loc[df["patient_id"] == partner_pid, "stage_split"] = STAGE1_LABEL
            swap_log.append({
                "weak_pid": weak_pid, "weak_moved_to": STAGE2_LABEL,
                "nonweak_pid": partner_pid, "nonweak_moved_to": STAGE1_LABEL,
            })
        else:
            weak_pid = sorted(s2_w)[0]
            partner_pid = _find_swap_partner(df, weak_pid, STAGE1_LABEL)
            if not partner_pid:
                return df, None
            df.loc[df["patient_id"] == weak_pid, "stage_split"] = STAGE1_LABEL
            df.loc[df["patient_id"] == partner_pid, "stage_split"] = STAGE2_LABEL
            swap_log.append({
                "weak_pid": weak_pid, "weak_moved_to": STAGE1_LABEL,
                "nonweak_pid": partner_pid, "nonweak_moved_to": STAGE2_LABEL,
            })

    s1_final = int(df[(df["weak_case_flag"] == 1) & (df["stage_split"] == STAGE1_LABEL)].shape[0])
    s2_final = int(df[(df["weak_case_flag"] == 1) & (df["stage_split"] == STAGE2_LABEL)].shape[0])
    if max(s1_final, s2_final) >= 5:
        return df, None
    return df, swap_log


def main() -> None:
    # --- 출력 파일 덮어쓰기 방지 ---
    existing = [str(p) for p in (OUT_CSV, OUT_JSON) if p.exists()]
    if existing:
        print("[ERROR] 출력 파일이 이미 존재합니다. 덮어쓰기 금지 — 실행 중단:")
        for e in existing:
            print(f"  - {e}")
        sys.exit(1)

    # --- 입력 로드 ---
    df = pd.read_csv(HIT_CSV, encoding="utf-8-sig", usecols=INPUT_COLS)
    print(f"[split] 입력 환자: {len(df)}명")

    # 입력 크기 검증
    if len(df) != EXPECTED_N:
        print(f"[WARN] 입력 환자 수 예상({EXPECTED_N}) 불일치: {len(df)}명")

    grp_counts = df["group"].value_counts()
    print(f"[split] 그룹 분포:\n{grp_counts.to_string()}")

    nsclc_n = int(grp_counts.get("NSCLC", 0))
    msd_n = int(grp_counts.get("MSD_Lung", 0))
    if nsclc_n != EXPECTED_NSCLC:
        print(f"[WARN] NSCLC 수 예상({EXPECTED_NSCLC}) 불일치: {nsclc_n}명")
    if msd_n != EXPECTED_MSD:
        print(f"[WARN] MSD_Lung 수 예상({EXPECTED_MSD}) 불일치: {msd_n}명")

    # --- weak_case_flag ---
    df["weak_case_flag"] = df["patient_id"].isin(WEAK_CASES).astype(int)
    found_weak = df[df["weak_case_flag"] == 1]["patient_id"].tolist()
    not_found = sorted(WEAK_CASES - set(found_weak))
    if not_found:
        print(f"[WARN] 약한 케이스 미발견: {not_found}")

    # --- stratified split ---
    df["stage_split"] = stratified_half_split(df, seed=SPLIT_SEED)

    # --- weak case 균형 조정 (swap) ---
    df, swap_log = balance_weak_cases(df)
    if swap_log is None:
        print("[ERROR] weak case 균형 조정 실패: 같은 group non-weak 없음 또는 5:1 이상. 저장 중단.")
        sys.exit(1)
    if swap_log:
        print(f"[split] weak case swap {len(swap_log)}건 수행")

    # --- 약한 케이스 분산 검사 ---
    weak_df = df[df["weak_case_flag"] == 1].copy()
    weak_dist_detail = (
        weak_df[["patient_id", "group", "stage_split"]]
        .sort_values("group")
        .reset_index(drop=True)
    )
    print(f"[split] 약한 케이스 분포:\n{weak_dist_detail.to_string(index=False)}")

    s1_weak = int((weak_df["stage_split"] == STAGE1_LABEL).sum())
    s2_weak = int((weak_df["stage_split"] == STAGE2_LABEL).sum())
    weak_case_balance_pass = max(s1_weak, s2_weak) <= 4
    print(f"[split] 약한 케이스 균형: stage1={s1_weak}, stage2={s2_weak}, pass={weak_case_balance_pass}")
    if not weak_case_balance_pass:
        print(
            f"[ERROR] 약한 케이스 5:1 이상 편중: stage1={s1_weak}, stage2={s2_weak}. 저장 중단."
        )
        sys.exit(1)

    # --- 출력 컬럼 구성 ---
    out_cols = [
        "patient_id",
        "safe_id",
        "group",
        "stage_split",
        "lesion_patch_count",
        "lesion_slice_count",
        "patient_patch_recall",
        "lesion_slice_recall",
        "continuous_hit_ratio",
        "missed_lesion_slice_count",
        "weak_case_flag",
    ]
    out_df = df[out_cols].copy()
    out_df["split_seed"] = SPLIT_SEED

    # --- 총합 검증 guard ---
    total_s1 = int((out_df["stage_split"] == STAGE1_LABEL).sum())
    total_s2 = int((out_df["stage_split"] == STAGE2_LABEL).sum())
    if total_s1 != TOTAL_STAGE1_TARGET or total_s2 != TOTAL_STAGE1_TARGET:
        print(
            f"[ERROR] split 총합 불일치: stage1={total_s1}, stage2={total_s2}. "
            f"{TOTAL_STAGE1_TARGET}/{TOTAL_STAGE1_TARGET} 필요. 저장 중단."
        )
        sys.exit(1)
    print(f"[split] 총합 검증 통과: stage1={total_s1}, stage2={total_s2}")

    # --- group별 목표 수 검증 ---
    group_split_verified = {}
    for grp in sorted(GROUP_STAGE1_TARGETS):
        grp_sub = out_df[out_df["group"] == grp]
        actual_s1 = int((grp_sub["stage_split"] == STAGE1_LABEL).sum())
        actual_s2 = int((grp_sub["stage_split"] == STAGE2_LABEL).sum())
        exp_s1 = GROUP_STAGE1_TARGETS[grp]
        exp_s2 = int(len(grp_sub)) - exp_s1
        ok = actual_s1 == exp_s1 and actual_s2 == exp_s2
        group_split_verified[str(grp)] = {
            "expected_stage1": exp_s1, "actual_stage1": actual_s1,
            "expected_stage2": exp_s2, "actual_stage2": actual_s2,
            "pass": ok,
        }
        if not ok:
            print(f"[ERROR] group {grp} 목표 불일치: 기대 s1={exp_s1} 실제={actual_s1}. 저장 중단.")
            sys.exit(1)
    print("[split] group별 목표 수 검증 통과")

    # --- 요약 통계 ---
    group_split = {}
    for grp, sub in out_df.groupby("group", sort=True):
        group_split[str(grp)] = {
            "total": int(len(sub)),
            STAGE1_LABEL: int((sub["stage_split"] == STAGE1_LABEL).sum()),
            STAGE2_LABEL: int((sub["stage_split"] == STAGE2_LABEL).sum()),
        }

    weak_case_assignment = {
        str(row["patient_id"]): str(row["stage_split"])
        for _, row in weak_dist_detail.iterrows()
    }

    summary = {
        "split_seed": SPLIT_SEED,
        "total_patients": int(len(out_df)),
        "stage1_dev_total": int((out_df["stage_split"] == STAGE1_LABEL).sum()),
        "stage2_holdout_total": int((out_df["stage_split"] == STAGE2_LABEL).sum()),
        "split_total_verified": {
            "stage1_dev": total_s1,
            "stage2_holdout": total_s2,
            "target_each": TOTAL_STAGE1_TARGET,
            "pass": total_s1 == TOTAL_STAGE1_TARGET and total_s2 == TOTAL_STAGE1_TARGET,
        },
        "group_split": group_split,
        "group_split_verified": group_split_verified,
        "group_stage1_targets": GROUP_STAGE1_TARGETS,
        "weak_case_assignment": weak_case_assignment,
        "weak_case_stage1_count": s1_weak,
        "weak_case_stage2_count": s2_weak,
        "weak_case_balance_pass": weak_case_balance_pass,
        "swap_log": swap_log,
        "stratification_keys": [
            "group",
            "patient_patch_recall (3-quantile bin)",
            "continuous_hit_ratio (2-quantile bin)",
        ],
        "interpretation_note": (
            "stage2_holdout은 완전한 외부 독립 test가 아님. "
            "308명 전체를 1차 분석에 이미 사용했으므로 내부 hold-out 검증으로만 해석. "
            "최종 일반화 성능 주장은 별도 독립 환자 데이터로만 가능."
        ),
        "data_leakage_note": (
            "이미 308명 전체로 PaDiM score를 생성·분석했으므로, "
            "stage2_holdout이 순수한 unseen test가 아님을 명시."
        ),
        "split_basis": (
            "inputs: lesion_hit_overlap_by_patient.csv (read-only). "
            "환자 단위 split. slice/patch 단위 분할 없음."
        ),
    }

    # --- 저장 ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n[split] 저장: {OUT_CSV}")
    print(f"[split] 저장: {OUT_JSON}")


if __name__ == "__main__":
    main()
