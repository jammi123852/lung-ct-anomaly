#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PaDiM (EfficientNet-B0 + v4_20) p90 threshold + z-track(min_run=2) 후보 기반
환자별 4지표 산출.

후보 패치 정의:
  - padim_score >= THRESHOLD_P90 (12.196394, normal_val patch p90)
  - 동일 위치(y0,x0,y1,x1)에서 연속 local_z run 길이 >= 2 (min_run=2)
  둘 다 만족하는 패치만 "후보".

지표 (모두 후보 집합 기준):
  1) 환자별 병변 히트율  : 후보에 has_lesion_patch=1 패치가 1개 이상인 환자 / 전체 환자
  2) 환자별 병변 포함률  : (병변 z-slice recall) 환자별 [병변 있는 z 중, 그 z에서
                          병변패치가 후보로 검출된 z 비율]의 환자 평균 (macro)
  3) 환자별 패치 포함률  : (병변 patch recall) 환자별 [병변패치 중 후보로 뽑힌 비율]의
                          환자 평균 (macro)
  4) 환자단위 AUROC      : 병변(stage2 154) vs 정상(normal_test 36).
                          환자 score = 후보 패치 max padim_score (후보 없으면 0.0)

입력은 모두 read-only. 출력은 신규 폴더에만 기록.
"""

import os
import sys
import csv
import glob
import json
import math
import datetime

import numpy as np

# ----------------------------- 설정 -----------------------------
THRESHOLD_P90 = 12.196394  # normal_val patch padim_score p90 (v4_20 branch 전용)
MIN_RUN = 2

BASE = "/home/jinhy/project/lung-ct-anomaly"
SCORE_ROOT = os.path.join(
    BASE,
    "experiments/efficientnet_b0_imagenet_chestwall_removed_roi_v1/outputs/scores",
)
LESION_DIR = os.path.join(SCORE_ROOT, "stage2_holdout_by_patient")
NORMAL_DIR = os.path.join(SCORE_ROOT, "normal_test_by_patient")

OUT_DIR = os.path.join(
    BASE,
    "outputs/position-aware-padim-v1/reports/padim_p90_ztrack_patient_metrics_v1",
)

# CSV 컬럼 인덱스 (0-based). 헤더 BOM 대비 이름 기반으로 재확인.
# 정상(normal_test) CSV에는 병변 컬럼(has_lesion_patch)이 없으므로 optional 처리.
COL_NAMES_REQUIRED = ["local_z", "y0", "x0", "y1", "x1", "padim_score"]
COL_NAMES_OPTIONAL = ["has_lesion_patch"]


def resolve_cols(header):
    """헤더(BOM 제거)에서 필요한 컬럼 인덱스를 이름으로 찾는다.
    required는 없으면 에러, optional은 없으면 None."""
    clean = [h.lstrip("﻿").strip() for h in header]
    idx = {}
    for name in COL_NAMES_REQUIRED:
        if name not in clean:
            raise KeyError(f"필수 컬럼 누락: {name} / 실제 헤더={clean}")
        idx[name] = clean.index(name)
    for name in COL_NAMES_OPTIONAL:
        idx[name] = clean.index(name) if name in clean else None
    return idx


def load_patient(path):
    """환자 CSV 1개를 읽어 필요한 배열만 반환. 전체 누적 안 함(메모리 안전)."""
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = resolve_cols(header)
        local_z, y0, x0, y1, x1, has_les, score = [], [], [], [], [], [], []
        for row in reader:
            if not row:
                continue
            local_z.append(int(float(row[idx["local_z"]])))
            y0.append(int(float(row[idx["y0"]])))
            x0.append(int(float(row[idx["x0"]])))
            y1.append(int(float(row[idx["y1"]])))
            x1.append(int(float(row[idx["x1"]])))
            if idx["has_lesion_patch"] is not None:
                has_les.append(int(float(row[idx["has_lesion_patch"]])))
            else:
                has_les.append(0)  # 정상 환자: 병변 패치 없음
            score.append(float(row[idx["padim_score"]]))
    return {
        "local_z": np.asarray(local_z, dtype=np.int32),
        "y0": np.asarray(y0, dtype=np.int32),
        "x0": np.asarray(x0, dtype=np.int32),
        "y1": np.asarray(y1, dtype=np.int32),
        "x1": np.asarray(x1, dtype=np.int32),
        "has_les": np.asarray(has_les, dtype=np.int8),
        "score": np.asarray(score, dtype=np.float64),
    }


def compute_candidate_mask(d):
    """
    후보 패치 boolean mask 반환.
    1) score >= threshold
    2) 동일 (y0,x0,y1,x1) 위치에서 연속 local_z run 길이 >= MIN_RUN
    run 판정은 threshold 통과 패치들 사이에서만 수행한다.
    """
    n = d["score"].shape[0]
    above = d["score"] >= THRESHOLD_P90
    cand = np.zeros(n, dtype=bool)

    # 위치 key별로 threshold 통과 패치의 z를 모아 연속 run 검사
    # key = (y0,x0,y1,x1)
    pos_key = {}
    for i in np.nonzero(above)[0]:
        k = (int(d["y0"][i]), int(d["x0"][i]), int(d["y1"][i]), int(d["x1"][i]))
        pos_key.setdefault(k, []).append((int(d["local_z"][i]), int(i)))

    for k, zlist in pos_key.items():
        zlist.sort()  # z 오름차순
        # 연속 run 그룹화
        run = [zlist[0]]
        for j in range(1, len(zlist)):
            if zlist[j][0] == run[-1][0] + 1:
                run.append(zlist[j])
            else:
                if len(run) >= MIN_RUN:
                    for _, ridx in run:
                        cand[ridx] = True
                run = [zlist[j]]
        if len(run) >= MIN_RUN:
            for _, ridx in run:
                cand[ridx] = True
    return cand


def patient_metrics(d, cand):
    """한 환자의 지표 원자료 계산."""
    has = d["has_les"] == 1
    n_lesion_patch = int(has.sum())

    # (1) hit: 후보 중 병변패치 1개 이상
    hit = bool((cand & has).any())

    # (3) patch recall: 병변패치 중 후보 비율
    if n_lesion_patch > 0:
        patch_recall = float((cand & has).sum()) / n_lesion_patch
    else:
        patch_recall = None

    # (2) lesion z-slice recall: 병변 있는 z 중, 그 z에서 병변패치가 후보로 검출된 z 비율
    lesion_z = set(int(z) for z in d["local_z"][has])
    detected_z = set(int(z) for z in d["local_z"][cand & has])
    if len(lesion_z) > 0:
        z_recall = len(detected_z) / len(lesion_z)
    else:
        z_recall = None

    # (4) AUROC용 환자 score: 후보 패치 max padim_score (후보 없으면 0.0)
    if cand.any():
        pscore = float(d["score"][cand].max())
    else:
        pscore = 0.0

    return {
        "n_patches": int(d["score"].shape[0]),
        "n_lesion_patch": n_lesion_patch,
        "n_candidate": int(cand.sum()),
        "n_candidate_lesion": int((cand & has).sum()),
        "hit": hit,
        "patch_recall": patch_recall,
        "z_recall": z_recall,
        "patient_score": pscore,
    }


def auroc(scores_pos, scores_neg):
    """Mann-Whitney U 기반 AUROC (tie=0.5). 외부 라이브러리 불필요."""
    pos = np.asarray(scores_pos, dtype=np.float64)
    neg = np.asarray(scores_neg, dtype=np.float64)
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return None
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), dtype=np.float64)
    sorted_v = allv[order]
    i = 0
    while i < len(sorted_v):
        j = i
        while j + 1 < len(sorted_v) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based 평균 순위
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    rank_pos_sum = ranks[:n_pos].sum()
    u_pos = rank_pos_sum - n_pos * (n_pos + 1) / 2.0
    return float(u_pos / (n_pos * n_neg))


def process_dir(score_dir, label):
    """폴더 내 모든 환자 CSV 처리. (per-patient rows, error rows) 반환."""
    rows, errors = [], []
    files = sorted(glob.glob(os.path.join(score_dir, "*.csv")))
    for path in files:
        pid = os.path.splitext(os.path.basename(path))[0]
        try:
            d = load_patient(path)
            cand = compute_candidate_mask(d)
            m = patient_metrics(d, cand)
            m["patient_id"] = pid
            m["group"] = label
            rows.append(m)
        except Exception as e:  # noqa
            errors.append({"patient_id": pid, "group": label, "error": repr(e)})
    return rows, errors


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = datetime.datetime.now()

    lesion_rows, lesion_err = process_dir(LESION_DIR, "stage2_holdout_lesion")
    normal_rows, normal_err = process_dir(NORMAL_DIR, "normal_test")
    all_rows = lesion_rows + normal_rows
    all_err = lesion_err + normal_err

    # ---- per-patient CSV ----
    pp_path = os.path.join(OUT_DIR, "patient_metrics.csv")
    fields = ["group", "patient_id", "n_patches", "n_lesion_patch",
              "n_candidate", "n_candidate_lesion", "hit",
              "patch_recall", "z_recall", "patient_score"]
    with open(pp_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k) for k in fields})

    # ---- error CSV ----
    err_path = os.path.join(OUT_DIR, "errors.csv")
    with open(err_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group", "patient_id", "error"])
        w.writeheader()
        for r in all_err:
            w.writerow(r)

    # ---- 집계 ----
    n_lesion = len(lesion_rows)
    hit_count = sum(1 for r in lesion_rows if r["hit"])
    hit_rate = hit_count / n_lesion if n_lesion else None

    z_recalls = [r["z_recall"] for r in lesion_rows if r["z_recall"] is not None]
    patch_recalls = [r["patch_recall"] for r in lesion_rows if r["patch_recall"] is not None]
    lesion_z_recall_macro = float(np.mean(z_recalls)) if z_recalls else None
    lesion_patch_recall_macro = float(np.mean(patch_recalls)) if patch_recalls else None

    pos_scores = [r["patient_score"] for r in lesion_rows]
    neg_scores = [r["patient_score"] for r in normal_rows]
    patient_auroc = auroc(pos_scores, neg_scores)

    summary = {
        "config": {
            "threshold_p90": THRESHOLD_P90,
            "threshold_source": "normal_val patch padim_score p90 (v4_20 branch)",
            "min_run": MIN_RUN,
            "candidate_def": "padim_score>=p90 AND same-(y0,x0,y1,x1) consecutive local_z run>=2",
            "lesion_set": "stage2_holdout (154)",
            "normal_set_for_auroc": "normal_test (36)",
            "patient_score_for_auroc": "max padim_score among candidate patches (0.0 if none)",
        },
        "counts": {
            "n_lesion_patients": n_lesion,
            "n_normal_patients": len(normal_rows),
            "n_lesion_with_lesion_patch": len(patch_recalls),
            "n_errors": len(all_err),
        },
        "metrics": {
            "patient_lesion_hit_rate": hit_rate,
            "patient_lesion_z_slice_recall_macro": lesion_z_recall_macro,
            "patient_lesion_patch_recall_macro": lesion_patch_recall_macro,
            "patient_auroc": patient_auroc,
        },
        "runtime_sec": (datetime.datetime.now() - t0).total_seconds(),
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary["metrics"], indent=2, ensure_ascii=False))
    print(f"\n[OK] out: {OUT_DIR}")
    print(f"  patients lesion={n_lesion} normal={len(normal_rows)} errors={len(all_err)}")


if __name__ == "__main__":
    main()
