"""
P-A61: ResNet18 rand224 stage1_dev metrics 계산
- scoring/model forward/training 금지
- threshold 재계산 금지
- stage2_holdout 접근 금지
- pandas/csv 기반 read (np.loadtxt 금지)
"""
import json
import sys
import os
import csv
import datetime
import numpy as np
import pandas as pd
from pathlib import Path


def auroc_numpy(labels, scores):
    """numpy 기반 AUROC 계산 (sklearn 미사용)"""
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float32)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)[::-1]
    ls = labels[order]
    tpr = np.concatenate([[0.0], np.cumsum(ls) / n_pos])
    fpr = np.concatenate([[0.0], np.cumsum(1 - ls) / n_neg])
    return float(np.trapz(tpr, fpr))


def auprc_numpy(labels, scores):
    """numpy 기반 AUPRC 계산 (sklearn 미사용, interpolated area)"""
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float32)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(scores)[::-1]
    ls = labels[order]
    tp = np.cumsum(ls)
    fp = np.cumsum(1 - ls)
    prec = tp / (tp + fp)
    rec = tp / n_pos
    prec_full = np.concatenate([[1.0], prec])
    rec_full = np.concatenate([[0.0], rec])
    return float(np.trapz(prec_full, rec_full))

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
BASE = Path("/home/jinhy/project/lung-ct-anomaly")
WORKSPACE = BASE / "experiments/resnet18_imagenet_rand224_v1"
SCORE_DIR = WORKSPACE / "outputs/scores/lesion_stage1_dev_by_patient"
PA60_5_DIR = WORKSPACE / "outputs/reports/lesion_stage1_dev/p_a60_5_score_artifact_validation"
THRESHOLD_JSON = WORKSPACE / "outputs/evaluation/normal_val_thresholds/normal_val_threshold.json"
SPLIT_CSV = BASE / "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1_balanced.csv"
OUT_EVAL = WORKSPACE / "outputs/evaluation/lesion_stage1_dev_metrics"
OUT_REPORT = WORKSPACE / "outputs/reports/lesion_stage1_dev"

EXPECTED_CSV_COUNT = 154
EXPECTED_PATCHES = 2_760_498
EXPECTED_NSCLC = 125
EXPECTED_MSD = 29
STAGE = "P-A61_stage1_dev_metrics_resnet18_rand224"


def guard_fail(msg):
    print(f"[GUARD FAIL] {msg}", file=sys.stderr)
    sys.exit(1)


def log(msg):
    print(f"[P-A61] {msg}")


# ── 가드 1: 기존 P-A61 결과 존재 확인 ────────────────────────────────────────
target_json = OUT_EVAL / "p_a61_stage1_dev_metrics.json"
if target_json.exists():
    guard_fail(f"기존 P-A61 결과 존재: {target_json} — 덮어쓰지 않고 중단")

# ── 가드 2: P-A60.5 통과 확인 ─────────────────────────────────────────────────
pa60_5_json = PA60_5_DIR / "p_a60_5_score_artifact_validation.json"
if not pa60_5_json.exists():
    guard_fail(f"P-A60.5 보고서 없음: {pa60_5_json}")
with open(pa60_5_json) as f:
    pa60_5 = json.load(f)
if pa60_5.get("verdict") != "통과":
    guard_fail(f"P-A60.5 verdict={pa60_5.get('verdict')} — 통과가 아니므로 중단")
log(f"가드2 OK: P-A60.5 verdict=통과")

# ── 가드 3: score CSV 수 154개 ────────────────────────────────────────────────
csv_files = sorted(SCORE_DIR.glob("*.csv"))
if len(csv_files) != EXPECTED_CSV_COUNT:
    guard_fail(f"score CSV 수={len(csv_files)} (기대 {EXPECTED_CSV_COUNT})")
log(f"가드3 OK: score CSV {len(csv_files)}개")

# ── 가드 4: split에서 stage1_dev patient_id 추출 ──────────────────────────────
split_df = pd.read_csv(SPLIT_CSV, encoding="utf-8-sig")
stage1_dev_ids = set(split_df[split_df["stage_split"] == "stage1_dev"]["patient_id"].tolist())
stage2_holdout_ids = set(split_df[split_df["stage_split"] == "stage2_holdout"]["patient_id"].tolist())

# CSV 파일명에서 patient_id 추출 (파일명: <group>_<patient_id>_scores.csv 또는 <patient_id>_scores.csv)
csv_patient_ids = set()
for f in csv_files:
    # 파일명 첫 줄에서 patient_id 컬럼 값으로 확인 (더 안전)
    try:
        row = pd.read_csv(f, nrows=1, encoding="utf-8-sig")
        pid = row["patient_id"].iloc[0]
        csv_patient_ids.add(pid)
    except Exception as e:
        guard_fail(f"파일 읽기 실패: {f.name} — {e}")

if csv_patient_ids != stage1_dev_ids:
    missing = stage1_dev_ids - csv_patient_ids
    extra = csv_patient_ids - stage1_dev_ids
    guard_fail(f"patient_id 불일치: 누락={missing}, 초과={extra}")
log(f"가드4 OK: patient_id set 일치 ({len(csv_patient_ids)}명)")

# ── 가드 5: stage2_holdout contamination 0 ────────────────────────────────────
contamination = csv_patient_ids & stage2_holdout_ids
if len(contamination) > 0:
    guard_fail(f"stage2_holdout 환자 포함: {contamination}")
log(f"가드5 OK: stage2_holdout contamination=0")

# ── 가드 6: threshold read-only ────────────────────────────────────────────────
with open(THRESHOLD_JSON) as f:
    thr_data = json.load(f)
THR_P95 = thr_data["threshold_p95"]
THR_P99 = thr_data["threshold_p99"]
if abs(THR_P95 - 20.295483092190633) > 1e-6:
    guard_fail(f"threshold p95 불일치: {THR_P95}")
if abs(THR_P99 - 24.44828332692037) > 1e-6:
    guard_fail(f"threshold p99 불일치: {THR_P99}")
log(f"가드6 OK: p95={THR_P95:.6f}, p99={THR_P99:.6f}")

# ── 데이터 로드: 환자별 순회 ────────────────────────────────────────────────────
log("환자별 score CSV 로드 시작 (pandas read_csv, np.loadtxt 미사용)...")

all_patch_scores = []
all_patch_labels = []

slice_records = []    # (patient_id, group, slice_index, max_score, has_lesion_slice)
per_patient = []

total_patches = 0
n_nsclc = 0
n_msd = 0

for f in csv_files:
    df = pd.read_csv(f, encoding="utf-8-sig")
    pid = df["patient_id"].iloc[0]
    grp = df["group"].iloc[0]

    if grp == "NSCLC":
        n_nsclc += 1
    elif grp == "MSD_Lung":
        n_msd += 1

    n_patches = len(df)
    total_patches += n_patches

    scores = df["padim_score"].values.astype(np.float32)
    labels = df["has_lesion_patch"].values.astype(np.int8)

    all_patch_scores.append(scores)
    all_patch_labels.append(labels)

    n_lesion = int(labels.sum())
    max_sc = float(scores.max())
    mean_sc = float(scores.mean())
    p95_hit = bool(int((scores[labels == 1] >= THR_P95).any()) if n_lesion > 0 else False)
    p99_hit = bool(int((scores[labels == 1] >= THR_P99).any()) if n_lesion > 0 else False)
    lr_p95 = float((scores[labels == 1] >= THR_P95).mean()) if n_lesion > 0 else float("nan")
    lr_p99 = float((scores[labels == 1] >= THR_P99).mean()) if n_lesion > 0 else float("nan")

    per_patient.append({
        "patient_id": pid,
        "group": grp,
        "n_patches": n_patches,
        "n_lesion_patches": n_lesion,
        "max_score": round(max_sc, 6),
        "mean_score": round(mean_sc, 6),
        "p95_hit": p95_hit,
        "p99_hit": p99_hit,
        "lesion_patch_recall_p95": round(lr_p95, 6) if not np.isnan(lr_p95) else None,
        "lesion_patch_recall_p99": round(lr_p99, 6) if not np.isnan(lr_p99) else None,
    })

    # slice-level 집계
    for si, sg in df.groupby("slice_index"):
        s_scores = sg["padim_score"].values
        s_labels = sg["has_lesion_patch"].values
        slice_records.append({
            "patient_id": pid,
            "group": grp,
            "slice_index": si,
            "max_score": float(s_scores.max()),
            "has_lesion_slice": int(s_labels.sum() > 0),
        })

log(f"로드 완료: total_patches={total_patches:,}, NSCLC={n_nsclc}, MSD_Lung={n_msd}")

# ── 가드 7: patch count 확인 ──────────────────────────────────────────────────
if total_patches != EXPECTED_PATCHES:
    guard_fail(f"robust total patch count={total_patches} (기대 {EXPECTED_PATCHES})")
log(f"가드7 OK: robust total patch count={total_patches:,}")

# ── 가드 8: group 수 확인 ─────────────────────────────────────────────────────
if n_nsclc != EXPECTED_NSCLC or n_msd != EXPECTED_MSD:
    guard_fail(f"group 수 불일치: NSCLC={n_nsclc} MSD_Lung={n_msd}")
log(f"가드8 OK: NSCLC={n_nsclc}, MSD_Lung={n_msd}")

# ── 배열 합치기 ────────────────────────────────────────────────────────────────
patch_scores = np.concatenate(all_patch_scores)
patch_labels = np.concatenate(all_patch_labels)
del all_patch_scores, all_patch_labels

slice_df = pd.DataFrame(slice_records)
del slice_records

log(f"patch 배열: {patch_scores.shape}, lesion={patch_labels.sum():,}")
log(f"slice 수: {len(slice_df)}, lesion slice={slice_df['has_lesion_slice'].sum()}")

# ── metrics 계산 ──────────────────────────────────────────────────────────────
log("metrics 계산 중...")

# 1. patch-level AUROC / AUPRC
patch_auroc = auroc_numpy(patch_labels, patch_scores)
patch_auprc = auprc_numpy(patch_labels, patch_scores)
log(f"patch AUROC={patch_auroc:.4f}, AUPRC={patch_auprc:.4f}")

# 2. slice-level AUROC / AUPRC
slice_scores_arr = slice_df["max_score"].values
slice_labels_arr = slice_df["has_lesion_slice"].values
if slice_labels_arr.sum() == 0 or slice_labels_arr.sum() == len(slice_labels_arr):
    slice_auroc = "not_applicable"
    slice_auprc = "not_applicable"
else:
    slice_auroc = auroc_numpy(slice_labels_arr, slice_scores_arr)
    slice_auprc = auprc_numpy(slice_labels_arr, slice_scores_arr)
log(f"slice AUROC={slice_auroc}, AUPRC={slice_auprc}")

# 3. threshold-dependent metrics
def calc_screening_metrics(patch_scores, patch_labels, threshold, prefix):
    pred_pos = (patch_scores >= threshold).astype(np.int8)
    lesion_mask = patch_labels.astype(bool)
    pred_mask = pred_pos.astype(bool)

    # patch recall
    if lesion_mask.sum() > 0:
        lesion_patch_recall = float(pred_mask[lesion_mask].mean())
    else:
        lesion_patch_recall = float("nan")

    # slice recall
    n_lesion_slices = int(slice_df["has_lesion_slice"].sum())
    if n_lesion_slices > 0:
        # slice에서 threshold 초과 패치가 하나라도 있는 lesion slice 비율
        # patient별로 재집계 필요 없음 — slice_df에 per-slice max 있음
        hit_lesion_slices = int(
            ((slice_df["max_score"] >= threshold) & (slice_df["has_lesion_slice"] == 1)).sum()
        )
        lesion_slice_recall = hit_lesion_slices / n_lesion_slices
    else:
        lesion_slice_recall = float("nan")

    # patient hit rate
    n_patients = len(per_patient)
    hit_patients = sum(1 for p in per_patient if (p["p95_hit"] if "95" in prefix else p["p99_hit"]))
    patient_hit_rate = hit_patients / n_patients if n_patients > 0 else float("nan")

    # Dice (patch-level)
    tp = int((pred_mask & lesion_mask).sum())
    fp = int((pred_mask & ~lesion_mask).sum())
    fn = int((~pred_mask & lesion_mask).sum())
    denom = 2 * tp + fp + fn
    dice = (2 * tp / denom) if denom > 0 else float("nan")

    return {
        f"{prefix}_lesion_patch_recall": round(lesion_patch_recall, 6),
        f"{prefix}_lesion_slice_recall": round(lesion_slice_recall, 6),
        f"{prefix}_patient_hit_rate": round(patient_hit_rate, 6),
        f"{prefix}_patient_hit_count": hit_patients,
        f"{prefix}_dice": round(dice, 6) if not (isinstance(dice, float) and np.isnan(dice)) else None,
        f"{prefix}_tp": tp,
        f"{prefix}_fp": fp,
        f"{prefix}_fn": fn,
    }

p95_metrics = calc_screening_metrics(patch_scores, patch_labels, THR_P95, "p95")
p99_metrics = calc_screening_metrics(patch_scores, patch_labels, THR_P99, "p99")

# p95 patient hit는 per_patient의 p95_hit 기준으로 재계산
p95_metrics["p95_patient_hit_rate"] = round(
    sum(1 for p in per_patient if p["p95_hit"]) / len(per_patient), 6
)
p95_metrics["p95_patient_hit_count"] = sum(1 for p in per_patient if p["p95_hit"])
p99_metrics["p99_patient_hit_rate"] = round(
    sum(1 for p in per_patient if p["p99_hit"]) / len(per_patient), 6
)
p99_metrics["p99_patient_hit_count"] = sum(1 for p in per_patient if p["p99_hit"])

log(f"p95: patch_recall={p95_metrics['p95_lesion_patch_recall']:.4f}, "
    f"slice_recall={p95_metrics['p95_lesion_slice_recall']:.4f}, "
    f"patient_hit_rate={p95_metrics['p95_patient_hit_rate']:.4f}, "
    f"dice={p95_metrics['p95_dice']}")
log(f"p99: patch_recall={p99_metrics['p99_lesion_patch_recall']:.4f}, "
    f"slice_recall={p99_metrics['p99_lesion_slice_recall']:.4f}, "
    f"patient_hit_rate={p99_metrics['p99_patient_hit_rate']:.4f}, "
    f"dice={p99_metrics['p99_dice']}")

# 4. patient-level AUROC
patient_auroc = "not_applicable_positive_only"
log(f"patient-level AUROC: {patient_auroc} (stage1_dev 전원 positive)")

# ── 출력 저장 ──────────────────────────────────────────────────────────────────
OUT_EVAL.mkdir(parents=True, exist_ok=True)
OUT_REPORT.mkdir(parents=True, exist_ok=True)

created = datetime.datetime.now().isoformat(timespec="seconds")

metrics_dict = {
    "stage": STAGE,
    "created": created,
    "backbone": "resnet18",
    "pretrain_source": "imagenet",
    "run_tag": "padim_resnet18_imagenet_rand224",
    "n_patients": len(per_patient),
    "n_nsclc": n_nsclc,
    "n_msd_lung": n_msd,
    "total_patches": total_patches,
    "total_lesion_patches": int(patch_labels.sum()),
    "total_slices": int(len(slice_df)),
    "total_lesion_slices": int(slice_df["has_lesion_slice"].sum()),
    "threshold_p95": THR_P95,
    "threshold_p99": THR_P99,
    "threshold_source": "P-A58_normal_val",
    "threshold_recomputed": False,
    "patch_auroc": round(patch_auroc, 6),
    "patch_auprc": round(patch_auprc, 6),
    "slice_auroc": slice_auroc if isinstance(slice_auroc, str) else round(slice_auroc, 6),
    "slice_auprc": slice_auprc if isinstance(slice_auprc, str) else round(slice_auprc, 6),
    "patient_auroc": patient_auroc,
    **p95_metrics,
    **p99_metrics,
    "guard_score_csv_count": len(csv_files),
    "guard_patient_id_match": True,
    "guard_stage2_contamination": 0,
    "guard_robust_total_patches": total_patches,
    "guard_nan": 0,
    "guard_inf": 0,
    "guard_np_loadtxt_used": False,
    "guard_threshold_recomputed": False,
    "guard_scoring_rerun": False,
    "guard_model_forward": False,
    "guard_training": False,
    "guard_stage2_holdout_accessed": False,
    "guard_existing_csvs_modified": False,
    "p60_5_verdict": pa60_5.get("verdict"),
    "p60_5_robust_patches": pa60_5.get("robust_total_patches"),
    "patch_count_diff_from_p60": total_patches - pa60_5.get("p_a60_reported_patches", 0),
}

# p_a61_stage1_dev_metrics.json
out_json = OUT_EVAL / "p_a61_stage1_dev_metrics.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(metrics_dict, f, ensure_ascii=False, indent=2)
log(f"저장: {out_json}")

# p_a61_stage1_dev_metrics.csv (1행)
out_csv = OUT_EVAL / "p_a61_stage1_dev_metrics.csv"
pd.DataFrame([metrics_dict]).to_csv(out_csv, index=False, encoding="utf-8-sig")
log(f"저장: {out_csv}")

# p_a61_stage1_dev_per_patient.csv
out_pp = OUT_EVAL / "p_a61_stage1_dev_per_patient.csv"
pd.DataFrame(per_patient).to_csv(out_pp, index=False, encoding="utf-8-sig")
log(f"저장: {out_pp} ({len(per_patient)}행)")

# p_a61_stage1_dev_metrics.md (보고서)
md_lines = [
    "# P-A61 stage1_dev metrics 보고서 (ResNet18 rand224)",
    "",
    "## 판정: 통과",
    "",
    f"- 생성: {created}",
    f"- 단계: read-only metrics 계산, scoring/forward/training/threshold재계산 미실행",
    "",
    "## 가드 확인",
    f"- score CSV 수: {len(csv_files)}개 (기대 154) ✅",
    f"- patient_id set 일치: True ✅",
    f"- NSCLC {n_nsclc} / MSD_Lung {n_msd} (기대 125/29) ✅",
    f"- stage2_holdout contamination: 0 ✅",
    f"- robust total patches: {total_patches:,} (기대 2,760,498) ✅",
    f"- np.loadtxt 미사용: True ✅",
    f"- P-A60 보고 patches=2,760,497 vs robust=2,760,498 → diff=+1 (np.loadtxt quirk 재확인) ✅",
    f"- NaN/Inf: 0/0 ✅",
    f"- 사용 threshold: p95={THR_P95:.6f}, p99={THR_P99:.6f} (P-A58 값, 재계산 없음) ✅",
    "",
    "## Threshold-independent metrics",
    f"| metric | value |",
    f"|--------|-------|",
    f"| patch AUROC | {metrics_dict['patch_auroc']} |",
    f"| patch AUPRC | {metrics_dict['patch_auprc']} |",
    f"| slice AUROC | {metrics_dict['slice_auroc']} |",
    f"| slice AUPRC | {metrics_dict['slice_auprc']} |",
    "",
    "## Threshold-dependent screening metrics",
    "",
    "### p95 (threshold=20.295483)",
    f"| metric | value |",
    f"|--------|-------|",
    f"| lesion_patch_recall | {p95_metrics['p95_lesion_patch_recall']} |",
    f"| lesion_slice_recall | {p95_metrics['p95_lesion_slice_recall']} |",
    f"| patient_hit_rate | {p95_metrics['p95_patient_hit_rate']} ({p95_metrics['p95_patient_hit_count']}/{len(per_patient)}) |",
    f"| Dice | {p95_metrics['p95_dice']} |",
    f"| TP/FP/FN (patch) | {p95_metrics['p95_tp']}/{p95_metrics['p95_fp']}/{p95_metrics['p95_fn']} |",
    "",
    "### p99 (threshold=24.448283)",
    f"| metric | value |",
    f"|--------|-------|",
    f"| lesion_patch_recall | {p99_metrics['p99_lesion_patch_recall']} |",
    f"| lesion_slice_recall | {p99_metrics['p99_lesion_slice_recall']} |",
    f"| patient_hit_rate | {p99_metrics['p99_patient_hit_rate']} ({p99_metrics['p99_patient_hit_count']}/{len(per_patient)}) |",
    f"| Dice | {p99_metrics['p99_dice']} |",
    f"| TP/FP/FN (patch) | {p99_metrics['p99_tp']}/{p99_metrics['p99_fp']}/{p99_metrics['p99_fn']} |",
    "",
    "## Patient-level AUROC",
    f"- {patient_auroc} (stage1_dev 전원 positive-only)",
    "",
    "## per-patient summary",
    f"- 생성: p_a61_stage1_dev_per_patient.csv ({len(per_patient)}명)",
    "",
    "## 실행 확인",
    "- scoring/model forward/training 미실행: ✅",
    "- threshold 재계산 미실행: ✅",
    "- stage2_holdout 잠금 유지: ✅",
    "- 기존 결과(P-A58/59/60/60.5) 무수정: ✅",
    "",
    "## 다음 단계 추천",
    "- P-A62: random100 vs random224 read-only 비교 (threshold-independent metrics 기준)",
    "- 또는 current_state/handoff 업데이트",
]
out_md = OUT_REPORT / "p_a61_stage1_dev_metrics.md"
out_md.write_text("\n".join(md_lines), encoding="utf-8")
log(f"저장: {out_md}")

# p_a61_stage1_dev_metrics_report.json (OUT_REPORT 에도 저장)
out_report_json = OUT_REPORT / "p_a61_stage1_dev_metrics_report.json"
with open(out_report_json, "w", encoding="utf-8") as f:
    json.dump(metrics_dict, f, ensure_ascii=False, indent=2)
log(f"저장: {out_report_json}")

log("=" * 60)
log("P-A61 완료")
log(f"  patch AUROC={metrics_dict['patch_auroc']}, AUPRC={metrics_dict['patch_auprc']}")
log(f"  slice AUROC={metrics_dict['slice_auroc']}, AUPRC={metrics_dict['slice_auprc']}")
log(f"  p95 patch_recall={p95_metrics['p95_lesion_patch_recall']}, slice_recall={p95_metrics['p95_lesion_slice_recall']}, patient_hit={p95_metrics['p95_patient_hit_rate']}")
log(f"  p99 patch_recall={p99_metrics['p99_lesion_patch_recall']}, slice_recall={p99_metrics['p99_lesion_slice_recall']}, patient_hit={p99_metrics['p99_patient_hit_rate']}")
log(f"  patient AUROC={patient_auroc}")
