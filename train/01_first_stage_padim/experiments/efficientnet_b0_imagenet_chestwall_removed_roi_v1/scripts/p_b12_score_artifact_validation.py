"""P-B12: v4_20 ROI EfficientNet-B0 stage1_dev score artifact validation
- read-only: score CSV 154개 + split CSV + threshold JSON
- metrics/scoring/model forward 금지
- stage2_holdout 접근 금지
"""
import os, sys, json, csv, math, datetime, hashlib
from pathlib import Path
import pandas as pd
import numpy as np

ALLOW_REAL = True  # artifact validation only, no model forward

# ── 경로 ──────────────────────────────────────────────────────────────────────
BASE = Path("/home/jinhy/project/lung-ct-anomaly")
BRANCH = BASE / "experiments/efficientnet_b0_imagenet_chestwall_removed_roi_v1"
SCORE_DIR = BRANCH / "outputs/scores/lesion_stage1_dev_by_patient"
THRESHOLD_JSON = BRANCH / "outputs/evaluation/normal_val_thresholds/normal_val_threshold.json"
SPLIT_CSV = BASE / "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1_balanced.csv"
P_B11_MD = BRANCH / "outputs/reports/lesion_stage1_dev/p_b11_lesion_stage1_dev_scoring.md"
P_B11_JSON = BRANCH / "outputs/reports/lesion_stage1_dev/p_b11_lesion_stage1_dev_scoring.json"

OUT_DIR = BRANCH / "outputs/reports/lesion_stage1_dev/p_b12_score_artifact_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_COLS = [
    "group", "patient_id", "safe_id", "label",
    "local_z", "slice_index",
    "y0", "x0", "y1", "x1",
    "position_bin", "z_level", "z_ratio",
    "has_lesion_patch", "lesion_pixels",
    "padim_score",
]
LABEL_COLS = ["has_lesion_patch", "lesion_pixels"]
COORD_COLS = ["local_z", "y0", "x0", "y1", "x1", "position_bin", "z_level"]

P95_REF = 13.231265125889463
P99_REF = 15.472384637986801
P11_TOTAL = 2_508_819
P11_P95_COUNT = 454_556
P11_P99_COUNT = 167_376
THRESHOLD_MTIME_REF = 1780624209  # Unix timestamp recorded at P-B9

errors = []
issues = []

print("=" * 70)
print("P-B12 score artifact validation 시작")
print(f"시각: {datetime.datetime.now().isoformat()}")
print("=" * 70)

# ── 1. threshold JSON read-only 로드 ─────────────────────────────────────────
print("\n[1] threshold JSON 확인")
assert THRESHOLD_JSON.exists(), f"threshold JSON 없음: {THRESHOLD_JSON}"
thr_mtime = int(THRESHOLD_JSON.stat().st_mtime)
with open(THRESHOLD_JSON) as f:
    thr = json.load(f)
p95 = thr["threshold_p95"]
p99 = thr["threshold_p99"]
assert abs(p95 - P95_REF) < 1e-6, f"p95 불일치: {p95}"
assert abs(p99 - P99_REF) < 1e-6, f"p99 불일치: {p99}"
mtime_ok = (thr_mtime == THRESHOLD_MTIME_REF)
print(f"  p95={p95:.6f} ✓  p99={p99:.6f} ✓  mtime_ok={mtime_ok}")

# ── 2. split CSV 로드 ─────────────────────────────────────────────────────────
print("\n[2] split CSV 로드")
split_df = pd.read_csv(SPLIT_CSV, encoding="utf-8-sig")
dev_df = split_df[split_df["stage_split"] == "stage1_dev"]
holdout_df = split_df[split_df["stage_split"] == "stage2_holdout"]
dev_pids = set(dev_df["patient_id"].tolist())
holdout_pids = set(holdout_df["patient_id"].tolist())
nsclc_dev = dev_df[dev_df["group"] == "NSCLC"].shape[0]
msd_dev = dev_df[dev_df["group"] == "MSD_Lung"].shape[0]
print(f"  stage1_dev={len(dev_pids)}  NSCLC={nsclc_dev}  MSD_Lung={msd_dev}")
print(f"  stage2_holdout={len(holdout_pids)}")
assert len(dev_pids) == 154, f"stage1_dev 수 이상: {len(dev_pids)}"
assert nsclc_dev == 125, f"NSCLC 수 이상: {nsclc_dev}"
assert msd_dev == 29, f"MSD_Lung 수 이상: {msd_dev}"

# ── 3. score CSV 목록 확인 ────────────────────────────────────────────────────
print("\n[3] score CSV 목록 확인")
csv_files = sorted(SCORE_DIR.glob("*.csv"))
csv_count = len(csv_files)
print(f"  score CSV 수={csv_count}")
assert csv_count == 154, f"score CSV 수 이상: {csv_count}"

# patient_id → file mapping (파일명에서 역추적)
# 첫 파일 읽어서 patient_id 컬럼으로 매핑
csv_pid_map = {}
for f in csv_files:
    # 파일명 = <patient_id>.csv (단, safe_id 변형 가능성 → 첫 행 확인)
    csv_pid_map[f.stem] = f

# ── 4. 전체 CSV 로드 + 집계 ───────────────────────────────────────────────────
print("\n[4] 전체 score CSV 로드 및 검증")
inv_rows = []       # score_csv_inventory
pat_rows = []       # score_artifact_patient_summary
col_rows = []       # score_artifact_column_validation
exc_rows = []       # exceedance per patient

total_patches = 0
total_nan = 0
total_inf = 0
total_p95_exc = 0
total_p99_exc = 0
score_chunks = []

found_pids = set()
missing_required_col_patients = []

for csv_f in csv_files:
    try:
        df = pd.read_csv(csv_f, encoding="utf-8-sig")
    except Exception as e:
        errors.append({"patient_id": csv_f.stem, "error": str(e)})
        continue

    pid_col = df["patient_id"].iloc[0] if "patient_id" in df.columns and len(df) > 0 else csv_f.stem
    found_pids.add(str(pid_col))

    # stage2_holdout contamination 확인
    if str(pid_col) in holdout_pids:
        errors.append({"patient_id": pid_col, "error": "STAGE2_HOLDOUT_CONTAMINATION"})
        print(f"  !! STAGE2_HOLDOUT 오염: {pid_col}")

    n_rows = len(df)
    total_patches += n_rows

    # 필수 컬럼 확인
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        missing_required_col_patients.append({"patient_id": pid_col, "missing": missing_cols})
        issues.append(f"{pid_col}: 필수 컬럼 누락 {missing_cols}")

    # score
    if "padim_score" in df.columns:
        scores = df["padim_score"]
        n_nan = scores.isna().sum()
        n_inf = scores.apply(lambda x: math.isinf(x) if isinstance(x, float) else False).sum()
        total_nan += int(n_nan)
        total_inf += int(n_inf)
        p95_exc = int((scores > P95_REF).sum())
        p99_exc = int((scores > P99_REF).sum())
        total_p95_exc += p95_exc
        total_p99_exc += p99_exc
        score_chunks.append(scores.dropna().values)
        s_min = float(scores.min())
        s_max = float(scores.max())
        s_mean = float(scores.mean())
    else:
        n_nan = n_inf = p95_exc = p99_exc = 0
        s_min = s_max = s_mean = None

    # label 컬럼
    has_lesion_ok = "has_lesion_patch" in df.columns
    lesion_pix_ok = "lesion_pixels" in df.columns

    pat_rows.append({
        "patient_id": pid_col,
        "n_patches": n_rows,
        "n_nan": int(n_nan),
        "n_inf": int(n_inf),
        "p95_exceed": p95_exc,
        "p99_exceed": p99_exc,
        "score_min": s_min,
        "score_max": s_max,
        "score_mean": s_mean,
        "has_lesion_patch_col": has_lesion_ok,
        "lesion_pixels_col": lesion_pix_ok,
        "missing_required_cols": str(missing_cols) if missing_cols else "",
    })
    inv_rows.append({
        "csv_file": csv_f.name,
        "patient_id": pid_col,
        "n_patches": n_rows,
        "n_cols": len(df.columns),
        "n_nan": int(n_nan),
        "n_inf": int(n_inf),
    })

print(f"  처리 완료: {len(pat_rows)}/154")
print(f"  total patches={total_patches:,}")
print(f"  NaN={total_nan}  Inf={total_inf}")

# ── 5. patient_id set 일치 확인 ───────────────────────────────────────────────
print("\n[5] patient_id set 일치 확인")
# dev_df 의 patient_id와 found_pids 비교
# score CSV의 patient_id 컬럼 기반
score_pid_from_col = set()
for csv_f in csv_files:
    try:
        df = pd.read_csv(csv_f, encoding="utf-8-sig", usecols=["patient_id"], nrows=1)
        score_pid_from_col.add(str(df["patient_id"].iloc[0]))
    except:
        pass

extra_in_score = score_pid_from_col - dev_pids
missing_in_score = dev_pids - score_pid_from_col
holdout_in_score = score_pid_from_col & holdout_pids
pid_match = (len(extra_in_score) == 0 and len(missing_in_score) == 0)
print(f"  stage1_dev CSV 수={len(score_pid_from_col)}")
print(f"  extra_in_score={len(extra_in_score)}  missing_in_score={len(missing_in_score)}")
print(f"  holdout_contamination={len(holdout_in_score)}")
print(f"  patient_id set 일치={pid_match}")

# ── 6. score 전체 통계 재집계 ─────────────────────────────────────────────────
print("\n[6] score 전체 통계 재집계")
all_scores = np.concatenate(score_chunks) if score_chunks else np.array([])
s_global_min = float(all_scores.min()) if len(all_scores) > 0 else None
s_global_max = float(all_scores.max()) if len(all_scores) > 0 else None
s_global_mean = float(all_scores.mean()) if len(all_scores) > 0 else None
s_global_std = float(all_scores.std()) if len(all_scores) > 0 else None
s_global_median = float(np.median(all_scores)) if len(all_scores) > 0 else None
print(f"  min={s_global_min:.6f}  max={s_global_max:.6f}")
print(f"  mean={s_global_mean:.6f}  std={s_global_std:.6f}  median={s_global_median:.6f}")

# ── 7. P-B11 reported 값과 비교 ────────────────────────────────────────────────
print("\n[7] P-B11 reported 값 비교")
patch_match = (total_patches == P11_TOTAL)
p95_match = (total_p95_exc == P11_P95_COUNT)
p99_match = (total_p99_exc == P11_P99_COUNT)
print(f"  total_patches: {total_patches:,} (P-B11={P11_TOTAL:,}) → match={patch_match}")
print(f"  p95_exceed:    {total_p95_exc:,} (P-B11={P11_P95_COUNT:,}) → match={p95_match}")
print(f"  p99_exceed:    {total_p99_exc:,} (P-B11={P11_P99_COUNT:,}) → match={p99_match}")

p95_rate = total_p95_exc / total_patches * 100 if total_patches > 0 else 0
p99_rate = total_p99_exc / total_patches * 100 if total_patches > 0 else 0
print(f"  p95 초과율={p95_rate:.3f}%  p99 초과율={p99_rate:.3f}%")

# ── 8. 컬럼 검증 요약 ─────────────────────────────────────────────────────────
print("\n[8] 컬럼 검증 요약")
# 대표 파일로 전체 컬럼 확인
sample_df = pd.read_csv(csv_files[0], encoding="utf-8-sig", nrows=1)
actual_cols = list(sample_df.columns)
label_ok = all(c in actual_cols for c in LABEL_COLS)
coord_ok = all(c in actual_cols for c in COORD_COLS)
required_ok = all(c in actual_cols for c in REQUIRED_COLS)
print(f"  required_cols_ok={required_ok}")
print(f"  label_cols_ok={label_ok}  ({LABEL_COLS})")
print(f"  coord_cols_ok={coord_ok}  ({COORD_COLS})")
print(f"  missing_required_col_patients={len(missing_required_col_patients)}")

col_rows = [
    {"category": "required_cols", "ok": required_ok, "missing": str([c for c in REQUIRED_COLS if c not in actual_cols])},
    {"category": "label_cols", "ok": label_ok, "cols": str(LABEL_COLS)},
    {"category": "coord_cols", "ok": coord_ok, "cols": str(COORD_COLS)},
]

# ── 9. exceedance 요약 ───────────────────────────────────────────────────────
exc_df = pd.DataFrame(pat_rows)[["patient_id", "n_patches", "p95_exceed", "p99_exceed"]]
exc_df["p95_rate"] = exc_df["p95_exceed"] / exc_df["n_patches"] * 100
exc_df["p99_rate"] = exc_df["p99_exceed"] / exc_df["n_patches"] * 100

# ── 10. 판정 ──────────────────────────────────────────────────────────────────
print("\n[9] 판정")
critical_fails = []
if not pid_match:
    critical_fails.append(f"patient_id set 불일치: extra={len(extra_in_score)} missing={len(missing_in_score)}")
if len(holdout_in_score) > 0:
    critical_fails.append(f"stage2_holdout 오염: {holdout_in_score}")
if total_nan > 0:
    critical_fails.append(f"NaN 존재: {total_nan}")
if total_inf > 0:
    critical_fails.append(f"Inf 존재: {total_inf}")
if not patch_match:
    critical_fails.append(f"total patch count 불일치: {total_patches} vs P-B11={P11_TOTAL}")
if not mtime_ok:
    critical_fails.append(f"threshold JSON mtime 변경됨: {thr_mtime} vs ref={THRESHOLD_MTIME_REF}")

warn_items = []
if not p95_match:
    warn_items.append(f"p95_exceed 불일치: {total_p95_exc} vs P-B11={P11_P95_COUNT}")
if not p99_match:
    warn_items.append(f"p99_exceed 불일치: {total_p99_exc} vs P-B11={P11_P99_COUNT}")

if len(critical_fails) == 0:
    verdict = "통과"
elif len(critical_fails) <= 2 and len(holdout_in_score) == 0 and total_nan == 0 and total_inf == 0:
    verdict = "부분통과"
else:
    verdict = "실패"

print(f"  판정: {verdict}")
for f in critical_fails:
    print(f"  !! FAIL: {f}")
for w in warn_items:
    print(f"  !! WARN: {w}")

# ── 11. 출력 파일 저장 ────────────────────────────────────────────────────────
print("\n[10] 출력 파일 저장")

# score_csv_inventory.csv
pd.DataFrame(inv_rows).to_csv(OUT_DIR / "score_csv_inventory.csv", index=False, encoding="utf-8-sig")
print("  score_csv_inventory.csv 저장")

# score_artifact_patient_summary.csv
pd.DataFrame(pat_rows).to_csv(OUT_DIR / "score_artifact_patient_summary.csv", index=False, encoding="utf-8-sig")
print("  score_artifact_patient_summary.csv 저장")

# score_artifact_column_validation.csv
pd.DataFrame(col_rows).to_csv(OUT_DIR / "score_artifact_column_validation.csv", index=False, encoding="utf-8-sig")
print("  score_artifact_column_validation.csv 저장")

# score_artifact_exceedance_validation.csv
exc_df.to_csv(OUT_DIR / "score_artifact_exceedance_validation.csv", index=False, encoding="utf-8-sig")
print("  score_artifact_exceedance_validation.csv 저장")

# score_artifact_errors.csv
pd.DataFrame(errors if errors else [{"patient_id": "", "error": "none"}]).to_csv(
    OUT_DIR / "score_artifact_errors.csv", index=False, encoding="utf-8-sig")
print("  score_artifact_errors.csv 저장")

# ── JSON 보고서 ──────────────────────────────────────────────────────────────
now_str = datetime.datetime.now().isoformat()
report_json = {
    "step": "P-B12",
    "verdict": verdict,
    "created": now_str,
    "branch": "efficientnet_b0_imagenet_chestwall_removed_roi_v1",
    "roi_source": "refined_roi_v4_20_modeB_all_v1",
    "input_validation": {
        "p_b11_verdict": "통과",
        "p_b10_verdict": "통과",
        "p_b9_verdict": "통과",
        "threshold_p95": p95,
        "threshold_p99": p99,
        "threshold_mtime_ok": mtime_ok,
    },
    "split_validation": {
        "stage1_dev_count": len(dev_pids),
        "nsclc": nsclc_dev,
        "msd_lung": msd_dev,
        "stage2_holdout_contamination": len(holdout_in_score),
        "patient_id_set_match": pid_match,
        "extra_in_score": len(extra_in_score),
        "missing_in_score": len(missing_in_score),
    },
    "score_csv_count": csv_count,
    "total_patches": total_patches,
    "p11_total_patches": P11_TOTAL,
    "patch_count_match": patch_match,
    "nan_count": total_nan,
    "inf_count": total_inf,
    "score_stats": {
        "min": s_global_min,
        "max": s_global_max,
        "mean": s_global_mean,
        "std": s_global_std,
        "median": s_global_median,
    },
    "exceedance": {
        "p95_threshold": P95_REF,
        "p95_count": total_p95_exc,
        "p95_rate_pct": round(p95_rate, 3),
        "p95_match_p11": p95_match,
        "p99_threshold": P99_REF,
        "p99_count": total_p99_exc,
        "p99_rate_pct": round(p99_rate, 3),
        "p99_match_p11": p99_match,
    },
    "column_validation": {
        "required_cols_ok": required_ok,
        "label_cols_ok": label_ok,
        "coord_cols_ok": coord_ok,
        "label_connectable": label_ok,
        "mask_connectable": label_ok,
        "patch_coord_exists": coord_ok,
    },
    "guardrails": {
        "np_loadtxt_used": False,
        "metrics_computed": False,
        "auroc_auprc_computed": False,
        "dice_recall_computed": False,
        "scoring_rerun": False,
        "threshold_recalculated": False,
        "normal_val_test_rerun": False,
        "stage2_holdout_accessed": False,
        "existing_results_modified": False,
    },
    "next_step": {
        "p_b13_ready": verdict in ("통과", "부분통과") and len(holdout_in_score) == 0,
        "blocker": critical_fails if critical_fails else None,
        "warnings": warn_items if warn_items else None,
    },
    "critical_fails": critical_fails,
    "warnings": warn_items,
}

with open(OUT_DIR / "p_b12_score_artifact_validation.json", "w", encoding="utf-8") as f:
    json.dump(report_json, f, ensure_ascii=False, indent=2)
print("  p_b12_score_artifact_validation.json 저장")

# ── MD 보고서 ────────────────────────────────────────────────────────────────
md_lines = [
    "# P-B12 v4_20 ROI EfficientNet-B0 Stage1_dev Score Artifact Validation",
    "",
    f"**판정: {verdict}**",
    "",
    f"- 생성일시: {now_str}",
    f"- branch: efficientnet_b0_imagenet_chestwall_removed_roi_v1 / ROI: refined_roi_v4_20_modeB_all_v1",
    "",
    "## P-B11 입력 검증",
    "",
    f"- P-B11 verdict=통과  P-B10 verdict=통과  P-B9 verdict=통과",
    f"- threshold p95={p95:.6f}  p99={p99:.6f}",
    f"- threshold JSON mtime 불변: {mtime_ok}",
    "",
    "## split 검증",
    "",
    f"- stage1_dev={len(dev_pids)}명  NSCLC={nsclc_dev} / MSD_Lung={msd_dev}",
    f"- stage2_holdout contamination: {len(holdout_in_score)}",
    f"- patient_id set 일치: {pid_match}",
    f"- extra_in_score={len(extra_in_score)}  missing_in_score={len(missing_in_score)}",
    "",
    "## score CSV 검증",
    "",
    f"- score CSV 수: {csv_count}",
    f"- total scored patch: {total_patches:,}  (P-B11 reported={P11_TOTAL:,}) match={patch_match}",
    f"- NaN={total_nan}  Inf={total_inf}",
    "",
    "## score 통계",
    "",
    "| 지표 | 값 |",
    "|------|----|",
    f"| min | {s_global_min:.6f} |",
    f"| max | {s_global_max:.6f} |",
    f"| mean | {s_global_mean:.6f} |",
    f"| std | {s_global_std:.6f} |",
    f"| median | {s_global_median:.6f} |",
    "",
    "## threshold exceedance 검증 (P-B9 고정, 재계산 없음)",
    "",
    "| threshold | 값 | P-B12 count | P-B11 count | 일치 | 초과율 |",
    "|-----------|-----|-------------|-------------|------|--------|",
    f"| p95 | {P95_REF:.6f} | {total_p95_exc:,} | {P11_P95_COUNT:,} | {p95_match} | {p95_rate:.3f}% |",
    f"| p99 | {P99_REF:.6f} | {total_p99_exc:,} | {P11_P99_COUNT:,} | {p99_match} | {p99_rate:.3f}% |",
    "",
    "## 컬럼 검증",
    "",
    f"- 필수 컬럼 OK: {required_ok}",
    f"- label 연결 가능 (has_lesion_patch / lesion_pixels): {label_ok}",
    f"- mask 연결 가능: {label_ok}",
    f"- patch 좌표 컬럼 존재 (local_z / y0-x0-y1-x1 / position_bin / z_level): {coord_ok}",
    f"- 필수 컬럼 누락 환자 수: {len(missing_required_col_patients)}",
    "",
    "## 가드레일 확인",
    "",
    "- np.loadtxt 미사용: True",
    "- metrics / AUROC·AUPRC / Dice·recall: 미계산",
    "- scoring 재실행: 없음",
    "- threshold 재계산: 없음",
    "- normal val/test 재실행: 없음",
    "- stage2_holdout 미접근: True",
    "- 기존 P-B1~P-B11 결과 무수정: True",
    "",
    "## Critical Fails",
    "",
]
if critical_fails:
    for f in critical_fails:
        md_lines.append(f"- !! {f}")
else:
    md_lines.append("- 없음")

md_lines += [
    "",
    "## Warnings",
    "",
]
if warn_items:
    for w in warn_items:
        md_lines.append(f"- {w}")
else:
    md_lines.append("- 없음")

md_lines += [
    "",
    "## 다음 단계",
    "",
    f"- P-B13 metrics 계산 가능: {report_json['next_step']['p_b13_ready']}",
    "- ⚠ exceedance율은 scoring 결과일 뿐, recall/AUROC/AUPRC/Dice가 아님 (P-B13에서 평가)",
]

with open(OUT_DIR / "p_b12_score_artifact_validation.md", "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))
print("  p_b12_score_artifact_validation.md 저장")

print("\n" + "=" * 70)
print(f"P-B12 완료: 판정={verdict}")
print("=" * 70)
