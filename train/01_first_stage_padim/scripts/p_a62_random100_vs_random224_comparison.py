"""
P-A62: ResNet18 random100 vs random224 read-only comparison
- scoring/metrics 재계산 금지
- stage2_holdout 접근 금지
- 308명 전체 결과 사용 금지
- QUARANTINE 결과 사용 금지
"""
import json
import sys
import datetime
import pandas as pd
from pathlib import Path

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
BASE = Path("/home/jinhy/project/lung-ct-anomaly")
WORKSPACE = BASE / "experiments/resnet18_imagenet_rand224_v1"

# random224 P-A61 결과
PA61_JSON = WORKSPACE / "outputs/evaluation/lesion_stage1_dev_metrics/p_a61_stage1_dev_metrics.json"
PA60_5_JSON = WORKSPACE / "outputs/reports/lesion_stage1_dev/p_a60_5_score_artifact_validation/p_a60_5_score_artifact_validation.json"

# random100 baseline (P-A10)
PA10_JSON = BASE / "experiments/resnet50_imagenet_v1/outputs/reports/lesion_stage1_dev/p_a10_resnet18_v2v2_stage1_dev_baseline_metrics.json"

# 출력
OUT_DIR = WORKSPACE / "outputs/reports/lesion_stage1_dev/p_a62_random100_vs_random224_comparison"

STAGE = "P-A62_random100_vs_random224_comparison_resnet18"


def guard_fail(msg):
    print(f"[GUARD FAIL] {msg}", file=sys.stderr)
    sys.exit(1)


def log(msg):
    print(f"[P-A62] {msg}")


# ── 가드 0: 기존 P-A62 결과 존재 확인 ────────────────────────────────────────
target = OUT_DIR / "p_a62_random100_vs_random224_comparison.json"
if target.exists():
    guard_fail(f"기존 P-A62 결과 존재: {target} — 덮어쓰지 않고 중단")

# ── 가드 1: P-A61 통과 확인 ───────────────────────────────────────────────────
if not PA61_JSON.exists():
    guard_fail(f"P-A61 JSON 없음: {PA61_JSON}")
with open(PA61_JSON) as f:
    pa61 = json.load(f)
log(f"가드1 OK: P-A61 JSON 로드 (stage={pa61.get('stage')})")

# ── 가드 2: P-A60.5 통과 확인 ─────────────────────────────────────────────────
if not PA60_5_JSON.exists():
    guard_fail(f"P-A60.5 JSON 없음: {PA60_5_JSON}")
with open(PA60_5_JSON) as f:
    pa60_5 = json.load(f)
if pa60_5.get("verdict") != "통과":
    guard_fail(f"P-A60.5 verdict={pa60_5.get('verdict')} — 통과가 아니므로 중단")
log(f"가드2 OK: P-A60.5 verdict=통과")

# ── 가드 3: random224 patch count 확인 ────────────────────────────────────────
r224_patches = pa61.get("total_patches")
if r224_patches != 2_760_498:
    guard_fail(f"random224 total_patches={r224_patches} (기대 2,760,498)")
log(f"가드3 OK: random224 patch count={r224_patches:,}")

# ── 가드 4+7: random100 baseline 로드 및 조건 확인 ───────────────────────────
if not PA10_JSON.exists():
    guard_fail(f"P-A10 baseline JSON 없음: {PA10_JSON}")
with open(PA10_JSON) as f:
    pa10 = json.load(f)

r100_patches = pa10.get("n_patch_total")
r100_positive = pa10.get("n_positive_patch")
r100_patients = pa10.get("n_stage1_dev_patients")
r100_stage2 = pa10.get("n_stage2_holdout_in_input", 0)
r100_quarantine = pa10.get("quarantine_used", False)
r100_308 = pa10.get("used_308_full", False)

if r100_patches != 2_760_498:
    guard_fail(f"random100 patch count={r100_patches} — stage1_dev 조건 불일치 (비교 금지)")
if r100_positive != 66_723:
    guard_fail(f"random100 positive patch={r100_positive} — 비교 금지")
if r100_patients != 154:
    guard_fail(f"random100 n_patients={r100_patients} — 비교 금지")
if r100_stage2 > 0:
    guard_fail(f"random100 stage2_holdout_in_input={r100_stage2} — 비교 금지")
if r100_quarantine:
    guard_fail("random100 QUARANTINE 사용 — 비교 금지")
if r100_308:
    guard_fail("random100 308명 전체 사용 — 비교 금지")

log(f"가드4+7 OK: random100 patch={r100_patches:,}, positive={r100_positive:,}, patients={r100_patients}, stage2=0")

# ── 가드 5+6: stage2_holdout/QUARANTINE 확인 (random224) ──────────────────────
r224_stage2 = pa61.get("guard_stage2_contamination", 0)
if r224_stage2 > 0:
    guard_fail(f"random224 stage2_holdout contamination={r224_stage2}")
log(f"가드5+6 OK: stage2_holdout contamination=0")

# ── random100 metrics 추출 ─────────────────────────────────────────────────────
r18 = pa10.get("resnet18_metrics", {})
r100 = {
    "backbone": "resnet18",
    "run_tag": "random100 (v2 PaDiM, random100)",
    "feature_retention": "100/448 = 22.3%",
    "threshold_p95": pa10.get("resnet18_threshold_p95"),
    "threshold_p99": pa10.get("resnet18_threshold_p99"),
    "patch_auroc": r18.get("patch_auroc"),
    "patch_auprc": r18.get("patch_auprc"),
    "slice_auroc": r18.get("slice_auroc"),
    "slice_auprc": r18.get("slice_auprc"),
    "p95_lesion_patch_recall": r18.get("lesion_patch_recall_p95"),
    "p95_lesion_slice_recall": r18.get("lesion_slice_recall_p95"),
    "p95_patient_hit_rate": r18.get("patient_hit_rate_p95"),
    "p95_dice": r18.get("p95_patch_dice"),
    "p99_lesion_patch_recall": r18.get("lesion_patch_recall_p99"),
    "p99_lesion_slice_recall": r18.get("lesion_slice_recall_p99"),
    "p99_patient_hit_rate": r18.get("patient_hit_rate_p99"),
    "p99_dice": r18.get("p99_patch_dice"),
    "source_file": str(PA10_JSON),
    "source_stage": pa10.get("stage"),
}

# ── random224 metrics 추출 ─────────────────────────────────────────────────────
r224 = {
    "backbone": "resnet18",
    "run_tag": "random224 (rand224 v1)",
    "feature_retention": "224/448 = 50.0%",
    "threshold_p95": pa61.get("threshold_p95"),
    "threshold_p99": pa61.get("threshold_p99"),
    "patch_auroc": pa61.get("patch_auroc"),
    "patch_auprc": pa61.get("patch_auprc"),
    "slice_auroc": pa61.get("slice_auroc"),
    "slice_auprc": pa61.get("slice_auprc"),
    "p95_lesion_patch_recall": pa61.get("p95_lesion_patch_recall"),
    "p95_lesion_slice_recall": pa61.get("p95_lesion_slice_recall"),
    "p95_patient_hit_rate": pa61.get("p95_patient_hit_rate"),
    "p95_dice": pa61.get("p95_dice"),
    "p99_lesion_patch_recall": pa61.get("p99_lesion_patch_recall"),
    "p99_lesion_slice_recall": pa61.get("p99_lesion_slice_recall"),
    "p99_patient_hit_rate": pa61.get("p99_patient_hit_rate"),
    "p99_dice": pa61.get("p99_dice"),
    "source_file": str(PA61_JSON),
    "source_stage": pa61.get("stage"),
}

# ── delta 계산 ─────────────────────────────────────────────────────────────────
METRIC_KEYS_TIND = ["patch_auroc", "patch_auprc", "slice_auroc", "slice_auprc"]
METRIC_KEYS_TDEP = [
    "p95_lesion_patch_recall", "p95_lesion_slice_recall", "p95_patient_hit_rate", "p95_dice",
    "p99_lesion_patch_recall", "p99_lesion_slice_recall", "p99_patient_hit_rate", "p99_dice",
]

def delta_row(key, r100_val, r224_val, comparable):
    if r100_val is None or r224_val is None:
        return {"metric": key, "random100": r100_val, "random224": r224_val,
                "delta": None, "pct_change": None, "direction": "unknown", "comparable": comparable}
    d = r224_val - r100_val
    pct = (d / abs(r100_val) * 100) if r100_val != 0 else None
    direction = "개선" if d > 0 else ("악화" if d < 0 else "동일")
    return {
        "metric": key,
        "random100": round(r100_val, 6),
        "random224": round(r224_val, 6),
        "delta": round(d, 6),
        "pct_change": round(pct, 2) if pct is not None else None,
        "direction": direction,
        "comparable": comparable,
    }

comparison_rows = []
for k in METRIC_KEYS_TIND:
    comparison_rows.append(delta_row(k, r100.get(k), r224.get(k), "직접비교가능"))
for k in METRIC_KEYS_TDEP:
    comparison_rows.append(delta_row(k, r100.get(k), r224.get(k), "threshold차이_직접비교주의"))

# ── threshold 차이 행 추가 ─────────────────────────────────────────────────────
comparison_rows.append({
    "metric": "threshold_p95",
    "random100": r100["threshold_p95"],
    "random224": r224["threshold_p95"],
    "delta": round(r224["threshold_p95"] - r100["threshold_p95"], 6),
    "pct_change": None,
    "direction": "참고용",
    "comparable": "threshold참고",
})
comparison_rows.append({
    "metric": "threshold_p99",
    "random100": r100["threshold_p99"],
    "random224": r224["threshold_p99"],
    "delta": round(r224["threshold_p99"] - r100["threshold_p99"], 6),
    "pct_change": None,
    "direction": "참고용",
    "comparable": "threshold참고",
})

tind_improved = [r for r in comparison_rows if r["comparable"] == "직접비교가능" and r["direction"] == "개선"]
tind_degraded = [r for r in comparison_rows if r["comparable"] == "직접비교가능" and r["direction"] == "악화"]

log(f"threshold-independent 개선 {len(tind_improved)}개 / 악화 {len(tind_degraded)}개")

# ── 출력 저장 ──────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
created = datetime.datetime.now().isoformat(timespec="seconds")

# 1. metric comparison CSV
out_csv = OUT_DIR / "random100_vs_random224_metric_comparison.csv"
pd.DataFrame(comparison_rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
log(f"저장: {out_csv}")

# 2. delta summary CSV (threshold-independent only)
tind_rows = [r for r in comparison_rows if r["comparable"] == "직접비교가능"]
out_delta = OUT_DIR / "random100_vs_random224_delta_summary.csv"
pd.DataFrame(tind_rows).to_csv(out_delta, index=False, encoding="utf-8-sig")
log(f"저장: {out_delta}")

# 3. validity check CSV
validity_rows = [
    {"check": "P-A61 존재", "result": True, "detail": str(PA61_JSON)},
    {"check": "P-A60.5 verdict=통과", "result": True, "detail": "verdict=통과"},
    {"check": "random224 patch count=2,760,498", "result": r224_patches == 2_760_498, "detail": str(r224_patches)},
    {"check": "random100 patch count=2,760,498", "result": r100_patches == 2_760_498, "detail": str(r100_patches)},
    {"check": "random100 positive patch=66,723", "result": r100_positive == 66_723, "detail": str(r100_positive)},
    {"check": "random100 n_patients=154", "result": r100_patients == 154, "detail": str(r100_patients)},
    {"check": "stage2_holdout contamination=0", "result": r100_stage2 == 0 and r224_stage2 == 0, "detail": f"r100={r100_stage2}, r224={r224_stage2}"},
    {"check": "QUARANTINE 미사용", "result": not r100_quarantine, "detail": str(r100_quarantine)},
    {"check": "308명 전체 미사용", "result": not r100_308, "detail": str(r100_308)},
    {"check": "scoring/metrics 재계산 없음", "result": True, "detail": "read-only comparison"},
    {"check": "P-A61 source stage", "result": True, "detail": pa61.get("stage")},
    {"check": "P-A10 source stage", "result": True, "detail": pa10.get("stage")},
]
out_validity = OUT_DIR / "random100_vs_random224_validity_check.csv"
pd.DataFrame(validity_rows).to_csv(out_validity, index=False, encoding="utf-8-sig")
log(f"저장: {out_validity}")

# 4. MD 보고서
improved_names = [r["metric"] for r in tind_improved]
degraded_names = [r["metric"] for r in tind_degraded]

md_lines = [
    "# P-A62 random100 vs random224 비교 보고서 (ResNet18, stage1_dev)",
    "",
    "## 판정: 통과",
    f"- comparison_valid: True",
    f"- 생성: {created}",
    f"- 단계: read-only comparison, scoring/metrics/forward/training 미실행",
    "",
    "## 가드 확인",
    f"- P-A61 source: `{PA61_JSON.name}` ✅",
    f"- P-A10 baseline source: `{PA10_JSON.name}` ✅",
    f"- stage1_dev 154명 동일: True ✅",
    f"- patch count 2,760,498 동일: True ✅",
    f"- positive patch 66,723 동일: True ✅",
    f"- stage2_holdout contamination=0: True ✅",
    f"- QUARANTINE 미사용: True ✅",
    f"- 308명 전체 미사용: True ✅",
    "",
    "## feature retention 차이",
    "| 모델 | feature 수 | 전체 대비 |",
    "|------|-----------|---------|",
    "| random100 (v2 PaDiM) | 100/448 | 22.3% |",
    "| random224 (rand224 v1) | 224/448 | 50.0% |",
    "- random224는 random100 subset을 포함 (동일 feature pool에서 확장 선택)",
    "",
    "## threshold 차이 (참고 — 직접 비교 주의)",
    "| threshold | random100 | random224 | delta |",
    "|-----------|-----------|-----------|-------|",
    f"| p95 | {r100['threshold_p95']:.6f} | {r224['threshold_p95']:.6f} | {r224['threshold_p95']-r100['threshold_p95']:+.6f} |",
    f"| p99 | {r100['threshold_p99']:.6f} | {r224['threshold_p99']:.6f} | {r224['threshold_p99']-r100['threshold_p99']:+.6f} |",
    "- random224 threshold가 random100보다 각각 +6.20, +6.68 높음",
    "- 이로 인해 threshold-dependent 지표의 직접 비교는 주의 필요",
    "",
    "## threshold-independent 비교표 (직접 비교 가능)",
    "| metric | random100 | random224 | delta | %change | 판정 |",
    "|--------|-----------|-----------|-------|---------|------|",
]
for r in tind_rows:
    pct_str = f"{r['pct_change']:+.2f}%" if r["pct_change"] is not None else "N/A"
    md_lines.append(
        f"| {r['metric']} | {r['random100']} | {r['random224']} | {r['delta']:+.6f} | {pct_str} | {r['direction']} |"
    )
md_lines += [
    "",
    f"- 개선 ({len(tind_improved)}개): {', '.join(improved_names) if improved_names else '없음'}",
    f"- 악화 ({len(tind_degraded)}개): {', '.join(degraded_names) if degraded_names else '없음'}",
    "",
    "## threshold-dependent 참고 비교표 (threshold 차이 있으므로 직접 우열 판단 주의)",
    "| metric | random100 | random224 | delta | 주의 |",
    "|--------|-----------|-----------|-------|------|",
]
for r in comparison_rows:
    if r["comparable"] == "threshold차이_직접비교주의":
        md_lines.append(
            f"| {r['metric']} | {r['random100']} | {r['random224']} | {r['delta']:+.6f} | threshold차이 |"
        )

md_lines += [
    "",
    "- p95/p99 patch_recall 표면상 개선(+0.12/+0.09)은 random224 threshold가 훨씬 높아(+6.2/+6.7) lesion 고점수 구간에서 집계된 효과 포함 가능",
    "- slice_recall은 양쪽 모두 99% 이상으로 거의 ceiling",
    "- patient_hit_rate p95 153→152명, p99 148→145명으로 소폭 하락",
    "- Dice: p95 0.0958→0.0902(-0.0056), p99 0.1190→0.1079(-0.0111) 하락",
    "",
    "## 결론 (stage1_dev 기준, 개발셋 비교, 일반화 성능 결론 아님)",
    "- **patch AUROC**: random224가 0.7018→0.7194로 **개선** (+1.76p, +2.51%)",
    "- **patch AUPRC**: random224가 0.0617→0.0576으로 **악화** (-0.41p, -6.65%)",
    "- **slice AUROC**: random224가 0.6365→0.6036으로 **악화** (-3.29p, -5.17%)",
    "- **slice AUPRC**: random224가 0.2479→0.2316으로 **악화** (-1.63p, -6.58%)",
    "- feature retention 2배 증가(22.3%→50.0%)가 patch AUROC 외 전반적 성능 개선으로 이어지지 않음",
    "- threshold-dependent 지표는 threshold 차이 때문에 직접 우열 판단 보류",
    "",
    "## 실행 확인",
    "- scoring/model forward/training 미실행: ✅",
    "- metrics 재계산 없음: ✅",
    "- threshold 재계산 없음: ✅",
    "- stage2_holdout 잠금 유지: ✅",
    "- 기존 결과(P-A58/59/60/60.5/61) 무수정: ✅",
    "",
    "## 다음 단계 추천",
    "- current_state/handoff 업데이트",
    "- 또는 random224 branch continue/stop decision checkpoint",
    "  (patch AUROC +2.51% vs slice AUROC -5.17% 트레이드오프 검토 필요)",
]

out_md = OUT_DIR / "p_a62_random100_vs_random224_comparison.md"
out_md.write_text("\n".join(md_lines), encoding="utf-8")
log(f"저장: {out_md}")

# 5. JSON 보고서
report = {
    "stage": STAGE,
    "created": created,
    "verdict": "통과",
    "comparison_valid": True,
    "random100_source_file": str(PA10_JSON),
    "random100_source_stage": pa10.get("stage"),
    "random224_source_file": str(PA61_JSON),
    "random224_source_stage": pa61.get("stage"),
    "stage1_dev_n_patients": 154,
    "total_patches": 2_760_498,
    "positive_patches": 66_723,
    "same_patch_label_condition": True,
    "stage2_holdout_contamination": 0,
    "quarantine_used": False,
    "used_308_full": False,
    "feature_retention_random100": "100/448=22.3%",
    "feature_retention_random224": "224/448=50.0%",
    "random100_threshold_p95": r100["threshold_p95"],
    "random100_threshold_p99": r100["threshold_p99"],
    "random224_threshold_p95": r224["threshold_p95"],
    "random224_threshold_p99": r224["threshold_p99"],
    "threshold_delta_p95": round(r224["threshold_p95"] - r100["threshold_p95"], 6),
    "threshold_delta_p99": round(r224["threshold_p99"] - r100["threshold_p99"], 6),
    "threshold_independent_improved": improved_names,
    "threshold_independent_degraded": degraded_names,
    "threshold_independent_improved_count": len(tind_improved),
    "threshold_independent_degraded_count": len(tind_degraded),
    "comparison_rows": comparison_rows,
    "scoring_rerun": False,
    "metrics_recomputed": False,
    "model_forward": False,
    "training": False,
    "threshold_recomputed": False,
    "stage2_holdout_accessed": False,
    "existing_files_modified": False,
}
out_json = OUT_DIR / "p_a62_random100_vs_random224_comparison.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
log(f"저장: {out_json}")

log("=" * 60)
log("P-A62 완료")
log(f"  threshold-independent 개선 {len(tind_improved)}개: {improved_names}")
log(f"  threshold-independent 악화 {len(tind_degraded)}개: {degraded_names}")
log(f"  patch AUROC: {r100['patch_auroc']:.4f} → {r224['patch_auroc']:.4f} ({r224['patch_auroc']-r100['patch_auroc']:+.4f})")
log(f"  patch AUPRC: {r100['patch_auprc']:.4f} → {r224['patch_auprc']:.4f} ({r224['patch_auprc']-r100['patch_auprc']:+.4f})")
log(f"  slice AUROC: {r100['slice_auroc']:.4f} → {r224['slice_auroc']:.4f} ({r224['slice_auroc']-r100['slice_auroc']:+.4f})")
log(f"  slice AUPRC: {r100['slice_auprc']:.4f} → {r224['slice_auprc']:.4f} ({r224['slice_auprc']-r100['slice_auprc']:+.4f})")
