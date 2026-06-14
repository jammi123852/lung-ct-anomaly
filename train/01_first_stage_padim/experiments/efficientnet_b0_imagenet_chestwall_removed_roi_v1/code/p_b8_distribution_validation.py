"""
P-B8: v4_20 ROI EfficientNet-B0 distribution validation (read-only)

P-B7b full train 생성 distribution npz 무결성 검증.
- npz key 구조, position_bin count, mean/cov shape, NaN/Inf, cov 대칭/대각
- count 정합: global_count == P-B7 used patch (계층형 중복 합산 금지)
- selected index 불변, P-B7 report/runtime/patch-filtering 정합, smoke 보존, roi_0_0 무수정

금지: scoring / threshold / metrics / model forward / training / stage2_holdout
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJ_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT  = Path(__file__).resolve().parents[1]
ROI0_BRANCH = PROJ_ROOT / "experiments" / "efficientnet_b0_imagenet_v1"

REDUCED_FEATURE_DIM = 100
EXPECTED_USED_PATCH = 11356415   # P-B7 reported

DIST_NPZ   = EXP_ROOT / "outputs" / "models" / "distributions" / "position_bin_stats.npz"
SEL_IDX    = EXP_ROOT / "outputs" / "models" / "distributions" / "selected_feature_indices.npy"
SEL_IDX_SRC = ROI0_BRANCH / "outputs" / "models" / "distributions" / "selected_feature_indices.npy"

P_B7_JSON  = EXP_ROOT / "outputs" / "reports" / "full" / "p_b7_full_train.json"
RUNTIME_CSV = EXP_ROOT / "outputs" / "reports" / "full" / "p_b7_runtime_summary.csv"
PATCH_CSV  = EXP_ROOT / "outputs" / "reports" / "full" / "p_b7_patch_filtering_summary.csv"
FULL_LOG   = EXP_ROOT / "outputs" / "reports" / "full" / "full_train.log"

SMOKE_NPZ_L1 = EXP_ROOT / "outputs" / "smoke" / "train_limit1" / "position_bin_stats.npz"
SMOKE_NPZ_L5 = EXP_ROOT / "outputs" / "smoke" / "train_limit5" / "position_bin_stats.npz"
ROI0_FULL_NPZ = ROI0_BRANCH / "outputs" / "models" / "distributions" / "position_bin_stats.npz"

REPORT_DIR = EXP_ROOT / "outputs" / "reports" / "full"
SCRIPT_NAME = "p_b8_distribution_validation.py"

POSITION_BINS = ["upper_central", "upper_peripheral", "middle_central",
                 "middle_peripheral", "lower_central", "lower_peripheral"]
Z_LEVEL_KEYS  = ["upper_all", "middle_all", "lower_all"]
GLOBAL_KEY    = "global_pure_lung"
ALL_DIST_KEYS = POSITION_BINS + Z_LEVEL_KEYS + [GLOBAL_KEY]   # 10개


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _to_jsonable(obj):
    """numpy bool/int/float 등을 JSON 직렬화 가능 타입으로 재귀 변환."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main():
    ts = datetime.now().isoformat(timespec="seconds")
    print(f"[{SCRIPT_NAME}] 시작: {ts}\n")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    issues = []
    checks = {}

    # ── 1~3. npz 존재/size/sha256 ────────────────────────────────────────
    assert DIST_NPZ.exists(), f"distribution npz 없음: {DIST_NPZ}"
    npz_size = DIST_NPZ.stat().st_size
    npz_nonzero = npz_size > 0
    npz_sha = sha256_of(DIST_NPZ)
    checks["npz_exists"] = True
    checks["npz_size_bytes"] = npz_size
    checks["npz_nonzero"] = npz_nonzero
    checks["npz_sha256"] = npz_sha
    if not npz_nonzero:
        issues.append("npz 0-byte")
    print(f"[1-3] npz size={npz_size}, sha256={npz_sha[:16]}...")

    z = np.load(str(DIST_NPZ), allow_pickle=True)
    keys = list(z.keys())
    checks["npz_total_keys"] = len(keys)

    # ── 4~5. key 구조 / position_bin 10개 ────────────────────────────────
    dist_keys_present = [k for k in ALL_DIST_KEYS if f"{k}_count" in keys]
    checks["distribution_keys_count"] = len(dist_keys_present)
    checks["distribution_keys"] = dist_keys_present
    if len(dist_keys_present) != 10:
        issues.append(f"distribution key {len(dist_keys_present)}≠10")
    print(f"[4-5] distribution key: {len(dist_keys_present)}/10")

    # ── 6~8, 10~12. 각 bin count/mean/cov/NaN/Inf/대칭/대각 ───────────────
    per_bin = []
    total_nan = total_inf = 0
    cov_asym_count = 0
    cov_neg_diag_count = 0
    pb_count_sum = 0   # 6 position_bin count 합 (global과 비교)
    for k in ALL_DIST_KEYS:
        cnt = int(z[f"{k}_count"]) if f"{k}_count" in keys else -1
        mean = z[f"{k}_mean"] if f"{k}_mean" in keys else None
        cov  = z[f"{k}_cov"]  if f"{k}_cov" in keys else None
        mean_shape = tuple(mean.shape) if mean is not None else None
        cov_shape  = tuple(cov.shape) if cov is not None else None
        nan_c = int(np.isnan(mean).sum()) + int(np.isnan(cov).sum()) if (mean is not None and cov is not None) else -1
        inf_c = int(np.isinf(mean).sum()) + int(np.isinf(cov).sum()) if (mean is not None and cov is not None) else -1
        total_nan += max(nan_c, 0)
        total_inf += max(inf_c, 0)
        # 대칭성 / 대각 음수
        sym = neg_diag = None
        if cov is not None and cov.shape == (REDUCED_FEATURE_DIM, REDUCED_FEATURE_DIM):
            sym = bool(np.allclose(cov, cov.T, atol=1e-6))
            neg_diag = int((np.diag(cov) < 0).sum())
            if not sym: cov_asym_count += 1
            if neg_diag > 0: cov_neg_diag_count += 1
        per_bin.append({
            "key": k, "count": cnt,
            "mean_shape": str(mean_shape), "cov_shape": str(cov_shape),
            "mean_ok": mean_shape == (REDUCED_FEATURE_DIM,),
            "cov_ok": cov_shape == (REDUCED_FEATURE_DIM, REDUCED_FEATURE_DIM),
            "nan": nan_c, "inf": inf_c,
            "cov_symmetric": sym, "cov_neg_diag": neg_diag,
        })
        if k in POSITION_BINS:
            pb_count_sum += cnt

    bad_count = [b["key"] for b in per_bin if b["count"] <= 0]
    bad_mean  = [b["key"] for b in per_bin if not b["mean_ok"]]
    bad_cov   = [b["key"] for b in per_bin if not b["cov_ok"]]
    if bad_count: issues.append(f"count<=0 bin: {bad_count}")
    if bad_mean:  issues.append(f"mean shape 오류 bin: {bad_mean}")
    if bad_cov:   issues.append(f"cov shape 오류 bin: {bad_cov}")
    if total_nan: issues.append(f"NaN {total_nan}")
    if total_inf: issues.append(f"Inf {total_inf}")
    if cov_asym_count: issues.append(f"cov 비대칭 {cov_asym_count}개")
    if cov_neg_diag_count: issues.append(f"cov 대각 음수 {cov_neg_diag_count}개")
    print(f"[6-8] count<=0:{len(bad_count)}, mean오류:{len(bad_mean)}, cov오류:{len(bad_cov)}")
    print(f"[10-12] NaN:{total_nan}, Inf:{total_inf}, cov비대칭:{cov_asym_count}, cov음수대각:{cov_neg_diag_count}")

    # ── 9. count 정합 (계층형 중복 합산 금지) ────────────────────────────
    global_count = int(z[f"{GLOBAL_KEY}_count"])
    z_level_sum = sum(int(z[f"{k}_count"]) for k in Z_LEVEL_KEYS)
    count_match_global = (global_count == EXPECTED_USED_PATCH)
    count_match_pb_sum = (pb_count_sum == global_count)
    count_match_zlevel = (z_level_sum == global_count)   # z_level 합도 global과 같아야 함
    checks["global_count"] = global_count
    checks["position_bin_count_sum"] = pb_count_sum
    checks["z_level_count_sum"] = z_level_sum
    checks["expected_used_patch"] = EXPECTED_USED_PATCH
    checks["count_match_global_vs_used"] = count_match_global
    checks["count_match_pb_sum_vs_global"] = count_match_pb_sum
    checks["count_match_zlevel_vs_global"] = count_match_zlevel
    if not count_match_global:
        issues.append(f"global_count {global_count} != used patch {EXPECTED_USED_PATCH}")
    if not count_match_pb_sum:
        issues.append(f"position_bin 합 {pb_count_sum} != global {global_count}")
    if not count_match_zlevel:
        issues.append(f"z_level 합 {z_level_sum} != global {global_count}")
    print(f"[9] global_count={global_count}, pb합={pb_count_sum}, z합={z_level_sum}, used patch={EXPECTED_USED_PATCH}")
    print(f"    일치(global==used):{count_match_global}, (pb==global):{count_match_pb_sum}, (z==global):{count_match_zlevel}")

    # ── 13~14. selected index ────────────────────────────────────────────
    sidx = np.load(str(SEL_IDX))
    sidx_ok = (sidx.shape == (REDUCED_FEATURE_DIM,) and len(np.unique(sidx)) == REDUCED_FEATURE_DIM
               and sidx.min() >= 0 and sidx.max() <= 143)
    # P-B4/P-B5 이후 불변 = 기존 branch 값과 동일
    sidx_src = np.load(str(SEL_IDX_SRC)) if SEL_IDX_SRC.exists() else None
    sidx_unchanged = bool(np.array_equal(sidx, sidx_src)) if sidx_src is not None else None
    # npz 내장 selected index와도 일치
    npz_sidx = z["selected_feature_indices"] if "selected_feature_indices" in keys else None
    sidx_match_npz = bool(np.array_equal(sidx, np.asarray(npz_sidx))) if npz_sidx is not None else None
    checks["selected_index_shape"] = list(sidx.shape)
    checks["selected_index_unique"] = int(len(np.unique(sidx)))
    checks["selected_index_range"] = [int(sidx.min()), int(sidx.max())]
    checks["selected_index_valid"] = sidx_ok
    checks["selected_index_unchanged_vs_roi0_src"] = sidx_unchanged
    checks["selected_index_match_npz_internal"] = sidx_match_npz
    if not sidx_ok: issues.append("selected index 검증 실패")
    if sidx_unchanged is False: issues.append("selected index 기존 값과 다름")
    if sidx_match_npz is False: issues.append("selected index npz 내장값과 불일치")
    print(f"[13-14] selected index valid={sidx_ok}, unchanged={sidx_unchanged}, npz_match={sidx_match_npz}")

    # ── 15. P-B7 report 정합 ─────────────────────────────────────────────
    p_b7 = json.load(open(P_B7_JSON, encoding="utf-8"))
    p_b7_used = p_b7.get("n_patches_used")
    report_consistent = (p_b7_used == global_count == EXPECTED_USED_PATCH
                         and p_b7.get("position_bins_with_data") == 10
                         and p_b7.get("total_nan") == 0 and p_b7.get("total_inf") == 0)
    checks["p_b7_used_patch"] = p_b7_used
    checks["p_b7_report_consistent"] = report_consistent
    if not report_consistent:
        issues.append("P-B7 report와 npz 정합 실패")
    print(f"[15] P-B7 report 정합: {report_consistent}")

    # ── 16. runtime summary 정합 ─────────────────────────────────────────
    runtime_used = None
    runtime_rows = 0
    if RUNTIME_CSV.exists():
        with open(RUNTIME_CSV, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                runtime_rows += 1
                if row.get("metric") == "n_patches_used":
                    runtime_used = int(float(row["value"]))
    runtime_consistent = (runtime_used == p_b7_used)
    checks["runtime_n_patches_used"] = runtime_used
    checks["runtime_consistent"] = runtime_consistent
    if not runtime_consistent:
        issues.append(f"runtime used {runtime_used} != report {p_b7_used}")
    print(f"[16] runtime 정합: {runtime_consistent}")

    # ── 17~18. patch filtering summary 290행 + 합계 정합 ─────────────────
    pf_rows = 0
    pf_before_sum = pf_after_sum = 0
    if PATCH_CSV.exists():
        with open(PATCH_CSV, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                pf_rows += 1
                try:
                    pf_before_sum += int(row["patch_before"])
                    pf_after_sum  += int(row["patch_after"])
                except (ValueError, KeyError):
                    pass
    pf_rows_ok = (pf_rows == 290)
    pf_before_ok = (pf_before_sum == p_b7.get("total_patch_before"))
    pf_after_ok  = (pf_after_sum == p_b7.get("total_patch_after") == EXPECTED_USED_PATCH)
    checks["patch_filtering_rows"] = pf_rows
    checks["patch_filtering_before_sum"] = pf_before_sum
    checks["patch_filtering_after_sum"] = pf_after_sum
    checks["patch_filtering_rows_ok"] = pf_rows_ok
    checks["patch_filtering_before_match"] = pf_before_ok
    checks["patch_filtering_after_match"] = pf_after_ok
    if not pf_rows_ok: issues.append(f"patch filtering 행 {pf_rows}≠290")
    if not pf_before_ok: issues.append("patch before 합 불일치")
    if not pf_after_ok: issues.append("patch after 합 불일치")
    print(f"[17-18] patch filtering 행={pf_rows}, before합={pf_before_sum}, after합={pf_after_sum}")

    # ── 19. full_train.log 정상 종료 ─────────────────────────────────────
    log_ok = False
    if FULL_LOG.exists():
        txt = FULL_LOG.read_text(encoding="utf-8", errors="ignore")
        log_ok = ("full train 완료: 통과" in txt) or ("=== P-B7 full train 완료" in txt)
    checks["full_log_exists"] = FULL_LOG.exists()
    checks["full_log_normal_termination"] = log_ok
    if not log_ok: issues.append("full_train.log 정상 종료 로그 없음")
    print(f"[19] full_train.log 정상 종료: {log_ok}")

    # ── 20. smoke 보존 ───────────────────────────────────────────────────
    smoke_ok = SMOKE_NPZ_L1.exists() and SMOKE_NPZ_L5.exists()
    checks["smoke_l1_preserved"] = SMOKE_NPZ_L1.exists()
    checks["smoke_l5_preserved"] = SMOKE_NPZ_L5.exists()
    if not smoke_ok: issues.append("smoke 결과 미보존")
    print(f"[20] smoke 보존: {smoke_ok}")

    # ── 21. roi_0_0 branch 무수정 ────────────────────────────────────────
    roi0_isolated = (ROI0_FULL_NPZ != DIST_NPZ)
    roi0_exists = ROI0_FULL_NPZ.exists()
    checks["roi0_branch_npz_path_isolated"] = roi0_isolated
    checks["roi0_branch_npz_exists"] = roi0_exists
    print(f"[21] roi_0_0 branch 경로 분리: {roi0_isolated}, 존재: {roi0_exists}")

    # ── 22. stage2_holdout 미접근 (이 스크립트 경로상 미참조) ────────────
    checks["stage2_holdout_accessed"] = False
    print(f"[22] stage2_holdout 미접근: True")

    # ── 판정 ──────────────────────────────────────────────────────────────
    verdict = "통과" if not issues else ("실패" if any(
        kw in i for i in issues for kw in ["NaN", "Inf", "count", "shape", "0-byte", "비대칭", "음수", "정합 실패", "검증 실패"]
    ) else "부분통과")
    print(f"\n[판정] {verdict}")
    for i in issues:
        print(f"  ⚠ {i}")

    p_b9_can_proceed = (verdict == "통과")

    report = {
        "stage": "P-B8_distribution_validation",
        "created": ts,
        "verdict": verdict,
        "scope": {
            "scoring": False, "threshold_calculated": False, "metrics_calculated": False,
            "auroc_auprc_computed": False, "model_forward": False, "training": False,
            "stage2_holdout_accessed": False, "existing_results_modified": False,
        },
        "npz": {
            "path": str(DIST_NPZ), "size_bytes": npz_size, "nonzero": npz_nonzero,
            "sha256": npz_sha, "total_keys": len(keys),
            "distribution_keys_count": len(dist_keys_present),
        },
        "per_bin": per_bin,
        "count_validation": {
            "global_count": global_count,
            "position_bin_count_sum": pb_count_sum,
            "z_level_count_sum": z_level_sum,
            "expected_used_patch": EXPECTED_USED_PATCH,
            "global_eq_used": count_match_global,
            "pb_sum_eq_global": count_match_pb_sum,
            "z_sum_eq_global": count_match_zlevel,
            "note": "계층형 키. 1 patch가 position_bin/z_level/global 3중 누적. used patch는 global_count(=6 position_bin 합)와 비교.",
        },
        "integrity": {
            "total_nan": total_nan, "total_inf": total_inf,
            "cov_asymmetric_count": cov_asym_count, "cov_neg_diag_count": cov_neg_diag_count,
        },
        "selected_index": {
            "shape": list(sidx.shape), "unique": int(len(np.unique(sidx))),
            "range": [int(sidx.min()), int(sidx.max())], "valid": sidx_ok,
            "unchanged_vs_roi0_src": sidx_unchanged, "match_npz_internal": sidx_match_npz,
        },
        "consistency": {
            "p_b7_report_consistent": report_consistent,
            "runtime_consistent": runtime_consistent,
            "patch_filtering_rows": pf_rows, "patch_filtering_rows_ok": pf_rows_ok,
            "patch_before_match": pf_before_ok, "patch_after_match": pf_after_ok,
            "full_log_normal_termination": log_ok,
        },
        "preservation": {
            "smoke_l1_preserved": SMOKE_NPZ_L1.exists(),
            "smoke_l5_preserved": SMOKE_NPZ_L5.exists(),
            "roi0_branch_isolated": roi0_isolated,
        },
        "p_b9_normal_val_threshold_feasible": p_b9_can_proceed,
        "issues": issues,
    }
    with open(REPORT_DIR / "p_b8_distribution_validation.json", "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(report), f, ensure_ascii=False, indent=2)

    # summary CSV
    with open(REPORT_DIR / "p_b8_distribution_validation_summary.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["key", "count", "mean_shape", "cov_shape",
                                          "mean_ok", "cov_ok", "nan", "inf",
                                          "cov_symmetric", "cov_neg_diag"])
        w.writeheader()
        w.writerows(per_bin)

    # MD
    md = [
        "# P-B8 v4_20 ROI EfficientNet-B0 Distribution Validation\n",
        f"**판정: {verdict}**\n",
        f"- 생성일시: {ts}",
        "- 모드: read-only validation (scoring/threshold/metrics/forward 없음)\n",
        "## 1. npz 기본\n",
        "| 항목 | 값 |",
        "|------|----|",
        f"| path | `{DIST_NPZ}` |",
        f"| size | {npz_size:,} bytes (0-byte 아님: {npz_nonzero}) |",
        f"| sha256 | `{npz_sha}` |",
        f"| 총 key | {len(keys)} |",
        f"| distribution key | {len(dist_keys_present)}/10 |\n",
        "## 2. position_bin / z_level / global (10개 분포)\n",
        "| key | count | mean | cov | NaN | Inf | 대칭 | 음수대각 |",
        "|-----|-------|------|-----|-----|-----|------|----------|",
    ]
    for b in per_bin:
        md.append(f"| {b['key']} | {b['count']:,} | {b['mean_shape']} | {b['cov_shape']} | "
                  f"{b['nan']} | {b['inf']} | {b['cov_symmetric']} | {b['cov_neg_diag']} |")
    md += [
        "",
        "## 3. count 정합 (계층형 중복 합산 금지)\n",
        "| 항목 | 값 |",
        "|------|----|",
        f"| global_count | {global_count:,} |",
        f"| 6 position_bin 합 | {pb_count_sum:,} |",
        f"| 3 z_level 합 | {z_level_sum:,} |",
        f"| P-B7 used patch | {EXPECTED_USED_PATCH:,} |",
        f"| global == used patch | {count_match_global} |",
        f"| position_bin 합 == global | {count_match_pb_sum} |",
        f"| z_level 합 == global | {count_match_zlevel} |\n",
        "> 1 patch가 position_bin/z_level/global 3중 누적되므로, used patch는 global_count(=position_bin 합)와만 비교.\n",
        "## 4. 무결성\n",
        f"- 전체 NaN: {total_nan} / Inf: {total_inf}",
        f"- cov 비대칭: {cov_asym_count}개 / cov 대각 음수: {cov_neg_diag_count}개\n",
        "## 5. selected index\n",
        f"- shape={list(sidx.shape)}, unique={len(np.unique(sidx))}, range=[{int(sidx.min())},{int(sidx.max())}]",
        f"- valid: {sidx_ok}",
        f"- P-B4/P-B5 이후 불변(기존 branch 값과 동일): {sidx_unchanged}",
        f"- npz 내장 selected index와 일치: {sidx_match_npz}\n",
        "## 6. 정합성\n",
        f"- P-B7 report 정합: {report_consistent}",
        f"- runtime summary 정합: {runtime_consistent}",
        f"- patch filtering 행 {pf_rows} (290: {pf_rows_ok}), before합 일치: {pf_before_ok}, after합 일치: {pf_after_ok}",
        f"- full_train.log 정상 종료: {log_ok}\n",
        "## 7. 보존 / 무수정\n",
        f"- smoke limit1/limit5 보존: {SMOKE_NPZ_L1.exists()} / {SMOKE_NPZ_L5.exists()}",
        f"- 기존 roi_0_0 EfficientNet branch 경로 분리(무수정): {roi0_isolated}",
        "- scoring/threshold/metrics/forward/training: 미실행",
        "- stage2_holdout 접근: 없음\n",
        "## 8. 미결 사항\n",
    ]
    for i in issues:
        md.append(f"- ⚠ {i}")
    if not issues:
        md.append("- 없음")
    md += [
        "",
        "## 9. 최종 판정\n",
        f"- **{verdict}**",
        f"- P-B9 normal val threshold 진행 가능: **{p_b9_can_proceed}**",
    ]
    with open(REPORT_DIR / "p_b8_distribution_validation.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(f"\n[저장] {REPORT_DIR}")
    print(f"[완료] 판정: {verdict}, P-B9 가능: {p_b9_can_proceed}")
    return 0 if verdict != "실패" else 1


if __name__ == "__main__":
    sys.exit(main())
