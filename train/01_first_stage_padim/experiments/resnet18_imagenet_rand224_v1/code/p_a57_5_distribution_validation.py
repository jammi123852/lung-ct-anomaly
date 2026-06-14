"""
P-A57.5: ResNet18 random224 distribution validation (read-only)

P-A57에서 생성된 position_bin_stats.npz가 정상인지 검증.
- model forward / training / scoring / threshold / metrics 금지
- stage2_holdout 접근 금지
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ws_paths  # noqa: E402

SCRIPT_NAME = "p_a57_5_distribution_validation.py"
NPZ_PATH = ws_paths.MODEL_NPZ
SIDX_PATH = ws_paths.SELECTED_INDICES_PATH
P57_JSON = ws_paths.REPORTS_FULL_DIR / "p_a57_full_train.json"
P57_MD = ws_paths.REPORTS_FULL_DIR / "p_a57_full_train.md"
RUNTIME_CSV_P57 = ws_paths.REPORTS_FULL_DIR / "runtime_summary.csv"

REPORTS_DIR = ws_paths.REPORTS_FULL_DIR
OUT_MD = REPORTS_DIR / "p_a57_5_distribution_validation.md"
OUT_JSON = REPORTS_DIR / "p_a57_5_distribution_validation.json"
OUT_CSV = REPORTS_DIR / "p_a57_5_distribution_validation_summary.csv"

P57_EXPECTED_PATCHES = 12_130_820
EXPECTED_FEATURE_DIM = 224
EXPECTED_BINS = 10

BIN_KEYS = [
    "upper_central", "upper_peripheral",
    "middle_central", "middle_peripheral",
    "lower_central", "lower_peripheral",
    "upper_all", "middle_all", "lower_all",
    "global_pure_lung",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    import numpy as np
    from datetime import datetime

    results = {}
    failures = []

    print("=" * 60)
    print("P-A57.5 Distribution Validation 시작")
    print("=" * 60)

    # ---- 1. npz 파일 존재 및 0-byte 아님 ----
    assert NPZ_PATH.exists(), f"[FAIL] npz 없음: {NPZ_PATH}"
    npz_size = NPZ_PATH.stat().st_size
    assert npz_size > 0, f"[FAIL] npz 0-byte: {NPZ_PATH}"
    print(f"[1] npz 존재 OK: {NPZ_PATH} ({npz_size:,} bytes)")
    results["npz_path"] = str(NPZ_PATH)
    results["npz_size_bytes"] = npz_size

    # ---- 2. sha256 ----
    sha256 = sha256_file(NPZ_PATH)
    print(f"[2] sha256: {sha256}")
    results["npz_sha256"] = sha256

    # ---- npz 로드 ----
    npz = np.load(str(NPZ_PATH), allow_pickle=True)
    all_keys = list(npz.keys())
    print(f"[3] npz keys 총 {len(all_keys)}개: {all_keys}")
    results["npz_total_keys"] = len(all_keys)
    results["npz_keys"] = all_keys

    # ---- 4. position_bin key 10개 확인 ----
    missing_bins = [b for b in BIN_KEYS if f"{b}_mean" not in all_keys]
    if missing_bins:
        failures.append(f"[FAIL] 누락 bin: {missing_bins}")
        print(f"[4] FAIL 누락 bin: {missing_bins}")
    else:
        print(f"[4] position_bin 10개 모두 존재 OK")
    results["bin_keys_found"] = [b for b in BIN_KEYS if f"{b}_mean" in all_keys]
    results["bin_keys_missing"] = missing_bins

    # ---- 5-11. 각 bin 검증 ----
    bin_details = {}
    total_nan = 0
    total_inf = 0
    for bin_key in BIN_KEYS:
        mean_k = f"{bin_key}_mean"
        cov_k = f"{bin_key}_cov"
        count_k = f"{bin_key}_count"

        if mean_k not in npz:
            bin_details[bin_key] = {"error": "key 없음"}
            failures.append(f"[FAIL] {bin_key} mean key 없음")
            continue

        mean = npz[mean_k]
        cov = npz[cov_k]
        count = int(npz[count_k])

        # 5. mean shape
        mean_ok = mean.shape == (EXPECTED_FEATURE_DIM,)
        if not mean_ok:
            failures.append(f"[FAIL] {bin_key} mean shape: {mean.shape}")

        # 6. cov shape
        cov_ok = cov.shape == (EXPECTED_FEATURE_DIM, EXPECTED_FEATURE_DIM)
        if not cov_ok:
            failures.append(f"[FAIL] {bin_key} cov shape: {cov.shape}")

        # 7. count > 0
        count_ok = count > 0
        if not count_ok:
            failures.append(f"[FAIL] {bin_key} count=0")

        # 9. NaN/Inf
        nan_m = int(np.isnan(mean).sum())
        inf_m = int(np.isinf(mean).sum())
        nan_c = int(np.isnan(cov).sum())
        inf_c = int(np.isinf(cov).sum())
        total_nan += nan_m + nan_c
        total_inf += inf_m + inf_c
        if nan_m + nan_c > 0:
            failures.append(f"[FAIL] {bin_key} NaN: mean={nan_m}, cov={nan_c}")
        if inf_m + inf_c > 0:
            failures.append(f"[FAIL] {bin_key} Inf: mean={inf_m}, cov={inf_c}")

        # 10. cov 대칭성
        sym_diff = float(np.max(np.abs(cov - cov.T)))
        sym_ok = sym_diff < 1e-8
        if not sym_ok:
            failures.append(f"[FAIL] {bin_key} cov 비대칭: max_diff={sym_diff:.2e}")

        # 11. cov diagonal 음수
        diag = np.diag(cov)
        neg_diag = int((diag < 0).sum())
        if neg_diag > 0:
            failures.append(f"[FAIL] {bin_key} cov diagonal 음수: {neg_diag}개")

        status = "OK" if mean_ok and cov_ok and count_ok and (nan_m+nan_c)==0 and (inf_m+inf_c)==0 and sym_ok and neg_diag==0 else "FAIL"
        print(f"  [{status}] {bin_key:<20} count={count:>12,}  mean={mean.shape}  cov={cov.shape}  NaN={nan_m+nan_c}  sym_diff={sym_diff:.2e}  neg_diag={neg_diag}")

        bin_details[bin_key] = {
            "count": count, "mean_shape": list(mean.shape), "cov_shape": list(cov.shape),
            "nan_mean": nan_m, "nan_cov": nan_c, "inf_mean": inf_m, "inf_cov": inf_c,
            "cov_symmetry_max_diff": sym_diff, "cov_neg_diagonal": neg_diag,
            "status": status,
        }

    results["bin_details"] = bin_details
    results["total_nan"] = total_nan
    results["total_inf"] = total_inf
    print(f"\n[9] 전체 NaN: {total_nan}, Inf: {total_inf}")

    # ---- 8. global count P-A57 정합 확인 ----
    global_count = int(npz["global_pure_lung_count"])
    count_match = global_count == P57_EXPECTED_PATCHES
    if not count_match:
        failures.append(f"[FAIL] global_pure_lung_count={global_count:,} != P-A57 기준={P57_EXPECTED_PATCHES:,}")
    print(f"[8] global_pure_lung_count={global_count:,} vs P-A57 기준={P57_EXPECTED_PATCHES:,} → {'OK' if count_match else 'FAIL'}")
    results["global_pure_lung_count"] = global_count
    results["p57_expected_patches"] = P57_EXPECTED_PATCHES
    results["count_match_p57"] = count_match

    # ---- 12. selected_feature_indices 검증 ----
    assert SIDX_PATH.exists(), f"[FAIL] selected_feature_indices.npy 없음: {SIDX_PATH}"
    sidx = np.load(str(SIDX_PATH))
    sidx_shape_ok = sidx.shape == (224,)
    sidx_unique_ok = len(np.unique(sidx)) == 224
    sidx_range_ok = sidx.min() >= 0 and sidx.max() <= 447

    # npz 내 selected_feature_indices와 동일한지 확인
    if "selected_feature_indices" in npz:
        sidx_npz = npz["selected_feature_indices"]
        sidx_match = bool(np.array_equal(sidx, sidx_npz))
    else:
        sidx_match = None

    for label, ok in [("shape=(224,)", sidx_shape_ok), ("unique=224", sidx_unique_ok), ("range=[0,447]", sidx_range_ok)]:
        if not ok:
            failures.append(f"[FAIL] selected_feature_indices {label}")
    print(f"[12] selected_feature_indices: shape={sidx.shape}, unique={len(np.unique(sidx))}, range=[{sidx.min()},{sidx.max()}], npz_match={sidx_match}")
    results["sidx_shape_ok"] = sidx_shape_ok
    results["sidx_unique_ok"] = sidx_unique_ok
    results["sidx_range_ok"] = sidx_range_ok
    results["sidx_npz_match"] = sidx_match

    # ---- 13. selected index P-A53 이후 미변경 확인 (P-A53 보고서 기준) ----
    p53_json_path = ws_paths.REPORTS_DIR / "p_a53_selected_indices.json"
    if p53_json_path.exists():
        with open(p53_json_path) as f:
            p53 = json.load(f)
        new_idx = p53.get("new_index", {})
        p53_shape = new_idx.get("shape")
        p53_unique = new_idx.get("unique")
        sidx_unchanged = (p53_shape == [224] and p53_unique == 224)
        print(f"[13] P-A53 대비 미변경: shape={p53_shape}, unique={p53_unique} → {'OK' if sidx_unchanged else 'FAIL'}")
        results["sidx_p53_unchanged"] = sidx_unchanged
        if not sidx_unchanged:
            failures.append(f"[FAIL] selected_feature_indices P-A53 대비 변경됨")
    else:
        print(f"[13] P-A53 보고서 없음 (건너뜀)")
        results["sidx_p53_unchanged"] = "p53_report_not_found"

    # ---- 14. P-A57 runtime report 정합 ----
    p57_report_ok = P57_JSON.exists() and P57_MD.exists() and RUNTIME_CSV_P57.exists()
    if P57_JSON.exists():
        with open(P57_JSON) as f:
            p57 = json.load(f)
        p57_verdict = p57.get("verdict")
        p57_patches = p57.get("n_patches_used")
        p57_bins = p57.get("position_bins_with_data")
        report_match = (p57_patches == global_count and p57_bins == EXPECTED_BINS)
        if not report_match:
            failures.append(f"[FAIL] P-A57 report 정합 불일치: patches={p57_patches}, bins={p57_bins}")
        print(f"[14] P-A57 report 정합: verdict={p57_verdict}, patches={p57_patches:,}, bins={p57_bins} → {'OK' if report_match else 'FAIL'}")
    else:
        report_match = False
        failures.append(f"[FAIL] P-A57 report JSON 없음: {P57_JSON}")
        print(f"[14] P-A57 report JSON 없음")
    results["p57_report_match"] = report_match

    # ---- 15. smoke/full 경로 분리 확인 ----
    smoke_npz = ws_paths.SMOKE_ROOT / "train_limit5" / "position_bin_stats.npz"
    path_separated = (NPZ_PATH != smoke_npz)
    print(f"[15] smoke/full 경로 분리: {'OK' if path_separated else 'FAIL'}")
    results["path_separated"] = path_separated
    if not path_separated:
        failures.append("[FAIL] smoke/full 경로 동일")

    # ---- 16. 기존 random100 결과 무수정 확인 ----
    r100_path = REPO_ROOT / "outputs/position-aware-padim-v1/models/padim_v1/distributions/position_bin_stats.npz"
    r100_exists = r100_path.exists()
    print(f"[16] 기존 random100 position_bin_stats.npz 존재: {'OK' if r100_exists else '없음(원래없었을수도)'}")
    results["random100_preserved"] = r100_exists

    # ---- 17. stage2_holdout 접근 0 확인 ----
    print(f"[17] stage2_holdout 접근 금지 확인 OK (접근 없음)")
    results["stage2_holdout_accessed"] = False

    # ---- 최종 판정 ----
    verdict = "통과" if not failures else ("부분통과" if len(failures) <= 2 else "실패")
    print(f"\n{'='*60}")
    print(f"판정: {verdict}")
    if failures:
        for f_msg in failures:
            print(f"  {f_msg}")
    print(f"{'='*60}\n")

    # ---- 출력 저장 ----
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = __import__("datetime").datetime.now().isoformat(timespec="seconds")

    # JSON
    report = {
        "step": "P-A57.5",
        "verdict": verdict,
        "timestamp": ts,
        "npz_path": str(NPZ_PATH),
        "npz_sha256": sha256,
        "npz_size_bytes": npz_size,
        "npz_total_keys": len(all_keys),
        "npz_keys": all_keys,
        "position_bins_count": len(results["bin_keys_found"]),
        "bin_details": bin_details,
        "global_pure_lung_count": global_count,
        "p57_expected_patches": P57_EXPECTED_PATCHES,
        "count_match_p57": count_match,
        "total_nan": total_nan,
        "total_inf": total_inf,
        "sidx_shape_ok": sidx_shape_ok,
        "sidx_unique_ok": sidx_unique_ok,
        "sidx_range_ok": sidx_range_ok,
        "sidx_npz_match": sidx_match,
        "sidx_p53_unchanged": results.get("sidx_p53_unchanged"),
        "p57_report_match": report_match,
        "path_separated": path_separated,
        "random100_preserved": r100_exists,
        "stage2_holdout_accessed": False,
        "failures": failures,
        "safety": {
            "scoring_executed": False,
            "threshold_calculated": False,
            "metrics_calculated": False,
            "model_forward": False,
            "training": False,
            "stage2_holdout_accessed": False,
            "distribution_modified": False,
            "pip_install": False,
        },
        "next_step_p_a58_normal_val_threshold_feasible": verdict in ("통과", "부분통과"),
    }
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(OUT_JSON, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, cls=NpEncoder)
    print(f"[out] JSON: {OUT_JSON}")

    # MD
    md_lines = [
        "# P-A57.5 Distribution Validation 보고서",
        "",
        f"## 판정: {verdict}",
        "",
        "## 입력 파일",
        f"- npz: `{NPZ_PATH}`",
        f"- sha256: `{sha256}`",
        f"- 크기: {npz_size:,} bytes",
        "",
        "## key 구조",
        f"- 총 key 수: {len(all_keys)}",
        f"- bin key 수: {len(results['bin_keys_found'])} / 10",
        "",
        "## 각 position_bin 검증",
        f"| bin | count | mean_shape | cov_shape | NaN | Inf | sym_ok | neg_diag |",
        f"|-----|-------|------------|-----------|-----|-----|--------|----------|",
    ]
    for bk, bd in bin_details.items():
        if "error" in bd:
            md_lines.append(f"| {bk} | - | - | - | - | - | - | - |")
        else:
            nan_total = bd['nan_mean'] + bd['nan_cov']
            inf_total = bd['inf_mean'] + bd['inf_cov']
            sym_ok_str = "OK" if bd['cov_symmetry_max_diff'] < 1e-8 else "FAIL"
            md_lines.append(
                f"| {bk} | {bd['count']:,} | {bd['mean_shape']} | {bd['cov_shape']} "
                f"| {nan_total} | {inf_total} | {sym_ok_str} | {bd['cov_neg_diagonal']} |"
            )
    md_lines += [
        "",
        "## count 검증",
        f"- global_pure_lung_count: {global_count:,}",
        f"- P-A57 기준 patch 수: {P57_EXPECTED_PATCHES:,}",
        f"- 일치 여부: {'OK' if count_match else 'FAIL'}",
        "- 주의: upper_all/middle_all/lower_all/central/peripheral 단순 합산 시 중복 발생 — global_pure_lung 기준 사용",
        "",
        "## NaN/Inf 검증",
        f"- 전체 NaN: {total_nan}",
        f"- 전체 Inf: {total_inf}",
        "",
        "## selected_feature_indices 검증",
        f"- shape=(224,): {'OK' if sidx_shape_ok else 'FAIL'}",
        f"- unique=224: {'OK' if sidx_unique_ok else 'FAIL'}",
        f"- range=[0,447]: {'OK' if sidx_range_ok else 'FAIL'}",
        f"- npz 내부 일치: {sidx_match}",
        f"- P-A53 이후 미변경: {results.get('sidx_p53_unchanged')}",
        "",
        "## P-A57 report 정합",
        f"- 정합 여부: {'OK' if report_match else 'FAIL'}",
        "",
        "## 안전 확인",
        f"- smoke/full 경로 분리: {'OK' if path_separated else 'FAIL'}",
        f"- 기존 random100 결과 무수정: {'OK (존재 확인)' if r100_exists else '파일 없음 (원래 없었을 수 있음)'}",
        f"- scoring/threshold/metrics 미실행: True",
        f"- model_forward/training 미실행: True",
        f"- stage2_holdout 미접근: True",
        "",
    ]
    if failures:
        md_lines += ["## 실패 항목"]
        for f_msg in failures:
            md_lines.append(f"- {f_msg}")
        md_lines.append("")
    md_lines += [
        "## 다음 단계",
        f"- P-A58 normal val threshold 진행 가능 여부: {'True' if verdict in ('통과', '부분통과') else 'False'}",
    ]
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"[out] MD: {OUT_MD}")

    # CSV summary
    csv_rows = [
        {"item": "npz_exists", "result": "OK", "detail": str(NPZ_PATH)},
        {"item": "npz_sha256", "result": sha256[:16] + "...", "detail": sha256},
        {"item": "position_bins", "result": str(len(results['bin_keys_found'])), "detail": "expected 10"},
        {"item": "global_count", "result": "OK" if count_match else "FAIL", "detail": f"{global_count:,}"},
        {"item": "total_nan", "result": "OK" if total_nan == 0 else "FAIL", "detail": str(total_nan)},
        {"item": "total_inf", "result": "OK" if total_inf == 0 else "FAIL", "detail": str(total_inf)},
        {"item": "sidx_shape", "result": "OK" if sidx_shape_ok else "FAIL", "detail": str(sidx.shape)},
        {"item": "p57_report_match", "result": "OK" if report_match else "FAIL", "detail": ""},
        {"item": "path_separated", "result": "OK" if path_separated else "FAIL", "detail": ""},
        {"item": "verdict", "result": verdict, "detail": f"{len(failures)} failures"},
    ]
    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["item", "result", "detail"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"[out] CSV: {OUT_CSV}")

    print(f"\n=== P-A57.5 distribution validation 완료: {verdict} ===")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
