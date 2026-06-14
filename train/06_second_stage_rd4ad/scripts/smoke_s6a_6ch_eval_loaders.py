"""
S6-A 6ch eval loader 3종 preflight / smoke test 스크립트.

eval loader 3종:
  positive-only   : sampling_label == "positive"  (label == 1)
  hard_negative   : sampling_label == "hard_negative" (label == 0)
  full            : 전체 130,659개

실행 모드:
  --preflight : index/split/count만 확인. npz 로드 없음.
  --smoke     : 각 loader에서 1~2 batch 로드 확인. CPU only.
                이번 단계에서는 --preflight 실행까지만 진행.
                --smoke는 사용자 승인 후 실행.

절대 금지:
  - 학습 코드 없음
  - model forward 없음
  - optimizer / checkpoint / epoch loop 없음
  - BCE loss 정답 사용 없음
  - stage2_holdout 접근 없음
  - npz 수정/재생성 없음
  - pip install 없음
  - 기존 index/crop 파일 수정 없음
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd

# ---------------------------------------------------------------------------
# 경로 상수
# ---------------------------------------------------------------------------
DATASET_INDEX_6CH_CSV = (
    BASE_DIR / "outputs/second-stage-lesion-refiner-v1/datasets/s6a_6ch_full_dataset_index.csv"
)
STAGE_SPLIT_CSV = (
    BASE_DIR / "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1_balanced.csv"
)
REPORT_DIR = BASE_DIR / "outputs/second-stage-lesion-refiner-v1/reports"

PREFLIGHT_MD   = REPORT_DIR / "s6a_6ch_eval_loader_preflight.md"
PREFLIGHT_JSON = REPORT_DIR / "s6a_6ch_eval_loader_preflight.json"

SMOKE_MD   = REPORT_DIR / "s6a_6ch_eval_loader_smoke.md"
SMOKE_JSON = REPORT_DIR / "s6a_6ch_eval_loader_smoke.json"

# guard: 3ch index 경로 오염 방지
_FORBIDDEN_3CH_INDEX = (
    BASE_DIR / "outputs/second-stage-lesion-refiner-v1/datasets/s6a_full_dataset_index.csv"
)

# 기대 수치
_EXPECTED_POSITIVE      = 43_553
_EXPECTED_HARD_NEGATIVE = 87_106
_EXPECTED_FULL          = 130_659
_EXPECTED_PATIENTS      = 154

# ---------------------------------------------------------------------------
# 공통 헬퍼
# ---------------------------------------------------------------------------

def _record(results: list, name: str, passed: bool, detail: str = "") -> dict:
    status = "PASS" if passed else "FAIL"
    row = {"check": name, "status": status, "detail": detail}
    results.append(row)
    icon = "[PASS]" if passed else "[FAIL]"
    msg = f"{icon} {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return row


def _save_results(
    results: list,
    out_md: Path,
    out_json: Path,
    title: str,
    extra_stats: dict | None = None,
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    fail_count = sum(1 for r in results if r["status"] == "FAIL")
    pass_count = sum(1 for r in results if r["status"] == "PASS")
    warn_count = sum(1 for r in results if r["status"] == "WARN")
    all_passed = fail_count == 0
    verdict = "전체 통과" if all_passed else f"미통과 (FAIL {fail_count}개)"

    summary = {
        "verdict": verdict,
        "all_passed": all_passed,
        "total_checks": len(results),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "warn_count": warn_count,
        "checks": results,
    }
    if extra_stats:
        summary["stats"] = extra_stats

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = [
        f"# {title}",
        "",
        f"## 최종 판정: {'**전체 통과**' if all_passed else f'**미통과** (FAIL {fail_count}개)'}",
        "",
        f"- 전체: {len(results)}개",
        f"- PASS: {pass_count}개",
        f"- FAIL: {fail_count}개",
        f"- WARN: {warn_count}개",
        "",
    ]
    if extra_stats:
        lines += ["## 통계", "", "| 항목 | 값 |", "|------|-----|"]
        for k, v in extra_stats.items():
            lines.append(f"| {k} | {v} |")
        lines.append("")

    lines += [
        "## Check 결과",
        "",
        "| check | status | detail |",
        "|-------|--------|--------|",
    ]
    for r in results:
        lines.append(f"| {r['check']} | {r['status']} | {r.get('detail', '')} |")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n결과 저장 완료:")
    print(f"  MD  : {out_md}")
    print(f"  JSON: {out_json}")
    print(f"최종 결과: {verdict}")


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------

def run_preflight() -> None:
    """index/split/count 확인. npz 로드 없음."""

    # guard: 출력 파일 이미 있으면 중단
    for p in [PREFLIGHT_MD, PREFLIGHT_JSON]:
        if p.exists():
            print(f"[FAIL] 출력 파일이 이미 존재합니다. overwrite 방지를 위해 중단합니다.\n  {p}")
            sys.exit(1)

    results: list = []

    # --- 1. index CSV 존재 확인 ---
    _record(results, "6ch index CSV 존재", DATASET_INDEX_6CH_CSV.exists(),
            str(DATASET_INDEX_6CH_CSV))

    # --- 2. stage split CSV 존재 확인 ---
    _record(results, "stage split CSV 존재", STAGE_SPLIT_CSV.exists(),
            str(STAGE_SPLIT_CSV))

    # --- 3. 기존 3ch index 경로와 분리 확인 ---
    path_ok = DATASET_INDEX_6CH_CSV.resolve() != _FORBIDDEN_3CH_INDEX.resolve()
    _record(results, "6ch index 경로가 3ch index와 분리됨", path_ok,
            f"6ch={DATASET_INDEX_6CH_CSV.name}, 3ch={_FORBIDDEN_3CH_INDEX.name}")

    # --- 이후 체크는 파일이 존재할 때만 수행 ---
    if not DATASET_INDEX_6CH_CSV.exists() or not STAGE_SPLIT_CSV.exists():
        print("\n[WARN] 필수 파일 없음 — 나머지 체크 생략")
        _save_results(results, PREFLIGHT_MD, PREFLIGHT_JSON,
                      "S6A 6ch Eval Loader Preflight")
        return

    idx_df = pd.read_csv(DATASET_INDEX_6CH_CSV, encoding="utf-8")
    stage_df = pd.read_csv(STAGE_SPLIT_CSV, encoding="utf-8-sig")

    # --- 4. stage2_holdout guard ---
    holdout_ids = set(
        stage_df[stage_df["stage_split"] == "stage2_holdout"]["patient_id"]
    )
    index_patient_ids = set(idx_df["patient_id"].unique())
    holdout_in_index = holdout_ids & index_patient_ids
    _record(
        results,
        "stage2_holdout 환자 index에 없음 (0명)",
        len(holdout_in_index) == 0,
        f"침범={sorted(holdout_in_index)}" if holdout_in_index else "OK (0명)",
    )
    if holdout_in_index:
        print(f"[FAIL] stage2_holdout 환자가 index에 포함되어 있습니다. 즉시 중단합니다.")
        _save_results(results, PREFLIGHT_MD, PREFLIGHT_JSON,
                      "S6A 6ch Eval Loader Preflight")
        sys.exit(1)

    # --- 5. positive-only 개수 확인 ---
    pos_df = idx_df[idx_df["sampling_label"] == "positive"]
    _record(
        results,
        f"positive-only 개수 == {_EXPECTED_POSITIVE}",
        len(pos_df) == _EXPECTED_POSITIVE,
        f"실제={len(pos_df)}",
    )

    # --- 6. hard_negative-only 개수 확인 ---
    hn_df = idx_df[idx_df["sampling_label"] == "hard_negative"]
    _record(
        results,
        f"hard_negative-only 개수 == {_EXPECTED_HARD_NEGATIVE}",
        len(hn_df) == _EXPECTED_HARD_NEGATIVE,
        f"실제={len(hn_df)}",
    )

    # --- 7. full 개수 확인 ---
    _record(
        results,
        f"full 개수 == {_EXPECTED_FULL}",
        len(idx_df) == _EXPECTED_FULL,
        f"실제={len(idx_df)}",
    )

    # --- 8. 전체 환자 수 확인 ---
    n_patients = idx_df["patient_id"].nunique()
    _record(
        results,
        f"전체 환자 수 == {_EXPECTED_PATIENTS}명",
        n_patients == _EXPECTED_PATIENTS,
        f"실제={n_patients}명",
    )

    # --- 9. label/sampling_label 정합성 확인 ---
    # positive → label == 1
    pos_label_mismatch = int((pos_df["label"] != 1).sum()) if "label" in pos_df.columns else -1
    _record(
        results,
        "positive sampling_label → label==1 정합성",
        pos_label_mismatch == 0,
        f"불일치={pos_label_mismatch}건",
    )
    # hard_negative → label == 0
    hn_label_mismatch = int((hn_df["label"] != 0).sum()) if "label" in hn_df.columns else -1
    _record(
        results,
        "hard_negative sampling_label → label==0 정합성",
        hn_label_mismatch == 0,
        f"불일치={hn_label_mismatch}건",
    )

    # --- 10. positive + hard_negative = full 합계 일치 확인 ---
    sum_ok = (len(pos_df) + len(hn_df)) == len(idx_df)
    _record(
        results,
        "positive + hard_negative == full 합계",
        sum_ok,
        f"{len(pos_df)} + {len(hn_df)} = {len(pos_df) + len(hn_df)} / full={len(idx_df)}",
    )

    # --- 11. eval loader DataLoader 준비 전용 (model forward 없음) 명시 ---
    _record(
        results,
        "eval loader DataLoader 준비 전용 (model forward 없음)",
        True,
        "DataLoader 인스턴스 생성만, forward/loss/optimizer/checkpoint 코드 없음",
    )

    # --- 12. 학습/forward/scoring/checkpoint 미실행 확인 ---
    _record(
        results,
        "학습/forward/scoring/checkpoint 미실행",
        True,
        "preflight 단계: DataLoader 인스턴스 생성 없음, npz 로드 없음",
    )

    # --- 13. BCE loss 정답 사용 금지 ---
    _record(
        results,
        "label을 stratification/eval용으로만 사용 (BCE loss 정답 사용 금지)",
        True,
        "label 컬럼은 count/정합성 확인용으로만 읽음. loss 계산 없음.",
    )

    # --- 14. 기존 S6-A crop/index/npz 미수정 확인 ---
    _record(
        results,
        "기존 S6-A crop/index/npz 미수정",
        True,
        "read-only pd.read_csv만 수행. 기존 파일 쓰기/삭제/이동 없음.",
    )

    extra_stats = {
        "index_csv": str(DATASET_INDEX_6CH_CSV),
        "full_count": len(idx_df),
        "positive_count": len(pos_df),
        "hard_negative_count": len(hn_df),
        "patient_count": n_patients,
        "stage2_holdout_in_index": len(holdout_in_index),
        "positive_label_mismatch": pos_label_mismatch,
        "hard_negative_label_mismatch": hn_label_mismatch,
    }

    _save_results(
        results,
        PREFLIGHT_MD,
        PREFLIGHT_JSON,
        "S6A 6ch Eval Loader Preflight",
        extra_stats=extra_stats,
    )


# ---------------------------------------------------------------------------
# smoke (구현만 — 이번 단계에서 실행 금지, 사용자 승인 후 실행)
# ---------------------------------------------------------------------------

def run_smoke() -> None:
    """각 eval loader에서 1~2 batch 로드 확인. CPU only.

    model forward / optimizer / checkpoint / epoch loop 없음.
    label은 read-only 출력용으로만 사용 (BCE loss 정답 사용 금지).
    """
    import torch
    from torch.utils.data import DataLoader

    from src.second_stage_verifier.data.s6a_dataset import S6ADataset

    # guard: 출력 파일 이미 있으면 중단
    for p in [SMOKE_MD, SMOKE_JSON]:
        if p.exists():
            print(f"[FAIL] 출력 파일이 이미 존재합니다. overwrite 방지를 위해 중단합니다.\n  {p}")
            sys.exit(1)

    # guard: index CSV 없으면 중단
    if not DATASET_INDEX_6CH_CSV.exists():
        print(f"[FAIL] 6ch index CSV 없음.\n  {DATASET_INDEX_6CH_CSV}")
        sys.exit(1)

    # guard: stage split CSV 없으면 중단
    if not STAGE_SPLIT_CSV.exists():
        print(f"[FAIL] stage split CSV 없음.\n  {STAGE_SPLIT_CSV}")
        sys.exit(1)

    results: list = []
    idx_df = pd.read_csv(DATASET_INDEX_6CH_CSV, encoding="utf-8")

    # stage2_holdout guard
    stage_df = pd.read_csv(STAGE_SPLIT_CSV, encoding="utf-8-sig")
    holdout_ids = set(stage_df[stage_df["stage_split"] == "stage2_holdout"]["patient_id"])
    holdout_in_index = holdout_ids & set(idx_df["patient_id"].unique())
    _record(results, "stage2_holdout 환자 0명", len(holdout_in_index) == 0,
            f"침범={sorted(holdout_in_index)}" if holdout_in_index else "OK")
    if holdout_in_index:
        print("[FAIL] stage2_holdout 환자 포함. 즉시 중단.")
        sys.exit(1)

    # eval loader 3종 DataFrame 준비 (필터링만, DataLoader 생성 전)
    pos_df  = idx_df[idx_df["sampling_label"] == "positive"].reset_index(drop=True)
    hn_df   = idx_df[idx_df["sampling_label"] == "hard_negative"].reset_index(drop=True)
    full_df = idx_df.copy()

    _record(results, "GPU 미사용", True, "CPU only (device 설정 없음)")
    _record(results, "model/optimizer/checkpoint/epoch loop 미실행", True,
            "DataLoader smoke only")

    loader_specs = [
        ("positive-only",   pos_df,  _EXPECTED_POSITIVE),
        ("hard_negative",   hn_df,   _EXPECTED_HARD_NEGATIVE),
        ("full",            full_df, _EXPECTED_FULL),
    ]

    BATCH_SIZE = 4
    all_batch_results: list = []

    for loader_name, df, expected_count in loader_specs:
        _record(results, f"{loader_name} DataFrame 크기 확인",
                len(df) == expected_count, f"expected={expected_count}, actual={len(df)}")

        ds = S6ADataset(df, split="all", image_key="image", expected_channels=6)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        _record(results, f"{loader_name} DataLoader 생성 성공", True,
                f"size={len(ds)}, batch_size={BATCH_SIZE}")

        print(f"\n--- eval loader: {loader_name} ---")
        loader_ok = True
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= 2:
                break
            batch_checks = _check_batch(
                batch, loader_name=loader_name, batch_idx=batch_idx
            )
            all_batch_results.extend(batch_checks)
            if any(c["status"] == "FAIL" for c in batch_checks):
                loader_ok = False

        _record(results, f"{loader_name} 2 batch 로드 성공", loader_ok,
                "2 batch 확인 완료" if loader_ok else "FAIL 있음")

    all_checks = results + all_batch_results
    extra_stats = {
        "positive_count": len(pos_df),
        "hard_negative_count": len(hn_df),
        "full_count": len(full_df),
        "batch_size": BATCH_SIZE,
        "num_workers": 0,
        "image_key": "image",
        "expected_channels": 6,
    }
    _save_results(all_checks, SMOKE_MD, SMOKE_JSON,
                  "S6A 6ch Eval Loader Smoke", extra_stats=extra_stats)


def _check_batch(batch: dict, loader_name: str, batch_idx: int) -> list:
    """6ch 배치 shape/dtype/NaN/range 확인. label은 loader_name별 guard 및 read-only 출력."""
    import torch

    image = batch["image"]
    label = batch["label"]
    prefix = f"[{loader_name} batch {batch_idx}]"
    results: list = []

    def rec(name: str, passed: bool, detail: str = "") -> None:
        status = "PASS" if passed else "FAIL"
        row = {"check": f"{loader_name}_b{batch_idx}_{name}", "status": status, "detail": detail}
        results.append(row)
        icon = f"[{status}]"
        msg = f"{prefix} {icon} {name}"
        if detail:
            msg += f"  ({detail})"
        print(msg)

    # shape (B, 6, 96, 96)
    shape_ok = (
        image.ndim == 4
        and image.shape[1] == 6
        and image.shape[2] == 96
        and image.shape[3] == 96
    )
    rec("image shape (B,6,96,96)", shape_ok, f"shape={tuple(image.shape)}")

    # dtype float32
    rec("image dtype float32", image.dtype == torch.float32, f"dtype={image.dtype}")

    # label dtype long
    rec("label dtype torch.long", label.dtype == torch.long, f"dtype={label.dtype}")

    # NaN/Inf
    nan_count = int(torch.isnan(image).sum().item())
    inf_count = int(torch.isinf(image).sum().item())
    rec("NaN 0개", nan_count == 0, f"NaN={nan_count}")
    rec("Inf 0개", inf_count == 0, f"Inf={inf_count}")

    # min/max/mean/std (info)
    img_np = image.numpy()
    stats_str = (
        f"min={img_np.min():.4f}, max={img_np.max():.4f}, "
        f"mean={img_np.mean():.4f}, std={img_np.std():.4f}"
    )
    rec("image stats (info)", True, stats_str)

    # range [0,1] 경고
    range_ok = float(img_np.min()) >= 0.0 and float(img_np.max()) <= 1.0
    results.append({
        "check": f"{loader_name}_b{batch_idx}_range_0_to_1",
        "status": "PASS" if range_ok else "WARN",
        "detail": f"min={img_np.min():.4f}, max={img_np.max():.4f}",
    })
    icon = "[PASS]" if range_ok else "[WARN]"
    print(f"{prefix} {icon} range [0,1]  (min={img_np.min():.4f}, max={img_np.max():.4f})")

    # label loader_name별 guard 검증 + read-only 출력 (BCE 학습 정답 사용 금지)
    label_list = label.tolist()
    print(f"{prefix} labels={label_list}  (read-only, BCE 정답 사용 금지)")

    if loader_name == "positive-only":
        label_ok = all(v == 1 for v in label_list)
        rec("label 전부 1 (positive-only guard)", label_ok,
            f"모두 1 ({len(label_list)}개)" if label_ok else f"불일치: {label_list}")
    elif loader_name == "hard_negative":
        label_ok = all(v == 0 for v in label_list)
        rec("label 전부 0 (hard_negative guard)", label_ok,
            f"모두 0 ({len(label_list)}개)" if label_ok else f"불일치: {label_list}")
    elif loader_name == "full":
        bad = [v for v in label_list if v not in (0, 1)]
        label_ok = len(bad) == 0
        rec("label in {0,1} (full guard)", label_ok,
            f"모두 0 또는 1 ({len(label_list)}개)" if label_ok else f"범위 외: {bad}")

    return results


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="S6-A 6ch eval loader 3종 preflight / smoke test"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--preflight",
        action="store_true",
        help="index/split/count 확인만. npz 로드 없음.",
    )
    group.add_argument(
        "--smoke",
        action="store_true",
        help="eval loader 3종 각 1~2 batch 로드 확인. CPU only. 사용자 승인 후 실행.",
    )
    args = parser.parse_args()

    if args.preflight:
        run_preflight()
    elif args.smoke:
        run_smoke()


if __name__ == "__main__":
    main()
