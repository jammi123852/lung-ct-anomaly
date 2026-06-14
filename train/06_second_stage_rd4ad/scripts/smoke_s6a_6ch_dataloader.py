"""
S6ADataset 6ch 모드 DataLoader smoke test 스크립트.

기존 smoke_s6a_dataloader.py 와 별개로 동작하는 6ch 전용 스크립트.
기존 3ch index/결과 파일을 일절 참조하지 않으며, 출력 경로도 분리되어 있다.

실행 모드:
  --preflight : 6ch 관련 파일 존재 및 경로 설계 확인. npz 로드 없음.
  --smoke     : 실제 DataLoader로 train/val 각 2 batch 로드하여
                shape(B,6,96,96)/dtype/NaN/range 확인. CPU에서만 실행.

절대 금지:
  - PNG 생성 없음
  - 모델/optimizer/checkpoint/epoch loop 없음
  - stage2_holdout 사용 없음
  - pip install 없음
  - 기존 3ch index(s6a_full_dataset_index.csv) 참조/덮어쓰기 없음
  - 6ch npz 생성/수정/삭제 없음
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 경로 상수 (6ch 전용 — 기존 3ch 경로와 분리)
# ---------------------------------------------------------------------------
CROPS_6CH_DIR = BASE_DIR / "outputs/second-stage-lesion-refiner-v1/crops_s6a_6ch_full"
DATASET_INDEX_6CH_CSV = BASE_DIR / "outputs/second-stage-lesion-refiner-v1/datasets/s6a_6ch_full_dataset_index.csv"
TRAIN_VAL_SPLIT_CSV = BASE_DIR / "outputs/second-stage-lesion-refiner-v1/datasets/s6a_stage1_train_val_split.csv"
STAGE_SPLIT_CSV = BASE_DIR / "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1_balanced.csv"

REPORT_DIR = BASE_DIR / "outputs/second-stage-lesion-refiner-v1/reports"

PREFLIGHT_CSV = REPORT_DIR / "s6a_6ch_loader_index_preflight_run_v1.csv"
PREFLIGHT_JSON = REPORT_DIR / "s6a_6ch_loader_index_preflight_run_v1.json"
PREFLIGHT_MD = REPORT_DIR / "s6a_6ch_loader_index_preflight_run_v1.md"

SMOKE_CSV = REPORT_DIR / "s6a_6ch_dataloader_smoke_summary.csv"
SMOKE_JSON = REPORT_DIR / "s6a_6ch_dataloader_smoke_summary.json"
SMOKE_MD = REPORT_DIR / "s6a_6ch_dataloader_smoke_summary.md"

# guard: 기존 3ch index 경로 (이 경로로 출력하는 것을 금지)
_FORBIDDEN_3CH_INDEX = BASE_DIR / "outputs/second-stage-lesion-refiner-v1/datasets/s6a_full_dataset_index.csv"

# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------

def run_preflight() -> None:
    """6ch 관련 파일 존재 및 경로 설계 확인. npz 로드 없음."""

    # guard: 출력 파일 이미 있으면 중단
    for p in [PREFLIGHT_CSV, PREFLIGHT_JSON, PREFLIGHT_MD]:
        if p.exists():
            print(f"[FAIL] 출력 파일이 이미 존재합니다. overwrite 방지를 위해 중단합니다.\n  {p}")
            sys.exit(1)

    results = []

    def check(name: str, passed: bool, detail: str = "") -> dict:
        status = "PASS" if passed else "FAIL"
        row = {"check": name, "status": status, "detail": detail}
        results.append(row)
        icon = "[PASS]" if passed else "[FAIL]"
        msg = f"{icon} {name}"
        if detail:
            msg += f"  ({detail})"
        print(msg)
        return row

    # 1. crops_s6a_6ch_full 폴더 존재
    check("crops_s6a_6ch_full 폴더 존재", CROPS_6CH_DIR.exists(), str(CROPS_6CH_DIR))

    # 2. 6ch dataset index CSV 존재
    check("6ch dataset index CSV 존재", DATASET_INDEX_6CH_CSV.exists(), str(DATASET_INDEX_6CH_CSV))

    # 3. 기존 3ch index 경로와 분리 확인
    path_separated = DATASET_INDEX_6CH_CSV.resolve() != _FORBIDDEN_3CH_INDEX.resolve()
    check(
        "6ch index 경로가 기존 3ch index와 분리됨",
        path_separated,
        f"6ch={DATASET_INDEX_6CH_CSV.name}, 3ch={_FORBIDDEN_3CH_INDEX.name}",
    )

    # 4. train/val split CSV 존재 (기존 3ch split 재사용 가능)
    check("train/val split CSV 존재", TRAIN_VAL_SPLIT_CSV.exists(), str(TRAIN_VAL_SPLIT_CSV))

    # 5. stage split CSV 존재
    check("stage split CSV 존재", STAGE_SPLIT_CSV.exists(), str(STAGE_SPLIT_CSV))

    # 6. stage2_holdout guard (split CSV 있으면 확인)
    if STAGE_SPLIT_CSV.exists():
        split_df = pd.read_csv(STAGE_SPLIT_CSV, encoding="utf-8-sig")
        if CROPS_6CH_DIR.exists():
            holdout_ids = set(split_df[split_df["stage_split"] == "stage2_holdout"]["patient_id"])
            crop_patient_folders = {p.name for p in CROPS_6CH_DIR.iterdir() if p.is_dir()}
            holdout_in_crops = holdout_ids & crop_patient_folders
            check(
                "stage2_holdout 환자가 6ch crops 폴더에 없음",
                len(holdout_in_crops) == 0,
                f"침범 환자={sorted(holdout_in_crops)}" if holdout_in_crops else "OK",
            )
        else:
            check("stage2_holdout 6ch crops 폴더 체크", False, "6ch crops 폴더 없어서 확인 불가")

    # 7. 6ch index CSV 내용 확인 (존재하는 경우)
    if DATASET_INDEX_6CH_CSV.exists():
        idx_df = pd.read_csv(DATASET_INDEX_6CH_CSV, encoding="utf-8")
        row_count = len(idx_df)
        patient_count = idx_df["patient_id"].nunique() if "patient_id" in idx_df.columns else -1
        check(
            "6ch index CSV row/patient 수 확인",
            row_count > 0,
            f"rows={row_count}, patients={patient_count}",
        )

    all_passed = all(r["status"] == "PASS" for r in results)
    _save_preflight_results(results, all_passed=all_passed)


def _save_preflight_results(results: list, all_passed: bool) -> None:
    """preflight 결과를 CSV / JSON / MD로 저장한다."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(PREFLIGHT_CSV, index=False, encoding="utf-8")

    summary = {
        "verdict": "전체 통과" if all_passed else "미통과",
        "all_passed": all_passed,
        "total_checks": len(results),
        "pass_count": sum(1 for r in results if r["status"] == "PASS"),
        "fail_count": sum(1 for r in results if r["status"] == "FAIL"),
        "checks": results,
    }
    with open(PREFLIGHT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = [
        "# S6A 6ch Loader Index Preflight Summary",
        "",
        f"## 최종 판정: {'**전체 통과**' if all_passed else '**미통과**'}",
        "",
        f"- 전체: {summary['total_checks']}개",
        f"- PASS: {summary['pass_count']}개",
        f"- FAIL: {summary['fail_count']}개",
        "",
        "| check | status | detail |",
        "|-------|--------|--------|",
    ]
    for r in results:
        lines.append(f"| {r['check']} | {r['status']} | {r.get('detail', '')} |")

    with open(PREFLIGHT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n[preflight] 결과 저장 완료:")
    print(f"  CSV : {PREFLIGHT_CSV}")
    print(f"  JSON: {PREFLIGHT_JSON}")
    print(f"  MD  : {PREFLIGHT_MD}")
    print(f"[preflight] 최종 결과: {'ALL PASS' if all_passed else '미통과'}")


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------

def run_smoke() -> None:
    """DataLoader 실제 로드 확인. CPU에서만 실행. 6ch 전용."""
    import torch
    from torch.utils.data import DataLoader

    from src.second_stage_verifier.data.s6a_dataset import S6ADataset

    # guard: smoke 결과 파일 이미 있으면 중단
    for p in [SMOKE_CSV, SMOKE_JSON, SMOKE_MD]:
        if p.exists():
            print(f"[FAIL] 출력 파일이 이미 존재합니다. overwrite 방지를 위해 중단합니다.\n  {p}")
            sys.exit(1)

    # guard: 6ch index CSV 없으면 중단
    if not DATASET_INDEX_6CH_CSV.exists():
        print(f"[FAIL] 6ch dataset index CSV 없음. 먼저 build_dataset_index를 실행하세요.\n  {DATASET_INDEX_6CH_CSV}")
        sys.exit(1)

    # guard: train/val split CSV 없으면 중단
    if not TRAIN_VAL_SPLIT_CSV.exists():
        print(f"[FAIL] train/val split CSV 없음.\n  {TRAIN_VAL_SPLIT_CSV}")
        sys.exit(1)

    checks = []

    def record(name: str, passed: bool, detail: str = "", warn: bool = False) -> dict:
        if warn:
            status = "WARN"
        else:
            status = "PASS" if passed else "FAIL"
        row = {"check": name, "status": status, "detail": detail}
        checks.append(row)
        icon = f"[{status}]"
        msg = f"{icon} {name}"
        if detail:
            msg += f"  ({detail})"
        print(msg)
        return row

    # 로드
    index_df = pd.read_csv(DATASET_INDEX_6CH_CSV, encoding="utf-8")
    split_df = pd.read_csv(TRAIN_VAL_SPLIT_CSV, encoding="utf-8")

    # row/patient 수 확인
    row_count = len(index_df)
    patient_count = index_df["patient_id"].nunique()
    record("6ch index row 수 > 0", row_count > 0, f"실제={row_count}")
    record("6ch index patient 수 확인", patient_count > 0, f"실제={patient_count}명")

    # stage2_holdout guard
    if "stage_split" in split_df.columns:
        holdout_in_split = split_df[split_df["stage_split"] == "stage2_holdout"]
        record("stage2_holdout 0명", len(holdout_in_split) == 0, f"실제={len(holdout_in_split)}명")
        if len(holdout_in_split) > 0:
            print(f"[FAIL] stage2_holdout 환자가 split CSV에 포함되어 있습니다.")
            sys.exit(1)

    # index_df에 train_val 컬럼 병합
    index_merged = index_df.merge(
        split_df[["patient_id", "train_val"]],
        on="patient_id",
        how="left",
    )

    # train_val NaN row 확인
    nan_row_count = int(index_merged["train_val"].isna().sum())
    record("train_val NaN row 0개", nan_row_count == 0, f"실제={nan_row_count}개")

    train_ids = set(split_df[split_df["train_val"] == "train"]["patient_id"])
    val_ids = set(split_df[split_df["train_val"] == "val"]["patient_id"])

    record("train 환자 수 확인", True, f"{len(train_ids)}명")
    record("val 환자 수 확인", True, f"{len(val_ids)}명")

    overlap = train_ids & val_ids
    record("train/val patient overlap 0명", len(overlap) == 0,
           f"overlap={sorted(overlap)}" if overlap else "OK")
    if overlap:
        print(f"[FAIL] train/val 환자 overlap: {sorted(overlap)}")
        sys.exit(1)

    # GPU 미사용 / model 미실행 명시
    record("GPU 미사용", True, "CPU only (device 설정 없음)")
    record("model/optimizer/checkpoint/epoch loop 미실행", True, "DataLoader smoke only")

    # 6ch Dataset 생성 (image_key="image", expected_channels=6)
    train_ds = S6ADataset(index_merged, split="train", image_key="image", expected_channels=6)
    val_ds = S6ADataset(index_merged, split="val", image_key="image", expected_channels=6)
    print(f"\n[INFO] 6ch train dataset size: {len(train_ds)}")
    print(f"[INFO] 6ch val   dataset size: {len(val_ds)}")

    record("S6ADataset(image_key='image', expected_channels=6) 생성 성공", True,
           f"train={len(train_ds)}, val={len(val_ds)}")

    # DataLoader
    BATCH_SIZE = 16
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"\n[INFO] batch_size={BATCH_SIZE}, num_workers=0 (CPU only)")

    # train loader 2 batch 확인
    print("\n--- 6ch train loader ---")
    train_batch_results = []
    train_success = True
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 2:
            break
        batch_checks = _check_batch_6ch(batch, loader_name="train", batch_idx=batch_idx)
        train_batch_results.extend(batch_checks)
        if any(c["status"] == "FAIL" for c in batch_checks):
            train_success = False

    record("train loader 2 batch 성공", train_success,
           "2 batch 확인 완료" if train_success else "FAIL 있음 — 위 batch 결과 확인")

    # val loader 2 batch 확인
    print("\n--- 6ch val loader ---")
    val_batch_results = []
    val_success = True
    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= 2:
            break
        batch_checks = _check_batch_6ch(batch, loader_name="val", batch_idx=batch_idx)
        val_batch_results.extend(batch_checks)
        if any(c["status"] == "FAIL" for c in batch_checks):
            val_success = False

    record("val loader 2 batch 성공", val_success,
           "2 batch 확인 완료" if val_success else "FAIL 있음 — 위 batch 결과 확인")

    all_checks = checks + train_batch_results + val_batch_results

    split_stats = {
        "dataset_index_row_count": row_count,
        "patient_count": patient_count,
        "train_patient_count": len(train_ids),
        "val_patient_count": len(val_ids),
        "train_crop_count": int((index_merged["train_val"] == "train").sum()),
        "val_crop_count": int((index_merged["train_val"] == "val").sum()),
        "mode": "6ch",
        "image_key": "image",
        "expected_channels": 6,
    }

    _save_smoke_results(all_checks, split_stats)


def _check_batch_6ch(batch: dict, loader_name: str, batch_idx: int) -> list:
    """6ch 배치 shape(B,6,96,96)/dtype/NaN/range 확인."""
    import torch

    image = batch["image"]
    label = batch["label"]

    prefix = f"[{loader_name} batch {batch_idx}]"
    results = []

    def record(name: str, passed: bool, detail: str = "") -> dict:
        status = "PASS" if passed else "FAIL"
        row = {"check": f"{loader_name}_batch{batch_idx}_{name}", "status": status, "detail": detail}
        results.append(row)
        icon = f"[{status}]"
        print(f"{prefix} {icon} {name}  ({detail})" if detail else f"{prefix} {icon} {name}")
        return row

    # shape 확인: (B, 6, 96, 96)
    expected_c, expected_h, expected_w = 6, 96, 96
    shape_ok = (
        image.ndim == 4
        and image.shape[1] == expected_c
        and image.shape[2] == expected_h
        and image.shape[3] == expected_w
    )
    record("image shape (B,6,96,96)", shape_ok, f"shape={tuple(image.shape)}")

    # dtype float32 확인
    dtype_ok = image.dtype == torch.float32
    record("image dtype float32", dtype_ok, f"dtype={image.dtype}")

    # label dtype torch.long 확인
    label_dtype_ok = label.dtype == torch.long
    record("label dtype torch.long", label_dtype_ok, f"dtype={label.dtype}")

    # NaN/Inf 확인
    nan_count = int(torch.isnan(image).sum().item())
    inf_count = int(torch.isinf(image).sum().item())
    record("NaN 0개", nan_count == 0, f"NaN={nan_count}")
    record("Inf 0개", inf_count == 0, f"Inf={inf_count}")

    # min/max/mean/std (info, 항상 PASS)
    img_np = image.numpy()
    stats_str = (
        f"min={img_np.min():.4f}, max={img_np.max():.4f}, "
        f"mean={img_np.mean():.4f}, std={img_np.std():.4f}"
    )
    record("image min/max/mean/std", True, stats_str)

    # range [0, 1] 확인 (경고로 처리 — normalize 여부 미확정)
    range_ok = float(img_np.min()) >= 0.0 and float(img_np.max()) <= 1.0
    results.append({
        "check": f"{loader_name}_batch{batch_idx}_range_0_to_1",
        "status": "PASS" if range_ok else "WARN",
        "detail": f"min={img_np.min():.4f}, max={img_np.max():.4f}",
    })
    icon = "[PASS]" if range_ok else "[WARN]"
    print(f"{prefix} {icon} range [0,1] check  (min={img_np.min():.4f}, max={img_np.max():.4f})")

    # label 분포 출력 (read-only, BCE 학습 정답 사용 금지)
    label_list = label.tolist()
    print(f"{prefix} labels={label_list}  (참고용 read-only)")

    return results


def _save_smoke_results(checks: list, split_stats: dict) -> None:
    """smoke 결과를 CSV / JSON / MD로 저장한다."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    fail_count = sum(1 for r in checks if r["status"] == "FAIL")
    warn_count = sum(1 for r in checks if r["status"] == "WARN")
    pass_count = sum(1 for r in checks if r["status"] == "PASS")
    all_passed = fail_count == 0
    verdict = "전체 통과" if all_passed else f"미통과 (FAIL {fail_count}개)"

    df = pd.DataFrame(checks)
    df.to_csv(SMOKE_CSV, index=False, encoding="utf-8")

    summary = {
        "verdict": verdict,
        "all_passed": all_passed,
        "total_checks": len(checks),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "warn_count": warn_count,
        "split_stats": split_stats,
        "checks": checks,
    }
    with open(SMOKE_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = [
        "# S6A 6ch DataLoader Smoke Summary",
        "",
        f"## 최종 판정: {'**전체 통과**' if all_passed else f'**미통과** (FAIL {fail_count}개)'}",
        "",
        "## Split 통계",
        "",
        "| 항목 | 값 |",
        "|------|-----|",
    ]
    for k, v in split_stats.items():
        lines.append(f"| {k} | {v} |")
    lines += [
        "",
        "## Check 결과",
        "",
        f"- 전체: {len(checks)}개",
        f"- PASS: {pass_count}개",
        f"- FAIL: {fail_count}개",
        f"- WARN: {warn_count}개",
        "",
        "| check | status | detail |",
        "|-------|--------|--------|",
    ]
    for r in checks:
        lines.append(f"| {r['check']} | {r['status']} | {r.get('detail', '')} |")

    with open(SMOKE_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n[smoke] 결과 저장 완료:")
    print(f"  CSV : {SMOKE_CSV}")
    print(f"  JSON: {SMOKE_JSON}")
    print(f"  MD  : {SMOKE_MD}")
    print(f"[smoke] 최종 결과: {verdict}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="S6ADataset 6ch 모드 preflight / smoke test"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true", help="파일 존재 및 경로 설계 확인 (npz 로드 없음)")
    group.add_argument("--smoke", action="store_true", help="DataLoader 실제 로드 확인 (CPU only, 6ch 전용)")
    args = parser.parse_args()

    if args.preflight:
        run_preflight()
    elif args.smoke:
        run_smoke()


if __name__ == "__main__":
    main()
