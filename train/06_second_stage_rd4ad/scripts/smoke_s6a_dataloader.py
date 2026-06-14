"""
S6ADataset / DataLoader smoke test + preflight 스크립트.

실행 모드:
  --preflight : 파일 존재, split 분포, stage2_holdout 포함 여부만 확인. npz 로드 없음.
  --smoke     : 실제 DataLoader로 train/val 각 2 batch 로드하여
                shape/label/dtype/NaN 확인. CPU에서만 실행.

절대 금지:
  - PNG 생성 없음
  - 모델/optimizer/checkpoint/epoch loop 없음
  - stage2_holdout 사용 없음
  - pip install 없음
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
# 경로 상수
# ---------------------------------------------------------------------------
CROPS_DIR = BASE_DIR / "outputs/second-stage-lesion-refiner-v1/crops_s6a_full"
SUMMARY_CSV = BASE_DIR / "outputs/second-stage-lesion-refiner-v1/reports/crop_s6a_full_summary.csv"
STAGE_SPLIT_CSV = BASE_DIR / "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1_balanced.csv"
VALIDATION_SUMMARY_JSON = BASE_DIR / "outputs/second-stage-lesion-refiner-v1/reports/crop_s6a_full_validation_summary.json"
DATASET_INDEX_CSV = BASE_DIR / "outputs/second-stage-lesion-refiner-v1/datasets/s6a_full_dataset_index.csv"
TRAIN_VAL_SPLIT_CSV = BASE_DIR / "outputs/second-stage-lesion-refiner-v1/datasets/s6a_stage1_train_val_split.csv"

PREFLIGHT_REPORT_DIR = BASE_DIR / "outputs/second-stage-lesion-refiner-v1/reports"
PREFLIGHT_CSV = PREFLIGHT_REPORT_DIR / "s6a_dataset_loader_preflight_summary.csv"
PREFLIGHT_JSON = PREFLIGHT_REPORT_DIR / "s6a_dataset_loader_preflight_summary.json"
PREFLIGHT_MD = PREFLIGHT_REPORT_DIR / "s6a_dataset_loader_preflight_summary.md"

SMOKE_SUMMARY_CSV = PREFLIGHT_REPORT_DIR / "s6a_dataloader_smoke_summary.csv"
SMOKE_SUMMARY_JSON = PREFLIGHT_REPORT_DIR / "s6a_dataloader_smoke_summary.json"
SMOKE_SUMMARY_MD = PREFLIGHT_REPORT_DIR / "s6a_dataloader_smoke_summary.md"


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------

def run_preflight() -> None:
    """파일 존재, split 분포, stage2_holdout 포함 여부 확인. npz 로드 없음."""

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

    # 1. crops_s6a_full 폴더 존재
    check("crops_s6a_full 폴더 존재", CROPS_DIR.exists(), str(CROPS_DIR))

    # 2. full crop summary CSV 존재
    check("full crop summary CSV 존재", SUMMARY_CSV.exists(), str(SUMMARY_CSV))

    # 3. stage split CSV 존재
    check("stage split CSV 존재", STAGE_SPLIT_CSV.exists(), str(STAGE_SPLIT_CSV))

    # 4. validation summary JSON 존재
    check("validation summary JSON 존재", VALIDATION_SUMMARY_JSON.exists(), str(VALIDATION_SUMMARY_JSON))

    # 이후 체크는 파일이 존재할 때만 진행
    if not STAGE_SPLIT_CSV.exists():
        print("[SKIP] stage split CSV 없어서 이후 체크 건너뜀.")
        _save_preflight_results(results, all_passed=False)
        return

    split_df = pd.read_csv(STAGE_SPLIT_CSV, encoding="utf-8-sig")

    # 5. stage1_dev 환자 수 154명 확인
    dev_count = (split_df["stage_split"] == "stage1_dev").sum()
    check(
        "stage1_dev 환자 수 154명",
        dev_count == 154,
        f"실제={dev_count}명",
    )

    # 6. stage2_holdout 환자가 crop 폴더에 있는지 확인 (있으면 FAIL)
    if CROPS_DIR.exists():
        holdout_ids = set(split_df[split_df["stage_split"] == "stage2_holdout"]["patient_id"])
        crop_patient_folders = {p.name for p in CROPS_DIR.iterdir() if p.is_dir()}
        holdout_in_crops = holdout_ids & crop_patient_folders
        check(
            "stage2_holdout 환자가 crops 폴더에 없음",
            len(holdout_in_crops) == 0,
            f"침범 환자={sorted(holdout_in_crops)}" if holdout_in_crops else "OK",
        )
    else:
        check("stage2_holdout crops 폴더 체크", False, "crops 폴더 없어서 확인 불가")

    # 7. train/val 예상 환자 수 계산 (80/20)
    dev_patients = split_df[split_df["stage_split"] == "stage1_dev"]
    dev_n = len(dev_patients)
    expected_train = round(dev_n * 0.8)
    expected_val = dev_n - expected_train
    check(
        "train/val 예상 환자 수 계산 (80/20)",
        dev_n > 0,
        f"total={dev_n}, expected_train~{expected_train}, expected_val~{expected_val}",
    )

    # 8. 예상 positive/hard_negative 수 계산 (summary CSV 기반)
    if SUMMARY_CSV.exists():
        summary_df = pd.read_csv(SUMMARY_CSV, encoding="utf-8-sig")
        # stage1_dev 환자에 해당하는 crop만
        dev_ids = set(dev_patients["patient_id"])
        dev_crops = summary_df[summary_df["patient_id"].isin(dev_ids)]
        pos_count = int((dev_crops["label_int"] == 1).sum()) if "label_int" in dev_crops.columns else -1
        hn_count = int((dev_crops["sampling_label"] == "hard_negative").sum()) if "sampling_label" in dev_crops.columns else -1
        check(
            "예상 positive/hard_negative 수 계산",
            True,
            f"stage1_dev crops: positive={pos_count}, hard_negative={hn_count}, total={len(dev_crops)}",
        )
    else:
        check("예상 positive/hard_negative 수 계산", False, "summary CSV 없어서 확인 불가")

    all_passed = all(r["status"] == "PASS" for r in results)
    _save_preflight_results(results, all_passed=all_passed)


def _save_preflight_results(results: list, all_passed: bool) -> None:
    """preflight 결과를 CSV / JSON / MD로 저장한다."""
    PREFLIGHT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # CSV
    df = pd.DataFrame(results)
    df.to_csv(PREFLIGHT_CSV, index=False, encoding="utf-8")

    # JSON
    summary = {
        "all_passed": all_passed,
        "total_checks": len(results),
        "pass_count": sum(1 for r in results if r["status"] == "PASS"),
        "fail_count": sum(1 for r in results if r["status"] == "FAIL"),
        "checks": results,
    }
    with open(PREFLIGHT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # MD
    lines = [
        "# S6A Dataset Loader Preflight Summary",
        "",
        f"- 전체: {summary['total_checks']}개",
        f"- PASS: {summary['pass_count']}개",
        f"- FAIL: {summary['fail_count']}개",
        f"- 결과: {'**ALL PASS**' if all_passed else '**FAIL 있음**'}",
        "",
        "| check | status | detail |",
        "|-------|--------|--------|",
    ]
    for r in results:
        lines.append(f"| {r['check']} | {r['status']} | {r['detail']} |")

    with open(PREFLIGHT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n[preflight] 결과 저장 완료:")
    print(f"  CSV : {PREFLIGHT_CSV}")
    print(f"  JSON: {PREFLIGHT_JSON}")
    print(f"  MD  : {PREFLIGHT_MD}")
    overall = "ALL PASS" if all_passed else "FAIL 있음"
    print(f"[preflight] 최종 결과: {overall}")


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------

def run_smoke() -> None:
    """DataLoader 실제 로드 확인. CPU에서만 실행."""
    import torch
    from torch.utils.data import DataLoader

    from src.second_stage_verifier.data.s6a_dataset import S6ADataset

    # guard: smoke 결과 파일 이미 있으면 중단
    for p in [SMOKE_SUMMARY_CSV, SMOKE_SUMMARY_JSON, SMOKE_SUMMARY_MD]:
        if p.exists():
            print(f"[FAIL] 출력 파일이 이미 존재합니다. overwrite 방지를 위해 중단합니다.\n  {p}")
            sys.exit(1)

    # guard: index CSV 없으면 중단
    if not DATASET_INDEX_CSV.exists():
        print(f"[FAIL] dataset index CSV 없음. 먼저 build_dataset_index를 실행하세요.\n  {DATASET_INDEX_CSV}")
        sys.exit(1)

    # guard: split CSV 없으면 중단
    if not TRAIN_VAL_SPLIT_CSV.exists():
        print(f"[FAIL] train/val split CSV 없음. 먼저 build_train_val_split을 실행하세요.\n  {TRAIN_VAL_SPLIT_CSV}")
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
    index_df = pd.read_csv(DATASET_INDEX_CSV, encoding="utf-8")
    split_df = pd.read_csv(TRAIN_VAL_SPLIT_CSV, encoding="utf-8")

    # dataset index row 수 확인
    row_count = len(index_df)
    record("dataset index row 수 130,659", row_count == 130_659, f"실제={row_count}")

    # patient 수 확인
    patient_count = index_df["patient_id"].nunique()
    record("patient 수 154", patient_count == 154, f"실제={patient_count}")

    # stage2_holdout guard
    if "stage_split" in split_df.columns:
        holdout_in_split = split_df[split_df["stage_split"] == "stage2_holdout"]
        record("stage2_holdout 0명", len(holdout_in_split) == 0, f"실제={len(holdout_in_split)}명")
        if len(holdout_in_split) > 0:
            print(f"[FAIL] stage2_holdout 환자가 split CSV에 포함되어 있습니다: {holdout_in_split['patient_id'].tolist()}")
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

    train_ids_from_split = set(split_df[split_df["train_val"] == "train"]["patient_id"])
    val_ids_from_split = set(split_df[split_df["train_val"] == "val"]["patient_id"])

    # train/val 환자 수
    train_patient_count = len(train_ids_from_split)
    val_patient_count = len(val_ids_from_split)
    record("train 환자 수", True, f"{train_patient_count}명")
    record("val 환자 수", True, f"{val_patient_count}명")

    # train/val overlap 확인
    overlap = train_ids_from_split & val_ids_from_split
    record("train/val patient overlap 0명", len(overlap) == 0,
           f"overlap={sorted(overlap)}" if overlap else "OK")
    if overlap:
        print(f"[FAIL] train/val 환자 overlap: {sorted(overlap)}")
        sys.exit(1)

    # LUNG1-140 소속 확인
    lung140_rows = split_df[split_df["patient_id"] == "LUNG1-140"]
    if len(lung140_rows) > 0:
        lung140_split = lung140_rows.iloc[0]["train_val"]
        record("LUNG1-140 train/val 소속", True, f"split={lung140_split}")
    else:
        record("LUNG1-140 train/val 소속", False, "LUNG1-140이 split CSV에 없음", warn=True)

    # train/val crop 수
    train_crop_count = int((index_merged["train_val"] == "train").sum())
    val_crop_count = int((index_merged["train_val"] == "val").sum())
    record("train crop 수", True, f"{train_crop_count}개")
    record("val crop 수", True, f"{val_crop_count}개")

    # positive/hard_negative 수
    if "label" in index_merged.columns and "sampling_label" in index_merged.columns:
        train_pos = int(((index_merged["train_val"] == "train") & (index_merged["label"] == 1)).sum())
        train_hn = int(((index_merged["train_val"] == "train") & (index_merged["sampling_label"] == "hard_negative")).sum())
        val_pos = int(((index_merged["train_val"] == "val") & (index_merged["label"] == 1)).sum())
        val_hn = int(((index_merged["train_val"] == "val") & (index_merged["sampling_label"] == "hard_negative")).sum())
        record("train positive/hard_negative 수", True, f"positive={train_pos}, hard_negative={train_hn}")
        record("val positive/hard_negative 수", True, f"positive={val_pos}, hard_negative={val_hn}")
    else:
        record("positive/hard_negative 수 확인", False, "label 또는 sampling_label 컬럼 없음", warn=True)

    # GPU 미사용 확인
    record("GPU 미사용", True, "CPU only (device 설정 없음, torch.cuda 미사용)")

    # model/optimizer/checkpoint/epoch loop 미실행 명시
    record("model/optimizer/checkpoint/epoch loop 미실행", True, "코드 구조상 없음 (DataLoader smoke only)")

    # Dataset 생성
    train_ds = S6ADataset(index_merged, split="train")
    val_ds = S6ADataset(index_merged, split="val")
    print(f"\n[INFO] train dataset size: {len(train_ds)}")
    print(f"[INFO] val   dataset size: {len(val_ds)}")

    # DataLoader
    BATCH_SIZE = 16
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"\n[INFO] batch_size={BATCH_SIZE}, num_workers=0 (CPU only)")

    # train loader 2 batch 확인
    print("\n--- train loader ---")
    train_batch_results = []
    train_success = True
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 2:
            break
        batch_checks = _check_batch(batch, loader_name="train", batch_idx=batch_idx)
        train_batch_results.extend(batch_checks)
        if any(c["status"] == "FAIL" for c in batch_checks):
            train_success = False

    record("train loader 2 batch 성공", train_success,
           "2 batch 확인 완료" if train_success else "FAIL 있음 — 위 batch 결과 확인")

    # val loader 2 batch 확인
    print("\n--- val loader ---")
    val_batch_results = []
    val_success = True
    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= 2:
            break
        batch_checks = _check_batch(batch, loader_name="val", batch_idx=batch_idx)
        val_batch_results.extend(batch_checks)
        if any(c["status"] == "FAIL" for c in batch_checks):
            val_success = False

    record("val loader 2 batch 성공", val_success,
           "2 batch 확인 완료" if val_success else "FAIL 있음 — 위 batch 결과 확인")

    # 전체 check 합산 (summary checks + batch checks)
    all_checks = checks + train_batch_results + val_batch_results

    # split 통계 dict
    split_stats = {
        "dataset_index_row_count": row_count,
        "patient_count": patient_count,
        "train_patient_count": train_patient_count,
        "val_patient_count": val_patient_count,
        "train_crop_count": train_crop_count,
        "val_crop_count": val_crop_count,
    }

    _save_smoke_results(all_checks, split_stats)


def _check_batch(batch: dict, loader_name: str, batch_idx: int) -> list:
    """배치 shape/dtype/NaN/Inf/통계 확인. check 결과 dict 리스트를 반환."""
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

    # shape 확인: (B, 3, 96, 96)
    expected_c, expected_h, expected_w = 3, 96, 96
    shape_ok = (
        image.ndim == 4
        and image.shape[1] == expected_c
        and image.shape[2] == expected_h
        and image.shape[3] == expected_w
    )
    record("image shape (B,3,96,96)", shape_ok, f"shape={tuple(image.shape)}")

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
        f"min={img_np.min():.2f}, max={img_np.max():.2f}, "
        f"mean={img_np.mean():.2f}, std={img_np.std():.2f}"
    )
    record("crop min/max/mean/std", True, stats_str)

    # label 분포 출력
    label_list = label.tolist()
    print(f"{prefix} labels={label_list}")

    return results


def _save_smoke_results(checks: list, split_stats: dict) -> None:
    """smoke 결과를 CSV / JSON / MD로 저장한다."""
    PREFLIGHT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    fail_count = sum(1 for r in checks if r["status"] == "FAIL")
    warn_count = sum(1 for r in checks if r["status"] == "WARN")
    pass_count = sum(1 for r in checks if r["status"] == "PASS")
    all_passed = fail_count == 0
    verdict = "전체 통과" if all_passed else f"미통과 (FAIL {fail_count}개)"

    # CSV
    df = pd.DataFrame(checks)
    df.to_csv(SMOKE_SUMMARY_CSV, index=False, encoding="utf-8")

    # JSON
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
    with open(SMOKE_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # MD
    lines = [
        "# S6A DataLoader Smoke Summary",
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

    with open(SMOKE_SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n[smoke] 결과 저장 완료:")
    print(f"  CSV : {SMOKE_SUMMARY_CSV}")
    print(f"  JSON: {SMOKE_SUMMARY_JSON}")
    print(f"  MD  : {SMOKE_SUMMARY_MD}")
    print(f"[smoke] 최종 결과: {verdict}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="S6ADataset preflight / smoke test"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true", help="파일 존재 및 split 분포 확인 (npz 로드 없음)")
    group.add_argument("--smoke", action="store_true", help="DataLoader 실제 로드 확인 (CPU only)")
    args = parser.parse_args()

    if args.preflight:
        run_preflight()
    elif args.smoke:
        run_smoke()


if __name__ == "__main__":
    main()
