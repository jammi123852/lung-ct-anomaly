#!/usr/bin/env python3
"""
generate_s4_crop_smoke.py

Second-Stage Lesion Refiner v1 — S4 Crop Smoke Test 전용 스크립트

목적:
- rule_c4_training_sampling_manifest_dryrun.csv에서 S4_patient_balanced만 필터링
- stage1_dev 3~5명 한정으로 2.5D crop (3, H, W) npz 생성
- 저장 구조, label, 2.5D shape, 좌표 crop 정상 여부 확인
- 전체 crop 생성 금지 (환자당 최대 --limit 개)

IMPORTANT:
- stage1_dev 환자만 처리 (stage2_holdout 자동 감지 후 즉시 중단)
- 전체 crop 생성 금지: 환자당 최대 50개 이내 (--limit 기본 30)
- 학습/평가/score 재계산 금지
- 기존 manifest 수정 금지
- v1 outputs 하위 쓰기 금지
- ChatGPT 검토 + 사용자 승인 후에만 실행 가능

2.5D crop 방식:
- 중심 slice (slice_index) 기준 z-1, z, z+1
- z boundary: edge-repeat (z<1 → z=0, z>=Z → z=Z-1)
- crop 좌표: y0,x0,y1,x1 중심 기반 crop_size×crop_size 고정 crop

출력 경로:
- crops:   outputs/second-stage-lesion-refiner-v1/crops_s4_smoke/{patient_id}/{idx:06d}.npz
- summary: outputs/second-stage-lesion-refiner-v1/reports/crop_s4_smoke_summary.csv
           outputs/second-stage-lesion-refiner-v1/reports/crop_s4_smoke_summary.json

권장 smoke 환자 (stage1_dev 전용):
- LUNG1-140: positive 최다 (627개)
- LUNG1-001: hard_negative 최다 (300개)
- LUNG1-156: no_positive_patient=True
- MSD_lung_079: tiny 병변 (mean lesion_pixels=1.0 px)

실행 명령 초안 (ChatGPT 검토 + 사용자 승인 후):
  [dry-run]:
    source ~/ai_env/bin/activate && \\
    python scripts/generate_s4_crop_smoke.py \\
      --patients LUNG1-140 LUNG1-001 LUNG1-156 MSD_lung_079 \\
      --limit 30 --dry-run

  [실제 smoke — 별도 승인 후]:
    source ~/ai_env/bin/activate && \\
    python scripts/generate_s4_crop_smoke.py \\
      --patients LUNG1-140 LUNG1-001 LUNG1-156 MSD_lung_079 \\
      --limit 30

syntax check (실행 아님):
  python -m py_compile scripts/generate_s4_crop_smoke.py
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# ============================================================
# 경로 상수
# ============================================================
REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = (
    REPO_ROOT
    / "outputs/second-stage-lesion-refiner-v1/candidates"
    / "rule_c4_training_sampling_manifest_dryrun.csv"
)
PATHS_CONFIG = REPO_ROOT / "configs/paths.local.yaml"
OUTPUT_BASE = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1"
CROPS_SMOKE_DIR = OUTPUT_BASE / "crops_s4_smoke"
REPORTS_DIR = OUTPUT_BASE / "reports"

# v1 outputs 쓰기 절대 금지
V1_OUTPUT_DIR = REPO_ROOT / "outputs/position-aware-padim-v1"

SCRIPT_NAME = "generate_s4_crop_smoke.py"
SAMPLING_RULE_TARGET = "S4_patient_balanced"
MAX_ALLOWED_LIMIT = 50

# ============================================================
# 봉인 환자 (stage2_holdout 안전장치)
# ============================================================
KNOWN_HOLDOUT_PREFIXES: List[str] = []  # split CSV에서 동적 로딩

# ============================================================
# 유틸
# ============================================================

def load_paths_config() -> Dict:
    if not PATHS_CONFIG.exists():
        print(f"[WARN] paths.local.yaml 없음: {PATHS_CONFIG}", file=sys.stderr)
        return {}
    with open(PATHS_CONFIG, "r") as f:
        return yaml.safe_load(f) or {}


def get_v2_volume_root(paths_cfg: Dict) -> Path:
    v2_path = paths_cfg.get("nsclc_msd_usable_only_v2", "")
    if not v2_path:
        raise RuntimeError(
            "paths.local.yaml에 nsclc_msd_usable_only_v2 경로가 없습니다."
        )
    p = Path(v2_path)
    if not p.exists():
        raise RuntimeError(f"v2 volume 경로가 존재하지 않습니다: {p}")
    return p


def build_patient_to_safeid(v2_root: Path) -> Dict[str, str]:
    """patient_manifest.csv에서 patient_id → safe_id 매핑 구성"""
    manifest_csv = v2_root / "manifests" / "patient_manifest.csv"
    if not manifest_csv.exists():
        raise RuntimeError(f"patient_manifest.csv 없음: {manifest_csv}")
    df = pd.read_csv(manifest_csv, usecols=["patient_id", "safe_id"])
    mapping = dict(zip(df["patient_id"].astype(str), df["safe_id"].astype(str)))
    return mapping


def check_not_v1_output(path: Path) -> None:
    try:
        path.relative_to(V1_OUTPUT_DIR)
        raise RuntimeError(f"v1 output 경로에 쓰기 시도 감지: {path} — 즉시 중단")
    except ValueError:
        pass  # 정상: v1 경로 아님


def load_split_csv() -> pd.DataFrame:
    split_csv = (
        REPO_ROOT
        / "outputs/second-stage-lesion-refiner-v1/splits"
        / "lesion_stage_split_v1_balanced.csv"
    )
    if not split_csv.exists():
        raise RuntimeError(f"split CSV 없음: {split_csv}")
    return pd.read_csv(split_csv, usecols=["patient_id", "stage_split"])


def get_holdout_patients(split_df: pd.DataFrame) -> set:
    return set(
        split_df.loc[split_df["stage_split"] == "stage2_holdout", "patient_id"]
    )


def check_no_holdout(patient_ids: List[str], holdout_set: set) -> None:
    bad = [p for p in patient_ids if p in holdout_set]
    if bad:
        raise RuntimeError(
            f"[ABORT] stage2_holdout 환자 포함 감지 — 즉시 중단: {bad}"
        )


# ============================================================
# crop 좌표 계산
# ============================================================

def compute_fixed_crop_coords(
    cy: float, cx: float, crop_size: int, img_h: int, img_w: int
) -> Tuple[int, int, int, int]:
    """중심 좌표 기반 crop_size×crop_size 고정 crop 좌표 반환 (경계 클램프)"""
    half = crop_size // 2
    y0 = max(0, int(cy) - half)
    x0 = max(0, int(cx) - half)
    y1 = min(img_h, y0 + crop_size)
    x1 = min(img_w, x0 + crop_size)
    # 경계에서 역방향 보정
    if y1 - y0 < crop_size:
        y0 = max(0, y1 - crop_size)
    if x1 - x0 < crop_size:
        x0 = max(0, x1 - crop_size)
    return y0, x0, y1, x1


def load_ct_volume(volume_dir: Path) -> np.ndarray:
    """ct_hu.npy 로딩 (Z, H, W) 형태 검증"""
    ct_path = volume_dir / "ct_hu.npy"
    if not ct_path.exists():
        raise FileNotFoundError(f"ct_hu.npy 없음: {ct_path}")
    ct = np.load(ct_path, mmap_mode="r")
    if ct.ndim != 3:
        raise ValueError(f"ct_hu.npy shape 오류 (Z,H,W) 기대, 실제: {ct.shape}")
    return ct


def extract_2p5d_crop(
    ct: np.ndarray,
    z: int,
    y0: int,
    x0: int,
    y1: int,
    x1: int,
) -> np.ndarray:
    """
    z-1, z, z+1 슬라이스에서 (y0:y1, x0:x1) 추출 → (3, H, W)
    z boundary: edge-repeat
    """
    Z = ct.shape[0]
    z_prev = max(0, z - 1)
    z_next = min(Z - 1, z + 1)
    slices = [
        ct[z_prev, y0:y1, x0:x1].astype(np.float32),
        ct[z, y0:y1, x0:x1].astype(np.float32),
        ct[z_next, y0:y1, x0:x1].astype(np.float32),
    ]
    return np.stack(slices, axis=0)  # (3, H, W)


# ============================================================
# 메인 처리
# ============================================================

def process_patient(
    patient_id: str,
    rows: pd.DataFrame,
    volume_dir: Path,
    out_dir: Path,
    crop_size: int,
    limit: int,
    dry_run: bool,
) -> List[Dict]:
    ct = load_ct_volume(volume_dir)
    _, img_h, img_w = ct.shape

    # positive / hard_negative 균형 (limit 절반씩)
    pos_rows = rows[rows["sampling_label"] == "positive"]
    neg_rows = rows[rows["sampling_label"] == "hard_negative"]
    half = limit // 2
    pos_sample = pos_rows.head(half)
    neg_sample = neg_rows.head(limit - len(pos_sample))
    sampled = pd.concat([pos_sample, neg_sample]).reset_index(drop=True)

    records = []
    for idx, row in sampled.iterrows():
        bbox_cy = (row["y0"] + row["y1"]) / 2.0
        bbox_cx = (row["x0"] + row["x1"]) / 2.0
        cy0, cx0, cy1, cx1 = compute_fixed_crop_coords(
            bbox_cy, bbox_cx, crop_size, img_h, img_w
        )
        # local_z guard
        if "local_z" not in row.index:
            raise ValueError("local_z 컬럼 없음 — 중단")
        if pd.isna(row["local_z"]):
            raise ValueError(f"local_z NaN (slice_index={row['slice_index']}) — 중단")
        z = int(row["local_z"])
        if z < 0:
            raise ValueError(f"local_z < 0: {z} — 중단")
        if z >= ct.shape[0]:
            raise ValueError(f"local_z={z} >= ct.shape[0]={ct.shape[0]} — 중단")
        crop_arr = extract_2p5d_crop(ct, z, cy0, cx0, cy1, cx1)

        rec = {
            "patient_id": patient_id,
            "row_idx": idx,
            "sampling_label": row["sampling_label"],
            "local_z": z,
            "slice_index": int(row["slice_index"]),
            "z_source": "local_z",
            "orig_y0": int(row["y0"]),
            "orig_x0": int(row["x0"]),
            "orig_y1": int(row["y1"]),
            "orig_x1": int(row["x1"]),
            "crop_y0": cy0,
            "crop_x0": cx0,
            "crop_y1": cy1,
            "crop_x1": cx1,
            "crop_shape": list(crop_arr.shape),
            "padim_score": float(row["padim_score"]),
            "lesion_overlap": float(row["lesion_overlap"]),
            "no_positive_patient": bool(row["no_positive_patient"]),
        }

        if not dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
            npz_path = out_dir / f"{idx:06d}.npz"
            check_not_v1_output(npz_path)
            np.savez_compressed(
                npz_path,
                crop=crop_arr,
                label=np.array(row["sampling_label"]),
                patient_id=np.array(patient_id),
                local_z=np.array(z),
                slice_index=np.array(int(row["slice_index"])),
                z_source=np.array("local_z"),
                crop_coords=np.array([cy0, cx0, cy1, cx1]),
                orig_bbox=np.array([int(row["y0"]), int(row["x0"]), int(row["y1"]), int(row["x1"])]),
                padim_score=np.array(float(row["padim_score"])),
            )
            rec["saved_path"] = str(npz_path)
        else:
            rec["saved_path"] = "[dry-run: 저장 안 함]"

        records.append(rec)

    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="S4 crop smoke test (stage1_dev 한정, 실행 전 승인 필수)"
    )
    parser.add_argument(
        "--patients",
        nargs="+",
        required=True,
        help="대상 환자 ID 목록 (stage1_dev만, stage2_holdout 금지)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=30,
        help=f"환자당 최대 crop 수 (기본 30, 최대 {MAX_ALLOWED_LIMIT})",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=96,
        help="crop 크기 px (기본 96)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="실제 npz 저장 없이 경로/shape 확인만",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="기존 crops_s4_smoke 폴더 덮어쓰기 허용 (사용자 명시 승인 시만)",
    )
    args = parser.parse_args()

    # ── 안전 검사 ────────────────────────────────────────
    if args.limit > MAX_ALLOWED_LIMIT:
        print(
            f"[ABORT] --limit {args.limit} > MAX {MAX_ALLOWED_LIMIT} — 전체 crop 방지",
            file=sys.stderr,
        )
        sys.exit(1)

    paths_cfg = load_paths_config()
    v2_root = get_v2_volume_root(paths_cfg)
    patient_to_safeid = build_patient_to_safeid(v2_root)

    split_df = load_split_csv()
    holdout_set = get_holdout_patients(split_df)
    check_no_holdout(args.patients, holdout_set)

    # ── manifest 로딩 및 S4 필터 ─────────────────────────
    if not MANIFEST_PATH.exists():
        print(f"[ABORT] manifest 없음: {MANIFEST_PATH}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(MANIFEST_PATH)
    s4_df = df[df["sampling_rule"] == SAMPLING_RULE_TARGET].copy()
    print(f"[INFO] S4 총 행: {len(s4_df):,} / 대상 환자: {args.patients}")

    # ── 기존 출력 폴더 존재 검사 ─────────────────────────
    if CROPS_SMOKE_DIR.exists() and any(CROPS_SMOKE_DIR.iterdir()) and not args.force:
        print(
            f"[ABORT] crops_s4_smoke 폴더에 기존 파일 존재: {CROPS_SMOKE_DIR}\n"
            f"        삭제 후 재실행하거나 --force 옵션 사용 (사용자 승인 필요)",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── 환자별 처리 ──────────────────────────────────────
    all_records = []
    run_start = datetime.now()

    for patient_id in args.patients:
        safe_id = patient_to_safeid.get(patient_id)
        if safe_id is None:
            print(f"[WARN] patient_id 매핑 없음: {patient_id} — 건너뜀", file=sys.stderr)
            continue

        volume_dir = v2_root / "volumes_npy" / safe_id
        if not volume_dir.exists():
            print(f"[WARN] volume_dir 없음: {volume_dir} — 건너뜀", file=sys.stderr)
            continue

        patient_rows = s4_df[s4_df["patient_id"] == patient_id]
        if patient_rows.empty:
            print(f"[WARN] S4 manifest에서 {patient_id} 없음 — 건너뜀", file=sys.stderr)
            continue

        stage = patient_rows["stage_split"].iloc[0]
        if stage == "stage2_holdout":
            print(f"[ABORT] stage2_holdout 감지: {patient_id}", file=sys.stderr)
            sys.exit(1)

        out_dir = CROPS_SMOKE_DIR / patient_id
        print(
            f"[{patient_id}] stage={stage}, S4 rows={len(patient_rows)}, "
            f"limit={args.limit}, volume={volume_dir}"
        )

        records = process_patient(
            patient_id=patient_id,
            rows=patient_rows,
            volume_dir=volume_dir,
            out_dir=out_dir,
            crop_size=args.crop_size,
            limit=args.limit,
            dry_run=args.dry_run,
        )
        all_records.extend(records)
        pos_cnt = sum(1 for r in records if r["sampling_label"] == "positive")
        neg_cnt = sum(1 for r in records if r["sampling_label"] == "hard_negative")
        print(f"  → crops: {len(records)} (pos={pos_cnt}, neg={neg_cnt})")

    # ── summary 저장 ─────────────────────────────────────
    run_end = datetime.now()
    elapsed = (run_end - run_start).total_seconds()

    summary_records_df = pd.DataFrame(all_records)
    summary_json = {
        "script": SCRIPT_NAME,
        "dry_run": args.dry_run,
        "patients": args.patients,
        "sampling_rule": SAMPLING_RULE_TARGET,
        "crop_size": args.crop_size,
        "limit_per_patient": args.limit,
        "total_crops": len(all_records),
        "positive_crops": int((summary_records_df["sampling_label"] == "positive").sum()) if len(all_records) > 0 else 0,
        "hard_negative_crops": int((summary_records_df["sampling_label"] == "hard_negative").sum()) if len(all_records) > 0 else 0,
        "z_boundary_method": "edge-repeat",
        "2p5d_shape": f"(3, {args.crop_size}, {args.crop_size})",
        "run_start": run_start.isoformat(),
        "run_end": run_end.isoformat(),
        "elapsed_seconds": round(elapsed, 2),
    }

    if not args.dry_run:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = REPORTS_DIR / "crop_s4_smoke_summary.csv"
        json_path = REPORTS_DIR / "crop_s4_smoke_summary.json"
        check_not_v1_output(csv_path)
        check_not_v1_output(json_path)
        if len(all_records) > 0:
            summary_records_df.to_csv(csv_path, index=False)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_json, f, ensure_ascii=False, indent=2)
        print(f"\n[DONE] summary CSV: {csv_path}")
        print(f"[DONE] summary JSON: {json_path}")
    else:
        print("\n[DRY-RUN] 파일 저장 없음")

    print(f"\n[SUMMARY]")
    for k, v in summary_json.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
