"""
generate_visualizations.py: Task 7.5 - top-k 후보 시각화 파일 생성 스크립트.

저장 파일:
  outputs/position-aware-padim-v1/visualizations/{model}/{patient_id}/
    top{rank}_overlay.png
    top{rank}_thumbnail.png
    top{rank}_metadata.json

금지:
  - full_export_heatmaps
  - 전체 slice 저장
  - 전체 환자 루프
  - cv2 / matplotlib
  - 병변 데이터 실행
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

# 프로젝트 루트를 sys.path에 추가
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from position_aware_padim.candidate_card_generator import CandidateCardGenerator
from position_aware_padim.candidate_ranker import CandidateRanker
from position_aware_padim.data_loader import DataLoader
from position_aware_padim.heatmap_generator import HeatmapGenerator
from position_aware_padim.path_resolver import PathResolver
from position_aware_padim.score_aggregator import ScoreAggregator

# ------------------------------------------------------------------
# 경로 상수
# ------------------------------------------------------------------
REPORTS_DIR = REPO_ROOT / "outputs" / "position-aware-padim-v1" / "reports"
RUNTIME_SUMMARY_CSV = REPORTS_DIR / "runtime_summary.csv"
ERROR_CSV = REPORTS_DIR / "error.csv"

RUNTIME_COLUMNS = ["timestamp", "script", "metric", "value"]
ERROR_COLUMNS = ["patient_id", "error_type", "error_msg", "file_logical"]

SCRIPT_NAME = "generate_visualizations.py"


# ------------------------------------------------------------------
# CSV 기록 헬퍼 (기존 4컬럼 스키마 유지)
# ------------------------------------------------------------------

def _append_runtime(metric: str, value: str) -> None:
    RUNTIME_SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = (
        not RUNTIME_SUMMARY_CSV.exists()
        or RUNTIME_SUMMARY_CSV.stat().st_size == 0
    )
    with open(RUNTIME_SUMMARY_CSV, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUNTIME_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "script": SCRIPT_NAME,
            "metric": metric,
            "value": value,
        })


def _append_error(
    patient_id: str, error_type: str, error_msg: str, file_logical: str
) -> None:
    ERROR_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = (
        not ERROR_CSV.exists() or ERROR_CSV.stat().st_size == 0
    )
    with open(ERROR_CSV, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ERROR_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "patient_id": patient_id,
            "error_type": error_type,
            "error_msg": error_msg,
            "file_logical": file_logical,
        })


# ------------------------------------------------------------------
# candidate-csv 모드
# ------------------------------------------------------------------

MAX_CANDIDATES_HARD_LIMIT = 50


def run_candidate_csv_mode(args) -> None:
    """candidate CSV(rank 정렬된 patch-level top-k 후보)를 받아 시각화한다.

    - --patient-id 모드와 별도로 동작한다.
    - 환자별 DataLoader 로드는 1회만 수행한다.
    - 출력: outputs/.../visualizations/{model}/topk_candidates/
      파일명: rank{NNN}_{patient_id}_z{local_z}_{overlay|thumbnail|metadata}.{png|json}
    """
    model = args.model
    score_col = args.score_col
    max_candidates = args.max_candidates
    csv_path = Path(args.candidate_csv)

    # --- 안전 가드 ---
    if max_candidates <= 0:
        msg = f"--max-candidates는 양의 정수여야 합니다: {max_candidates}"
        print(f"ERROR: {msg}")
        sys.exit(1)
    if max_candidates > MAX_CANDIDATES_HARD_LIMIT:
        msg = (
            f"--max-candidates({max_candidates})가 안전 상한"
            f"({MAX_CANDIDATES_HARD_LIMIT})을 초과합니다."
        )
        _append_error("__candidate_csv__", "max_candidates_too_large", msg, str(csv_path))
        print(f"ERROR: {msg}")
        sys.exit(1)

    if not csv_path.exists():
        msg = f"candidate CSV 없음: {csv_path}"
        _append_error("__candidate_csv__", "file_not_found", msg, str(csv_path))
        print(f"ERROR: {msg}")
        sys.exit(1)

    # --- candidate CSV 로드 + 필수 컬럼 검증 ---
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    required_cols = [
        "patient_id", "safe_id", "local_z",
        "y0", "x0", "y1", "x1",
        "position_bin", "rank", "model", "score_col", score_col,
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        msg = f"candidate CSV 필수 컬럼 누락: {missing}"
        _append_error("__candidate_csv__", "missing_columns", msg, str(csv_path))
        print(f"ERROR: {msg}")
        sys.exit(1)

    # --- CLI vs CSV 컬럼 값 일치 검증 ---
    csv_models = sorted(set(df["model"].astype(str).unique().tolist()))
    if csv_models != [model]:
        msg = (
            f"candidate CSV의 model 값({csv_models})이 CLI --model({model})과 "
            f"일치하지 않습니다."
        )
        _append_error("__candidate_csv__", "model_mismatch", msg, str(csv_path))
        print(f"ERROR: {msg}")
        sys.exit(1)
    csv_score_cols = sorted(set(df["score_col"].astype(str).unique().tolist()))
    if csv_score_cols != [score_col]:
        msg = (
            f"candidate CSV의 score_col 값({csv_score_cols})이 CLI --score-col"
            f"({score_col})과 일치하지 않습니다."
        )
        _append_error("__candidate_csv__", "score_col_mismatch", msg, str(csv_path))
        print(f"ERROR: {msg}")
        sys.exit(1)

    # --- rank 오름차순 정렬 + max_candidates개만 ---
    df_sorted = df.sort_values(by="rank", ascending=True, kind="stable").reset_index(drop=True)
    if max_candidates > len(df_sorted):
        print(
            f"[WARN] --max-candidates({max_candidates}) > CSV rows({len(df_sorted)}); "
            f"CSV rows 전체를 처리합니다."
        )
    df_top = df_sorted.head(max_candidates).copy()

    # --- 경로 설정 ---
    cfg_path = REPO_ROOT / "configs" / "paths.local.yaml"
    with open(cfg_path, encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)
    base_path = cfg["normal_training_ready"]
    manifest_path = str(Path(base_path) / "manifests" / "patient_manifest.csv")

    # candidate-csv 모드 출력 하위 폴더 결정 (기존 기본값 "topk_candidates" 보존)
    output_subdir = getattr(args, "output_subdir", None) or "topk_candidates"
    if "/" in output_subdir or "\\" in output_subdir or output_subdir.startswith("."):
        msg = f"--output-subdir에는 경로 구분자/상대경로 금지: {output_subdir!r}"
        _append_error("__candidate_csv__", "invalid_output_subdir", msg, str(csv_path))
        print(f"ERROR: {msg}")
        sys.exit(1)

    out_dir = (
        REPO_ROOT
        / "outputs"
        / "position-aware-padim-v1"
        / "visualizations"
        / model
        / output_subdir
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[generate_visualizations] mode=candidate-csv, model={model}, "
          f"score_col={score_col}, max_candidates={max_candidates}")
    print(f"  candidate_csv : {csv_path}")
    print(f"  out_dir       : {out_dir}")
    print(f"  처리 후보 수   : {len(df_top)} (전체 {len(df)} 중)")
    print()

    # --- DataLoader / HeatmapGenerator / CandidateCardGenerator ---
    pr = PathResolver(manifest_path=manifest_path, base_path=base_path)
    loader = DataLoader(
        manifest_path=manifest_path,
        path_resolver=pr,
        error_csv_path=str(ERROR_CSV),
        use_mmap=True,
    )
    hgen = HeatmapGenerator(score_col=score_col, overlap_mode="max")
    cgen = CandidateCardGenerator(score_col=score_col)

    saved_count = 0

    # --- patient_id별로 그룹화 (CT 1회 로드) ---
    for patient_id, grp in df_top.groupby("patient_id", sort=False):
        # CT 로드
        patient_data = loader.load_patient_data(patient_id)
        if patient_data is None:
            msg = f"DataLoader.load_patient_data 실패: {patient_id}"
            _append_error(patient_id, "load_error", msg, "ct_hu")
            print(f"  [FAIL] {patient_id}: {msg}")
            continue
        ct_hu = patient_data["ct_hu"]
        print(f"  CT 로드 : {patient_id} shape={ct_hu.shape}")

        # 해당 환자의 score CSV (slice 단위 overlay 생성에 필요)
        score_csv_path = (
            REPO_ROOT
            / "outputs"
            / "position-aware-padim-v1"
            / "scores"
            / model
            / "by_patient"
            / f"{patient_id}.csv"
        )
        if not score_csv_path.exists():
            msg = f"score CSV 없음: {score_csv_path}"
            _append_error(patient_id, "score_csv_not_found", msg, str(score_csv_path))
            print(f"  [FAIL] {patient_id}: {msg}")
            continue
        score_df = pd.read_csv(score_csv_path, encoding="utf-8-sig")

        # 후보별 처리
        for _, row in grp.iterrows():
            rank = int(row["rank"])
            local_z = int(row["local_z"])
            try:
                ct_slice = ct_hu[local_z]
                df_slice = score_df[score_df["local_z"] == local_z].copy()

                overlay = hgen.create_overlay(ct_slice, df_slice, alpha=0.5)
                candidate = row.to_dict()
                card = cgen.make_card(candidate, overlay, rank=rank, lesion_mask=None)

                # 파일명: rank{NNN}_{patient_id}_z{local_z}_{type}
                stem = f"rank{rank:03d}_{patient_id}_z{local_z}"
                overlay_path = out_dir / f"{stem}_overlay.png"
                thumb_path = out_dir / f"{stem}_thumbnail.png"
                meta_path = out_dir / f"{stem}_metadata.json"

                Image.fromarray(card["overlay_with_box"]).save(str(overlay_path))
                thumb = card["thumbnail"]
                if thumb.size > 0:
                    Image.fromarray(thumb).save(str(thumb_path))
                else:
                    Image.fromarray(np.zeros((1, 1, 3), dtype=np.uint8)).save(str(thumb_path))

                metadata = {
                    "rank": rank,
                    "patient_id": str(row["patient_id"]),
                    "safe_id": str(row["safe_id"]),
                    "local_z": local_z,
                    "y0": int(row["y0"]),
                    "x0": int(row["x0"]),
                    "y1": int(row["y1"]),
                    "x1": int(row["x1"]),
                    "position_bin": str(row["position_bin"]),
                    "model": model,
                    "score_col": score_col,
                    "score": float(row[score_col]),
                }
                with open(meta_path, "w", encoding="utf-8") as jf:
                    json.dump(metadata, jf, ensure_ascii=False, indent=2)

                print(
                    f"    rank={rank:>3} z={local_z} score={metadata['score']:.4f} "
                    f"overlay={card['overlay_with_box'].shape} "
                    f"thumb={card['thumbnail'].shape}"
                )
                saved_count += 1

            except Exception as exc:
                _append_error(
                    patient_id, "visualization_error", str(exc),
                    f"rank{rank}_local_z{local_z}",
                )
                print(f"    rank={rank} ERROR: {exc}")

    # --- runtime_summary 기록 (기존 4컬럼 스키마) ---
    _append_runtime("mode", "candidate-csv")
    _append_runtime("model", model)
    _append_runtime("score_col", score_col)
    _append_runtime("max_candidates", str(max_candidates))
    _append_runtime("candidates_processed", str(len(df_top)))
    _append_runtime("candidates_saved", str(saved_count))
    _append_runtime("candidate_csv", str(csv_path))
    _append_runtime("output_dir", str(out_dir))

    print(f"\n완료: {saved_count}건 저장 → {out_dir}")


# ------------------------------------------------------------------
# 메인
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="top-k 후보 시각화 파일 생성")
    parser.add_argument("--model", required=True, help="모델 이름 (예: padim_v1)")

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--patient-id", dest="patient_id",
        help="단일 환자 ID (예: normal023). --candidate-csv와 동시 사용 불가.",
    )
    mode_group.add_argument(
        "--candidate-csv", dest="candidate_csv",
        help="candidate CSV 경로. --patient-id와 동시 사용 불가.",
    )

    parser.add_argument("--score-col", required=True, dest="score_col",
                        help="score 컬럼명 (예: padim_score)")
    parser.add_argument("--top-k", type=int, default=1, dest="top_k",
                        help="단일 환자 모드 상위 후보 수 (기본: 1)")
    parser.add_argument(
        "--max-candidates", type=int, default=10, dest="max_candidates",
        help=f"candidate-csv 모드 처리 후보 수 (기본: 10, 최대: {MAX_CANDIDATES_HARD_LIMIT})",
    )
    parser.add_argument(
        "--output-subdir", default=None, dest="output_subdir",
        help=(
            "candidate-csv 모드 출력 하위 폴더명 (기본: topk_candidates). "
            "다른 후보 셋과 분리해 저장할 때 사용. 단일 환자 모드에서는 무시됨."
        ),
    )
    args = parser.parse_args()

    # --- candidate-csv 모드 분기 ---
    if args.candidate_csv:
        run_candidate_csv_mode(args)
        return

    # --- 단일 환자 모드 (기존 로직 보존) ---
    model      = args.model
    patient_id = args.patient_id
    score_col  = args.score_col
    top_k      = args.top_k

    # --- 경로 설정 ---
    cfg_path = REPO_ROOT / "configs" / "paths.local.yaml"
    with open(cfg_path, encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)
    base_path     = cfg["normal_training_ready"]
    manifest_path = str(Path(base_path) / "manifests" / "patient_manifest.csv")

    score_csv_path = (
        REPO_ROOT
        / "outputs"
        / "position-aware-padim-v1"
        / "scores"
        / model
        / "by_patient"
        / f"{patient_id}.csv"
    )
    out_dir = (
        REPO_ROOT
        / "outputs"
        / "position-aware-padim-v1"
        / "visualizations"
        / model
        / patient_id
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[generate_visualizations] model={model}, patient_id={patient_id}, "
          f"score_col={score_col}, top_k={top_k}")

    # --- score CSV 로드 ---
    if not score_csv_path.exists():
        msg = f"score CSV 없음: {score_csv_path}"
        _append_error(patient_id, "file_not_found", msg, "score_csv")
        print(f"ERROR: {msg}")
        sys.exit(1)

    aggregator = ScoreAggregator(score_col=score_col)
    df = aggregator.load_csv(str(score_csv_path))
    print(f"score CSV 로드: {score_csv_path} ({len(df)} rows)")

    # --- 상위 후보 선택 ---
    ranker   = CandidateRanker(score_col=score_col)
    top_df   = ranker.rank_patches(df, top_k=top_k)
    print(f"top_{top_k} 후보 선택 완료")

    # --- CT 로드 ---
    pr     = PathResolver(manifest_path=manifest_path, base_path=base_path)
    loader = DataLoader(
        manifest_path=manifest_path,
        path_resolver=pr,
        error_csv_path=str(ERROR_CSV),
        use_mmap=True,
    )
    patient_data = loader.load_patient_data(patient_id)
    if patient_data is None:
        msg = f"DataLoader.load_patient_data 실패: {patient_id}"
        _append_error(patient_id, "load_error", msg, "ct_hu")
        print(f"ERROR: {msg}")
        sys.exit(1)

    ct_hu = patient_data["ct_hu"]
    print(f"CT 로드 완료: shape={ct_hu.shape}")

    # --- 후보별 시각화 생성 및 저장 ---
    hgen   = HeatmapGenerator(score_col=score_col, overlap_mode="max")
    cgen   = CandidateCardGenerator(score_col=score_col)
    saved_count = 0

    for rank, (_, row) in enumerate(top_df.iterrows(), start=1):
        local_z = int(row["local_z"])
        try:
            ct_slice = ct_hu[local_z]
            df_slice = df[df["local_z"] == local_z].copy()

            # overlay 생성
            overlay = hgen.create_overlay(ct_slice, df_slice, alpha=0.5)

            # 카드 생성
            candidate = row.to_dict()
            card = cgen.make_card(candidate, overlay, rank=rank, lesion_mask=None)

            # 저장 경로
            overlay_path  = out_dir / f"top{rank}_overlay.png"
            thumb_path    = out_dir / f"top{rank}_thumbnail.png"
            meta_path     = out_dir / f"top{rank}_metadata.json"

            # overlay_with_box → PNG
            Image.fromarray(card["overlay_with_box"]).save(str(overlay_path))

            # thumbnail → PNG
            thumb = card["thumbnail"]
            if thumb.size > 0:
                Image.fromarray(thumb).save(str(thumb_path))
            else:
                # 빈 thumbnail이면 1×1 검정 이미지 저장
                Image.fromarray(np.zeros((1, 1, 3), dtype=np.uint8)).save(str(thumb_path))

            # metadata → JSON (ndarray 제외한 직렬화 가능 항목만)
            metadata = {
                k: v for k, v in card.items()
                if not isinstance(v, np.ndarray)
            }
            with open(meta_path, "w", encoding="utf-8") as jf:
                json.dump(metadata, jf, ensure_ascii=False, indent=2)

            print(f"  rank={rank} local_z={local_z} score={card['score']:.4f}")
            print(f"    overlay  → {overlay_path.name} "
                  f"shape={card['overlay_with_box'].shape} dtype={card['overlay_with_box'].dtype}")
            print(f"    thumbnail→ {thumb_path.name} "
                  f"shape={card['thumbnail'].shape} dtype={card['thumbnail'].dtype}")
            print(f"    metadata → {meta_path.name} keys={list(metadata.keys())}")
            saved_count += 1

        except Exception as exc:
            _append_error(patient_id, "visualization_error", str(exc), f"rank{rank}_local_z{local_z}")
            print(f"  rank={rank} local_z={local_z} ERROR: {exc}")

    # --- runtime_summary 기록 ---
    _append_runtime("patients_processed", "1")
    _append_runtime("candidates_saved", str(saved_count))
    _append_runtime("score_col", score_col)
    _append_runtime("model", model)
    _append_runtime("top_k", str(top_k))

    print(f"\n완료: {saved_count}건 저장 → {out_dir}")


if __name__ == "__main__":
    main()
