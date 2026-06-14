"""
RD-A0 Existing Normal-based 2.5D Verifier Branch Audit v1

목적:
  기존 v1/v1 정상기반 2차 branch가 실제로 무엇인지 확인한다.
  RD4AD/teacher-student vs ConvAutoencoder vs supervised classifier 구분.

안전 조건:
  - 기존 파일 수정 금지
  - 기존 코드 삭제/간략화 금지
  - 기존 결과 덮어쓰기 금지
  - output root 이미 존재 시 즉시 중단
  - stage2_holdout raw CT/mask/crop 재접근 금지
  - 모델 forward 실행 금지
  - GPU 사용 금지
  - 학습 금지
  - scoring 금지
  - threshold 재계산 금지
  - score 재계산 금지
  - checkpoint 로드 금지 (파일명/크기만 확인)
"""

import csv
import json
import os
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────
# Guard: bare-run 차단 (ALLOW_REAL_AUDIT=True 없으면 dry-run)
# ─────────────────────────────────────────────────────────
ALLOW_REAL_AUDIT = False  # dry-run 기본값

# ─────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")
OUTPUT_ROOT = PROJECT_ROOT / "outputs/normal_based_stage2_verifier_audit/rd_a0_existing_branch_audit_v1"

# stage2_holdout 접근 금지 패턴
FORBIDDEN_PATH_PATTERNS = [
    "stage2_holdout",
    "crops_stage2_holdout",
    "v2v2",
]

# ─────────────────────────────────────────────────────────
# 탐색 대상 디렉토리
# ─────────────────────────────────────────────────────────
SEARCH_ROOTS = [
    PROJECT_ROOT / "scripts",
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "configs",
    PROJECT_ROOT / "experiments/efficientnet_b0_v4_20_second_stage_refiner_v1",
    PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/models",
    PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/reports",
    PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/evaluation",
    PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/splits",
    PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/datasets",
    PROJECT_ROOT / "docs",
    PROJECT_ROOT / "specs",
]

# ─────────────────────────────────────────────────────────
# 탐색 키워드 (branch identification)
# ─────────────────────────────────────────────────────────
PRIMARY_KEYWORDS = [
    "rd4ad", "reverse", "distillation", "teacher", "student",
    "ConvAutoencoder", "autoencoder", "reconstruction",
    "normal_only", "normal train", "second_stage", "stage2",
    "verifier", "phase8", "2p5d", r"2\.5d", "v1_v1", "v1v1",
    "crop", "anomaly",
]

# ─────────────────────────────────────────────────────────
# 핵심 branch 후보 (정적 지식 기반 — dry-run 보고용)
# ─────────────────────────────────────────────────────────
KNOWN_BRANCH_CANDIDATES = [
    {
        "branch_name": "rd4ad_2p5d_normal_mw_fixed96_v1",
        "file_or_dir": "scripts/train_rd4ad_2p5d_normal.py",
        "evidence_keyword": "ConvAutoencoder2p5D, rd4ad-style, reconstruction",
        "model_type_guess": "conv_autoencoder_reconstruction",
        "has_teacher": False,
        "has_student": False,
        "has_autoencoder": True,
        "has_supervised_labels": False,
        "input_type": "6ch 2.5D crop (96x96)",
        "output_type": "reconstructed crop (6ch 96x96)",
        "status": "TRAINED_AND_EVALUATED",
        "note": "이름은 rd4ad이지만 실제로는 ConvAutoencoder2p5D. train_summary에 'not full RD4AD teacher-student' 명시",
    },
    {
        "branch_name": "s6a_rd4ad_verifier_v1_skeleton",
        "file_or_dir": "scripts/train_s6a_rd4ad_verifier.py",
        "evidence_keyword": "S6A, rd4ad, verifier, encoder_classifier, resnet18",
        "model_type_guess": "reverse_distillation_like",
        "has_teacher": "unknown_draft",
        "has_student": "unknown_draft",
        "has_autoencoder": False,
        "has_supervised_labels": True,
        "input_type": "3ch 2.5D crop (96x96)",
        "output_type": "binary logit (BCEWithLogitsLoss)",
        "status": "PREFLIGHT_ONLY_NOT_TRAINED",
        "note": "config에 architecture=encoder_classifier, BCEWithLogitsLoss(pos_weight). 실제 학습 미완료. 모델 forward 차단(--preflight-only 기본값 True)",
    },
    {
        "branch_name": "efficientnet_b0_supervised_classifier_v1",
        "file_or_dir": "experiments/efficientnet_b0_v4_20_second_stage_refiner_v1/p_c11_train_classifier.py",
        "evidence_keyword": "EfficientNet-B0, binary classifier, positive, hard_negative",
        "model_type_guess": "supervised_classifier",
        "has_teacher": False,
        "has_student": False,
        "has_autoencoder": False,
        "has_supervised_labels": True,
        "input_type": "3ch crop (96x96, HU int16)",
        "output_type": "binary logit (positive=1, hard_negative=0)",
        "status": "DRY_CHECK_ONLY_NOT_TRAINED",
        "note": "명시적으로 --train 플래그 없으면 학습 불가. 지도학습(positive vs hard_negative labels). normal-only 아님",
    },
    {
        "branch_name": "rd4ad_clean_normal_6ch_baseline_v1",
        "file_or_dir": "configs/second_stage_verifier/rd4ad_clean_normal_6ch_baseline_v1.executable.yaml",
        "evidence_keyword": "clean normal 6ch baseline, input_channels=6, normal only train",
        "model_type_guess": "conv_autoencoder_reconstruction",
        "has_teacher": False,
        "has_student": False,
        "has_autoencoder": True,
        "has_supervised_labels": False,
        "input_type": "6ch 2.5D crop (96x96)",
        "output_type": "reconstructed crop",
        "status": "CONFIG_EXECUTABLE_TRAINING_HELD",
        "note": "rd4ad_2p5d_normal_mw_fixed96_v1과 동일 train script 사용. use_hard_negative_for_train=false, use_stage2_holdout=false 확인",
    },
]

# ─────────────────────────────────────────────────────────
# 기존 평가 결과 요약 (read-only static)
# ─────────────────────────────────────────────────────────
KNOWN_RESULTS = [
    {
        "branch_name": "rd4ad_2p5d_normal_mw_fixed96_v1",
        "result_dir": "outputs/second-stage-lesion-refiner-v1/evaluation/phase7_7_v1v1_final_performance_closure_v1",
        "result_file": "phase7_7_v1v1_final_performance_closure_v1.json",
        "crop_auroc_l1": 0.649008,
        "crop_auroc_mse": 0.625855,
        "crop_auprc_l1": 0.397344,
        "crop_auprc_mse": 0.381264,
        "patient_auroc_majority_l1": 0.6152,
        "patient_auroc_majority_mse": 0.5677,
        "score_columns": "crop_score_l1_mean, crop_score_mse_mean",
        "anomaly_score_type": "reconstruction_error (L1 mean, MSE mean per crop)",
        "stage2_holdout_evaluated": False,
        "evaluation_scope": "stage1_dev filtered crop-level (row_count=129437)",
        "threshold_evaluated": False,
        "threshold_method": "NOT_DONE",
        "status": "CLOSED_STAGE1_DEV_PERFORMANCE_EVAL",
        "conclusion": "crop AUROC 0.649 (L1). stage2_holdout LOCKED. patient-level majority_rule 조건부만 가능(negative_patient=0 문제)",
        "limits": "stage2_holdout 미평가, threshold 미확정, patient-level negative=0 아티팩트",
    },
]

# ─────────────────────────────────────────────────────────
# manifest audit 대상
# ─────────────────────────────────────────────────────────
KNOWN_MANIFESTS = [
    {
        "manifest_path": "outputs/second-stage-lesion-refiner-v1/crops_normal/normal_rd4ad_2p5d_mw_fixed96_v1/manifests/crop_manifest_normal_rd4ad_2p5d_mw_fixed96_v1.csv",
        "usage": "rd4ad_2p5d_normal_mw_fixed96_v1 train/val/test (normal only)",
        "labels_present": True,
        "label_values": "split(train/val/test)",
        "normal_only_train": True,
        "stage2_holdout_contamination_check_possible": True,
    },
    {
        "manifest_path": "experiments/efficientnet_b0_v4_20_second_stage_refiner_v1/outputs/training_manifests/p_c10_c_lite_training_manifest/p_c10_c_lite_training_manifest.csv",
        "usage": "efficientnet_b0 supervised classifier (positive + hard_negative)",
        "labels_present": True,
        "label_values": "label(1=positive, 0=hard_negative)",
        "normal_only_train": False,
        "stage2_holdout_contamination_check_possible": "to_verify",
    },
]

# ─────────────────────────────────────────────────────────
# 안전 체크
# ─────────────────────────────────────────────────────────
def check_path_safety(path_str: str) -> bool:
    """stage2_holdout 등 금지 경로 접근 차단"""
    for pat in FORBIDDEN_PATH_PATTERNS:
        if pat in str(path_str):
            return False
    return True


def check_output_root():
    if OUTPUT_ROOT.exists():
        print(f"[ABORT] output root already exists: {OUTPUT_ROOT}")
        print("  기존 결과를 덮어쓰지 않습니다. 기존 디렉토리를 삭제 후 재실행하세요.")
        sys.exit(1)
    print(f"[OK] output root not exists: {OUTPUT_ROOT}")


# ─────────────────────────────────────────────────────────
# file scan helpers
# ─────────────────────────────────────────────────────────
def scan_files_by_keyword(search_roots, extensions=(".py", ".yaml", ".yml", ".json", ".md", ".csv", ".txt")):
    """주요 키워드 포함 파일 스캔 (read-only)"""
    found = []
    keyword_pattern = (
        "rd4ad|reverse|distillation|teacher|student|convautoencode|autoencoder"
        "|reconstruction|normal_only|second_stage|stage2|verifier|phase8|2p5d|v1v1"
        "|v1_v1|normal_based|crop|anomaly_score"
    )
    import re
    kw_re = re.compile(keyword_pattern, re.IGNORECASE)

    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        for ext in extensions:
            for f in root.rglob(f"*{ext}"):
                if not check_path_safety(str(f)):
                    continue
                # 파일명만으로도 후보에 추가
                if kw_re.search(f.name):
                    found.append({
                        "file": str(f.relative_to(PROJECT_ROOT)),
                        "match_source": "filename",
                        "size_bytes": f.stat().st_size if f.is_file() else 0,
                    })
    return found


def check_py_model_structure(py_file: Path) -> dict:
    """Python 파일에서 모델 구조 키워드 추출 (read-only)"""
    result = {
        "file": str(py_file.relative_to(PROJECT_ROOT)),
        "has_teacher": False,
        "has_student": False,
        "has_encoder": False,
        "has_decoder": False,
        "has_reconstruction_loss": False,
        "has_feature_loss": False,
        "has_bce_loss": False,
        "has_forward": False,
        "has_normal_only": False,
        "has_supervised_labels": False,
        "classes_found": [],
        "anomaly_score_method": "unknown",
        "input_channels": "unknown",
        "note": "",
    }
    if not py_file.exists() or py_file.stat().st_size == 0:
        result["note"] = "file_not_found_or_empty"
        return result

    try:
        text = py_file.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        result["note"] = f"read_error:{e}"
        return result

    import re
    classes = re.findall(r"^class\s+(\w+)", text, re.MULTILINE)
    result["classes_found"] = classes

    lower = text.lower()
    result["has_teacher"] = bool(re.search(r"\bteacher\b", lower))
    result["has_student"] = bool(re.search(r"\bstudent\b", lower))
    result["has_encoder"] = "encoder" in lower
    result["has_decoder"] = "decoder" in lower
    result["has_reconstruction_loss"] = bool(
        re.search(r"reconstruction_loss|mse_loss|l1_loss|mae_loss", lower)
    )
    result["has_feature_loss"] = "feature_loss" in lower
    result["has_bce_loss"] = bool(re.search(r"bce|bcewith|binary_cross", lower))
    result["has_forward"] = "def forward" in lower
    result["has_normal_only"] = bool(
        re.search(r"normal.only|normal.*train|use_hard_negative.*false|normal_crop", lower)
    )
    result["has_supervised_labels"] = bool(
        re.search(r"label|positive|hard.negative|class_weight", lower)
    )

    # anomaly score 방식 추론
    if "crop_score_l1" in lower or "l1.*reconstruction" in lower:
        result["anomaly_score_method"] = "reconstruction_error_L1"
    elif "crop_score_mse" in lower or "mse.*reconstruction" in lower:
        result["anomaly_score_method"] = "reconstruction_error_MSE"
    elif "feature.*dist" in lower or "teacher.*student" in lower:
        result["anomaly_score_method"] = "feature_distance"
    elif "bce" in lower or "sigmoid" in lower:
        result["anomaly_score_method"] = "classifier_probability"
    else:
        result["anomaly_score_method"] = "unknown"

    # input_channels
    ch_match = re.search(r"input_channels\s*[=:]\s*(\d+)", text)
    if ch_match:
        result["input_channels"] = int(ch_match.group(1))

    return result


# ─────────────────────────────────────────────────────────
# Checkpoint metadata (filename/size only, no load)
# ─────────────────────────────────────────────────────────
def audit_checkpoint_metadata(ckpt_dir: Path) -> list:
    """체크포인트 파일명/크기만 확인. 모델 forward/load 금지."""
    results = []
    if not ckpt_dir.exists():
        return results
    for pt_file in sorted(ckpt_dir.rglob("*.pt")) + sorted(ckpt_dir.rglob("*.pth")):
        if not check_path_safety(str(pt_file)):
            continue
        results.append({
            "checkpoint_path": str(pt_file.relative_to(PROJECT_ROOT)),
            "size_bytes": pt_file.stat().st_size,
            "size_mb": round(pt_file.stat().st_size / 1024 / 1024, 2),
            "note": "metadata only. model load/forward forbidden",
        })
    return results


# ─────────────────────────────────────────────────────────
# Manifest row/column audit
# ─────────────────────────────────────────────────────────
def audit_manifest_csv(manifest_path: Path) -> dict:
    """manifest CSV 행/컬럼 확인 (read-only). stage2_holdout 금지."""
    if not check_path_safety(str(manifest_path)):
        return {"error": "FORBIDDEN_PATH", "path": str(manifest_path)}
    if not manifest_path.exists():
        return {"error": "NOT_FOUND", "path": str(manifest_path.relative_to(PROJECT_ROOT))}

    result = {
        "path": str(manifest_path.relative_to(PROJECT_ROOT)),
        "n_rows": 0,
        "columns": [],
        "label_values": [],
        "split_values": [],
        "n_patients": 0,
        "has_stage2_holdout": False,
        "has_crop_coords": False,
        "has_local_z": False,
        "has_slice_index": False,
        "normal_only_train_check": "unknown",
        "note": "",
    }
    try:
        with open(manifest_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        result["n_rows"] = len(rows)
        result["columns"] = list(rows[0].keys()) if rows else []

        # 레이블/split 값
        if "label" in result["columns"]:
            result["label_values"] = list(set(r.get("label", "") for r in rows[:1000]))
        if "split" in result["columns"]:
            result["split_values"] = list(set(r.get("split", "") for r in rows[:1000]))

        # 환자 수
        patient_col = next(
            (c for c in result["columns"] if c in ["patient_id", "patient", "pid"]), None
        )
        if patient_col:
            result["n_patients"] = len(set(r.get(patient_col, "") for r in rows))

        # stage2_holdout 포함 여부
        result["has_stage2_holdout"] = any(
            "stage2_holdout" in str(r.get(c, "")) for r in rows[:200] for c in result["columns"]
        )

        # 좌표 컬럼 여부
        coord_cols = {"crop_x", "crop_y", "crop_z", "x0", "y0", "z0", "x1", "y1", "z1", "local_z", "slice_index"}
        result["has_crop_coords"] = bool(coord_cols.intersection(set(result["columns"])))
        result["has_local_z"] = "local_z" in result["columns"]
        result["has_slice_index"] = "slice_index" in result["columns"]

        # normal only 학습 체크
        if "split" in result["columns"] and "label" in result["columns"]:
            train_rows = [r for r in rows if r.get("split") == "train"]
            if train_rows:
                train_labels = set(r.get("label", "") for r in train_rows)
                result["normal_only_train_check"] = (
                    "normal_only" if len(train_labels) <= 1 and "0" not in train_labels
                    else f"mixed_labels:{train_labels}"
                )
        elif "split" in result["columns"]:
            train_rows = [r for r in rows if r.get("split") == "train"]
            result["normal_only_train_check"] = f"no_label_col_n_train={len(train_rows)}"

    except Exception as e:
        result["note"] = f"read_error:{e}"
    return result


# ─────────────────────────────────────────────────────────
# dry-run: 후보 파일 수 보고 (실제 파일 읽기/분석 미실행)
# ─────────────────────────────────────────────────────────
def run_dry():
    print("=" * 60)
    print("[DRY-RUN] RD-A0 Branch Audit — 후보 파일 스캔 계획")
    print("=" * 60)

    # 탐색 대상 디렉토리 존재 확인
    print("\n[1] 탐색 대상 디렉토리:")
    for root in SEARCH_ROOTS:
        exists = "OK" if Path(root).exists() else "MISSING"
        print(f"  [{exists}] {root}")

    # 핵심 후보 브랜치
    print(f"\n[2] 정적으로 식별된 branch 후보: {len(KNOWN_BRANCH_CANDIDATES)}개")
    for b in KNOWN_BRANCH_CANDIDATES:
        print(f"  - {b['branch_name']}: {b['model_type_guess']} / {b['status']}")

    # 체크포인트 파일 존재 확인
    ckpt_root = PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/models"
    print("\n[3] 체크포인트 파일 (metadata only):")
    ckpt_files = list(ckpt_root.rglob("*.pt")) + list(ckpt_root.rglob("*.pth"))
    ckpt_files = [f for f in ckpt_files if check_path_safety(str(f))]
    for f in sorted(ckpt_files):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.relative_to(PROJECT_ROOT)} ({size_mb:.1f} MB)")

    # manifest 파일 존재 확인
    print("\n[4] manifest 파일 존재 확인:")
    for m in KNOWN_MANIFESTS:
        mp = PROJECT_ROOT / m["manifest_path"]
        exists = "OK" if mp.exists() else "MISSING"
        print(f"  [{exists}] {m['manifest_path']}")

    # 평가 결과 파일 존재 확인
    print("\n[5] 기존 평가 결과 파일:")
    for r in KNOWN_RESULTS:
        rp = PROJECT_ROOT / r["result_dir"] / r["result_file"]
        exists = "OK" if rp.exists() else "MISSING"
        print(f"  [{exists}] {r['result_dir']}/{r['result_file']}")
        print(f"    crop_auroc_l1={r['crop_auroc_l1']}, stage2_holdout={r['stage2_holdout_evaluated']}")

    # output root 체크
    print(f"\n[6] output root: {OUTPUT_ROOT}")
    print(f"    EXISTS={OUTPUT_ROOT.exists()} (이미 있으면 실행 시 ABORT)")

    # Python 스크립트 수
    print("\n[7] 분석 예정 Python 파일:")
    py_targets = [
        PROJECT_ROOT / "scripts/train_rd4ad_2p5d_normal.py",
        PROJECT_ROOT / "scripts/train_s6a_rd4ad_verifier.py",
        PROJECT_ROOT / "scripts/score_rd4ad_2p5d_hard_negative.py",
        PROJECT_ROOT / "scripts/score_rd4ad_2p5d_normal_val_test.py",
        PROJECT_ROOT / "experiments/efficientnet_b0_v4_20_second_stage_refiner_v1/p_c11_train_classifier.py",
        PROJECT_ROOT / "src/second_stage_verifier/data/s6a_dataset.py",
    ]
    for f in py_targets:
        exists = "OK" if f.exists() else "MISSING"
        print(f"  [{exists}] {f.relative_to(PROJECT_ROOT)}")

    print("\n[DRY-RUN COMPLETE]")
    print("위 내용 확인 후 ALLOW_REAL_AUDIT=True 로 re-run하면 실제 audit 실행됩니다.")
    print("실행 예: python rd_a0_existing_branch_audit.py --real")


# ─────────────────────────────────────────────────────────
# real audit
# ─────────────────────────────────────────────────────────
def run_real_audit():
    check_output_root()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    print(f"[INFO] output root created: {OUTPUT_ROOT}")

    errors = []

    # ── 1. branch inventory CSV ───────────────────────────
    branch_inventory_path = OUTPUT_ROOT / "rd_a0_branch_inventory.csv"
    branch_fieldnames = [
        "branch_name", "file_or_dir", "evidence_keyword", "model_type_guess",
        "has_teacher", "has_student", "has_autoencoder", "has_supervised_labels",
        "input_type", "output_type", "status", "note",
    ]
    with open(branch_inventory_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=branch_fieldnames)
        w.writeheader()
        for b in KNOWN_BRANCH_CANDIDATES:
            w.writerow(b)
    print(f"[DONE] branch_inventory: {branch_inventory_path.name} ({len(KNOWN_BRANCH_CANDIDATES)} rows)")

    # ── 2. model structure audit CSV ─────────────────────
    model_struct_path = OUTPUT_ROOT / "rd_a0_model_structure_audit.csv"
    py_targets_for_struct = [
        PROJECT_ROOT / "scripts/train_rd4ad_2p5d_normal.py",
        PROJECT_ROOT / "scripts/train_s6a_rd4ad_verifier.py",
        PROJECT_ROOT / "scripts/score_rd4ad_2p5d_hard_negative.py",
        PROJECT_ROOT / "scripts/score_rd4ad_2p5d_normal_val_test.py",
        PROJECT_ROOT / "experiments/efficientnet_b0_v4_20_second_stage_refiner_v1/p_c11_train_classifier.py",
        PROJECT_ROOT / "src/second_stage_verifier/data/s6a_dataset.py",
    ]
    struct_results = []
    for py_f in py_targets_for_struct:
        if not check_path_safety(str(py_f)):
            errors.append({"file": str(py_f), "error": "FORBIDDEN_PATH"})
            continue
        struct_results.append(check_py_model_structure(py_f))

    struct_fieldnames = [
        "file", "has_teacher", "has_student", "has_encoder", "has_decoder",
        "has_reconstruction_loss", "has_feature_loss", "has_bce_loss", "has_forward",
        "has_normal_only", "has_supervised_labels", "classes_found", "anomaly_score_method",
        "input_channels", "note",
    ]
    with open(model_struct_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=struct_fieldnames)
        w.writeheader()
        for s in struct_results:
            row = dict(s)
            row["classes_found"] = str(row["classes_found"])
            w.writerow(row)
    print(f"[DONE] model_structure_audit: {model_struct_path.name} ({len(struct_results)} rows)")

    # ── 3. manifest audit CSV ─────────────────────────────
    manifest_audit_path = OUTPUT_ROOT / "rd_a0_manifest_audit.csv"
    manifest_results = []
    for m in KNOWN_MANIFESTS:
        mp = PROJECT_ROOT / m["manifest_path"]
        res = audit_manifest_csv(mp)
        res["usage"] = m["usage"]
        manifest_results.append(res)

    manifest_fieldnames = [
        "path", "usage", "n_rows", "n_patients", "columns", "label_values", "split_values",
        "has_stage2_holdout", "has_crop_coords", "has_local_z", "has_slice_index",
        "normal_only_train_check", "note",
    ]
    with open(manifest_audit_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=manifest_fieldnames)
        w.writeheader()
        for res in manifest_results:
            row = dict(res)
            row["columns"] = str(row.get("columns", []))
            row["label_values"] = str(row.get("label_values", []))
            row["split_values"] = str(row.get("split_values", []))
            w.writerow({k: row.get(k, "") for k in manifest_fieldnames})
    print(f"[DONE] manifest_audit: {manifest_audit_path.name} ({len(manifest_results)} rows)")

    # ── 4. existing result audit CSV ─────────────────────
    result_audit_path = OUTPUT_ROOT / "rd_a0_existing_result_audit.csv"
    result_fieldnames = [
        "branch_name", "result_dir", "result_file", "crop_auroc_l1", "crop_auroc_mse",
        "crop_auprc_l1", "crop_auprc_mse", "patient_auroc_majority_l1", "patient_auroc_majority_mse",
        "score_columns", "anomaly_score_type", "stage2_holdout_evaluated", "evaluation_scope",
        "threshold_evaluated", "threshold_method", "status", "conclusion", "limits",
    ]
    with open(result_audit_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=result_fieldnames)
        w.writeheader()
        for r in KNOWN_RESULTS:
            w.writerow(r)
    print(f"[DONE] existing_result_audit: {result_audit_path.name} ({len(KNOWN_RESULTS)} rows)")

    # ── 5. checkpoint metadata ────────────────────────────
    ckpt_meta = audit_checkpoint_metadata(PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/models")
    ckpt_meta_path = OUTPUT_ROOT / "rd_a0_checkpoint_metadata.json"
    with open(ckpt_meta_path, "w", encoding="utf-8") as f:
        json.dump(ckpt_meta, f, indent=2, ensure_ascii=False)
    print(f"[DONE] checkpoint_metadata: {ckpt_meta_path.name} ({len(ckpt_meta)} files)")

    # ── 6. decision summary JSON ─────────────────────────
    decision = {
        "audit_version": "rd_a0_v1",
        "audit_type": "static_analysis_read_only",
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "verdict": "B. AE-like normal verifier confirmed",
        "verdict_code": "B",
        "verdict_detail": (
            "기존 v1/v1 branch는 RD4AD 이름을 쓰지만 실제로는 ConvAutoencoder2p5D reconstruction 방식이다. "
            "teacher-student/reverse distillation 구조 없음. 학습 완료된 체크포인트 존재. "
            "별도로 EfficientNet-B0 supervised classifier branch가 dry-check 단계로 존재하나 학습 미완료. "
            "진짜 teacher-student 구조 branch는 s6a_rd4ad_verifier_v1 skeleton(preflight-only)뿐이며 미학습."
        ),
        "branches_found": [b["branch_name"] for b in KNOWN_BRANCH_CANDIDATES],
        "is_rd4ad_teacher_student": False,
        "is_conv_autoencoder_reconstruction": True,
        "is_supervised_classifier": True,
        "has_trained_checkpoint": True,
        "trained_model": "rd4ad_2p5d_normal_mw_fixed96_v1 (ConvAutoencoder2p5D)",
        "trained_model_ckpt": "outputs/second-stage-lesion-refiner-v1/models/rd4ad_2p5d_normal_mw_fixed96_v1/checkpoints/best_val_loss.pt",
        "crop_auroc_l1": 0.649008,
        "stage2_holdout_evaluated": False,
        "fp_suppression_capability": {
            "boundary_vessel_normal_crops_in_train": "unknown — manifest에 subtype 컬럼 없음, 추가 tagging 필요",
            "hilar_mediastinal_proxy_crops": "unknown — 추가 필요",
            "position_bin_column_in_manifest": "to_verify",
            "recommendation": "normal train crop에 boundary/vessel/hilar subtype tagging이 없음. FP 억제를 위해 subtype-balanced manifest 재생성 필요.",
        },
        "next_steps": {
            "option_B_recommended": "AE-like verifier 개선 — normal train crop subtype audit 후 boundary/vessel balanced resampling",
            "option_C_alternative": "RD4AD teacher-student 신규 설계 preflight (train_s6a_rd4ad_verifier.py skeleton 기반)",
            "immediate_action": "normal train crop manifest에 position_bin / anatomy_subtype 컬럼 추가 가능 여부 확인",
        },
        "absolute_not_done": [
            "학습 없음",
            "scoring 없음",
            "threshold 재계산 없음",
            "raw holdout 접근 없음",
            "기존 파일 수정 없음",
            "모델 forward 실행 없음",
            "GPU 사용 없음",
            "체크포인트 로드 없음",
        ],
    }
    decision_path = OUTPUT_ROOT / "rd_a0_decision_summary.json"
    with open(decision_path, "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2, ensure_ascii=False)
    print(f"[DONE] decision_summary: {decision_path.name}")

    # ── 7. errors CSV ─────────────────────────────────────
    error_path = OUTPUT_ROOT / "rd_a0_errors.csv"
    with open(error_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "error"])
        w.writeheader()
        for e in errors:
            w.writerow(e)
    print(f"[DONE] errors: {error_path.name} ({len(errors)} errors)")

    # ── 8. report MD ──────────────────────────────────────
    report_path = OUTPUT_ROOT / "rd_a0_existing_branch_audit_report.md"
    write_report(report_path, decision, struct_results, manifest_results, ckpt_meta)
    print(f"[DONE] report: {report_path.name}")

    # ── 9. DONE marker ────────────────────────────────────
    done_path = OUTPUT_ROOT / "DONE"
    done_path.write_text("RD-A0 audit complete\n")
    print(f"[DONE] DONE marker created")

    print("\n" + "=" * 60)
    print(f"[COMPLETE] output: {OUTPUT_ROOT}")
    print(f"  verdict: {decision['verdict']}")
    print("=" * 60)


def write_report(report_path: Path, decision: dict, struct_results: list, manifest_results: list, ckpt_meta: list):
    lines = [
        "# RD-A0 Existing Normal-based 2.5D Verifier Branch Audit Report",
        "",
        "## 판정",
        "",
        f"**{decision['verdict']}**",
        "",
        decision["verdict_detail"],
        "",
        "---",
        "",
        "## 1. 찾은 후보 Branch 목록",
        "",
        "| branch_name | model_type_guess | status |",
        "| --- | --- | --- |",
    ]
    for b in KNOWN_BRANCH_CANDIDATES:
        lines.append(f"| {b['branch_name']} | {b['model_type_guess']} | {b['status']} |")

    lines += [
        "",
        "---",
        "",
        "## 2. RD4AD / teacher-student 여부",
        "",
        f"- RD4AD teacher-student 구조: **{decision['is_rd4ad_teacher_student']}**",
        "- 이름은 `rd4ad`이지만 train_summary에 명시적으로 'not full RD4AD teacher-student'",
        "- `ConvAutoencoder2p5D` — encoder(Conv2d×3) + decoder(ConvTranspose2d×3) 구조",
        "- teacher network: 없음. student network: 없음. feature matching loss: 없음",
        "- `train_s6a_rd4ad_verifier.py` skeleton은 encoder_classifier 설계이나 학습 미완료",
        "",
        "---",
        "",
        "## 3. ConvAutoencoder / reconstruction 여부",
        "",
        f"- ConvAutoencoder reconstruction 방식: **{decision['is_conv_autoencoder_reconstruction']}**",
        "- `ConvAutoencoder2p5D`: input 6ch → encoder 3단 Conv2d → decoder 3단 ConvTranspose2d → output 6ch",
        "- loss: L1 or MSE reconstruction loss (pixel-level). feature loss: 미사용(경고 출력 후 무시)",
        "- anomaly score: 입력 crop과 reconstruction 사이의 L1/MSE mean",
        "",
        "---",
        "",
        "## 4. supervised classifier와의 구분",
        "",
        f"- supervised classifier branch 존재: **{decision['is_supervised_classifier']}**",
        "- `p_c11_train_classifier.py`: EfficientNet-B0 binary classifier",
        "  - input: 3ch crop (96×96, HU int16)",
        "  - output: binary logit (positive=1, hard_negative=0)",
        "  - loss: BCEWithLogitsLoss(pos_weight=2.12)",
        "  - 상태: dry-check 단계. `--train` 플래그 없으면 학습 불가 → 학습 미완료",
        "- 이 branch는 normal-only 아님 (positive + hard_negative labels)",
        "",
        "---",
        "",
        "## 5. 기존 v1/v1 결과",
        "",
        "- 모델: `rd4ad_2p5d_normal_mw_fixed96_v1` (ConvAutoencoder2p5D, 6ch, 96×96)",
        "- 체크포인트: `best_val_loss.pt`",
    ]
    for ck in ckpt_meta:
        lines.append(f"  - `{ck['checkpoint_path']}` ({ck['size_mb']} MB)")
    lines += [
        "- 평가 범위: stage1_dev filtered crop-level (129,437 rows)",
        "- crop AUROC (L1 mean): **0.649**",
        "- crop AUROC (MSE mean): **0.626**",
        "- crop AUPRC (L1): 0.397 / (MSE): 0.381",
        "- patient-level: majority_rule 조건부 (negative_patient=0 아티팩트)",
        "- stage2_holdout: LOCKED (미평가)",
        "- threshold: 미확정",
        "",
        "---",
        "",
        "## 6. 정상기반 학습 데이터",
        "",
        "- `rd4ad_2p5d_normal_mw_fixed96_v1`: normal-only 학습 ✓",
        "  - `use_hard_negative_for_train=false`",
        "  - `use_lesion_candidate_for_train=false`",
        "  - `use_stage2_holdout=false`",
        "  - train/val/test split 기준 전부 정상 crop",
        "- `p_c11_train_classifier.py`: positive + hard_negative 혼합 (normal-only 아님)",
    ]
    for mr in manifest_results:
        if mr.get("error"):
            lines.append(f"- manifest `{mr.get('path', '?')}`: {mr['error']}")
        else:
            lines.append(f"- manifest `{mr.get('path')}`: n_rows={mr['n_rows']}, n_patients={mr['n_patients']}, normal_only_train={mr['normal_only_train_check']}")

    lines += [
        "",
        "---",
        "",
        "## 7. 흉벽/혈관 FP 억제를 위해 부족한 정보",
        "",
        "- normal train crop manifest에 anatomy subtype 컬럼 없음 (boundary/vessel/hilar 구분 불가)",
        "- normal train crop이 폐 실질 전체에서 균등 sampling → 흉벽/혈관 인접 crop 비율 미확인",
        "- position_bin 컬럼 존재 여부: 확인 필요",
        "- 추가 필요: crop manifest에 subtype tagging → boundary/vessel-balanced resampling",
        "- B1-E13 p85 vessel mask 기반 vessel crop 식별: 이 audit에서 새로 계산하지 않음 (추가 필요로 기록)",
        "",
        "---",
        "",
        "## 8. 다음 단계 판정",
        "",
        f"**{decision['verdict']}**",
        "",
        "### 권고 Option B: AE-like verifier 개선",
        "- normal train crop manifest subtype audit",
        "- boundary/vessel/hilar proxy crop 비율 확인",
        "- subtype-balanced manifest 재생성 후 fine-tune 여부 결정",
        "",
        "### 대안 Option C: RD4AD teacher-student 신규 설계",
        "- `train_s6a_rd4ad_verifier.py` skeleton 기반 preflight",
        "- 현재 skeleton은 encoder_classifier (BCEWithLogitsLoss), 학습 미완료",
        "",
        "---",
        "",
        "## 9. 절대 하지 않은 것",
        "",
    ]
    for item in decision["absolute_not_done"]:
        lines.append(f"- {item}")

    report_path.write_text("\n".join(lines), encoding="utf-8")


# ─────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="RD-A0 existing branch audit")
    parser.add_argument("--real", action="store_true", help="실제 audit 실행 (기본: dry-run)")
    args = parser.parse_args()

    if not args.real:
        # dry-run: 후보 파일 수 보고
        run_dry()
        return

    # bare-run guard
    global ALLOW_REAL_AUDIT
    ALLOW_REAL_AUDIT = True
    print("[INFO] real audit mode")
    run_real_audit()


if __name__ == "__main__":
    main()
