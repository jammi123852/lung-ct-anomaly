"""
RD-B6a-2: Tiny Forward Check
목적: teacher(ResNet18 local weight) / student(reverse decoder) forward shape 확인
     학습, backward, optimizer, checkpoint, scoring 일체 금지
모드:
  bare run   → exit 2 (파일 생성 금지)
  --dry-plan → crop 목록 출력만 (파일 생성 없음)
  --run      → forward check 실행 (사용자 승인 후)
안전 조건:
  stage2_holdout/lesion 경로 접근 금지
  backward/optimizer/checkpoint/scoring 금지
  GPU 사용 금지 (CPU only)
  인터넷 다운로드 금지 / local weight만 사용
  기존 파일 수정/삭제 금지
  output root 이미 존재 시 즉시 중단
"""

import sys
import csv
import json
import math
import time
import random
from pathlib import Path

# ─── bare-run guard ──────────────────────────────────────────────────────────
ALLOWED_MODES = {"--dry-plan", "--run"}
if not any(m in sys.argv for m in ALLOWED_MODES):
    print("오류: 실행 모드 지정 필요.")
    print("  --dry-plan : crop 목록 출력 (파일 생성 없음)")
    print("  --run      : forward check 실행 (사용자 승인 후)")
    sys.exit(2)

IS_DRY_PLAN = "--dry-plan" in sys.argv
IS_RUN = "--run" in sys.argv

# ─── 경로 설정 ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_b6a2_tiny_forward_check_v1"
)
SMOKE_SUBSET_MANIFEST = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_b6a_tiny_smoke_train_preflight_v1"
    / "rd_b6a_smoke_subset_manifest.csv"
)
PATIENT_MANIFEST_PATH = Path(
    "/mnt/c/Users/jinhy/Desktop/"
    "Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1/"
    "manifests/patient_manifest.csv"
)
LOCAL_WEIGHT_PATH = Path(
    "/home/jinhy/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
)

# ─── 안전 금지 키워드 ────────────────────────────────────────────────────────
FORBIDDEN_KEYWORDS = [
    "stage2_holdout",
    "lesion",
    "test_lesion",
    "second-stage-lesion-refiner",
]

# ─── 설계 상수 ────────────────────────────────────────────────────────────────
CROP_SIZE = 96
N_CHANNELS = 3
MIP_RADIUS = 3
HU_CLIP_MIN = -1000.0
HU_CLIP_MAX = 600.0
HU_RANGE = 1600.0
SIX_BIN_LABELS = [
    "lower_boundary",
    "lower_interior",
    "middle_boundary",
    "middle_interior",
    "upper_boundary",
    "upper_interior",
]
LOW_Z_BOUNDARY_WARN_THRESHOLD = 7
BATCH_SIZE = 1
SEED = 42

# expected feature shapes (배치 dim 제외)
EXPECTED_TEACHER_SHAPES = {
    "layer1": (64,  24, 24),
    "layer2": (128, 12, 12),
    "layer3": (256,  6,  6),
}
EXPECTED_STUDENT_SHAPES = {
    "de_layer3": (256,  6,  6),
    "de_layer2": (128, 12, 12),
    "de_layer1": (64,  24, 24),
}

# CSV 컬럼 정의
FORWARD_ROWS_FIELDS = [
    "check_id", "manifest_id", "patient_id", "safe_id",
    "six_bin_label", "local_z", "low_z_boundary_warning",
    "input_shape", "input_min", "input_max", "input_nan_count", "input_inf_count",
    "teacher_layer1_shape", "teacher_layer2_shape", "teacher_layer3_shape",
    "student_de_layer1_shape", "student_de_layer2_shape", "student_de_layer3_shape",
    "total_loss", "loss_finite", "note",
]
FEATURE_SHAPE_FIELDS = [
    "check_id", "six_bin_label",
    "input_shape_ok",
    "teacher_layer1_shape", "teacher_layer1_shape_ok",
    "teacher_layer2_shape", "teacher_layer2_shape_ok",
    "teacher_layer3_shape", "teacher_layer3_shape_ok",
    "student_de_layer3_shape", "student_de_layer3_shape_ok",
    "student_de_layer2_shape", "student_de_layer2_shape_ok",
    "student_de_layer1_shape", "student_de_layer1_shape_ok",
]
LOSS_FINITE_FIELDS = [
    "check_id", "six_bin_label",
    "loss_layer1", "loss_layer2", "loss_layer3", "total_loss", "loss_finite",
]


# =============================================================================
# 안전 검사
# =============================================================================
def assert_path_safe(path_str: str) -> None:
    for kw in FORBIDDEN_KEYWORDS:
        if kw.lower() in str(path_str).lower():
            raise RuntimeError(
                f"[SAFETY] 금지 경로 접근 차단: {path_str!r} (keyword={kw!r})"
            )


# =============================================================================
# 공통 함수
# =============================================================================
def normalize_hu(hu_array):
    import numpy as np
    clipped = np.clip(hu_array, HU_CLIP_MIN, HU_CLIP_MAX)
    return ((clipped - HU_CLIP_MIN) / HU_RANGE).astype("float32")


def compute_mip_slab_indices(center_z: int, direction: str, z_max: int) -> list:
    if direction == "lower":
        raw = [center_z - MIP_RADIUS + i for i in range(MIP_RADIUS)]
    elif direction == "upper":
        raw = [center_z + 1 + i for i in range(MIP_RADIUS)]
    else:
        raise ValueError(f"direction must be 'lower' or 'upper', got {direction!r}")
    return [max(0, min(idx, z_max - 1)) for idx in raw]


def has_low_z_boundary_warning(center_z: int) -> bool:
    return center_z <= LOW_Z_BOUNDARY_WARN_THRESHOLD


def select_forward_check_crops(smoke_manifest_path: Path) -> list:
    """
    6-bin에서 각 1개씩 총 6개 crop 선택.
    low_z_boundary_warning=True crop 가능하면 우선 포함.
    """
    rng = random.Random(SEED)
    rows_by_bin: dict = {lbl: [] for lbl in SIX_BIN_LABELS}
    with open(smoke_manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lbl = row.get("six_bin_label", "")
            if lbl in rows_by_bin:
                rows_by_bin[lbl].append(row)

    selected = []
    for lbl in SIX_BIN_LABELS:
        pool = rows_by_bin[lbl]
        if not pool:
            continue
        rng.shuffle(pool)
        low_z = [r for r in pool if has_low_z_boundary_warning(int(r["local_z"]))]
        others = [r for r in pool if not has_low_z_boundary_warning(int(r["local_z"]))]
        if low_z:
            selected.append(low_z[0])
        else:
            selected.append(others[0])
    return selected


def load_patient_paths(patient_manifest_path: Path, target_safe_ids: set) -> dict:
    """patient_manifest에서 대상 환자 경로만 로드 (npy 값 로딩 금지)."""
    patient_paths = {}
    with open(patient_manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row.get("safe_id", "")
            if sid in target_safe_ids:
                ct_path = row.get("ct_hu_npy", "")
                roi_path = row.get("roi_0_0_npy", "")
                assert_path_safe(ct_path)
                assert_path_safe(roi_path)
                patient_paths[sid] = {
                    "ct_hu_npy": ct_path,
                    "roi_0_0_npy": roi_path,
                }
    return patient_paths


def build_crop_tensor(ct_arr, center_z, crop_y0, crop_x0, crop_y1, crop_x1):
    """on-the-fly 3ch crop 생성 → torch tensor (저장 금지)."""
    import numpy as np
    import torch
    z_max = ct_arr.shape[0]
    ch0_raw = ct_arr[center_z, crop_y0:crop_y1, crop_x0:crop_x1].copy()
    lower_idxs = compute_mip_slab_indices(center_z, "lower", z_max)
    upper_idxs = compute_mip_slab_indices(center_z, "upper", z_max)
    ch1_raw = ct_arr[lower_idxs].max(axis=0)[crop_y0:crop_y1, crop_x0:crop_x1].copy()
    ch2_raw = ct_arr[upper_idxs].max(axis=0)[crop_y0:crop_y1, crop_x0:crop_x1].copy()
    crop_np = np.stack([
        normalize_hu(ch0_raw),
        normalize_hu(ch1_raw),
        normalize_hu(ch2_raw),
    ], axis=0)  # (3, H, W) float32
    tensor = torch.from_numpy(crop_np).unsqueeze(0)  # (1, 3, H, W)
    return tensor


# =============================================================================
# Teacher / Student 빌드 (--run 에서만 호출)
# =============================================================================

def build_teacher(local_weight_path: Path):
    """ResNet18 local weight 로드 → eval + frozen (자동 다운로드 금지)."""
    import torch
    import torchvision.models as models
    resnet = models.resnet18(weights=None)
    state_dict = torch.load(str(local_weight_path), map_location="cpu", weights_only=True)
    resnet.load_state_dict(state_dict)
    resnet.eval()
    resnet.requires_grad_(False)
    return resnet


def build_student_decoder():
    """Student reverse decoder (RD-B5 설계 기준, random init)."""
    import torch.nn as nn

    class _StudentDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            # de_layer3: 256×6×6 → 256×6×6
            self.de_layer3 = nn.Sequential(
                nn.Conv2d(256, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )
            # de_layer2: 256×6×6 → 128×12×12
            self.de_layer2 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(256, 128, 3, 1, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )
            # de_layer1: 128×12×12 → 64×24×24
            self.de_layer1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(128, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )

        def forward(self, layer3_feat):
            x = self.de_layer3(layer3_feat)   # B×256×6×6
            de3 = x
            x = self.de_layer2(x)             # B×128×12×12
            de2 = x
            x = self.de_layer1(x)             # B×64×24×24
            de1 = x
            return de3, de2, de1

    decoder = _StudentDecoder()
    decoder.eval()
    return decoder


# =============================================================================
# dry-plan
# =============================================================================

def run_dry_plan() -> None:
    """crop 목록 출력만 (파일 생성 없음)."""
    print("\n[DRY-PLAN] forward check 대상 crop 목록:")
    print(f"  smoke manifest: {SMOKE_SUBSET_MANIFEST}")
    print(f"  local weight  : {LOCAL_WEIGHT_PATH} (exists={LOCAL_WEIGHT_PATH.exists()})")
    print(f"  output root   : {OUTPUT_ROOT}")
    print(f"  output root 존재: {OUTPUT_ROOT.exists()}")

    if not SMOKE_SUBSET_MANIFEST.exists():
        print(f"  [ERROR] smoke manifest 없음: {SMOKE_SUBSET_MANIFEST}")
        return

    crops = select_forward_check_crops(SMOKE_SUBSET_MANIFEST)
    print(f"\n  선택된 crops: {len(crops)}개 (6-bin × 각 1개)")
    print()
    print(f"  {'#':>4}  {'six_bin_label':>22}  {'safe_id':>32}  {'local_z':>8}  {'low_z_warn':>11}")
    print("  " + "-" * 86)
    for i, row in enumerate(crops):
        lbl = row.get("six_bin_label", "?")
        sid = row.get("safe_id", "?")
        lz = row.get("local_z", "?")
        warn = has_low_z_boundary_warning(int(lz)) if lz != "?" else "?"
        print(f"  {i+1:>4}  {lbl:>22}  {sid:>32}  {lz:>8}  {str(warn):>11}")

    n_low_z = sum(1 for r in crops if has_low_z_boundary_warning(int(r["local_z"])))
    print()
    print(f"  low_z_boundary_warning 포함: {n_low_z}개 (최소 1개 조건: {'충족' if n_low_z >= 1 else '미충족'})")
    print(f"  output root 없음: {not OUTPUT_ROOT.exists()}")
    if OUTPUT_ROOT.exists():
        print(f"  [ABORT 조건] output root가 이미 존재합니다. --run 실행 불가.")
    else:
        print(f"  [OK] output root 없음 → --run 실행 가능 (사용자 승인 필요)")
    print("\n[DRY-PLAN 완료] 파일 생성 없음.")


# =============================================================================
# forward check run
# =============================================================================

def run_forward_check() -> None:
    """teacher/student forward shape + loss finite check."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    import datetime

    # output root guard
    if OUTPUT_ROOT.exists():
        print(f"[ABORT] output root 이미 존재 → 즉시 중단: {OUTPUT_ROOT}")
        sys.exit(1)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"  output root 생성: {OUTPUT_ROOT}")

    device = "cpu"
    print(f"  device: {device}")

    # local weight 메타 저장
    wt_stat = LOCAL_WEIGHT_PATH.stat() if LOCAL_WEIGHT_PATH.exists() else None
    local_weight_info = {
        "local_weight_available": LOCAL_WEIGHT_PATH.exists(),
        "local_weight_path": str(LOCAL_WEIGHT_PATH),
        "local_weight_size_mb": round(wt_stat.st_size / 1024 / 1024, 2) if wt_stat else 0.0,
    }
    with open(OUTPUT_ROOT / "rd_b6a2_local_weight_check.json", "w", encoding="utf-8") as f:
        json.dump(local_weight_info, f, ensure_ascii=False, indent=2)

    if not LOCAL_WEIGHT_PATH.exists():
        print(f"[ERROR] local weight 없음: {LOCAL_WEIGHT_PATH}")
        sys.exit(1)

    # teacher 빌드
    print("  teacher 빌드 (local weight 로드) ...")
    teacher = build_teacher(LOCAL_WEIGHT_PATH)
    teacher_features: dict = {}

    def make_hook(name: str):
        def hook(module, input, output):
            teacher_features[name] = output.detach()
        return hook

    teacher.layer1.register_forward_hook(make_hook("layer1"))
    teacher.layer2.register_forward_hook(make_hook("layer2"))
    teacher.layer3.register_forward_hook(make_hook("layer3"))
    print("  teacher: eval+frozen, hook 등록 완료")

    # student 빌드
    print("  student decoder 빌드 (random init) ...")
    student = build_student_decoder()
    print("  student: random init, eval")

    # crop 선택 및 경로 로드
    crops = select_forward_check_crops(SMOKE_SUBSET_MANIFEST)
    target_ids = set(r["safe_id"] for r in crops)
    patient_paths = load_patient_paths(PATIENT_MANIFEST_PATH, target_ids)

    forward_rows = []
    feature_shape_rows = []
    loss_finite_rows = []
    errors_list = []

    for check_idx, row in enumerate(crops):
        check_id = f"chk{check_idx+1:03d}"
        sid   = row.get("safe_id", "")
        mid   = row.get("manifest_id", "")
        pid   = row.get("patient_id", "")
        lbl   = row.get("six_bin_label", "")
        lz    = int(row["local_z"])
        low_z_warn = has_low_z_boundary_warning(lz)
        y0 = int(row["crop_y0"]); x0 = int(row["crop_x0"])
        y1 = int(row["crop_y1"]); x1 = int(row["crop_x1"])

        result_row: dict = {
            "check_id": check_id,
            "manifest_id": mid, "patient_id": pid, "safe_id": sid,
            "six_bin_label": lbl, "local_z": lz,
            "low_z_boundary_warning": low_z_warn,
            "input_shape": "", "input_min": "", "input_max": "",
            "input_nan_count": "", "input_inf_count": "",
            "teacher_layer1_shape": "", "teacher_layer2_shape": "", "teacher_layer3_shape": "",
            "student_de_layer1_shape": "", "student_de_layer2_shape": "", "student_de_layer3_shape": "",
            "total_loss": "", "loss_finite": "", "note": "",
        }

        try:
            if sid not in patient_paths:
                result_row["note"] = "SKIP_NO_PATH"
                forward_rows.append(result_row)
                errors_list.append({"check_id": check_id, "error": f"patient path 없음: {sid}"})
                continue

            ct_path = patient_paths[sid]["ct_hu_npy"]
            assert_path_safe(ct_path)
            ct_arr = np.load(ct_path, mmap_mode="r")

            # on-the-fly crop
            input_tensor = build_crop_tensor(ct_arr, lz, y0, x0, y1, x1)
            del ct_arr

            # input 확인
            inp_shape = tuple(input_tensor.shape)
            inp_min = float(input_tensor.min())
            inp_max = float(input_tensor.max())
            inp_nan = int(torch.isnan(input_tensor).sum())
            inp_inf = int(torch.isinf(input_tensor).sum())
            inp_shape_ok = inp_shape == (BATCH_SIZE, N_CHANNELS, CROP_SIZE, CROP_SIZE)

            result_row["input_shape"] = str(inp_shape)
            result_row["input_min"] = round(inp_min, 6)
            result_row["input_max"] = round(inp_max, 6)
            result_row["input_nan_count"] = inp_nan
            result_row["input_inf_count"] = inp_inf

            # teacher forward (backward 금지)
            teacher_features.clear()
            with torch.no_grad():
                _ = teacher(input_tensor)

            t_l1 = teacher_features["layer1"]
            t_l2 = teacher_features["layer2"]
            t_l3 = teacher_features["layer3"]
            t_l1_shape = tuple(t_l1.shape)
            t_l2_shape = tuple(t_l2.shape)
            t_l3_shape = tuple(t_l3.shape)

            result_row["teacher_layer1_shape"] = str(t_l1_shape)
            result_row["teacher_layer2_shape"] = str(t_l2_shape)
            result_row["teacher_layer3_shape"] = str(t_l3_shape)

            # student forward (backward 금지)
            with torch.no_grad():
                de3, de2, de1 = student(t_l3)

            s_de3_shape = tuple(de3.shape)
            s_de2_shape = tuple(de2.shape)
            s_de1_shape = tuple(de1.shape)

            result_row["student_de_layer3_shape"] = str(s_de3_shape)
            result_row["student_de_layer2_shape"] = str(s_de2_shape)
            result_row["student_de_layer1_shape"] = str(s_de1_shape)

            # loss 계산만 (backward 금지)
            # cosine_similarity: dim=1 → channel 방향 → (B, H, W) → .mean()
            with torch.no_grad():
                loss_l1 = 1.0 - F.cosine_similarity(t_l1, de1, dim=1).mean()
                loss_l2 = 1.0 - F.cosine_similarity(t_l2, de2, dim=1).mean()
                loss_l3 = 1.0 - F.cosine_similarity(t_l3, de3, dim=1).mean()
                total_loss = loss_l1 + loss_l2 + loss_l3

            total_loss_val = float(total_loss)
            loss_finite = math.isfinite(total_loss_val)

            result_row["total_loss"] = round(total_loss_val, 6)
            result_row["loss_finite"] = loss_finite
            result_row["note"] = "OK"

            # feature shape check 행
            t_l1_ok = t_l1_shape[1:] == EXPECTED_TEACHER_SHAPES["layer1"]
            t_l2_ok = t_l2_shape[1:] == EXPECTED_TEACHER_SHAPES["layer2"]
            t_l3_ok = t_l3_shape[1:] == EXPECTED_TEACHER_SHAPES["layer3"]
            s_de3_ok = s_de3_shape[1:] == EXPECTED_STUDENT_SHAPES["de_layer3"]
            s_de2_ok = s_de2_shape[1:] == EXPECTED_STUDENT_SHAPES["de_layer2"]
            s_de1_ok = s_de1_shape[1:] == EXPECTED_STUDENT_SHAPES["de_layer1"]

            feature_shape_rows.append({
                "check_id": check_id, "six_bin_label": lbl,
                "input_shape_ok": inp_shape_ok,
                "teacher_layer1_shape": str(t_l1_shape), "teacher_layer1_shape_ok": t_l1_ok,
                "teacher_layer2_shape": str(t_l2_shape), "teacher_layer2_shape_ok": t_l2_ok,
                "teacher_layer3_shape": str(t_l3_shape), "teacher_layer3_shape_ok": t_l3_ok,
                "student_de_layer3_shape": str(s_de3_shape), "student_de_layer3_shape_ok": s_de3_ok,
                "student_de_layer2_shape": str(s_de2_shape), "student_de_layer2_shape_ok": s_de2_ok,
                "student_de_layer1_shape": str(s_de1_shape), "student_de_layer1_shape_ok": s_de1_ok,
            })

            # loss finite 행
            loss_finite_rows.append({
                "check_id": check_id, "six_bin_label": lbl,
                "loss_layer1": round(float(loss_l1), 6),
                "loss_layer2": round(float(loss_l2), 6),
                "loss_layer3": round(float(loss_l3), 6),
                "total_loss": round(total_loss_val, 6),
                "loss_finite": loss_finite,
            })

        except Exception as e:
            result_row["note"] = f"ERROR: {e}"
            errors_list.append({"check_id": check_id, "error": str(e)})

        forward_rows.append(result_row)
        print(
            f"  [{check_id}] {lbl:22s} | inp={result_row['input_shape']}"
            f" | t_l3={result_row['teacher_layer3_shape']}"
            f" | s_de1={result_row['student_de_layer1_shape']}"
            f" | loss={result_row['total_loss']} finite={result_row['loss_finite']}"
            f" | low_z={low_z_warn}"
        )

    # ── CSV 저장 ──────────────────────────────────────────────────────────────
    _write_csv(OUTPUT_ROOT / "rd_b6a2_forward_check_rows.csv",
               FORWARD_ROWS_FIELDS, forward_rows)
    _write_csv(OUTPUT_ROOT / "rd_b6a2_feature_shape_check.csv",
               FEATURE_SHAPE_FIELDS, feature_shape_rows)
    _write_csv(OUTPUT_ROOT / "rd_b6a2_loss_finite_check.csv",
               LOSS_FINITE_FIELDS, loss_finite_rows)
    _write_csv(OUTPUT_ROOT / "rd_b6a2_errors.csv",
               ["check_id", "error"], errors_list)

    # ── summary JSON ──────────────────────────────────────────────────────────
    ok_rows = [r for r in forward_rows if r["note"] == "OK"]
    inp_pass = len(ok_rows) > 0 and all(
        r["input_shape"] == str((BATCH_SIZE, N_CHANNELS, CROP_SIZE, CROP_SIZE))
        for r in ok_rows
    )
    t_pass = len(feature_shape_rows) > 0 and all(
        r["teacher_layer1_shape_ok"] and
        r["teacher_layer2_shape_ok"] and
        r["teacher_layer3_shape_ok"]
        for r in feature_shape_rows
    )
    s_pass = len(feature_shape_rows) > 0 and all(
        r["student_de_layer3_shape_ok"] and
        r["student_de_layer2_shape_ok"] and
        r["student_de_layer1_shape_ok"]
        for r in feature_shape_rows
    )
    lf_pass = len(loss_finite_rows) > 0 and all(
        r["loss_finite"] for r in loss_finite_rows
    )
    all_pass = (len(ok_rows) == len(forward_rows) and
                inp_pass and t_pass and s_pass and lf_pass)

    summary = {
        "version": "rd_b6a2_v1",
        "timestamp": ts,
        "n_checked_crops": len(forward_rows),
        "batch_size": BATCH_SIZE,
        "device_used": device,
        "local_weight_available": local_weight_info["local_weight_available"],
        "local_weight_path": local_weight_info["local_weight_path"],
        "local_weight_size_mb": local_weight_info["local_weight_size_mb"],
        "input_shape_pass": inp_pass,
        "teacher_shape_pass": t_pass,
        "student_shape_pass": s_pass,
        "loss_finite_pass": lf_pass,
        "backward_called": False,
        "optimizer_created": False,
        "checkpoint_saved": False,
        "training_started": False,
        "scoring_started": False,
        "stage2_holdout_access": 0,
        "all_checks_passed": all_pass,
        "verdict": "통과" if all_pass else "경고",
    }
    with open(OUTPUT_ROOT / "rd_b6a2_tiny_forward_check_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    _write_report_md(OUTPUT_ROOT, ts, summary, forward_rows, feature_shape_rows, loss_finite_rows)

    if all_pass:
        (OUTPUT_ROOT / "DONE").write_text(f"rd_b6a2 tiny-forward-check completed: {ts}\n")
    else:
        print("[NO_DONE] all_checks_passed=False → DONE marker not created")

    print(f"\n판정: {summary['verdict']}")
    print(f"  checked={len(forward_rows)}  ok={len(ok_rows)}")
    print(f"  input_shape_pass={inp_pass}  teacher_shape_pass={t_pass}")
    print(f"  student_shape_pass={s_pass}  loss_finite_pass={lf_pass}")
    print(f"  backward=False  optimizer=False  checkpoint=False")
    print(f"\n생성 파일:")
    for fn in sorted(OUTPUT_ROOT.iterdir()):
        print(f"  {fn.name}")


# =============================================================================
# CSV / report 헬퍼
# =============================================================================

def _write_csv(path: Path, fieldnames: list, rows: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"  → {path.name}")


def _write_report_md(
    out_dir: Path, ts: str, summary: dict,
    forward_rows: list, feature_shape_rows: list, loss_finite_rows: list
) -> None:
    verdict = summary["verdict"]
    n = summary["n_checked_crops"]
    n_low_z = sum(1 for r in forward_rows if r.get("low_z_boundary_warning") is True)

    lines = [
        "# RD-B6a-2 Tiny Forward Check Report",
        f"- 버전: rd_b6a2_v1",
        f"- 날짜: {ts}",
        f"- 판정: **{verdict}**",
        "",
        "---",
        "## 1. RD-B6a 결과 요약",
        "",
        "| 항목 | 결과 |",
        "|------|------|",
        "| smoke subset | 60 crops = 2 patients × 6-bin × 5 crops |",
        "| crop loading | 3/3 OK |",
        "| crop shape | (3,96,96) ✓ |",
        "| value range | [0,1] ✓ |",
        "| NaN/Inf | 없음 ✓ |",
        "| low_z_boundary_warning | 1개 포함 ✓ |",
        "| local ResNet18 weight | 44.66MB 존재 ✓ |",
        "",
        "---",
        "## 2. local ResNet18 weight 확인 결과",
        "",
        f"- 경로: `{summary['local_weight_path']}`",
        f"- 존재: **{summary['local_weight_available']}**",
        f"- 크기: {summary['local_weight_size_mb']} MB",
        "- 자동 다운로드: **금지** — local cache 사용",
        "",
        "---",
        "## 3. input tensor shape/value range 결과",
        "",
        f"- 확인 crop 수: {n}",
        f"- input shape pass: **{summary['input_shape_pass']}** (expected: (1,3,96,96))",
        "",
        "| check_id | six_bin_label | input_shape | min | max | nan | inf | low_z |",
        "|----------|---------------|-------------|-----|-----|-----|-----|-------|",
    ]
    for r in forward_rows:
        lines.append(
            f"| {r['check_id']} | {r['six_bin_label']} | {r['input_shape']} "
            f"| {r['input_min']} | {r['input_max']} | {r['input_nan_count']} "
            f"| {r['input_inf_count']} | {r['low_z_boundary_warning']} |"
        )

    lines.extend([
        "",
        "---",
        "## 4. teacher layer1/layer2/layer3 shape 결과",
        "",
        f"- teacher shape pass: **{summary['teacher_shape_pass']}**",
        "- expected: layer1=(B,64,24,24) / layer2=(B,128,12,12) / layer3=(B,256,6,6)",
        "",
        "| check_id | bin | layer1 | l1_ok | layer2 | l2_ok | layer3 | l3_ok |",
        "|----------|-----|--------|-------|--------|-------|--------|-------|",
    ])
    for r in feature_shape_rows:
        lines.append(
            f"| {r['check_id']} | {r['six_bin_label']} "
            f"| {r['teacher_layer1_shape']} | {r['teacher_layer1_shape_ok']} "
            f"| {r['teacher_layer2_shape']} | {r['teacher_layer2_shape_ok']} "
            f"| {r['teacher_layer3_shape']} | {r['teacher_layer3_shape_ok']} |"
        )

    lines.extend([
        "",
        "---",
        "## 5. student de_layer1/de_layer2/de_layer3 shape 결과",
        "",
        f"- student shape pass: **{summary['student_shape_pass']}**",
        "- expected: de_layer3=(B,256,6,6) / de_layer2=(B,128,12,12) / de_layer1=(B,64,24,24)",
        "",
        "| check_id | bin | de_layer3 | de3_ok | de_layer2 | de2_ok | de_layer1 | de1_ok |",
        "|----------|-----|-----------|--------|-----------|--------|-----------|--------|",
    ])
    for r in feature_shape_rows:
        lines.append(
            f"| {r['check_id']} | {r['six_bin_label']} "
            f"| {r['student_de_layer3_shape']} | {r['student_de_layer3_shape_ok']} "
            f"| {r['student_de_layer2_shape']} | {r['student_de_layer2_shape_ok']} "
            f"| {r['student_de_layer1_shape']} | {r['student_de_layer1_shape_ok']} |"
        )

    lines.extend([
        "",
        "---",
        "## 6. loss finite 결과",
        "",
        f"- loss finite pass: **{summary['loss_finite_pass']}**",
        "- 수식: loss_k = 1 - cosine_similarity(teacher_k, student_de_k, dim=1).mean()",
        "- total_loss = loss_layer1 + loss_layer2 + loss_layer3",
        "- backward 호출: **없음**",
        "",
        "| check_id | bin | loss_l1 | loss_l2 | loss_l3 | total | finite |",
        "|----------|-----|---------|---------|---------|-------|--------|",
    ])
    for r in loss_finite_rows:
        lines.append(
            f"| {r['check_id']} | {r['six_bin_label']} "
            f"| {r['loss_layer1']} | {r['loss_layer2']} | {r['loss_layer3']} "
            f"| {r['total_loss']} | {r['loss_finite']} |"
        )

    lines.extend([
        "",
        "---",
        "## 7. low_z_boundary_warning crop 포함 여부",
        "",
        f"- low_z_boundary_warning=True crop: **{n_low_z}개**",
        f"- 최소 1개 포함 조건: {'충족' if n_low_z >= 1 else '미충족 ⚠️'}",
        "",
        "---",
        "## 8. 다음 단계",
        "",
        "- **RD-B6b**: tiny smoke train",
        "  - batch_size=24 (bin당 4), n_epochs=5, lr=1e-4",
        "  - 60 crops smoke subset 사용",
        "  - teacher frozen, student decoder 학습",
        "  - 사용자 승인 후 진행",
        "",
        "---",
        "## 9. 절대 하지 않은 것",
        "",
        "| 항목 | 확인 |",
        "|------|------|",
        "| backward 없음 | ✓ |",
        "| optimizer 없음 | ✓ |",
        "| checkpoint 저장 없음 | ✓ |",
        "| training 없음 | ✓ |",
        "| scoring 없음 | ✓ |",
        "| full crop NPZ 생성 없음 | ✓ |",
        "| stage2_holdout 접근 없음 | ✓ |",
        "| 기존 파일 수정 없음 | ✓ |",
        "| GPU 사용 없음 | ✓ (CPU only) |",
        "| 자동 다운로드 없음 | ✓ (local weight 사용) |",
    ])

    report_path = out_dir / "rd_b6a2_tiny_forward_check_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  → {report_path.name}")


# =============================================================================
# main
# =============================================================================

def main() -> None:
    print("=" * 70)
    print("RD-B6a-2 Tiny Forward Check")
    print("=" * 70)

    if IS_DRY_PLAN:
        run_dry_plan()
        return

    if IS_RUN:
        print("\n[RUN] teacher/student forward shape + loss finite check 실행 ...")
        run_forward_check()
        return


if __name__ == "__main__":
    main()
