"""
RD-B3: True RD4AD / Reverse Distillation Architecture Preflight

목적:
- true RD4AD teacher-student 구조 설계 (architecture preflight 전용)
- 기존 v1/v1 ConvAE baseline과 구조 차이 명확히 기록
- backbone 후보 비교 및 추천
- 6-bin balanced sampler, score calibration 설계 기록

금지 사항:
- crop NPZ 생성 금지
- 학습 금지
- scoring 금지
- model forward 금지
- checkpoint 로드 금지
- GPU 사용 금지
- stage2_holdout 접근 금지
- 기존 파일 수정 금지
- 기존 파일 덮어쓰기 금지
"""

import argparse
import csv
import json
import os
import sys
import time
import py_compile
from pathlib import Path

# ─── 경로 상수 ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "normal_based_stage2_verifier_audit"
    / "rd_b3_true_rd4ad_architecture_preflight_v1"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "normal_based_stage2_verifier_audit"
    / "rd_b1_6bin_balanced_manifest_preflight_v1"
    / "rd_b1_6bin_balanced_normal_train_coordinate_manifest.csv"
)

# ─── BARE-RUN GUARD ───────────────────────────────────────────────────────────
if __name__ == "__main__" and len(sys.argv) == 1:
    print("[BARE-RUN GUARD] 인자 없이 실행 불가.")
    print("사용법: python rd_b3_true_rd4ad_architecture_preflight.py --dry-run")
    print("        python rd_b3_true_rd4ad_architecture_preflight.py --run")
    sys.exit(1)

# ─── 금지 접근 경로 검사 ──────────────────────────────────────────────────────
FORBIDDEN_PATHS = [
    PROJECT_ROOT / "data" / "stage2_holdout",
    PROJECT_ROOT / "data" / "lesion",
]

def check_forbidden_access():
    errors = []
    for fp in FORBIDDEN_PATHS:
        if fp.exists():
            # 존재 확인만, 접근하지 않음 — 단지 경고
            errors.append(f"FORBIDDEN PATH EXISTS (접근 금지): {fp}")
    return errors


# ─── Backbone 이론 계산 ───────────────────────────────────────────────────────
# ResNet stem: conv(stride=2) + maxpool(stride=2) → 입력의 1/4
# ResNet layer stride: layer1=1, layer2=2, layer3=2, layer4=2

def compute_resnet_feature_shapes(input_size=96):
    """
    ResNet 계열의 feature map shape를 이론값으로 계산.
    실제 model.forward() 없이 stride 기반 계산.
    """
    after_stem = input_size // 4  # conv(s2) + maxpool(s2)
    layer1_h = after_stem         # stride=1
    layer2_h = layer1_h // 2     # stride=2
    layer3_h = layer2_h // 2     # stride=2
    layer4_h = layer3_h // 2     # stride=2
    return {
        "after_stem": (after_stem, after_stem),
        "layer1": (layer1_h, layer1_h),
        "layer2": (layer2_h, layer2_h),
        "layer3": (layer3_h, layer3_h),
        "layer4": (layer4_h, layer4_h),
    }


def build_backbone_candidates(input_size=96):
    shapes = compute_resnet_feature_shapes(input_size)

    # ResNet18 채널
    r18_channels = {"layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512}
    # ResNet50 / WideResNet50 채널 (bottleneck)
    r50_channels = {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048}
    # EfficientNet-B0 주요 feature 채널 (MBConv blocks, 공식 구현 기준)
    eff_b0_channels = {
        "stage1": 24,   # after first MBConv, input/4
        "stage2": 40,   # stride 누적 후 input/8
        "stage3": 112,  # input/16
        "stage4": 1280, # last conv, input/32
    }
    eff_b0_shapes = {
        "stage1": (input_size // 4, input_size // 4),
        "stage2": (input_size // 8, input_size // 8),
        "stage3": (input_size // 16, input_size // 16),
        "stage4": (input_size // 32, input_size // 32),
    }

    candidates = []

    # ResNet18
    candidates.append({
        "backbone": "ResNet18_ImageNet",
        "pretrained": "ImageNet",
        "params_M": 11.7,
        "feature_layers_used": "layer1,layer2,layer3",
        "layer1_shape": f"{r18_channels['layer1']}x{shapes['layer1'][0]}x{shapes['layer1'][1]}",
        "layer2_shape": f"{r18_channels['layer2']}x{shapes['layer2'][0]}x{shapes['layer2'][1]}",
        "layer3_shape": f"{r18_channels['layer3']}x{shapes['layer3'][0]}x{shapes['layer3'][1]}",
        "layer4_shape": f"{r18_channels['layer4']}x{shapes['layer4'][0]}x{shapes['layer4'][1]}",
        "gpu_mem_est_GB": "~1.5 (batch32 96x96)",
        "rd4ad_fit": "GOOD",
        "multiscale_feasibility": "HIGH",
        "complexity": "LOW",
        "padim_compat": "HIGH (기존 PaDiM도 ResNet 계열)",
        "note": "RD4AD 원논문 구현 다수. 96x96 소형 입력에서 layer3=6x6으로 충분히 세밀. 추천 후보.",
        "recommendation": "1차 추천",
    })

    # ResNet50
    candidates.append({
        "backbone": "ResNet50_ImageNet",
        "pretrained": "ImageNet",
        "params_M": 25.6,
        "feature_layers_used": "layer1,layer2,layer3",
        "layer1_shape": f"{r50_channels['layer1']}x{shapes['layer1'][0]}x{shapes['layer1'][1]}",
        "layer2_shape": f"{r50_channels['layer2']}x{shapes['layer2'][0]}x{shapes['layer2'][1]}",
        "layer3_shape": f"{r50_channels['layer3']}x{shapes['layer3'][0]}x{shapes['layer3'][1]}",
        "layer4_shape": f"{r50_channels['layer4']}x{shapes['layer4'][0]}x{shapes['layer4'][1]}",
        "gpu_mem_est_GB": "~3.5 (batch32 96x96)",
        "rd4ad_fit": "GOOD",
        "multiscale_feasibility": "HIGH",
        "complexity": "MEDIUM",
        "padim_compat": "HIGH (기존 PaDiM ResNet50 동일 backbone)",
        "note": "기존 PaDiM ResNet50과 동일 backbone. feature 호환성 최고. 채널수 많아 student decoder 복잡도 증가.",
        "recommendation": "2차 추천",
    })

    # WideResNet50
    candidates.append({
        "backbone": "WideResNet50_ImageNet",
        "pretrained": "ImageNet",
        "params_M": 68.9,
        "feature_layers_used": "layer1,layer2,layer3",
        "layer1_shape": f"{r50_channels['layer1']}x{shapes['layer1'][0]}x{shapes['layer1'][1]}",
        "layer2_shape": f"{r50_channels['layer2']}x{shapes['layer2'][0]}x{shapes['layer2'][1]}",
        "layer3_shape": f"{r50_channels['layer3']}x{shapes['layer3'][0]}x{shapes['layer3'][1]}",
        "layer4_shape": f"{r50_channels['layer4']}x{shapes['layer4'][0]}x{shapes['layer4'][1]}",
        "gpu_mem_est_GB": "~6.0 (batch32 96x96)",
        "rd4ad_fit": "VERY_GOOD",
        "multiscale_feasibility": "HIGH",
        "complexity": "HIGH",
        "padim_compat": "MEDIUM",
        "note": "RD4AD 원논문 WideResNet50 사용. 파라미터 많고 GPU mem 높음. 96x96 소형 입력 대비 over-parameterized 위험.",
        "recommendation": "3차 후보 (자원 여유 시)",
    })

    # EfficientNet-B0
    candidates.append({
        "backbone": "EfficientNet_B0_ImageNet",
        "pretrained": "ImageNet",
        "params_M": 5.3,
        "feature_layers_used": "stage1,stage2,stage3",
        "layer1_shape": f"{eff_b0_channels['stage1']}x{eff_b0_shapes['stage1'][0]}x{eff_b0_shapes['stage1'][1]}",
        "layer2_shape": f"{eff_b0_channels['stage2']}x{eff_b0_shapes['stage2'][0]}x{eff_b0_shapes['stage2'][1]}",
        "layer3_shape": f"{eff_b0_channels['stage3']}x{eff_b0_shapes['stage3'][0]}x{eff_b0_shapes['stage3'][1]}",
        "layer4_shape": f"{eff_b0_channels['stage4']}x{eff_b0_shapes['stage4'][0]}x{eff_b0_shapes['stage4'][1]}",
        "gpu_mem_est_GB": "~1.2 (batch32 96x96)",
        "rd4ad_fit": "CONDITIONAL",
        "multiscale_feasibility": "MEDIUM (MBConv 중간 레이어 추출 복잡)",
        "complexity": "HIGH (feature hook 구현 복잡)",
        "padim_compat": "LOW (기존 PaDiM과 다른 구조)",
        "note": "가장 경량. 단, RD4AD teacher feature 추출을 위한 hook 설계가 ResNet보다 복잡. 기존 프로젝트 PaDiM ResNet50과 구조 단절.",
        "recommendation": "4차 후보 (경량화 필요 시)",
    })

    return candidates


# ─── Architecture Design Table ───────────────────────────────────────────────
def build_architecture_table():
    rows = [
        {
            "component": "Teacher Encoder",
            "type": "Pretrained CNN (ResNet18 권장)",
            "trainable": "False",
            "mode": "eval",
            "role": "multi-scale feature extraction",
            "layers": "layer1, layer2, layer3",
            "note": "ImageNet pretrained, frozen. 학습 불가. feature만 추출.",
        },
        {
            "component": "Student Decoder",
            "type": "Reverse Distillation Decoder",
            "trainable": "True",
            "mode": "train",
            "role": "teacher feature 복원 / 따라가기",
            "layers": "de_layer3, de_layer2, de_layer1 (역순 upsample)",
            "note": "teacher 출력과 동일 resolution/channel로 복원. bottleneck 거쳐 역순 upsampling.",
        },
        {
            "component": "Bottleneck (OCBE)",
            "type": "One-Class Bottleneck Embedding",
            "trainable": "True",
            "mode": "train",
            "role": "teacher layer3 feature → compact code → student 입력",
            "layers": "1-3개 conv block",
            "note": "RD4AD 원논문 구조. student가 teacher high-level feature를 compact code로 받아 역방향 복원 시작.",
        },
        {
            "component": "Loss Function",
            "type": "Multi-scale Cosine Similarity Loss",
            "trainable": "N/A",
            "mode": "train",
            "role": "teacher-student feature 거리 최소화 (normal-only)",
            "layers": "layer1, layer2, layer3 각각",
            "note": "1 - cos_sim(teacher_feat, student_feat) 합산. pixel loss 기반 ConvAE와 근본 차이.",
        },
        {
            "component": "Anomaly Score",
            "type": "Feature Distance Map",
            "trainable": "N/A",
            "mode": "inference",
            "role": "crop-level / patch-level 이상 점수",
            "layers": "multi-scale 합산",
            "note": "test 시 teacher-student 거리. normal에서는 낮고 이상에서 높아야 함.",
        },
        {
            "component": "Input",
            "type": "mixed_3ch 96x96",
            "trainable": "N/A",
            "mode": "both",
            "role": "3채널 CT crop",
            "layers": "ch1=CT_center, ch2=lower_3mm_MIP, ch3=upper_3mm_MIP",
            "note": "HU [-1000,600] → [0,1] 정규화. RD-B2b 결정.",
        },
    ]
    return rows


# ─── Training Sampler Design ─────────────────────────────────────────────────
def build_sampler_design():
    rows = []
    bins = [
        "upper_boundary", "upper_interior",
        "middle_boundary", "middle_interior",
        "lower_boundary", "lower_interior",
    ]
    batch_configs = [
        {"batch_size": 24, "per_bin": 4},
        {"batch_size": 48, "per_bin": 8},
        {"batch_size": 12, "per_bin": 2},
    ]
    for bc in batch_configs:
        for b in bins:
            rows.append({
                "batch_size": bc["batch_size"],
                "six_bin_label": b,
                "target_per_bin": bc["per_bin"],
                "oversample_allowed": "False",
                "shortage_handling": "다른 환자/bin에서 보완 (duplicate oversampling 금지)",
                "patient_leakage_prevention": "True",
                "train_val_test_split_preserved": "True",
                "note": (
                    "lower_boundary 부족 시 middle_boundary로 보완 가능"
                    if "lower" in b else ""
                ),
            })
    return rows


# ─── Score Calibration Design ────────────────────────────────────────────────
def build_score_calibration_design():
    rows = [
        {
            "calibration_type": "global_score_distribution",
            "description": "전체 normal 학습 crop의 teacher-student 거리 분포",
            "groupby": "None",
            "usage": "global threshold 기준",
            "note": "전체 normal score의 p50/p90/p95/p99 기록",
        },
        {
            "calibration_type": "six_bin_score_distribution",
            "description": "6-bin label별 normal score 분포",
            "groupby": "six_bin_label",
            "usage": "bin별 threshold 후보 산출",
            "note": "test crop은 자신의 six_bin_label 분포와 비교",
        },
        {
            "calibration_type": "boundary_vs_interior_distribution",
            "description": "boundary vs interior normal score 분포",
            "groupby": "boundary_status",
            "usage": "boundary/interior 성능 분석",
            "note": "boundary가 interior보다 score 높으면 FP 위험",
        },
        {
            "calibration_type": "z_level_distribution",
            "description": "z-level (upper/middle/lower)별 normal score 분포",
            "groupby": "z_level",
            "usage": "z-level별 정상 분포 차이 확인",
            "note": "lower_boundary low-z 케이스 모니터링 포함",
        },
        {
            "calibration_type": "low_z_warning_flag",
            "description": "z<=10인 lower_boundary crop의 warning flag",
            "groupby": "low_z_boundary_warning",
            "usage": "학습 제외 아닌 분석용 flag",
            "note": "diaphragm_saturation_risk, roi_ratio_low_warning 포함",
        },
    ]
    return rows


# ─── Input Pipeline Design ───────────────────────────────────────────────────
def build_input_pipeline_design():
    rows = [
        {
            "step": 1,
            "stage": "crop_extraction",
            "description": "RD-B1 manifest (crop_y0,x0,y1,x1,slice_index)로 crop 좌표 참조",
            "input": "manifest CSV + CT npy",
            "output": "96x96x3 numpy array",
            "note": "crop NPZ 사전생성 없이 on-the-fly 방식 설계 권장. 단, RD-B4에서 확정.",
            "forbidden": "crop NPZ 사전생성은 RD-B4 이후",
        },
        {
            "step": 2,
            "stage": "channel_assembly",
            "description": "ch1=CT center slice, ch2=lower 3mm MIP, ch3=upper 3mm MIP",
            "input": "CT volume npy",
            "output": "3채널 crop",
            "note": "3mm MIP = 해당 z±1~±N slice max projection. 두께는 CT spacing 기준 계산.",
            "forbidden": "",
        },
        {
            "step": 3,
            "stage": "normalization",
            "description": "HU clip [-1000, 600] → (HU + 1000) / 1600 → [0, 1]",
            "input": "raw HU crop",
            "output": "normalized [0,1] float32",
            "note": "new_RD_style. RD-B2b 결정.",
            "forbidden": "",
        },
        {
            "step": 4,
            "stage": "augmentation_train",
            "description": "RandomHorizontalFlip, RandomVerticalFlip, RandomRotation(±10)",
            "input": "normalized crop",
            "output": "augmented crop",
            "note": "색상/밝기 augmentation 금지 (HU 정규화 이후이므로 물리적 의미 왜곡).",
            "forbidden": "ColorJitter, GaussianBlur (HU 왜곡 위험)",
        },
        {
            "step": 5,
            "stage": "low_z_warning_flag",
            "description": "local_z <= 10인 lower_boundary crop에 flag 부착",
            "input": "manifest six_bin_label, local_z",
            "output": "low_z_boundary_warning=True/False",
            "note": "학습 제외 아님. 분석/모니터링용.",
            "forbidden": "",
        },
        {
            "step": 6,
            "stage": "six_bin_balanced_sampler",
            "description": "batch 내 6-bin 균등 sampling",
            "input": "manifest six_bin_label",
            "output": "balanced batch",
            "note": "duplicate oversampling 금지. patient leakage 방지.",
            "forbidden": "duplicate oversampling",
        },
    ]
    return rows


# ─── 경고 플래그 설계 ─────────────────────────────────────────────────────────
WARNING_FLAGS = [
    {
        "flag": "low_z_boundary_warning",
        "condition": "six_bin_label == 'lower_boundary' AND local_z <= 10",
        "action": "학습 유지, 분석용 flag 부착",
        "note": "RD-B2b에서 z=7 케이스 borderline 확인됨",
    },
    {
        "flag": "diaphragm_saturation_risk",
        "condition": "lower_boundary AND new_RD_style normalization에서 HU 높은 횡격막 포화",
        "action": "학습 유지, 분석용 flag 부착",
        "note": "횡격막 과포화 케이스 모니터링",
    },
    {
        "flag": "roi_ratio_low_warning",
        "condition": "refined_roi_ratio < 0.3",
        "action": "학습 유지, 분석용 flag 부착",
        "note": "ROI 비율이 낮은 crop은 폐 밖 영역 비율 높음",
    },
]


# ─── 기존 ConvAE vs RD4AD 차이 요약 ─────────────────────────────────────────
CONV_AE_VS_RD4AD = {
    "existing_v1_v1": {
        "architecture": "ConvAutoencoder (encoder-bottleneck-decoder)",
        "loss": "pixel reconstruction loss (MSE/L1)",
        "anomaly_score": "pixel-level reconstruction error",
        "teacher": "None (no teacher)",
        "student": "None (single network)",
        "supervised": "False",
        "is_true_rd4ad": "False",
        "note": "기존 v1/v1 branch는 ConvAE reconstruction baseline이었음. true RD4AD 아님.",
    },
    "new_rd_b3": {
        "architecture": "Teacher-Student (Reverse Distillation)",
        "loss": "multi-scale cosine similarity feature matching loss",
        "anomaly_score": "teacher-student feature distance (multi-scale 합산)",
        "teacher": "Pretrained CNN (frozen, eval mode)",
        "student": "Reverse Distillation Decoder (trainable)",
        "supervised": "False (normal-only training)",
        "is_true_rd4ad": "True",
        "note": "RD4AD 구조: OCBE bottleneck → student decoder → multi-scale feature distance",
    },
}


# ─── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="RD-B3 True RD4AD Architecture Preflight"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="파일 생성 없이 계획만 출력",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="실제 preflight 실행 (파일 생성)",
    )
    args = parser.parse_args()

    if not args.dry_run and not args.run:
        print("[ERROR] --dry-run 또는 --run 인자가 필요합니다.")
        sys.exit(1)

    print("=" * 70)
    print("RD-B3: True RD4AD Architecture Preflight")
    print(f"Mode: {'DRY-RUN (파일 생성 없음)' if args.dry_run else 'RUN (파일 생성)'}")
    print("=" * 70)

    t0 = time.time()
    errors = []

    # ── 안전 검사 ─────────────────────────────────────────────────────────────
    print("\n[1/8] 안전 검사...")
    forbidden_errors = check_forbidden_access()
    # forbidden path가 존재해도 접근은 안 함 — 경고만
    for fe in forbidden_errors:
        print(f"  WARNING: {fe}")

    # output root 중복 검사
    if OUTPUT_ROOT.exists():
        print(f"\n[ABORT] 출력 폴더가 이미 존재합니다: {OUTPUT_ROOT}")
        print("기존 결과를 덮어쓰지 않습니다. 즉시 중단.")
        sys.exit(1)
    print(f"  출력 root 없음: OK ({OUTPUT_ROOT})")

    # manifest 존재 확인
    if not MANIFEST_PATH.exists():
        errors.append(f"MANIFEST_NOT_FOUND: {MANIFEST_PATH}")
        print(f"  [ERROR] manifest 없음: {MANIFEST_PATH}")
    else:
        print(f"  manifest 확인: OK ({MANIFEST_PATH})")

    # ── Backbone 후보 계산 ────────────────────────────────────────────────────
    print("\n[2/8] Backbone 후보 이론 계산 (model forward 없음)...")
    backbone_candidates = build_backbone_candidates(input_size=96)
    recommended = next(c for c in backbone_candidates if c["recommendation"] == "1차 추천")
    print(f"  후보 수: {len(backbone_candidates)}")
    print(f"  추천 backbone: {recommended['backbone']}")
    print(f"  layer1: {recommended['layer1_shape']}")
    print(f"  layer2: {recommended['layer2_shape']}")
    print(f"  layer3: {recommended['layer3_shape']}")

    # ── Architecture Table ────────────────────────────────────────────────────
    print("\n[3/8] Architecture Design Table 생성...")
    arch_table = build_architecture_table()
    print(f"  컴포넌트 수: {len(arch_table)}")

    # ── Sampler Design ────────────────────────────────────────────────────────
    print("\n[4/8] Training Sampler Design 생성...")
    sampler_design = build_sampler_design()
    print(f"  설계 항목 수: {len(sampler_design)}")

    # ── Score Calibration ─────────────────────────────────────────────────────
    print("\n[5/8] Score Calibration Design 생성...")
    calibration_design = build_score_calibration_design()
    print(f"  calibration 항목 수: {len(calibration_design)}")

    # ── Input Pipeline ────────────────────────────────────────────────────────
    print("\n[6/8] Input Pipeline Design 생성...")
    input_pipeline = build_input_pipeline_design()
    print(f"  pipeline 단계 수: {len(input_pipeline)}")

    # ── Summary 구성 ──────────────────────────────────────────────────────────
    elapsed = round(time.time() - t0, 2)
    summary = {
        "version": "rd_b3_v1",
        "integrity_passed": len(errors) == 0,
        "errors": errors,
        "dry_run": args.dry_run,
        "recommended_backbone": recommended["backbone"],
        "recommended_layers": recommended["feature_layers_used"],
        "recommended_gpu_mem_est": recommended["gpu_mem_est_GB"],
        "input_design": {
            "channels": "mixed_3ch",
            "ch1": "CT_center",
            "ch2": "lower_3mm_MIP",
            "ch3": "upper_3mm_MIP",
            "crop_size": "96x96",
            "normalization": "new_RD_style: HU clip [-1000,600] → [0,1]",
        },
        "architecture": {
            "type": "true_RD4AD_reverse_distillation",
            "teacher": "ResNet18 (ImageNet, frozen, eval)",
            "bottleneck": "OCBE (One-Class Bottleneck Embedding)",
            "student": "Reverse Distillation Decoder (trainable)",
            "loss": "multi-scale cosine similarity feature matching",
            "anomaly_score": "teacher-student feature distance",
        },
        "six_bin_labels": [
            "upper_boundary", "upper_interior",
            "middle_boundary", "middle_interior",
            "lower_boundary", "lower_interior",
        ],
        "warning_flags": [f["flag"] for f in WARNING_FLAGS],
        "existing_v1_architecture": CONV_AE_VS_RD4AD["existing_v1_v1"]["architecture"],
        "existing_v1_is_true_rd4ad": CONV_AE_VS_RD4AD["existing_v1_v1"]["is_true_rd4ad"],
        "absolute_not_done": [
            "crop NPZ 생성 없음",
            "학습 없음",
            "scoring 없음",
            "model forward 없음",
            "checkpoint 로드 없음",
            "GPU 사용 없음",
            "stage2_holdout 접근 없음",
            "lesion 접근 없음",
            "기존 파일 수정 없음",
            "threshold 재계산 없음",
            "score 재계산 없음",
        ],
        "next_steps": [
            "RD-B4: crop generation preflight (on-the-fly vs pre-generation 결정)",
            "RD-B5: model code skeleton / static check (py_compile, shape dry test)",
            "RD-B6: tiny smoke train (10 crops, 2 epoch, CPU)",
        ],
        "elapsed_seconds": elapsed,
    }

    # ── DRY-RUN 출력 ──────────────────────────────────────────────────────────
    print("\n[7/8] 설계 요약...")
    print(f"\n  === 추천 Backbone ===")
    print(f"  {recommended['backbone']}")
    print(f"  layer1: {recommended['layer1_shape']}")
    print(f"  layer2: {recommended['layer2_shape']}")
    print(f"  layer3: {recommended['layer3_shape']}")
    print(f"  GPU mem: {recommended['gpu_mem_est_GB']}")
    print(f"  이유: {recommended['note']}")
    print(f"\n  === 기존 v1 vs 새 RD4AD ===")
    print(f"  기존: {CONV_AE_VS_RD4AD['existing_v1_v1']['architecture']}")
    print(f"  신규: {CONV_AE_VS_RD4AD['new_rd_b3']['architecture']}")
    print(f"\n  === 안전 검증 ===")
    print(f"  오류 수: {len(errors)}")
    print(f"  출력 root 중복: 없음 (안전)")
    print(f"  forbidden path 접근: 없음")

    if args.dry_run:
        print("\n[DRY-RUN] 파일 생성 없이 종료합니다.")
        print("실제 실행: python rd_b3_true_rd4ad_architecture_preflight.py --run")
        sys.exit(0)

    # ── 실제 파일 생성 ────────────────────────────────────────────────────────
    print("\n[8/8] 파일 생성...")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)

    # errors.csv
    errors_path = OUTPUT_ROOT / "rd_b3_errors.csv"
    with open(errors_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "error"])
        writer.writeheader()
        for i, e in enumerate(errors):
            writer.writerow({"index": i, "error": e})
    print(f"  rd_b3_errors.csv 생성 ({len(errors)} 오류)")

    # backbone candidate comparison
    backbone_path = OUTPUT_ROOT / "rd_b3_backbone_candidate_comparison.csv"
    if backbone_candidates:
        fieldnames = list(backbone_candidates[0].keys())
        with open(backbone_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(backbone_candidates)
    print(f"  rd_b3_backbone_candidate_comparison.csv 생성")

    # architecture design table
    arch_path = OUTPUT_ROOT / "rd_b3_architecture_design_table.csv"
    if arch_table:
        fieldnames = list(arch_table[0].keys())
        with open(arch_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(arch_table)
    print(f"  rd_b3_architecture_design_table.csv 생성")

    # training sampler design
    sampler_path = OUTPUT_ROOT / "rd_b3_training_sampler_design.csv"
    if sampler_design:
        fieldnames = list(sampler_design[0].keys())
        with open(sampler_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sampler_design)
    print(f"  rd_b3_training_sampler_design.csv 생성")

    # score calibration design
    calib_path = OUTPUT_ROOT / "rd_b3_score_calibration_design.csv"
    if calibration_design:
        fieldnames = list(calibration_design[0].keys())
        with open(calib_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(calibration_design)
    print(f"  rd_b3_score_calibration_design.csv 생성")

    # input pipeline design
    input_path = OUTPUT_ROOT / "rd_b3_input_pipeline_design.csv"
    if input_pipeline:
        fieldnames = list(input_pipeline[0].keys())
        with open(input_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(input_pipeline)
    print(f"  rd_b3_input_pipeline_design.csv 생성")

    # summary JSON
    summary_path = OUTPUT_ROOT / "rd_b3_true_rd4ad_architecture_preflight_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  rd_b3_true_rd4ad_architecture_preflight_summary.json 생성")

    # report.md
    report_path = OUTPUT_ROOT / "rd_b3_true_rd4ad_architecture_preflight_report.md"
    _write_report(report_path, summary, backbone_candidates, recommended)
    print(f"  rd_b3_true_rd4ad_architecture_preflight_report.md 생성")

    # DONE
    done_path = OUTPUT_ROOT / "DONE"
    done_path.write_text("rd_b3_true_rd4ad_architecture_preflight_v1 DONE\n")
    print(f"  DONE 파일 생성")

    print(f"\n{'='*70}")
    print(f"RD-B3 Architecture Preflight 완료")
    print(f"판정: {'통과' if len(errors) == 0 else '부분 통과 (' + str(len(errors)) + ' 오류)'}")
    print(f"추천 backbone: {recommended['backbone']}")
    print(f"출력: {OUTPUT_ROOT}")
    print(f"소요: {elapsed}초")
    print(f"{'='*70}")


def _write_report(path, summary, backbone_candidates, recommended):
    lines = []
    lines.append("# RD-B3 True RD4AD Architecture Preflight Report\n")
    lines.append(f"- 버전: rd_b3_v1\n")
    lines.append(f"- 날짜: 2026-06-07\n")
    lines.append(f"- 판정: {'통과' if summary['integrity_passed'] else '부분 통과'}\n")
    lines.append("\n---\n")

    lines.append("## 1. 기존 v1/v1 ConvAE — true RD4AD 아님\n\n")
    lines.append(
        "기존 `v1/v1 branch`는 **ConvAutoencoder(ConvAE) reconstruction baseline**이었다.\n"
    )
    lines.append(
        "- Architecture: encoder → bottleneck → decoder (단일 네트워크)\n"
    )
    lines.append(
        "- Loss: pixel-level reconstruction loss (MSE/L1)\n"
    )
    lines.append(
        "- Anomaly score: pixel reconstruction error\n"
    )
    lines.append(
        "- Teacher network: 없음\n"
    )
    lines.append(
        "- 결론: **true RD4AD / reverse distillation 구조가 아니었음.** "
        "이번 RD-B3에서 새로 설계한다.\n"
    )
    lines.append("\n---\n")

    lines.append("## 2. 새 RD4AD Teacher-Student 구조 정의\n\n")
    lines.append(
        "**Reverse Distillation for Anomaly Detection (RD4AD)**:\n"
        "정상 데이터만으로 teacher의 multi-scale feature를 student가 복원하도록 학습.\n"
        "이상 데이터에서는 student가 teacher feature를 복원하지 못해 거리가 커짐.\n\n"
    )
    lines.append(
        "```\n"
        "입력 (mixed_3ch 96×96)\n"
        "   ↓\n"
        "Teacher (ResNet18, frozen, eval)\n"
        "   → layer1 feature  (64×24×24)\n"
        "   → layer2 feature  (128×12×12)\n"
        "   → layer3 feature  (256×6×6)  ← OCBE bottleneck 입력\n"
        "                          ↓\n"
        "            OCBE Bottleneck (compact code)\n"
        "                          ↓\n"
        "Student Decoder (trainable, reverse distillation)\n"
        "   → de_layer3 (256×6×6)\n"
        "   → de_layer2 (128×12×12)\n"
        "   → de_layer1 (64×24×24)\n"
        "                          ↓\n"
        "Loss: Σ (1 - cos_sim(teacher_layerN, student_de_layerN))\n"
        "Anomaly Score: multi-scale 거리 합산 → crop-level score\n"
        "```\n\n"
    )
    lines.append("\n---\n")

    lines.append("## 3. Teacher / Student / Feature Matching / Anomaly Score\n\n")
    lines.append(
        "### Teacher\n"
        "- ResNet18 (ImageNet pretrained)\n"
        "- `requires_grad = False` (완전 frozen)\n"
        "- `model.eval()` 고정\n"
        "- feature hook: layer1, layer2, layer3 출력 추출\n\n"
    )
    lines.append(
        "### Bottleneck (OCBE)\n"
        "- teacher layer3 feature (256×6×6) 입력\n"
        "- 1~3개 conv block으로 compact code 생성\n"
        "- student decoder의 입력으로 전달\n\n"
    )
    lines.append(
        "### Student Decoder\n"
        "- OCBE 출력부터 역순으로 upsample\n"
        "- de_layer3 (256×6×6) → de_layer2 (128×12×12) → de_layer1 (64×24×24)\n"
        "- 각 단계에서 teacher feature와 동일 resolution/channel 복원 목표\n\n"
    )
    lines.append(
        "### Feature Matching Loss\n"
        "```python\n"
        "loss = sum(1 - F.cosine_similarity(t_feat, s_feat, dim=1).mean()\n"
        "           for t_feat, s_feat in zip(teacher_feats, student_feats))\n"
        "```\n\n"
    )
    lines.append(
        "### Anomaly Score\n"
        "- test 시 teacher-student cosine distance 계산\n"
        "- 각 scale별 거리 맵 → 합산 → crop-level score\n"
        "- optional: 각 scale 거리 맵 upsample → patch heatmap\n\n"
    )
    lines.append("\n---\n")

    lines.append("## 4. Mixed_3ch 입력 설계 (RD-B2b 결정)\n\n")
    lines.append(
        "| 채널 | 내용 |\n"
        "|------|------|\n"
        "| ch1  | CT center slice (원본 HU 보존) |\n"
        "| ch2  | lower 3mm MIP (z-1~-N slice max projection) |\n"
        "| ch3  | upper 3mm MIP (z+1~+N slice max projection) |\n\n"
        "- crop size: 96×96\n"
        "- 3mm MIP 두께: CT spacing 기준 계산 (보통 z±1~±2 slice)\n\n"
    )
    lines.append("\n---\n")

    lines.append("## 5. HU [-1000, 600] 정규화 설계 (new_RD_style)\n\n")
    lines.append(
        "```\n"
        "normalized = (HU.clip(-1000, 600) + 1000) / 1600  # → [0, 1]\n"
        "```\n"
        "- lower_boundary z<=10 케이스에서 횡격막 과포화 모니터링 필요 (RD-B2b 확인)\n\n"
    )
    lines.append("\n---\n")

    lines.append("## 6. 6-bin Balanced Sampler 설계\n\n")
    lines.append(
        "```\n"
        "6-bin = upper_boundary, upper_interior,\n"
        "        middle_boundary, middle_interior,\n"
        "        lower_boundary, lower_interior\n"
        "```\n\n"
        "- batch_size=24 → bin당 4개\n"
        "- batch_size=48 → bin당 8개\n"
        "- shortage 시 duplicate oversampling 금지 → 다른 환자/bin 보완\n"
        "- patient leakage 방지\n"
        "- train/val/test split 유지\n\n"
    )
    lines.append("\n---\n")

    lines.append("## 7. Backbone 후보 비교 (96×96 입력)\n\n")
    lines.append(
        "| Backbone | Params | layer1 | layer2 | layer3 | GPU(B32) | RD4AD fit | 추천 |\n"
        "|----------|--------|--------|--------|--------|----------|-----------|------|\n"
    )
    for c in backbone_candidates:
        lines.append(
            f"| {c['backbone']} | {c['params_M']}M | "
            f"{c['layer1_shape']} | {c['layer2_shape']} | {c['layer3_shape']} | "
            f"{c['gpu_mem_est_GB']} | {c['rd4ad_fit']} | {c['recommendation']} |\n"
        )
    lines.append("\n---\n")

    lines.append("## 8. 추천 Backbone\n\n")
    lines.append(f"**{recommended['backbone']}**\n\n")
    lines.append(f"- {recommended['note']}\n")
    lines.append(
        f"- feature layers: {recommended['feature_layers_used']}\n"
        f"- GPU mem 예상: {recommended['gpu_mem_est_GB']}\n"
        f"- PaDiM 호환성: {recommended['padim_compat']}\n\n"
    )
    lines.append("\n---\n")

    lines.append("## 9. 예상 메모리/시간 위험\n\n")
    lines.append(
        "| 항목 | 예상 | 위험 |\n"
        "|------|------|------|\n"
        "| Teacher (ResNet18) GPU mem | ~0.5 GB (frozen) | 낮음 |\n"
        "| Student Decoder GPU mem | ~1.0 GB (batch32) | 낮음 |\n"
        "| 전체 학습 GPU mem | ~1.5 GB (batch32) | 낮음 |\n"
        "| lower_boundary shortage | bin당 충분히 없을 수 있음 | 중간 |\n"
        "| lower_z_boundary 횡격막 | 정규화 포화 | 낮음 (모니터링) |\n\n"
    )
    lines.append("\n---\n")

    lines.append("## 10. 다음 단계\n\n")
    for ns in summary["next_steps"]:
        lines.append(f"- {ns}\n")
    lines.append("\n---\n")

    lines.append("## 11. 절대 하지 않은 것\n\n")
    for nd in summary["absolute_not_done"]:
        lines.append(f"- {nd}\n")

    path.write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
