"""
P-A53: ResNet18 random224 selected_feature_indices generation
- 기존 100개 index를 포함하는 superset random224
- remaining 348개 후보 중 seed=42로 124개 추가
- training/scoring/model forward 금지
"""
import numpy as np
import json
import hashlib
import os
from pathlib import Path

EXISTING_IDX_PATH = "outputs/position-aware-padim-v1/models/padim_v1/distributions/selected_feature_indices.npy"
OUTPUT_IDX_PATH = "experiments/resnet18_imagenet_rand224_v1/outputs/models/distributions/selected_feature_indices.npy"
REPORT_MD_PATH = "experiments/resnet18_imagenet_rand224_v1/outputs/reports/p_a53_selected_indices.md"
REPORT_JSON_PATH = "experiments/resnet18_imagenet_rand224_v1/outputs/reports/p_a53_selected_indices.json"

RAW_FEATURE_DIM = 448
TARGET_COUNT = 224
SEED = 42

# 1. 기존 index read-only 로드
existing_idx = np.load(EXISTING_IDX_PATH)
existing_mtime_before = os.path.getmtime(EXISTING_IDX_PATH)

# 2. raw feature dim 확인
assert RAW_FEATURE_DIM == 448, f"raw_feature_dim 불일치: {RAW_FEATURE_DIM}"

# 3. 기존 100개 범위 확인
assert existing_idx.min() >= 0, f"min 범위 초과: {existing_idx.min()}"
assert existing_idx.max() <= RAW_FEATURE_DIM - 1, f"max 범위 초과: {existing_idx.max()}"

# 4. unique=100 확인
assert len(np.unique(existing_idx)) == 100, f"unique 수 불일치: {len(np.unique(existing_idx))}"

# 5. remaining index 계산
all_indices = set(range(RAW_FEATURE_DIM))
existing_set = set(existing_idx.tolist())
remaining = sorted(all_indices - existing_set)
assert len(remaining) == 348, f"remaining 수 불일치: {len(remaining)}"

# 6. seed=42로 124개 random choice
rng = np.random.default_rng(SEED)
added_indices = rng.choice(remaining, size=124, replace=False)

# 7. 합산 후 정렬
combined = np.array(sorted(existing_set | set(added_indices.tolist())), dtype=np.int64)

# 8. 검증
assert combined.shape == (TARGET_COUNT,), f"shape 불일치: {combined.shape}"
assert len(np.unique(combined)) == TARGET_COUNT, f"unique 불일치: {len(np.unique(combined))}"
assert combined.min() >= 0 and combined.max() <= RAW_FEATURE_DIM - 1
assert existing_set.issubset(set(combined.tolist())), "기존 100개 미포함"
assert len(set(combined.tolist()) - existing_set) == 124

# 재현성 확인
rng2 = np.random.default_rng(SEED)
added_check = rng2.choice(remaining, size=124, replace=False)
combined_check = np.array(sorted(existing_set | set(added_check.tolist())), dtype=np.int64)
assert np.array_equal(combined, combined_check), "seed=42 재현성 실패"

# layer별 분포
l1_new = int((combined < 64).sum())
l2_new = int(((combined >= 64) & (combined < 192)).sum())
l3_new = int((combined >= 192).sum())

l1_old = int((existing_idx < 64).sum())
l2_old = int(((existing_idx >= 64) & (existing_idx < 192)).sum())
l3_old = int((existing_idx >= 192).sum())

# 기존 index sha256
with open(EXISTING_IDX_PATH, "rb") as f:
    existing_sha256 = hashlib.sha256(f.read()).hexdigest()

# 저장
np.save(OUTPUT_IDX_PATH, combined)

# 저장 후 기존 mtime 변경 없음 확인
existing_mtime_after = os.path.getmtime(EXISTING_IDX_PATH)
assert existing_mtime_before == existing_mtime_after, "기존 index mtime 변경됨"

# 새 index sha256
with open(OUTPUT_IDX_PATH, "rb") as f:
    new_sha256 = hashlib.sha256(f.read()).hexdigest()

print("=== P-A53 selected_feature_indices generation ===")
print(f"판정: 통과")
print(f"기존 index 경로: {EXISTING_IDX_PATH}")
print(f"새 index 경로:   {OUTPUT_IDX_PATH}")
print(f"기존 shape: {existing_idx.shape}, min={existing_idx.min()}, max={existing_idx.max()}, unique={len(np.unique(existing_idx))}")
print(f"새 shape:   {combined.shape}, min={combined.min()}, max={combined.max()}, unique={len(np.unique(combined))}")
print(f"기존 100개 포함: {existing_set.issubset(set(combined.tolist()))}")
print(f"추가 124개: {len(set(combined.tolist()) - existing_set)}")
print(f"retention: {existing_idx.shape[0]}/{RAW_FEATURE_DIM}={existing_idx.shape[0]/RAW_FEATURE_DIM*100:.1f}% → {TARGET_COUNT}/{RAW_FEATURE_DIM}={TARGET_COUNT/RAW_FEATURE_DIM*100:.1f}%")
print(f"layer 분포 (기존): layer1={l1_old}, layer2={l2_old}, layer3={l3_old}")
print(f"layer 분포 (신규): layer1={l1_new}, layer2={l2_new}, layer3={l3_new}")
print(f"기존 index sha256: {existing_sha256}")
print(f"새 index sha256:   {new_sha256}")
print(f"seed=42 재현성: True")
print(f"기존 index mtime 변경: False")

# 보고서 생성
report = {
    "step": "P-A53",
    "verdict": "통과",
    "existing_index": {
        "path": EXISTING_IDX_PATH,
        "shape": list(existing_idx.shape),
        "dtype": str(existing_idx.dtype),
        "min": int(existing_idx.min()),
        "max": int(existing_idx.max()),
        "unique": int(len(np.unique(existing_idx))),
        "sha256": existing_sha256,
        "mtime_changed": False,
        "layer1_count": l1_old,
        "layer2_count": l2_old,
        "layer3_count": l3_old,
    },
    "new_index": {
        "path": OUTPUT_IDX_PATH,
        "shape": list(combined.shape),
        "dtype": str(combined.dtype),
        "min": int(combined.min()),
        "max": int(combined.max()),
        "unique": int(len(np.unique(combined))),
        "sha256": new_sha256,
        "layer1_count": l1_new,
        "layer2_count": l2_new,
        "layer3_count": l3_new,
    },
    "generation": {
        "method": "superset",
        "existing_100_included": True,
        "added_count": 124,
        "remaining_pool": 348,
        "seed": SEED,
        "reproducible": True,
    },
    "retention": {
        "before": f"{existing_idx.shape[0]}/{RAW_FEATURE_DIM}={existing_idx.shape[0]/RAW_FEATURE_DIM*100:.1f}%",
        "after": f"{TARGET_COUNT}/{RAW_FEATURE_DIM}={TARGET_COUNT/RAW_FEATURE_DIM*100:.1f}%",
    },
    "safety": {
        "existing_index_modified": False,
        "existing_results_modified": False,
        "training_executed": False,
        "scoring_executed": False,
        "model_forward_executed": False,
        "threshold_calculated": False,
        "metrics_calculated": False,
        "stage2_holdout_accessed": False,
        "quarantine_accessed": False,
        "pip_install": False,
        "download": False,
    },
    "next_step_p_a54_feasible": True,
}

with open(REPORT_JSON_PATH, "w") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"\n보고서 JSON: {REPORT_JSON_PATH}")
print("완료.")
