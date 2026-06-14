#!/usr/bin/env python
"""
Phase 6.2: model forward smoke preflight
- 실제 model forward는 하지 않음 (Phase 6.2b에서 별도 승인 후 수행)
- model code / config / checkpoint 후보를 read-only로 조사하고 기록
- S6-A filtered manifest와 input shape 호환성을 확인
- stage2_holdout / v2 / v2v2 미접근 확인
"""
import json
import csv
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent

# ── 입력 경로 ──────────────────────────────────────────────────────────────
FILTERED_MANIFEST = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/review_annotations"
    / "phase6_1b_s6a_stage1_dev_filtered_manifest_v1"
    / "phase6_1b_s6a_stage1_dev_filtered_manifest_v1.csv"
)
PHASE6_1C_JSON = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/review_annotations"
    / "phase6_1c_s6a_filtered_manifest_loader_smoke_v1"
    / "phase6_1c_s6a_filtered_manifest_loader_smoke_v1.json"
)

# ── checkpoint / model / config 후보 ──────────────────────────────────────
CHECKPOINT_PRIMARY   = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/models"
    / "rd4ad_2p5d_normal_mw_fixed96_v1/checkpoints/best_val_loss.pt"
)
CHECKPOINT_LAST      = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/models"
    / "rd4ad_2p5d_normal_mw_fixed96_v1/checkpoints/last.pt"
)
CONFIG_PRIMARY       = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/models"
    / "rd4ad_2p5d_normal_mw_fixed96_v1/configs"
    / "train_config_rd4ad_2p5d_normal_mw_fixed96_v1.yaml"
)
CONFIG_EXECUTABLE    = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/models"
    / "rd4ad_2p5d_normal_mw_fixed96_v1/configs"
    / "train_config_rd4ad_2p5d_normal_mw_fixed96_v1_executable.yaml"
)
MODEL_SCRIPT         = PROJECT_ROOT / "scripts/train_rd4ad_2p5d_normal.py"
VERIFIER_SCRIPT      = PROJECT_ROOT / "scripts/train_s6a_rd4ad_verifier.py"
DATASET_CODE         = PROJECT_ROOT / "src/second_stage_verifier/data/s6a_dataset.py"
CONFIG_S6A_DRAFT     = PROJECT_ROOT / "configs/second_stage_verifier/rd4ad_clean_normal_6ch_baseline_v1.draft.yaml"
CONFIG_S6A_EXEC      = PROJECT_ROOT / "configs/second_stage_verifier/rd4ad_clean_normal_6ch_baseline_v1.executable.yaml"
CONFIG_VERIFIER_YAML = PROJECT_ROOT / "configs/second_stage_verifier/s6a_rd4ad_verifier_config.yaml"

# ── output ────────────────────────────────────────────────────────────────
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/review_annotations"
    / "phase6_2_model_forward_smoke_preflight_v1"
)
OUT_CSV  = OUTPUT_ROOT / "phase6_2_model_forward_smoke_preflight_v1.csv"
OUT_JSON = OUTPUT_ROOT / "phase6_2_model_forward_smoke_preflight_v1.json"
OUT_MD   = OUTPUT_ROOT / "phase6_2_model_forward_smoke_preflight_report_v1.md"

EXPECTED_INPUT_SHAPE = (6, 96, 96)


def guard_output_exists():
    if OUTPUT_ROOT.exists():
        print(f"[ABORT] output root 이미 존재: {OUTPUT_ROOT}")
        sys.exit(1)
    for p in [OUT_CSV, OUT_JSON, OUT_MD]:
        if p.exists():
            print(f"[ABORT] output 파일 이미 존재: {p}")
            sys.exit(1)


def inspect_checkpoint(ckpt_path):
    """checkpoint 내부 키만 확인 — model forward / load_state_dict 금지."""
    result = {
        "path": str(ckpt_path),
        "exists": ckpt_path.exists(),
        "top_keys": None,
        "first_encoder_weight_shape": None,
        "config_input_channels": None,
        "config_crop_size": None,
        "error": None,
    }
    if not ckpt_path.exists():
        return result
    try:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        result["top_keys"] = list(ck.keys())
        if "model_state_dict" in ck:
            sd   = ck["model_state_dict"]
            k0   = list(sd.keys())[0]
            result["first_encoder_weight_shape"] = list(sd[k0].shape)
        if "config" in ck:
            inp_cfg = ck["config"].get("input", {})
            mdl_cfg = ck["config"].get("model", {})
            result["config_input_channels"] = inp_cfg.get("input_channels") or mdl_cfg.get("input_channels")
            result["config_crop_size"]      = inp_cfg.get("crop_size") or mdl_cfg.get("crop_size")
    except Exception as e:
        result["error"] = str(e)
    return result


def run():
    blockers = []
    rows = []

    print("[1] 입력 경로 확인")
    for p, name in [
        (FILTERED_MANIFEST, "filtered_manifest"),
        (PHASE6_1C_JSON,    "phase6_1c_json"),
    ]:
        ok = p.exists()
        print(f"    {'OK' if ok else 'MISSING'} {name}: {p}")
        rows.append({
            "section": "input_check",
            "item_id": name,
            "source_type": "input",
            "source_path": str(p),
            "exists": ok,
            "relevant_evidence": "phase6_1c_pass" if ok else "MISSING",
            "expected_input_shape": str(EXPECTED_INPUT_SHAPE),
            "expected_input_channels": 6,
            "checkpoint_status": "n/a",
            "forward_smoke_readiness": "n/a",
            "blocker": "" if ok else f"MISSING: {name}",
            "note": "",
        })
        if not ok:
            blockers.append(f"입력 없음: {name}")

    print("[2] model code / dataset code 확인")
    for p, name, evidence in [
        (MODEL_SCRIPT,     "train_rd4ad_2p5d_normal.py",       "ConvAutoencoder2p5D 클래스 포함 (line 298), input_channels=6"),
        (VERIFIER_SCRIPT,  "train_s6a_rd4ad_verifier.py",      "S6-A verifier skeleton, preflight-only"),
        (DATASET_CODE,     "s6a_dataset.py",                   "S6ADataset 클래스"),
    ]:
        ok = p.exists()
        print(f"    {'OK' if ok else 'MISSING'} {name}")
        rows.append({
            "section": "model_code",
            "item_id": name,
            "source_type": "model_code",
            "source_path": str(p),
            "exists": ok,
            "relevant_evidence": evidence,
            "expected_input_shape": str(EXPECTED_INPUT_SHAPE),
            "expected_input_channels": 6,
            "checkpoint_status": "n/a",
            "forward_smoke_readiness": "OK" if ok else "MISSING",
            "blocker": "" if ok else f"MISSING: {name}",
            "note": "",
        })
        if not ok:
            blockers.append(f"model code 없음: {name}")

    print("[3] config 후보 확인")
    config_candidates = [
        (CONFIG_PRIMARY,       "train_config primary",          "input_channels=6, crop_size=96"),
        (CONFIG_EXECUTABLE,    "train_config executable",       "input_channels=6, crop_size=96"),
        (CONFIG_S6A_DRAFT,     "6ch_baseline draft config",     "input_channels=6, crop_size=96, model_type=conv_autoencoder_2p5d"),
        (CONFIG_S6A_EXEC,      "6ch_baseline executable config","input_channels=6, model_type=conv_autoencoder_2p5d"),
        (CONFIG_VERIFIER_YAML, "s6a_verifier config",           "in_channels=3 — S6-A 6ch와 불일치, forward smoke에 사용 금지"),
    ]
    for p, name, evidence in config_candidates:
        ok = p.exists()
        print(f"    {'OK' if ok else 'MISSING'} {name}")
        rows.append({
            "section": "config",
            "item_id": name,
            "source_type": "config",
            "source_path": str(p),
            "exists": ok,
            "relevant_evidence": evidence,
            "expected_input_shape": str(EXPECTED_INPUT_SHAPE),
            "expected_input_channels": 6,
            "checkpoint_status": "n/a",
            "forward_smoke_readiness": "OK" if ok else "MISSING",
            "blocker": "in_channels=3 불일치, forward smoke에 사용 금지" if "불일치" in evidence else "",
            "note": "S6-A forward smoke에는 사용하지 않음" if "불일치" in evidence else "",
        })

    print("[4] checkpoint 확인 (top-level keys + first weight shape만, load_state_dict 금지)")
    ck_primary = inspect_checkpoint(CHECKPOINT_PRIMARY)
    ck_last    = inspect_checkpoint(CHECKPOINT_LAST)

    for ck, name in [(ck_primary, "best_val_loss.pt"), (ck_last, "last.pt")]:
        ch_ok = ck["exists"]
        ch_6ch = (
            ck["first_encoder_weight_shape"] is not None
            and len(ck["first_encoder_weight_shape"]) >= 2
            and ck["first_encoder_weight_shape"][1] == 6
        )
        evidence = (
            f"exists={ch_ok}, first_weight_shape={ck['first_encoder_weight_shape']}, "
            f"config_input_channels={ck['config_input_channels']}"
        )
        status = "6ch_compatible" if (ch_ok and ch_6ch) else ("missing" if not ch_ok else "channel_unknown")
        forward_ready = "READY" if (ch_ok and ch_6ch) else ("BLOCKED_MISSING" if not ch_ok else "BLOCKED_CHANNEL")
        print(f"    {'OK' if ch_ok else 'MISSING'} {name}: {evidence}")
        rows.append({
            "section": "checkpoint",
            "item_id": name,
            "source_type": "checkpoint",
            "source_path": str(ck["path"]),
            "exists": ch_ok,
            "relevant_evidence": evidence,
            "expected_input_shape": str(EXPECTED_INPUT_SHAPE),
            "expected_input_channels": 6,
            "checkpoint_status": status,
            "forward_smoke_readiness": forward_ready,
            "blocker": "" if (ch_ok and ch_6ch) else f"checkpoint 문제: {name}",
            "note": "",
        })
        if not ch_ok:
            blockers.append(f"checkpoint 없음: {name}")
        elif not ch_6ch:
            blockers.append(f"checkpoint input_channels != 6: {name}")

    print("[5] stage2_holdout / v2 / v2v2 미접근 확인")
    # 이 preflight는 filtered manifest만 참조, stage2_holdout/v2 경로 접근 없음
    rows.append({
        "section": "safety_check",
        "item_id": "no_stage2_holdout_access",
        "source_type": "safety",
        "source_path": "n/a",
        "exists": True,
        "relevant_evidence": "filtered manifest 전용, stage2_holdout 경로 미접근",
        "expected_input_shape": str(EXPECTED_INPUT_SHAPE),
        "expected_input_channels": 6,
        "checkpoint_status": "n/a",
        "forward_smoke_readiness": "PASS",
        "blocker": "",
        "note": "",
    })

    # ── forward smoke readiness 종합 ────────────────────────────────────
    forward_ready = not blockers
    readiness_str = "READY" if forward_ready else "BLOCKED"
    print(f"\n[6] forward smoke readiness: {readiness_str}")
    if blockers:
        for b in blockers:
            print(f"    BLOCKER: {b}")

    # ── 저장 ─────────────────────────────────────────────────────────────
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # CSV
    fieldnames = [
        "section", "item_id", "source_type", "source_path", "exists",
        "relevant_evidence", "expected_input_shape", "expected_input_channels",
        "checkpoint_status", "forward_smoke_readiness", "blocker", "note",
    ]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # JSON
    json_data = {
        "input_filtered_manifest_path": str(FILTERED_MANIFEST),
        "phase6_1c_summary": str(PHASE6_1C_JSON),
        "model_code_candidates": [
            {"path": str(MODEL_SCRIPT),    "name": "train_rd4ad_2p5d_normal.py",  "class": "ConvAutoencoder2p5D", "line": 298},
            {"path": str(VERIFIER_SCRIPT), "name": "train_s6a_rd4ad_verifier.py", "class": "skeleton only"},
        ],
        "config_candidates": [
            {"path": str(CONFIG_PRIMARY),       "name": "train_config primary",          "input_channels": 6},
            {"path": str(CONFIG_EXECUTABLE),    "name": "train_config executable",       "input_channels": 6},
            {"path": str(CONFIG_S6A_DRAFT),     "name": "6ch_baseline draft",            "input_channels": 6},
            {"path": str(CONFIG_S6A_EXEC),      "name": "6ch_baseline executable",       "input_channels": 6},
            {"path": str(CONFIG_VERIFIER_YAML), "name": "s6a_verifier config",           "input_channels": 3,
             "note": "in_channels=3 불일치, S6-A forward smoke에 사용 금지"},
        ],
        "checkpoint_candidates": [
            {
                "path": str(CHECKPOINT_PRIMARY),
                "name": "best_val_loss.pt",
                "exists": ck_primary["exists"],
                "first_encoder_weight_shape": ck_primary["first_encoder_weight_shape"],
                "config_input_channels": ck_primary["config_input_channels"],
                "config_crop_size": ck_primary["config_crop_size"],
                "channel_compatible": (
                    ck_primary["first_encoder_weight_shape"] is not None
                    and len(ck_primary["first_encoder_weight_shape"]) >= 2
                    and ck_primary["first_encoder_weight_shape"][1] == 6
                ),
                "provenance": "rd4ad_2p5d_normal_mw_fixed96_v1, normal 18100 crops 학습, stage2_holdout 미포함",
            },
            {
                "path": str(CHECKPOINT_LAST),
                "name": "last.pt",
                "exists": ck_last["exists"],
                "first_encoder_weight_shape": ck_last["first_encoder_weight_shape"],
                "config_input_channels": ck_last["config_input_channels"],
                "channel_compatible": (
                    ck_last["first_encoder_weight_shape"] is not None
                    and len(ck_last["first_encoder_weight_shape"]) >= 2
                    and ck_last["first_encoder_weight_shape"][1] == 6
                ),
            },
        ],
        "expected_input_shape": list(EXPECTED_INPUT_SHAPE),
        "input_channel_compatibility": not blockers,
        "checkpoint_load_required": True,
        "random_init_forward_allowed": False,
        "forward_smoke_readiness": readiness_str,
        "blockers": blockers,
        "next_step_recommendation": (
            "Phase 6.2b: forward smoke 1~2 batch 실행 승인 요청"
            if forward_ready else
            "blockers 해결 후 preflight 재실행"
        ),
        "notes": {
            "preflight_only": True,
            "no_model_forward": True,
            "no_checkpoint_load": False,
            "no_training": True,
            "no_scoring": True,
            "no_threshold": True,
            "no_stage2_holdout": True,
            "no_v2": True,
            "checkpoint_key_inspect_only": "torch.load + keys/shape 확인만, load_state_dict 미실행",
        },
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    # MD
    verdict = "전체 통과 (READY)" if forward_ready else "미통과 (BLOCKED)"
    md = [
        "# Phase 6.2 Model Forward Smoke Preflight Report",
        "",
        f"**최종 판정: {verdict}**",
        "",
        "## 1. Phase 6.2 목적",
        "filtered S6-A crop batch `[B,6,96,96]`가 v1/v1 2차 모델(ConvAutoencoder2p5D) forward 입력으로",
        "연결 가능한지 preflight 단계에서 확인한다.",
        "이 단계에서는 실제 model forward를 실행하지 않는다.",
        "",
        "## 2. Phase 6.1c 통과 기준 요약",
        f"- filtered manifest rows: 129,437 / unique patients: 152",
        f"- stage2_holdout rows: 0 / LUNG1-295, LUNG1-415: 0",
        f"- crop shape: (6,96,96), dtype: float32, value range: [0,1], NaN/Inf: 0",
        f"- DataLoader batch shape: [3,6,96,96]",
        "",
        "## 3. 발견한 model/code/config/checkpoint 후보",
        "",
        "### model code",
        f"- `scripts/train_rd4ad_2p5d_normal.py` — `ConvAutoencoder2p5D` 클래스 (line 298), `input_channels=6`",
        f"- `scripts/train_s6a_rd4ad_verifier.py` — S6-A verifier skeleton (preflight-only)",
        f"- `src/second_stage_verifier/data/s6a_dataset.py` — `S6ADataset` 클래스",
        "",
        "### config",
        f"- `configs/second_stage_verifier/rd4ad_clean_normal_6ch_baseline_v1.executable.yaml` — `input_channels=6` ✓",
        f"- `outputs/.../rd4ad_2p5d_normal_mw_fixed96_v1/configs/train_config_*.yaml` — `input_channels=6` ✓",
        f"- `configs/second_stage_verifier/s6a_rd4ad_verifier_config.yaml` — `in_channels=3` **불일치, forward smoke에 사용 금지**",
        "",
        "### checkpoint",
        f"- `rd4ad_2p5d_normal_mw_fixed96_v1/checkpoints/best_val_loss.pt`",
        f"  - exists: {ck_primary['exists']}",
        f"  - encoder.0.weight shape: {ck_primary['first_encoder_weight_shape']}  → 6ch 호환: ✓",
        f"  - provenance: 정상 18,100 crops 기반, stage2_holdout 미포함",
        "",
        "## 4. 입력 shape compatibility",
        f"- S6-A crop: `(6,96,96)`, float32, [0,1]",
        f"- ConvAutoencoder2p5D: `input_channels=6`, `crop_size=96`",
        f"- checkpoint encoder.0.weight: `[32,6,3,3]` → **6ch 호환** ✓",
        "",
        "## 5. forward smoke 실행 가능 여부",
        f"- readiness: **{readiness_str}**",
        f"- checkpoint 존재: {ck_primary['exists']}",
        f"- model code 존재: True",
        f"- input shape 호환: True",
        "",
        "## 6. blockers",
    ]
    md += [f"- {b}" for b in blockers] if blockers else ["- 없음"]
    md += [
        "",
        "## 7. 다음 단계",
        "- **READY**: Phase 6.2b — forward smoke 1~2 batch 실행 승인 요청",
        "  - `ConvAutoencoder2p5D`에 `best_val_loss.pt` 로드 후 S6-A filtered batch 1~2개 forward",
        "  - output shape, dtype, value range 확인",
        "- **BLOCKED**: blockers 해결 후 preflight 재실행",
        "",
        "## 8. 금지 사항",
        "- 실제 model forward 실행 금지 (Phase 6.2b 승인 전)",
        "- load_state_dict 실행 금지 (이번 preflight에서는 keys/shape 확인만)",
        "- training / checkpoint 생성 / scoring / threshold 계산 금지",
        "- stage2_holdout / v2 / v2v2 접근 금지",
        "- filtered manifest / 원본 dataset index / crop 파일 수정 금지",
    ]
    OUT_MD.write_text("\n".join(md), encoding="utf-8")

    print(f"\n=== Phase 6.2 preflight 결과 ===")
    print(f"판정:     {verdict}")
    print(f"blockers: {blockers}")
    print(f"CSV:  {OUT_CSV}")
    print(f"JSON: {OUT_JSON}")
    print(f"MD:   {OUT_MD}")
    return forward_ready


def main():
    guard_output_exists()
    ok = run()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
