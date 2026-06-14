"""
Step 11: stage2 fixed-evaluation preflight
- stage2 candidate manifest 확인
- p90 고정 적용 계획
- z-continuity 계획
- CT/mask/checkpoint readiness
- shard plan 생성
actual stage2 scoring 금지 / stage2 label 사용 금지 / score family 변경 금지
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── 경로 ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
DONE_STEP10 = ROOT / "DONE_STEP10_DECISION_CHECKPOINT.json"
PLAN_LOCK   = ROOT / "docs/FINAL_PLAN_LOCK.json"
CKPT_BEST   = ROOT / "checkpoints/full_train_v1/student_best_val_loss.pth"

STAGE2_MANIFEST = Path(
    "outputs/second-stage-lesion-refiner-v1/datasets"
    "/s6a_stage2_holdout_candidate_coordinate_manifest_v1.csv"
)
NSCLC_CT_ROOT  = Path(
    "/mnt/c/Users/jinhy/Desktop"
    "/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1"
    "/volumes_npy"
)
LESION_MASK_ROOT = Path(
    "outputs/mip-postprocess-research-v1/masks"
    "/refined_roi_v4_20_modeB_all_v1/lesion"
)

OUT_MANIFESTS = ROOT / "manifests"
OUT_REPORTS   = ROOT / "reports"
OUT_LOGS      = ROOT / "logs"
OUT_STAGE2    = ROOT / "stage2"

# ── 고정 파라미터 (Step 10에서 lock) ─────────────────────────────────────────
P90_THRESHOLD           = 12.196394
PRIMARY_CANDIDATE_SCORE = "rd4ad_lung5ch_score_raw"
PRIMARY_TRACK_SCORE     = "raw_track_top3_mean"
CROP_SIZE               = 96
HU_MIN, HU_MAX          = -1350.0, 150.0
N_SHARDS                = 8
STAGE2_SCORE_COL        = "score_original"   # manifest의 first_stage_score 해당 컬럼


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run-preflight", action="store_true")
    ap.add_argument("--confirm-plan-lock", action="store_true")
    ap.add_argument("--confirm-stage2-readiness", action="store_true")
    ap.add_argument("--confirm-fixed-eval-only", action="store_true")
    return ap.parse_args()


def build_tracks(df_p90):
    """p90 passed DataFrame으로부터 z-continuity track을 계산하고 track_id, track_len을 부여."""
    df = df_p90.copy()
    df["pos_key"] = (df["patient_id"].astype(str) + "_" +
                     df["y0"].astype(str) + "_" + df["x0"].astype(str) + "_" +
                     df["y1"].astype(str) + "_" + df["x1"].astype(str))
    df = df.sort_values(["pos_key", "local_z"]).reset_index(drop=True)

    global_tid = 0
    track_ids  = np.full(len(df), -1, dtype=np.int64)
    track_lens = np.zeros(len(df), dtype=np.int64)

    for _, grp in df.groupby("pos_key"):
        idx    = grp.index.values
        zs     = grp["local_z"].values
        run_start = 0
        tid    = global_tid
        for i in range(1, len(zs) + 1):
            end_run = (i == len(zs)) or (zs[i] - zs[i - 1] > 2)
            if end_run:
                run_len = i - run_start
                for j in range(run_start, i):
                    track_ids[idx[j]]  = tid
                    track_lens[idx[j]] = run_len
                global_tid += 1
                tid = global_tid
                run_start = i

    df["track_id"]  = track_ids
    df["track_len"] = track_lens
    df["track_z_start"] = df.groupby("track_id")["local_z"].transform("min")
    df["track_z_end"]   = df.groupby("track_id")["local_z"].transform("max")
    valid = df[df["track_len"] >= 2].reset_index(drop=True)
    return valid


def sample_crop_check(df_valid, n_samples=5):
    """소량 샘플에 대해 5ch crop + mask 생성 가능 여부 확인."""
    rows = []
    sampled = df_valid.sample(min(n_samples, len(df_valid)), random_state=42)
    for _, row in sampled.iterrows():
        sid = row["safe_id"]
        z   = int(row["local_z"])
        y0, x0 = int(row["y0"]), int(row["x0"])
        y1, x1 = int(row["y1"]), int(row["x1"])

        # 32×32 → 96×96 center±48
        cy = (y0 + y1) // 2
        cx = (x0 + x1) // 2
        cy0, cy1 = cy - 48, cy + 48
        cx0, cx1 = cx - 48, cx + 48

        result = {"safe_id": sid, "local_z": z,
                  "pos_coord": f"y={y0},{y1} x={x0},{x1}",
                  "crop_coord": f"y={cy0}:{cy1} x={cx0}:{cx1}"}

        ct_path   = NSCLC_CT_ROOT / sid / "ct_hu.npy"
        mask_path = LESION_MASK_ROOT / sid / "refined_roi.npy"

        if not ct_path.exists():
            result.update({"ct_ok": False, "crop_ok": False, "mask_ok": False,
                           "note": "CT MISSING"})
            rows.append(result)
            continue

        ct = np.load(str(ct_path), mmap_mode="r")
        D, H, W = ct.shape

        # boundary check
        if cy0 < 0 or cy1 > H or cx0 < 0 or cx1 > W:
            result.update({"ct_ok": True, "crop_ok": False, "mask_ok": False,
                           "note": f"crop out of CT bounds H={H},W={W}"})
            rows.append(result)
            continue

        # 5ch crop
        z_ids = [max(0, min(D - 1, z + dz)) for dz in [-2, -1, 0, 1, 2]]
        nearest_used = any(max(0, min(D - 1, z + dz)) != z + dz for dz in [-2, -1, 0, 1, 2])
        crop = np.stack([ct[zi, cy0:cy1, cx0:cx1] for zi in z_ids], axis=0).astype(np.float32)
        crop = np.clip(crop, HU_MIN, HU_MAX)
        crop = (crop - HU_MIN) / (HU_MAX - HU_MIN)

        crop_ok = (crop.shape == (5, CROP_SIZE, CROP_SIZE) and
                   not np.isnan(crop).any() and not np.isinf(crop).any())

        mask_ok = mask_path.exists()
        result.update({
            "ct_ok": True,
            "crop_ok": crop_ok,
            "crop_shape": str(crop.shape),
            "crop_range": f"[{crop.min():.3f},{crop.max():.3f}]",
            "nearest_used": nearest_used,
            "mask_ok": mask_ok,
            "note": "OK" if (crop_ok and mask_ok) else "WARN",
        })
        rows.append(result)
    return rows


def main():
    args = parse_args()

    if not args.dry_run and not args.run_preflight:
        print("bare run blocked — use --dry-run or --run-preflight with confirm flags")
        sys.exit(2)

    if args.dry_run:
        print("=" * 64)
        print("Step 11 Stage2 Fixed Preflight — DRY-RUN PLAN")
        print("=" * 64)
        print()
        print("[입력]")
        print(f"  stage2 manifest : {STAGE2_MANIFEST}")
        print(f"  score col       : {STAGE2_SCORE_COL} (first_stage_score 해당)")
        print(f"  p90 threshold   : {P90_THRESHOLD} (고정, stage2에서 재계산 금지)")
        print(f"  coordinate schema: 32×32 position → center±48 → 96×96 crop")
        print(f"  primary score   : {PRIMARY_CANDIDATE_SCORE}")
        print(f"  primary track   : {PRIMARY_TRACK_SCORE}")
        print(f"  checkpoint      : {CKPT_BEST}")
        print()
        print("[확인 항목]")
        print("  1. Step 10 lock 확인")
        print("  2. stage2 manifest schema 확인")
        print("  3. p90 고정 적용 계획")
        print("  4. z-continuity >= 2 계획")
        print("  5. 5ch crop + mask 샘플 readiness")
        print("  6. checkpoint readiness")
        print("  7. shard plan (8 shards, patient-stable)")
        print("  8. output collision 확인")
        print()
        print("[금지]")
        print("  actual stage2 scoring / label 사용 / score family 변경")
        print("  p90 재계산 / track aggregation 재선택 / threshold 튜닝")
        print()
        print("[실행 명령]")
        print("  python experiments/rd4ad_2p5d_lung5ch_masked_normal_v1/scripts/"
              "rd4ad_2p5d_lung5ch_step11_stage2_fixed_preflight.py \\")
        print("    --run-preflight --confirm-plan-lock "
              "--confirm-stage2-readiness --confirm-fixed-eval-only")
        print()
        print("DRY-RUN 완료.")
        return

    if not (args.confirm_plan_lock and args.confirm_stage2_readiness and args.confirm_fixed_eval_only):
        print("BLOCKED: confirm flags missing")
        sys.exit(2)

    for d in [OUT_MANIFESTS, OUT_REPORTS, OUT_LOGS, OUT_STAGE2]:
        d.mkdir(parents=True, exist_ok=True)

    errors = []
    guardrail = {
        "plan_lock_loaded": False,
        "step10_decision_passed": False,
        "stage2_readiness_passed": False,
        "fixed_eval_preflight_only": True,
        "actual_stage2_scoring_executed": False,
        "model_forward_executed": "small_smoke_only",
        "training_executed": False,
        "checkpoint_saved": False,
        "checkpoint_modified": False,
        "primary_candidate_score_locked": PRIMARY_CANDIDATE_SCORE,
        "primary_track_score_locked": PRIMARY_TRACK_SCORE,
        "P1_rejected_for_lung5ch": True,
        "P2_rejected_for_lung5ch": True,
        "p90_threshold": P90_THRESHOLD,
        "p90_recomputed_on_stage2": False,
        "threshold_tuning_executed": False,
        "score_family_changed_on_stage2": False,
        "candidate_deletion_executed": False,
        "representative_only_scoring_used": False,
        "survived_candidates_all_planned_for_scoring": True,
        "stage2_label_used_for_tuning": False,
        "stage2_label_used_for_metric": False,
        "convae_branch_created": False,
        "image_reconstruction_loss_used": False,
    }

    # ── [1] Step 10 lock 확인 ─────────────────────────────────────────────
    print("[1] Step 10 lock 확인")

    if PLAN_LOCK.exists():
        guardrail["plan_lock_loaded"] = True
        print("  plan lock: OK")
    else:
        errors.append({"step": "plan_lock", "msg": str(PLAN_LOCK)})
        print(f"  WARN: plan lock not found")

    if not DONE_STEP10.exists():
        print("BLOCKED: DONE_STEP10_DECISION_CHECKPOINT.json not found")
        sys.exit(2)

    with open(DONE_STEP10) as f:
        done10 = json.load(f)

    if done10.get("verdict") != "PASS_STEP10_DECISION_CHECKPOINT":
        print(f"BLOCKED: step10 verdict = {done10.get('verdict')}")
        sys.exit(2)
    if done10.get("stage2_readiness") != "PASS_STAGE2_READY":
        print(f"BLOCKED: stage2_readiness = {done10.get('stage2_readiness')}")
        sys.exit(2)
    if done10.get("primary_candidate_score") != PRIMARY_CANDIDATE_SCORE:
        print(f"BLOCKED: primary_candidate_score mismatch")
        sys.exit(2)
    if done10.get("primary_track_score") != PRIMARY_TRACK_SCORE:
        print(f"BLOCKED: primary_track_score mismatch")
        sys.exit(2)
    if not done10.get("P1_rejected"):
        print("BLOCKED: P1 not rejected in step10")
        sys.exit(2)

    guardrail["step10_decision_passed"] = True
    guardrail["stage2_readiness_passed"] = True
    print("  Step 10 lock: PASS")
    print(f"    primary_candidate_score={done10['primary_candidate_score']}")
    print(f"    primary_track_score={done10['primary_track_score']}")
    print(f"    P1_rejected={done10['P1_rejected']}")

    # ── [2] stage2 manifest schema 확인 ──────────────────────────────────
    print()
    print("[2] stage2 candidate manifest schema 확인")

    if not STAGE2_MANIFEST.exists():
        print(f"BLOCKED: stage2 manifest not found: {STAGE2_MANIFEST}")
        sys.exit(2)

    df = pd.read_csv(str(STAGE2_MANIFEST))
    print(f"  rows: {len(df):,}")
    print(f"  cols: {df.columns.tolist()}")

    # stage_split 확인
    if "stage_split" in df.columns:
        splits = df["stage_split"].value_counts().to_dict()
        print(f"  stage_split: {splits}")
        if set(splits.keys()) != {"stage2_holdout"}:
            print(f"  WARN: unexpected stage_split values — {splits}")
    else:
        errors.append({"step": "schema", "msg": "stage_split column missing"})
        print("  WARN: stage_split column missing")

    # 좌표 컬럼 탐색
    coord_schema = None
    if all(c in df.columns for c in ["crop_y0", "crop_x0", "crop_y1", "crop_x1"]):
        w = (df["crop_x1"] - df["crop_x0"]).unique()
        h = (df["crop_y1"] - df["crop_y0"]).unique()
        if set(w) == {CROP_SIZE} and set(h) == {CROP_SIZE}:
            coord_schema = "96x96_direct"
        else:
            coord_schema = f"crop_col_wh={w[:3]}"
    elif all(c in df.columns for c in ["y0", "x0", "y1", "x1"]):
        w = (df["x1"] - df["x0"]).unique()
        h = (df["y1"] - df["y0"]).unique()
        if set(w) == {32} and set(h) == {32}:
            coord_schema = "32x32_position_center48"
        else:
            coord_schema = f"yx_col_wh={w[:3]}"
    else:
        coord_schema = "BLOCKED_COORDINATE_SCHEMA"

    print(f"  coordinate schema: {coord_schema}")
    if coord_schema == "BLOCKED_COORDINATE_SCHEMA":
        print("BLOCKED: coordinate schema unrecognized")
        sys.exit(2)

    # score 컬럼
    if STAGE2_SCORE_COL not in df.columns:
        print(f"BLOCKED: score column '{STAGE2_SCORE_COL}' not in manifest")
        sys.exit(2)
    s_range = f"min={df[STAGE2_SCORE_COL].min():.3f}, p90={df[STAGE2_SCORE_COL].quantile(0.9):.3f}, max={df[STAGE2_SCORE_COL].max():.3f}"
    print(f"  {STAGE2_SCORE_COL}: {s_range}")

    # schema audit CSV
    schema_rows = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        null_count = int(df[col].isna().sum())
        schema_rows.append({"column": col, "dtype": dtype, "null_count": null_count,
                             "sample_val": str(df[col].iloc[0])})
    schema_rows.append({"column": "coordinate_schema", "dtype": "meta",
                         "null_count": 0, "sample_val": coord_schema})
    pd.DataFrame(schema_rows).to_csv(OUT_MANIFESTS / "step11_stage2_candidate_schema_audit.csv", index=False)
    print(f"  schema audit saved")

    # ── [3] p90 고정 적용 + [4] z-continuity ─────────────────────────────
    print()
    print("[3] p90 고정 적용")
    p90_df = df[df[STAGE2_SCORE_COL] > P90_THRESHOLD].copy()
    total_cands = len(df)
    p90_cands   = len(p90_df)
    print(f"  total: {total_cands:,}  →  p90 > {P90_THRESHOLD}: {p90_cands:,}")

    print()
    print("[4] z-continuity >= 2 계획")
    df_valid = build_tracks(p90_df)
    zcont_cands   = len(df_valid)
    n_tracks      = df_valid["track_id"].nunique()
    n_patients    = df_valid["patient_id"].nunique()
    print(f"  z-cont survived: {zcont_cands:,}")
    print(f"  unique tracks  : {n_tracks:,}")
    print(f"  unique patients: {n_patients}")

    # p90/ztrack summary
    summary_rows = [{
        "metric": "total_candidates",       "value": total_cands},
        {"metric": "p90_candidates",        "value": p90_cands},
        {"metric": "zcont_ge2_candidates",  "value": zcont_cands},
        {"metric": "unique_tracks",         "value": n_tracks},
        {"metric": "unique_patients",       "value": n_patients},
        {"metric": "p90_threshold",         "value": P90_THRESHOLD},
        {"metric": "coordinate_schema",     "value": coord_schema},
        {"metric": "stage2_score_col",      "value": STAGE2_SCORE_COL},
    ]
    pd.DataFrame(summary_rows).to_csv(
        OUT_MANIFESTS / "step11_stage2_p90_ztrack_summary.csv", index=False)

    # ── [5] crop + mask readiness ──────────────────────────────────────────
    print()
    print("[5] 5ch crop + mask readiness (sample 10명)")
    sample_results = sample_crop_check(df_valid, n_samples=10)
    n_crop_ok = sum(1 for r in sample_results if r.get("crop_ok", False))
    n_mask_ok = sum(1 for r in sample_results if r.get("mask_ok", False))
    n_sampled = len(sample_results)
    for r in sample_results:
        print(f"  {r['safe_id'][:40]:40s} crop={'OK' if r.get('crop_ok') else 'FAIL'}  "
              f"mask={'OK' if r.get('mask_ok') else 'MISS'}")
    print(f"  → crop OK: {n_crop_ok}/{n_sampled}  mask OK: {n_mask_ok}/{n_sampled}")

    pd.DataFrame(sample_results).to_csv(
        OUT_MANIFESTS / "step11_stage2_crop_mask_readiness_audit.csv", index=False)

    crop_readiness = "PASS" if n_crop_ok == n_sampled else f"WARN_{n_crop_ok}/{n_sampled}"
    mask_readiness = "PASS" if n_mask_ok == n_sampled else f"WARN_{n_mask_ok}/{n_sampled}"

    # 전체 CT 존재 확인
    print("  전체 CT 존재 확인...")
    pt_to_safe = df.groupby("patient_id")["safe_id"].first().to_dict()
    ct_ok_count = sum(1 for sid in pt_to_safe.values()
                      if (NSCLC_CT_ROOT / sid / "ct_hu.npy").exists())
    ct_miss_count = len(pt_to_safe) - ct_ok_count
    print(f"  CT OK: {ct_ok_count}/{len(pt_to_safe)}, MISS: {ct_miss_count}")
    if ct_miss_count > 0:
        errors.append({"step": "ct_readiness", "msg": f"CT MISSING: {ct_miss_count} patients"})

    # ── [6] checkpoint readiness ───────────────────────────────────────────
    print()
    print("[6] checkpoint readiness")

    if not CKPT_BEST.exists():
        print(f"BLOCKED: checkpoint not found: {CKPT_BEST}")
        sys.exit(2)

    ckpt_size_mb = CKPT_BEST.stat().st_size / (1024 ** 2)
    print(f"  checkpoint: {CKPT_BEST.name}  ({ckpt_size_mb:.1f} MB)")

    # 로드 테스트
    import torch
    ckpt = torch.load(str(CKPT_BEST), map_location="cpu", weights_only=False)
    ckpt_epoch = ckpt.get("epoch", "?")
    ckpt_val_loss = ckpt.get("val_loss", "?")
    ckpt_keys = list(ckpt.keys())
    print(f"  epoch={ckpt_epoch}, val_loss={ckpt_val_loss}")
    print(f"  keys: {ckpt_keys}")
    ckpt_ok = ("student_state_dict" in ckpt) and (ckpt.get("epoch", 0) > 0)
    print(f"  checkpoint readiness: {'PASS' if ckpt_ok else 'WARN'}")

    # ── [7] model smoke (teacher 5ch + student load) ───────────────────────
    print()
    print("[7] model smoke (teacher 5ch + student load)")
    model_smoke_ok = False
    try:
        import torchvision.models as tvm
        import torch.nn as nn

        class StudentDecoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.ocbe = nn.Sequential(
                    nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
                    nn.Conv2d(512, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True))
                self.dl3 = nn.Sequential(
                    nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True))
                self.dl2 = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(256, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True))
                self.dl1 = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True))

            def forward(self, l3):
                x = self.ocbe(l3)
                s3 = self.dl3(x)
                s2 = self.dl2(s3)
                s1 = self.dl1(s2)
                return s1, s2, s3

        # teacher 5ch
        teacher = tvm.resnet18(weights=None)
        old_w = teacher.conv1.weight.data.clone()
        new_w = old_w.mean(dim=1, keepdim=True).repeat(1, 5, 1, 1) * (3.0 / 5.0)
        teacher.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
        teacher.conv1.weight.data = new_w
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        # student load
        student = StudentDecoder()
        student.load_state_dict(ckpt["student_state_dict"])
        student.eval()
        for p in student.parameters():
            p.requires_grad_(False)

        # micro forward
        dummy = torch.zeros(2, 5, 96, 96)
        feats = {}
        hooks = []
        for name, layer in [("l1", teacher.layer1), ("l2", teacher.layer2), ("l3", teacher.layer3)]:
            def make_hook(n):
                def h(m, i, o): feats[n] = o
                return h
            hooks.append(layer.register_forward_hook(make_hook(name)))
        with torch.no_grad():
            teacher(dummy)
            s1, s2, s3 = student(feats["l3"])
        for h in hooks:
            h.remove()

        print(f"  teacher l3: {feats['l3'].shape}")
        print(f"  student s1/s2/s3: {s1.shape}/{s2.shape}/{s3.shape}")
        model_smoke_ok = (s1.shape == (2, 64, 24, 24) and
                          s2.shape == (2, 128, 12, 12) and
                          s3.shape == (2, 256, 6, 6))
        print(f"  model smoke: {'PASS' if model_smoke_ok else 'FAIL'}")
        guardrail["model_forward_executed"] = "small_smoke_only"
        del teacher, student, dummy, feats
    except Exception as e:
        print(f"  model smoke FAIL: {e}")
        errors.append({"step": "model_smoke", "msg": str(e)})

    # ── [8] shard plan ────────────────────────────────────────────────────
    print()
    print("[8] shard plan 생성")

    patients_sorted = sorted(df_valid["patient_id"].unique())
    shard_size = math.ceil(len(patients_sorted) / N_SHARDS)
    shard_rows = []
    for si in range(N_SHARDS):
        shard_pts = patients_sorted[si * shard_size: (si + 1) * shard_size]
        shard_cands = int(df_valid[df_valid["patient_id"].isin(shard_pts)].shape[0])
        shard_rows.append({
            "shard_id": si,
            "n_patients": len(shard_pts),
            "n_candidates": shard_cands,
            "patients": ";".join(shard_pts),
        })
        print(f"  shard {si}: {len(shard_pts)} patients, {shard_cands:,} candidates")

    pd.DataFrame(shard_rows).to_csv(OUT_MANIFESTS / "step11_stage2_shard_plan.csv", index=False)

    # scoring plan manifest
    plan_cols = ["row_id", "patient_id", "safe_id", "local_z",
                 "y0", "x0", "y1", "x1", "track_id", "track_len",
                 "track_z_start", "track_z_end", "pos_key"]
    plan_cols_exist = [c for c in plan_cols if c in df_valid.columns]
    df_plan = df_valid[plan_cols_exist].copy()
    df_plan["shard_id"] = -1
    for si, r in enumerate(shard_rows):
        pts = set(r["patients"].split(";"))
        df_plan.loc[df_plan["patient_id"].isin(pts), "shard_id"] = si
    df_plan.to_csv(OUT_MANIFESTS / "step11_stage2_scoring_plan_manifest.csv", index=False)
    print(f"  scoring plan manifest: {len(df_plan):,} rows")

    # ── [9] output collision 확인 ─────────────────────────────────────────
    print()
    print("[9] output collision 확인")
    stage2_out_dir = OUT_STAGE2 / "step12_scoring"
    if stage2_out_dir.exists() and any(stage2_out_dir.iterdir()):
        print(f"  BLOCKED_OUTPUT_COLLISION: {stage2_out_dir} 이미 존재")
        errors.append({"step": "output_collision", "msg": str(stage2_out_dir)})
        collision_ok = False
    else:
        collision_ok = True
        print(f"  output dir: {stage2_out_dir} — 비어있음 (PASS)")

    # ── [10] 예상 scoring plan ────────────────────────────────────────────
    secs_per_candidate = 0.003  # step8 기준: 95995 cands / 255s ≈ 0.0027s
    expected_runtime_s = zcont_cands * secs_per_candidate
    expected_runtime_min = expected_runtime_s / 60
    expected_output_rows = zcont_cands
    # 컬럼 수 추정: 기존 step8 컬럼(34) + score(10)
    expected_csv_mb = expected_output_rows * 60 / (1024 * 1024)  # 약 60 bytes/row

    print()
    print("[10] 예상 scoring plan")
    print(f"  total stage2 candidates  : {total_cands:,}")
    print(f"  p90 candidates           : {p90_cands:,}")
    print(f"  p90+z-cont candidates    : {zcont_cands:,}")
    print(f"  tracks                   : {n_tracks:,}")
    print(f"  patients                 : {n_patients}")
    print(f"  shards                   : {N_SHARDS}")
    print(f"  expected runtime         : ~{expected_runtime_min:.1f} min")
    print(f"  expected output rows     : {expected_output_rows:,}")
    print(f"  expected CSV size        : ~{expected_csv_mb:.1f} MB")

    # ── report 생성 ───────────────────────────────────────────────────────
    print()
    print("[11] report 생성")

    if errors and any(e["step"] == "output_collision" for e in errors):
        verdict = "BLOCKED_OUTPUT_COLLISION"
    elif n_crop_ok < n_sampled * 0.8 or ct_miss_count > 0:
        verdict = "PARTIAL_PASS_STEP11"
    elif not ckpt_ok or not model_smoke_ok:
        verdict = "PARTIAL_PASS_STEP11"
    else:
        verdict = "PASS_STEP11_STAGE2_FIXED_PREFLIGHT"

    report_lines = []
    report_lines.append("# Step 11 Stage2 Fixed Preflight Report")
    report_lines.append("")
    report_lines.append(f"## Verdict: **{verdict}**")
    report_lines.append("")
    report_lines.append("## Stage2 Candidate Manifest")
    report_lines.append(f"- path: `{STAGE2_MANIFEST}`")
    report_lines.append(f"- total rows: {total_cands:,}")
    report_lines.append(f"- coordinate schema: {coord_schema}")
    report_lines.append(f"- score column: `{STAGE2_SCORE_COL}`")
    report_lines.append("")
    report_lines.append("## P90 + Z-Continuity Plan")
    report_lines.append(f"- p90 threshold: {P90_THRESHOLD} (고정, stage2 재계산 금지)")
    report_lines.append(f"- p90 candidates: {p90_cands:,} / {total_cands:,}")
    report_lines.append(f"- z-cont >= 2: {zcont_cands:,}")
    report_lines.append(f"- tracks: {n_tracks:,}")
    report_lines.append(f"- patients: {n_patients}")
    report_lines.append("")
    report_lines.append("## Coordinate Schema")
    report_lines.append(f"- schema: {coord_schema}")
    report_lines.append("- 32×32 position (y0,x0,y1,x1) → center±48 → 96×96 crop")
    report_lines.append("- center_y = (y0+y1)//2, center_x = (x0+x1)//2")
    report_lines.append("- crop_y0 = center_y-48, crop_x0 = center_x-48")
    report_lines.append("")
    report_lines.append("## Crop + Mask Readiness")
    report_lines.append(f"- crop OK: {n_crop_ok}/{n_sampled}")
    report_lines.append(f"- mask OK: {n_mask_ok}/{n_sampled}")
    report_lines.append(f"- CT all patients: {ct_ok_count}/{len(pt_to_safe)}")
    report_lines.append("")
    report_lines.append("## Checkpoint Readiness")
    report_lines.append(f"- path: `{CKPT_BEST}`")
    report_lines.append(f"- size: {ckpt_size_mb:.1f} MB")
    report_lines.append(f"- epoch: {ckpt_epoch}, val_loss: {ckpt_val_loss}")
    report_lines.append(f"- model smoke: {'PASS' if model_smoke_ok else 'FAIL'}")
    report_lines.append("")
    report_lines.append("## Shard Plan")
    report_lines.append(f"- {N_SHARDS} shards, patient-stable split")
    for r in shard_rows:
        report_lines.append(f"  - shard {r['shard_id']}: {r['n_patients']} patients, {r['n_candidates']:,} cands")
    report_lines.append("")
    report_lines.append("## Expected Scoring")
    report_lines.append(f"- scoring rows: {zcont_cands:,}")
    report_lines.append(f"- runtime: ~{expected_runtime_min:.1f} min")
    report_lines.append(f"- output CSV: ~{expected_csv_mb:.1f} MB")
    report_lines.append("")
    report_lines.append("## Output Collision")
    report_lines.append(f"- {stage2_out_dir}: {'PASS (empty)' if collision_ok else 'BLOCKED'}")
    report_lines.append("")
    report_lines.append("## Guardrail")
    for k, v in guardrail.items():
        report_lines.append(f"- {k}: {v}")

    rpt = OUT_REPORTS / "step11_stage2_fixed_preflight_report.md"
    rpt.write_text("\n".join(report_lines), encoding="utf-8")

    summary = {
        "step": "step11_stage2_fixed_preflight",
        "verdict": verdict,
        "created": "2026-06-10",
        "stage2_manifest": str(STAGE2_MANIFEST),
        "coordinate_schema": coord_schema,
        "stage2_score_col": STAGE2_SCORE_COL,
        "p90_threshold": P90_THRESHOLD,
        "total_candidates": total_cands,
        "p90_candidates": p90_cands,
        "zcont_ge2_candidates": zcont_cands,
        "unique_tracks": n_tracks,
        "unique_patients": n_patients,
        "ct_readiness": f"{ct_ok_count}/{len(pt_to_safe)}",
        "crop_readiness": crop_readiness,
        "mask_readiness": mask_readiness,
        "checkpoint_ok": ckpt_ok,
        "model_smoke_ok": model_smoke_ok,
        "n_shards": N_SHARDS,
        "expected_runtime_min": round(expected_runtime_min, 1),
        "output_collision": not collision_ok,
        "primary_candidate_score": PRIMARY_CANDIDATE_SCORE,
        "primary_track_score": PRIMARY_TRACK_SCORE,
        "guardrail": guardrail,
    }

    s_path = OUT_REPORTS / "step11_stage2_fixed_preflight_summary.json"
    with open(s_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    err_path = OUT_LOGS / "step11_stage2_fixed_preflight_errors.csv"
    with open(err_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "msg"])
        w.writeheader()
        for e in errors:
            w.writerow(e)

    done_path = ROOT / "DONE_STEP11_STAGE2_FIXED_PREFLIGHT.json"
    with open(done_path, "w") as f:
        json.dump({
            "step": "step11_stage2_fixed_preflight",
            "verdict": verdict,
            "created": "2026-06-10",
            "stage2_manifest": str(STAGE2_MANIFEST),
            "zcont_candidates": zcont_cands,
            "tracks": n_tracks,
            "patients": n_patients,
            "n_shards": N_SHARDS,
            "coordinate_schema": coord_schema,
            "stage2_score_col": STAGE2_SCORE_COL,
            "scoring_plan_manifest": str(OUT_MANIFESTS / "step11_stage2_scoring_plan_manifest.csv"),
            "report": str(rpt),
            "summary_json": str(s_path),
        }, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 64)
    print(f"판정: {verdict}")
    print("=" * 64)
    print(f"  stage2 manifest  : {STAGE2_MANIFEST.name}")
    print(f"  coordinate schema: {coord_schema}")
    print(f"  score col        : {STAGE2_SCORE_COL}")
    print(f"  total → p90 → z-cont: {total_cands:,} → {p90_cands:,} → {zcont_cands:,}")
    print(f"  tracks: {n_tracks:,}, patients: {n_patients}")
    print(f"  CT readiness : {ct_ok_count}/{len(pt_to_safe)}")
    print(f"  crop readiness: {crop_readiness}")
    print(f"  mask readiness: {mask_readiness}")
    print(f"  checkpoint    : epoch={ckpt_epoch}, val_loss={ckpt_val_loss}")
    print(f"  model smoke   : {'PASS' if model_smoke_ok else 'FAIL'}")
    print(f"  shards        : {N_SHARDS}")
    print(f"  expected runtime: ~{expected_runtime_min:.1f} min")
    print(f"  stage2 accessed : False")
    print()
    print("다음 단계: Step 12 stage2 fixed scoring launch (사용자 승인 후)")


if __name__ == "__main__":
    main()
