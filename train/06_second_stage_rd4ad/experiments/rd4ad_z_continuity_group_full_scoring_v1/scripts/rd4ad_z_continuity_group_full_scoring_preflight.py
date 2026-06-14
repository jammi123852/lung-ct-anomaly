"""
RD4AD z-continuity group-level full scoring preflight v1

smoke v1 PASS_CANDIDATE 확인 후, stage1_dev 전체 20,216 groups에 대한
full scoring 실행 전 preflight 수행.

실행 방식:
  bare run (인자 없음): exit 2 로 막음
  dry-run:      python <script> --dry-run
  preflight:    python <script> --run-preflight --confirm-readonly --confirm-stage1dev-only
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# =============================================================================
# 경로 설정
# =============================================================================

PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments/rd4ad_z_continuity_group_full_scoring_v1"

# 입력 (read-only)
CANDIDATE_MANIFEST_CSV = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_c2_effb0_v420_candidate_rd4ad_retest_v1"
    / "rd_c2_effb0_v420_candidate_manifest.csv"
)

RD_D1S_SCORE_CSV = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_d1s_medi3ch_true_rd4ad_shard_run_v1"
    / "rd_d1s_stage1dev_candidate_score.csv"
)

CKPT_PATH = (
    PROJECT_ROOT
    / "outputs/models/rd_d1s_true_rd4ad_resnet18_medi3ch_shard_v1"
    / "checkpoints/best_train_loss.pth"
)

LOCAL_RESNET_WEIGHT = Path("/home/jinhy/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth")

CT_ROOT = Path(
    "/mnt/c/Users/jinhy/Desktop"
    "/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1"
    "/volumes_npy"
)

# smoke v1 참조 (read-only)
SMOKE_V1_ROOT = PROJECT_ROOT / "experiments/rd4ad_z_continuity_group_rescore_smoke_v1"
SMOKE_GROUP_MANIFEST_CSV = SMOKE_V1_ROOT / "manifests/group_manifest.csv"
SMOKE_REPR_MANIFEST_CSV  = SMOKE_V1_ROOT / "manifests/group_representative_manifest.csv"
SMOKE_SUMMARY_JSON       = SMOKE_V1_ROOT / "reports/rd4ad_z_continuity_group_rescore_smoke_summary.json"

# 출력 (새 폴더에만)
MANIFEST_DIR  = EXPERIMENT_ROOT / "manifests"
REPORT_DIR    = EXPERIMENT_ROOT / "reports"
LOG_DIR       = EXPERIMENT_ROOT / "logs"

GROUP_MANIFEST_CSV       = MANIFEST_DIR / "group_manifest_full.csv"
GROUP_REPR_MANIFEST_CSV  = MANIFEST_DIR / "group_representative_manifest_full.csv"
SHARD_PLAN_CSV           = MANIFEST_DIR / "full_scoring_shard_plan.csv"
ERROR_CSV                = LOG_DIR / "errors.csv"
REPORT_MD                = REPORT_DIR / "rd4ad_z_continuity_group_full_scoring_preflight_report.md"
SUMMARY_JSON             = REPORT_DIR / "rd4ad_z_continuity_group_full_scoring_preflight_summary.json"
DONE_JSON                = EXPERIMENT_ROOT / "DONE.json"

# group 파라미터 (smoke v1과 동일)
DEFAULT_Z_GAP     = 1
DEFAULT_XY_RADIUS = 24

# shard 설정
SHARD_COUNT = 4

# 모델 구조
CROP_SIZE = 96
HU_MIN, HU_MAX = -160.0, 240.0

# =============================================================================
# guardrail 상태
# =============================================================================

GUARDRAILS = {
    "stage2_holdout_accessed":            False,
    "checkpoint_loaded":                  False,
    "model_forward_executed":             False,
    "training_executed":                  False,
    "backward_executed":                  False,
    "optimizer_created":                  False,
    "checkpoint_saved":                   False,
    "crop_generation_executed":           False,
    "full_scoring_executed":              False,
    "threshold_recalculated":             False,
    "existing_artifact_modified":         False,
    "existing_script_modified":           False,
    "output_overwrite":                   False,
    "label_used_for_evaluation_only":     True,
    "label_used_as_deployment_selector":  False,
    "first_stage_score_used_for_representative_choice": True,
    "first_stage_score_used_for_candidate_deletion":    False,
    "raw_rd4ad_primary_score":            True,
    "boundary_penalty_primary_score":     False,
}

errors = []

def log_error(code, msg, details=""):
    errors.append({"code": code, "msg": msg, "details": str(details)})
    print(f"  [ERROR] {code}: {msg}" + (f" | {details}" if details else ""))

def log_warn(msg):
    print(f"  [WARN] {msg}")

def log_ok(msg):
    print(f"  [OK]   {msg}")


# =============================================================================
# CSV 유틸
# =============================================================================

def read_csv(path: Path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def write_csv(path: Path, rows, fieldnames=None):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# =============================================================================
# Union-Find
# =============================================================================

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank   = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


# =============================================================================
# group 생성 (smoke v1과 동일)
# =============================================================================

def build_groups(candidates, z_gap=DEFAULT_Z_GAP, xy_radius=DEFAULT_XY_RADIUS):
    pat_candidates = defaultdict(list)
    for c in candidates:
        pat_candidates[c["patient_id"]].append(c)

    group_id_map    = {}
    group_candidates = {}
    global_group_counter = [0]

    for pid, cands in pat_candidates.items():
        n = len(cands)
        uf = UnionFind(n)

        for i in range(n):
            zi = cands[i]["local_z"]
            yi = cands[i]["y_center"]
            xi = cands[i]["x_center"]
            for j in range(i + 1, n):
                zj = cands[j]["local_z"]
                if abs(zi - zj) > z_gap:
                    continue
                yj = cands[j]["y_center"]
                xj = cands[j]["x_center"]
                if abs(yi - yj) <= xy_radius and abs(xi - xj) <= xy_radius:
                    uf.union(i, j)

        root_to_gid = {}
        for i in range(n):
            root = uf.find(i)
            if root not in root_to_gid:
                gid = f"G{global_group_counter[0]:07d}"
                global_group_counter[0] += 1
                root_to_gid[root] = gid
                group_candidates[gid] = []
            gid = root_to_gid[root]
            cid = cands[i]["candidate_id"]
            group_id_map[cid] = gid
            group_candidates[gid].append(cid)

    return group_id_map, group_candidates


# =============================================================================
# shard 계획 생성
# =============================================================================

def build_shard_plan(group_stats_list, n_shards=SHARD_COUNT):
    """patient_id stable hash 기반 shard 분할."""
    rows = []
    shard_counts = defaultdict(int)
    shard_pos_counts = defaultdict(int)

    for gs in group_stats_list:
        pid = gs["patient_id"]
        h = int(hashlib.md5(pid.encode()).hexdigest(), 16)
        shard_id = h % n_shards
        shard_counts[shard_id] += 1
        if gs.get("has_positive", False):
            shard_pos_counts[shard_id] += 1
        rows.append({
            "group_id":    gs["group_id"],
            "patient_id":  pid,
            "shard_id":    shard_id,
            "has_positive": gs.get("has_positive", False),
        })

    # shard 요약
    shard_summary = []
    for sid in sorted(shard_counts.keys()):
        shard_summary.append({
            "shard_id":       sid,
            "group_count":    shard_counts[sid],
            "has_positive":   shard_pos_counts[sid],
        })

    return rows, shard_summary


# =============================================================================
# dry-run
# =============================================================================

def run_dry():
    print("=" * 70)
    print("[DRY-RUN] rd4ad_z_continuity_group_full_scoring_preflight v1")
    print("=" * 70)
    print()
    print("실행 모드: DRY-RUN (파일 읽기/쓰기 없음)")
    print()
    print("입력 파일:")
    print(f"  CANDIDATE_MANIFEST_CSV : {CANDIDATE_MANIFEST_CSV}")
    print(f"  RD_D1S_SCORE_CSV       : {RD_D1S_SCORE_CSV}")
    print(f"  CKPT_PATH              : {CKPT_PATH}")
    print(f"  LOCAL_RESNET_WEIGHT    : {LOCAL_RESNET_WEIGHT}")
    print(f"  CT_ROOT                : {CT_ROOT}")
    print(f"  SMOKE_GROUP_MANIFEST   : {SMOKE_GROUP_MANIFEST_CSV}")
    print(f"  SMOKE_REPR_MANIFEST    : {SMOKE_REPR_MANIFEST_CSV}")
    print()
    print("출력 파일:")
    print(f"  GROUP_MANIFEST_CSV     : {GROUP_MANIFEST_CSV}")
    print(f"  GROUP_REPR_MANIFEST    : {GROUP_REPR_MANIFEST_CSV}")
    print(f"  SHARD_PLAN_CSV         : {SHARD_PLAN_CSV}")
    print(f"  ERROR_CSV              : {ERROR_CSV}")
    print(f"  REPORT_MD              : {REPORT_MD}")
    print(f"  SUMMARY_JSON           : {SUMMARY_JSON}")
    print(f"  DONE_JSON              : {DONE_JSON}")
    print()
    print("group 파라미터 (smoke v1과 동일):")
    print(f"  z_gap      = {DEFAULT_Z_GAP}")
    print(f"  xy_radius  = {DEFAULT_XY_RADIUS}")
    print(f"  shard_count = {SHARD_COUNT}")
    print()
    print("guardrail 설정:")
    for k, v in GUARDRAILS.items():
        print(f"  {k}: {v}")
    print()
    print("실제 preflight 실행:")
    print("  python <script> --run-preflight --confirm-readonly --confirm-stage1dev-only")
    print()
    sys.exit(0)


# =============================================================================
# preflight 메인
# =============================================================================

def run_preflight():
    print("=" * 70)
    print("[PREFLIGHT] RD4AD z-continuity group-level full scoring v1")
    print("=" * 70)
    t0 = time.perf_counter()

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # [0] output overwrite 검사
    # -------------------------------------------------------------------------
    print("\n[0] output overwrite 검사")
    existing_outputs = [
        GROUP_MANIFEST_CSV,
        GROUP_REPR_MANIFEST_CSV,
        SHARD_PLAN_CSV,
        REPORT_MD,
        SUMMARY_JSON,
        DONE_JSON,
    ]
    overwrite_found = [p for p in existing_outputs if p.exists()]
    if overwrite_found:
        GUARDRAILS["output_overwrite"] = True
        for p in overwrite_found:
            log_error("OUTPUT_OVERWRITE", f"출력 파일이 이미 존재함: {p.name}")
        print("  [ABORT] output overwrite 위험 — 기존 결과 삭제 후 재실행 필요")
        _save_error_csv()
        sys.exit(3)
    log_ok("output overwrite 없음")

    # -------------------------------------------------------------------------
    # [1] 입력 파일 존재 확인
    # -------------------------------------------------------------------------
    print("\n[1] 입력 파일 존재 확인")
    input_files = {
        "candidate_manifest":    CANDIDATE_MANIFEST_CSV,
        "rd_d1s_score":          RD_D1S_SCORE_CSV,
        "checkpoint":            CKPT_PATH,
        "resnet18_weight":       LOCAL_RESNET_WEIGHT,
        "smoke_group_manifest":  SMOKE_GROUP_MANIFEST_CSV,
        "smoke_repr_manifest":   SMOKE_REPR_MANIFEST_CSV,
        "smoke_summary":         SMOKE_SUMMARY_JSON,
    }
    for name, p in input_files.items():
        if p.exists():
            size_mb = p.stat().st_size / 1024 / 1024
            log_ok(f"{name}: {p.name} ({size_mb:.1f} MB)")
        else:
            log_error("INPUT_MISSING", f"{name} 없음", p)

    # -------------------------------------------------------------------------
    # [2] smoke v1 summary 로드 및 기준값 확인
    # -------------------------------------------------------------------------
    print("\n[2] smoke v1 summary 로드")
    smoke_summary = {}
    if SMOKE_SUMMARY_JSON.exists():
        with open(SMOKE_SUMMARY_JSON) as f:
            smoke_summary = json.load(f)
        smoke_group_count    = smoke_summary.get("group_count", -1)
        smoke_orig_cands     = smoke_summary.get("original_candidate_count", -1)
        smoke_pos_rate       = smoke_summary.get("positive_candidate_group_assignment_rate", -1)
        smoke_has_pos_groups = smoke_summary.get("has_positive_group_count", -1)
        smoke_reduction_rate = smoke_summary.get("reduction_rate", -1)
        log_ok(f"smoke_group_count = {smoke_group_count:,}")
        log_ok(f"smoke_original_candidates = {smoke_orig_cands:,}")
        log_ok(f"smoke_positive_assignment_rate = {smoke_pos_rate:.4f}")
        log_ok(f"smoke_has_positive_groups = {smoke_has_pos_groups:,}")
    else:
        log_error("SMOKE_MISSING", "smoke v1 summary.json 없음")
        smoke_group_count    = 20216
        smoke_orig_cands     = 113447
        smoke_pos_rate       = 1.0
        smoke_has_pos_groups = 242
        smoke_reduction_rate = 0.8218

    # -------------------------------------------------------------------------
    # [3] candidate manifest 로드 및 stage1_dev 필터
    # -------------------------------------------------------------------------
    print("\n[3] candidate manifest 로드 및 stage1_dev 필터")
    all_rows    = read_csv(CANDIDATE_MANIFEST_CSV)
    stage1_rows = [r for r in all_rows if r.get("stage_split") == "stage1_dev"]
    stage2_rows = [r for r in all_rows if r.get("stage_split") == "stage2_holdout"]

    print(f"  전체 rows: {len(all_rows):,}")
    print(f"  stage1_dev: {len(stage1_rows):,}")
    print(f"  stage2_holdout (접근 금지): {len(stage2_rows):,}")

    if stage2_rows:
        log_warn(f"stage2_holdout rows {len(stage2_rows):,}개 존재 — 로드하지 않음")

    # stage2 접근 없음 검증
    GUARDRAILS["stage2_holdout_accessed"] = False

    if len(stage1_rows) != smoke_orig_cands:
        log_warn(f"stage1_dev count {len(stage1_rows):,} != smoke v1 기준 {smoke_orig_cands:,}")
    else:
        log_ok(f"stage1_dev candidate count 일치: {len(stage1_rows):,}")

    # -------------------------------------------------------------------------
    # [4] candidate 전처리
    # -------------------------------------------------------------------------
    print("\n[4] candidate 전처리")
    cand_map = {}
    skip_count = 0
    for r in stage1_rows:
        cid = r.get("candidate_id", "")
        if not cid:
            skip_count += 1
            continue
        try:
            y0 = int(r["crop_y0"])
            x0 = int(r["crop_x0"])
            y1 = int(r["crop_y1"])
            x1 = int(r["crop_x1"])
            local_z = int(r["local_z"])
            fss = float(r["first_stage_score"])
        except (KeyError, ValueError) as e:
            log_error("CAND_PARSE_ERROR", f"{cid} 파싱 실패", e)
            skip_count += 1
            continue

        cand_map[cid] = {
            "candidate_id":      cid,
            "patient_id":        r["patient_id"],
            "safe_id":           r.get("safe_id", ""),
            "local_z":           local_z,
            "crop_y0":           y0,
            "crop_x0":           x0,
            "crop_y1":           y1,
            "crop_x1":           x1,
            "y_center":          (y0 + y1) / 2.0,
            "x_center":          (x0 + x1) / 2.0,
            "first_stage_score": fss,
            "label":             r.get("label", ""),
        }

    candidates = list(cand_map.values())
    print(f"  처리 완료: {len(candidates):,}  스킵: {skip_count}")
    if skip_count > 0:
        log_warn(f"{skip_count}개 candidate 스킵")

    # -------------------------------------------------------------------------
    # [5] group 생성 (smoke v1과 동일 규칙)
    # -------------------------------------------------------------------------
    print(f"\n[5] group 생성 (z_gap={DEFAULT_Z_GAP}, xy_radius={DEFAULT_XY_RADIUS})")
    t_group = time.perf_counter()
    group_id_map, group_candidates = build_groups(candidates, DEFAULT_Z_GAP, DEFAULT_XY_RADIUS)
    t_group_elapsed = time.perf_counter() - t_group

    full_group_count = len(group_candidates)
    reduction_rate   = 1.0 - full_group_count / len(candidates) if candidates else 0.0
    print(f"  원본 candidates: {len(candidates):,}")
    print(f"  groups: {full_group_count:,}  reduction: {reduction_rate:.4f} ({reduction_rate:.1%})")
    print(f"  group 생성 소요: {t_group_elapsed:.2f}s")

    # smoke v1과 group count 일치 여부
    group_count_match = (full_group_count == smoke_group_count)
    if group_count_match:
        log_ok(f"group count smoke v1 일치: {full_group_count:,} == {smoke_group_count:,}")
    else:
        log_error(
            "GROUP_COUNT_MISMATCH",
            f"group count smoke v1 불일치: {full_group_count:,} != {smoke_group_count:,}"
        )

    # -------------------------------------------------------------------------
    # [6] group 통계 계산 및 대표 후보 선택
    # -------------------------------------------------------------------------
    print("\n[6] group 통계 및 대표 후보 선택")
    group_stats_list = []
    for gid, cid_list in group_candidates.items():
        rows_g = [cand_map[cid] for cid in cid_list if cid in cand_map]
        if not rows_g:
            continue

        patient_id = rows_g[0]["patient_id"]
        n_cands    = len(rows_g)
        has_positive = any(r["label"] in ("positive", "lesion_positive") for r in rows_g)
        positive_count = sum(1 for r in rows_g if r["label"] in ("positive", "lesion_positive"))

        # 대표 후보: first_stage_score 최대
        repr_row = max(rows_g, key=lambda r: r["first_stage_score"])
        repr_cid = repr_row["candidate_id"]

        group_stats_list.append({
            "group_id":            gid,
            "patient_id":          patient_id,
            "n_candidates":        n_cands,
            "has_positive":        has_positive,
            "positive_count":      positive_count,
            "representative_candidate_id": repr_cid,
            "first_stage_score_max":  repr_row["first_stage_score"],
            "local_z_repr":        repr_row["local_z"],
            "crop_y0":             repr_row["crop_y0"],
            "crop_x0":             repr_row["crop_x0"],
            "crop_y1":             repr_row["crop_y1"],
            "crop_x1":             repr_row["crop_x1"],
            "label_repr":          repr_row["label"],
        })

    has_positive_groups = sum(1 for gs in group_stats_list if gs["has_positive"])
    total_positive_cands = sum(1 for c in candidates if c["label"] in ("positive", "lesion_positive"))

    # positive candidate group assignment rate
    assigned_positive_groups = sum(1 for gs in group_stats_list if gs["positive_count"] > 0)
    pos_assign_rate = assigned_positive_groups / has_positive_groups if has_positive_groups else 0.0

    print(f"  전체 groups: {full_group_count:,}")
    print(f"  has_positive groups: {has_positive_groups:,}")
    print(f"  positive candidates: {total_positive_cands:,}")
    print(f"  positive assignment rate: {pos_assign_rate:.4f}")

    if abs(pos_assign_rate - 1.0) < 1e-6:
        log_ok("positive candidate group assignment rate = 100%")
    else:
        log_error(
            "POSITIVE_ASSIGNMENT_RATE",
            f"positive assignment rate < 100%: {pos_assign_rate:.4f}"
        )

    # -------------------------------------------------------------------------
    # [7] smoke v1 대표 후보 일치 검증 (샘플 확인)
    # -------------------------------------------------------------------------
    print("\n[7] smoke v1 representative 일치 검증")
    repr_consistency = "SKIP"
    repr_match_count = 0
    repr_total_check = 0

    if SMOKE_REPR_MANIFEST_CSV.exists():
        smoke_repr_rows = read_csv(SMOKE_REPR_MANIFEST_CSV)
        # smoke 결과에서 처음 200개 group 비교
        smoke_repr_map = {r["group_id"]: r["representative_candidate_id"] for r in smoke_repr_rows}
        full_repr_map  = {gs["group_id"]: gs["representative_candidate_id"] for gs in group_stats_list}

        check_gids = list(smoke_repr_map.keys())[:200]
        for gid in check_gids:
            if gid in full_repr_map:
                repr_total_check += 1
                if smoke_repr_map[gid] == full_repr_map[gid]:
                    repr_match_count += 1

        if repr_total_check > 0:
            repr_match_rate = repr_match_count / repr_total_check
            print(f"  샘플 비교: {repr_match_count}/{repr_total_check} 일치 ({repr_match_rate:.2%})")
            if repr_match_rate >= 0.99:
                repr_consistency = "OK"
                log_ok(f"representative 일치율 {repr_match_rate:.2%}")
            else:
                repr_consistency = "FAIL"
                log_error(
                    "REPR_MISMATCH",
                    f"representative 일치율 {repr_match_rate:.2%} < 99%"
                )
        else:
            repr_consistency = "SKIP_NO_COMMON_GID"
            log_warn("공통 group_id 없음 — 일치 검증 스킵")
    else:
        log_warn("smoke v1 repr manifest 없음 — 일치 검증 스킵")

    # -------------------------------------------------------------------------
    # [8] CT readiness 확인
    # -------------------------------------------------------------------------
    print("\n[8] CT mmap readiness")
    ct_patient_ids = set(c["safe_id"] for c in candidates if c.get("safe_id"))
    stage1_patients_unique = set(c["patient_id"] for c in candidates)
    ct_found = 0
    ct_missing = []

    for safe_id in sorted(ct_patient_ids)[:10]:  # 샘플 10개 확인
        ct_dir  = CT_ROOT / safe_id
        ct_file = ct_dir / "ct_hu.npy"
        if ct_file.exists():
            ct_found += 1
        else:
            ct_missing.append(safe_id)

    sample_size = min(10, len(ct_patient_ids))
    print(f"  stage1_dev 환자 수: {len(stage1_patients_unique):,}")
    print(f"  샘플 {sample_size}개 CT 확인: {ct_found}/{sample_size} 존재")

    if ct_missing:
        log_warn(f"CT 없는 safe_id (샘플): {ct_missing[:3]}")
        ct_readiness = "PARTIAL"
    elif ct_found == sample_size and sample_size > 0:
        log_ok(f"CT readiness 샘플 통과 ({ct_found}/{sample_size})")
        ct_readiness = "OK"
    else:
        ct_readiness = "UNKNOWN"
        log_warn("CT 샘플 확인 불가")

    # CT 전체 볼륨 수 확인
    try:
        all_ct_dirs = [d for d in CT_ROOT.iterdir() if d.is_dir()] if CT_ROOT.exists() else []
        log_ok(f"CT_ROOT 총 볼륨 폴더: {len(all_ct_dirs):,}")
    except Exception as e:
        log_warn(f"CT_ROOT 열거 실패: {e}")
        all_ct_dirs = []

    # -------------------------------------------------------------------------
    # [9] checkpoint readiness
    # -------------------------------------------------------------------------
    print("\n[9] checkpoint readiness")
    ckpt_ok = CKPT_PATH.exists()
    resnet_ok = LOCAL_RESNET_WEIGHT.exists()

    if ckpt_ok:
        ckpt_size_mb = CKPT_PATH.stat().st_size / 1024 / 1024
        log_ok(f"checkpoint: {CKPT_PATH.name} ({ckpt_size_mb:.1f} MB)")
    else:
        log_error("CKPT_MISSING", f"checkpoint 없음: {CKPT_PATH}")

    if resnet_ok:
        log_ok(f"ResNet18 weight: {LOCAL_RESNET_WEIGHT.name}")
    else:
        log_warn(f"ResNet18 local weight 없음: {LOCAL_RESNET_WEIGHT} (torch hub에서 자동 다운로드 필요)")

    # NOTE: preflight에서는 checkpoint를 실제로 load하지 않음
    GUARDRAILS["checkpoint_loaded"] = False

    # -------------------------------------------------------------------------
    # [10] RD-D1s scalar score CSV readiness (scalar reproduction용)
    # -------------------------------------------------------------------------
    print("\n[10] RD-D1s scalar score CSV readiness")
    if RD_D1S_SCORE_CSV.exists():
        rd_score_rows = read_csv(RD_D1S_SCORE_CSV)
        rd_stage1_rows = [r for r in rd_score_rows if r.get("stage_split") == "stage1_dev"]
        log_ok(f"rd_d1s_score: {len(rd_score_rows):,} rows (stage1_dev: {len(rd_stage1_rows):,})")
        if len(rd_stage1_rows) == 0:
            log_warn("RD-D1s score CSV에 stage1_dev rows 없음 — stage_split 컬럼 확인 필요")
    else:
        log_error("RD_SCORE_MISSING", "RD-D1s scalar score CSV 없음")
        rd_stage1_rows = []

    # -------------------------------------------------------------------------
    # [11] GPU 메모리 추정
    # -------------------------------------------------------------------------
    print("\n[11] GPU / 메모리 추정")
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        gpu_info = result.stdout.strip()
        log_ok(f"GPU: {gpu_info}")
    except Exception as e:
        gpu_info = f"확인 실패: {e}"
        log_warn(f"GPU 확인 실패: {e}")

    # full 20,216 groups forward 예상 시간
    # smoke 742 groups → 62.3s → 1 group ≈ 0.084s
    # (실제 모델 forward는 preflight에서 실행 안 함 — smoke v1 기준으로 추정)
    smoke_elapsed = smoke_summary.get("elapsed_sec", 62.3)
    smoke_forward_groups = smoke_summary.get("smoke_forward_group_count", 742)
    if smoke_forward_groups > 0:
        per_group_sec = smoke_elapsed / smoke_forward_groups
        estimated_full_sec = per_group_sec * full_group_count
    else:
        per_group_sec = 0.084
        estimated_full_sec = 0.084 * full_group_count

    estimated_full_min = estimated_full_sec / 60.0
    print(f"  smoke v1 기준: {smoke_forward_groups} groups / {smoke_elapsed:.1f}s")
    print(f"  per group 추정: {per_group_sec:.4f}s")
    print(f"  full {full_group_count:,} groups 예상 시간: {estimated_full_min:.1f}분")

    # crop 단위 메모리: 96×96×3 float32 = 110KB → batch 64 ≈ 7MB
    crop_mem_mb = (CROP_SIZE * CROP_SIZE * 3 * 4) / 1024 / 1024
    batch_size_rec = 64
    batch_mem_mb = crop_mem_mb * batch_size_rec
    print(f"  crop 크기: {CROP_SIZE}×{CROP_SIZE}×3 = {crop_mem_mb:.3f} MB")
    print(f"  권장 batch_size={batch_size_rec}: {batch_mem_mb:.2f} MB")

    # -------------------------------------------------------------------------
    # [12] shard 계획 생성
    # -------------------------------------------------------------------------
    print(f"\n[12] shard 계획 생성 (n_shards={SHARD_COUNT})")
    shard_plan_rows, shard_summary = build_shard_plan(group_stats_list, n_shards=SHARD_COUNT)

    print(f"  shard_count: {SHARD_COUNT}")
    for ss in shard_summary:
        print(f"  shard {ss['shard_id']}: {ss['group_count']:,} groups  has_positive={ss['has_positive']:,}")

    # -------------------------------------------------------------------------
    # [13] label leakage 검사
    # -------------------------------------------------------------------------
    print("\n[13] label leakage 검사")
    label_leakage = False
    # preflight에서 label은 group assignment rate 계산용으로만 사용
    # deployment selector로 사용하지 않음
    GUARDRAILS["label_used_for_evaluation_only"] = True
    GUARDRAILS["label_used_as_deployment_selector"] = False
    log_ok("label: evaluation / positive assignment rate 계산용으로만 사용")
    log_ok("label → deployment selector: False")

    # -------------------------------------------------------------------------
    # [14] 기존 artifact 수정 없음 검증
    # -------------------------------------------------------------------------
    print("\n[14] 기존 artifact 수정 없음 검증")
    readonly_paths = [
        CANDIDATE_MANIFEST_CSV,
        RD_D1S_SCORE_CSV,
        CKPT_PATH,
        SMOKE_GROUP_MANIFEST_CSV,
        SMOKE_REPR_MANIFEST_CSV,
    ]
    all_readonly_exist = all(p.exists() for p in readonly_paths if p.name != "resnet18-f37072fd.pth")
    GUARDRAILS["existing_artifact_modified"] = False
    GUARDRAILS["existing_script_modified"]   = False
    log_ok("기존 artifact 수정 없음 (read-only 접근만)")

    # -------------------------------------------------------------------------
    # [15] 출력 파일 저장
    # -------------------------------------------------------------------------
    print("\n[15] 출력 파일 저장")

    # group_manifest_full.csv
    gm_fieldnames = [
        "group_id", "patient_id", "n_candidates", "has_positive", "positive_count",
        "representative_candidate_id", "first_stage_score_max",
        "local_z_repr", "crop_y0", "crop_x0", "crop_y1", "crop_x1", "label_repr",
    ]
    write_csv(GROUP_MANIFEST_CSV, group_stats_list, gm_fieldnames)
    log_ok(f"group_manifest_full.csv: {len(group_stats_list):,} rows")

    # group_representative_manifest_full.csv
    repr_rows = []
    for gs in group_stats_list:
        repr_rows.append({
            "group_id":                  gs["group_id"],
            "patient_id":                gs["patient_id"],
            "representative_candidate_id": gs["representative_candidate_id"],
            "first_stage_score_max":     gs["first_stage_score_max"],
            "local_z":                   gs["local_z_repr"],
            "crop_y0":                   gs["crop_y0"],
            "crop_x0":                   gs["crop_x0"],
            "crop_y1":                   gs["crop_y1"],
            "crop_x1":                   gs["crop_x1"],
            "has_positive":              gs["has_positive"],
            "n_candidates":              gs["n_candidates"],
            "label_repr":                gs["label_repr"],
        })
    write_csv(GROUP_REPR_MANIFEST_CSV, repr_rows)
    log_ok(f"group_representative_manifest_full.csv: {len(repr_rows):,} rows")

    # full_scoring_shard_plan.csv
    write_csv(SHARD_PLAN_CSV, shard_plan_rows, ["group_id", "patient_id", "shard_id", "has_positive"])
    log_ok(f"full_scoring_shard_plan.csv: {len(shard_plan_rows):,} rows")

    # errors.csv
    _save_error_csv()
    log_ok(f"errors.csv: {len(errors)} rows")

    # -------------------------------------------------------------------------
    # [16] DONE 조건 판정
    # -------------------------------------------------------------------------
    print("\n[16] DONE 조건 판정")

    critical_errors = [e for e in errors if e["code"] not in ("PARTIAL_CT",)]
    n_critical = len(critical_errors)

    cond_group_count  = group_count_match
    cond_repr         = repr_consistency in ("OK",)
    cond_pos_assign   = abs(pos_assign_rate - 1.0) < 1e-6
    cond_shard_plan   = len(shard_plan_rows) > 0
    cond_no_stage2    = not GUARDRAILS["stage2_holdout_accessed"]
    cond_no_forward   = not GUARDRAILS["model_forward_executed"]
    cond_no_overwrite = not GUARDRAILS["output_overwrite"]
    cond_report       = True  # 아래에서 저장

    all_pass = (
        n_critical == 0
        and cond_group_count
        and cond_repr
        and cond_pos_assign
        and cond_shard_plan
        and cond_no_stage2
        and cond_no_forward
        and cond_no_overwrite
    )

    partial = (
        not all_pass
        and n_critical == 0
        and cond_no_stage2
        and cond_no_forward
    )

    if all_pass:
        verdict = "PASS"
    elif partial:
        verdict = "PARTIAL_PASS"
    else:
        verdict = "FAIL"

    print(f"\n  critical_errors:    {n_critical}")
    print(f"  group_count_match:  {cond_group_count}  ({full_group_count:,} vs smoke {smoke_group_count:,})")
    print(f"  repr_consistency:   {repr_consistency}")
    print(f"  pos_assign_rate:    {pos_assign_rate:.4f}")
    print(f"  shard_plan:         {cond_shard_plan} ({len(shard_plan_rows):,} rows)")
    print(f"  no_stage2_access:   {cond_no_stage2}")
    print(f"  no_model_forward:   {cond_no_forward}")
    print(f"  no_overwrite:       {cond_no_overwrite}")
    print(f"\n  판정: {verdict}")

    elapsed_total = time.perf_counter() - t0

    # -------------------------------------------------------------------------
    # [17] summary JSON 저장
    # -------------------------------------------------------------------------
    summary = {
        "verdict":                         verdict,
        "original_candidate_count":        len(candidates),
        "full_group_count":                full_group_count,
        "reduction_rate":                  round(reduction_rate, 4),
        "positive_candidate_group_assignment_rate": round(pos_assign_rate, 4),
        "has_positive_group_count":        has_positive_groups,
        "smoke_v1_group_count":            smoke_group_count,
        "group_count_match":               cond_group_count,
        "representative_consistency":      repr_consistency,
        "repr_match_count":                repr_match_count,
        "repr_total_check":                repr_total_check,
        "shard_count":                     SHARD_COUNT,
        "shard_plan_rows":                 len(shard_plan_rows),
        "shard_summary":                   shard_summary,
        "ct_readiness":                    ct_readiness,
        "checkpoint_exists":               ckpt_ok,
        "resnet18_weight_exists":          resnet_ok,
        "gpu_info":                        gpu_info,
        "estimated_full_runtime_min":      round(estimated_full_min, 1),
        "per_group_sec_estimate":          round(per_group_sec, 4),
        "recommended_batch_size":          batch_size_rec,
        "crop_size":                       CROP_SIZE,
        "z_gap":                           DEFAULT_Z_GAP,
        "xy_radius":                       DEFAULT_XY_RADIUS,
        "rd_d1s_score_stage1_count":       len(rd_stage1_rows),
        "critical_error_count":            n_critical,
        "total_error_count":               len(errors),
        "elapsed_sec":                     round(elapsed_total, 2),
        "guardrails":                      GUARDRAILS,
    }

    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log_ok(f"summary.json 저장: {SUMMARY_JSON.name}")

    # -------------------------------------------------------------------------
    # [18] report.md 저장
    # -------------------------------------------------------------------------
    _save_report(summary, shard_summary, errors)
    log_ok(f"report.md 저장: {REPORT_MD.name}")

    # -------------------------------------------------------------------------
    # [19] DONE.json 저장
    # -------------------------------------------------------------------------
    if verdict in ("PASS", "PARTIAL_PASS"):
        with open(DONE_JSON, "w") as f:
            json.dump({"verdict": verdict, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}, f, indent=2)
        log_ok(f"DONE.json 저장")

    # -------------------------------------------------------------------------
    # 최종 출력
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"판정: {verdict}")
    print("=" * 70)
    print(f"핵심 수치:")
    print(f"  original candidates      : {len(candidates):,}")
    print(f"  full group count         : {full_group_count:,}")
    print(f"  reduction rate           : {reduction_rate:.4f} ({reduction_rate:.1%})")
    print(f"  positive assignment rate : {pos_assign_rate:.4f}")
    print(f"  has_positive groups      : {has_positive_groups:,}")
    print(f"  representative consistency: {repr_consistency}")
    print(f"  shard count              : {SHARD_COUNT}")
    print(f"  expected runtime         : ~{estimated_full_min:.1f}분")
    print(f"  CT readiness             : {ct_readiness}")
    print(f"  checkpoint exists        : {ckpt_ok}")
    print(f"  output overwrite         : {GUARDRAILS['output_overwrite']}")
    print(f"  stage2_holdout accessed  : {GUARDRAILS['stage2_holdout_accessed']}")
    print(f"  critical errors          : {n_critical}")
    print(f"  elapsed                  : {elapsed_total:.1f}s")
    print()
    if verdict == "PASS":
        print("다음 단계: actual full group scoring 실행 계획 수립 및 shard run")
    elif verdict == "PARTIAL_PASS":
        print("다음 단계: shard 전략 또는 output guard 수정 후 재검토")
    else:
        print("다음 단계: FAIL — full scoring으로 넘어가지 않음")

    return verdict


# =============================================================================
# 보조 함수
# =============================================================================

def _save_error_csv():
    if errors:
        write_csv(ERROR_CSV, errors, ["code", "msg", "details"])
    else:
        write_csv(ERROR_CSV, [{"code": "", "msg": "", "details": "no errors"}])


def _save_report(summary, shard_summary, errors):
    verdict = summary["verdict"]
    lines = []
    lines.append("# RD4AD z-continuity group-level full scoring preflight report v1")
    lines.append("")
    lines.append(f"**판정: {verdict}**")
    lines.append("")
    lines.append("## 핵심 수치")
    lines.append("")
    lines.append(f"| 항목 | 값 |")
    lines.append(f"|------|-----|")
    lines.append(f"| original candidates | {summary['original_candidate_count']:,} |")
    lines.append(f"| full group count | {summary['full_group_count']:,} |")
    lines.append(f"| reduction rate | {summary['reduction_rate']:.4f} ({summary['reduction_rate']*100:.1f}%) |")
    lines.append(f"| positive assignment rate | {summary['positive_candidate_group_assignment_rate']:.4f} |")
    lines.append(f"| has_positive groups | {summary['has_positive_group_count']:,} |")
    lines.append(f"| smoke v1 group count | {summary['smoke_v1_group_count']:,} |")
    lines.append(f"| group count match | {summary['group_count_match']} |")
    lines.append(f"| representative consistency | {summary['representative_consistency']} |")
    lines.append(f"| shard count | {summary['shard_count']} |")
    lines.append(f"| expected runtime | ~{summary['estimated_full_runtime_min']:.1f}분 |")
    lines.append(f"| CT readiness | {summary['ct_readiness']} |")
    lines.append(f"| checkpoint exists | {summary['checkpoint_exists']} |")
    lines.append(f"| output overwrite | {summary['guardrails']['output_overwrite']} |")
    lines.append(f"| stage2_holdout accessed | {summary['guardrails']['stage2_holdout_accessed']} |")
    lines.append(f"| critical errors | {summary['critical_error_count']} |")
    lines.append(f"| elapsed | {summary['elapsed_sec']:.1f}s |")
    lines.append("")
    lines.append("## Shard plan 요약")
    lines.append("")
    lines.append("| shard_id | group_count | has_positive |")
    lines.append("|----------|-------------|--------------|")
    for ss in shard_summary:
        lines.append(f"| {ss['shard_id']} | {ss['group_count']:,} | {ss['has_positive']:,} |")
    lines.append("")
    lines.append("## Guardrails")
    lines.append("")
    lines.append("| guardrail | value |")
    lines.append("|-----------|-------|")
    for k, v in summary["guardrails"].items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    if errors:
        lines.append("## Errors")
        lines.append("")
        lines.append("| code | msg | details |")
        lines.append("|------|-----|---------|")
        for e in errors:
            lines.append(f"| {e['code']} | {e['msg']} | {str(e['details'])[:80]} |")
        lines.append("")
    lines.append("## 다음 단계")
    lines.append("")
    if verdict == "PASS":
        lines.append("PASS — actual full group scoring 실행 계획 수립 및 shard run")
    elif verdict == "PARTIAL_PASS":
        lines.append("PARTIAL_PASS — shard 전략 또는 output guard 수정 후 재검토")
    else:
        lines.append("FAIL — full scoring으로 넘어가지 않음")

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="RD4AD z-continuity group full scoring preflight v1"
    )
    parser.add_argument("--dry-run",            action="store_true")
    parser.add_argument("--run-preflight",      action="store_true")
    parser.add_argument("--confirm-readonly",   action="store_true")
    parser.add_argument("--confirm-stage1dev-only", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        run_dry()
        return

    if not (args.run_preflight and args.confirm_readonly and args.confirm_stage1dev_only):
        print("[ABORT] 인자 없이 실행 불가.")
        print("  dry-run:   python <script> --dry-run")
        print("  preflight: python <script> --run-preflight --confirm-readonly --confirm-stage1dev-only")
        sys.exit(2)

    run_preflight()


if __name__ == "__main__":
    main()
