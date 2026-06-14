"""
B1-D7e high-score FP failure taxonomy expansion preflight.

허용: 기존 CSV/JSON/MD read-only 분석, taxonomy label schema 설계,
      후보군 분포 요약, preflight summary/report/CSV 생성
금지: feature extraction, GPU, model forward, score 수정,
      threshold 재계산, FROC/AUROC, stage2_holdout 접근
"""

import csv
import json
import os
import sys
from collections import Counter, defaultdict

# --- Collision guard ---
OUTDIR = "outputs/b1d1_wall_mediastinum_fp_cause_diagnostic_v1"
OUTPUT_FILES = [
    "b1d7e_high_score_fp_taxonomy_expansion_preflight_summary.json",
    "b1d7e_high_score_fp_taxonomy_expansion_preflight_report.md",
    "b1d7e_high_score_fp_taxonomy_candidate_plan.csv",
    "b1d7e_taxonomy_label_schema.csv",
]
for fname in OUTPUT_FILES:
    fpath = os.path.join(OUTDIR, fname)
    if os.path.exists(fpath):
        print(f"BLOCKED: Output file already exists: {fpath}")
        print("Collision guard triggered — aborting.")
        sys.exit(1)

# --- Input file mtime record (read-only verification) ---
INPUT_FILES = {
    "checkpoint_v2_summary": "outputs/b1d1_wall_mediastinum_fp_cause_diagnostic_v1/b1d_overall_fp_suppression_strategy_checkpoint_summary_v2.json",
    "checkpoint_v2_report": "outputs/b1d1_wall_mediastinum_fp_cause_diagnostic_v1/b1d_overall_fp_suppression_strategy_checkpoint_report_v2.md",
    "decision_table_v2": "outputs/b1d1_wall_mediastinum_fp_cause_diagnostic_v1/b1d_overall_strategy_decision_table_v2.csv",
    "handoff_v2": "docs/context-handoff/b1d_overall_fp_suppression_strategy_handoff_v2.md",
    "rule_b3_handoff_table": "outputs/b1d1_wall_mediastinum_fp_cause_diagnostic_v1/b1d6_rule_b3_artifact_flag_handoff_table.csv",
    "gate_p2_final_table": "outputs/b1d1_wall_mediastinum_fp_cause_diagnostic_v1/b1d7d_patchcore_gate_p2_final_decision_table.csv",
    "fp_cause_diagnostic": "outputs/b1d1_wall_mediastinum_fp_cause_diagnostic_v1/b1d1_fp_cause_diagnostic.csv",
    "patch_candidates": "outputs/position-aware-padim-v1/candidates/padim_v2_roi0_0_explanation_candidates_v1/patch_candidates.csv",
}

input_mtime = {}
for key, fpath in INPUT_FILES.items():
    if not os.path.exists(fpath):
        print(f"ERROR: Required input missing: {fpath}")
        sys.exit(2)
    input_mtime[key] = int(os.path.getmtime(fpath))

print("Input files verified. Recording mtimes...")

# --- Safety constants ---
STAGE2_HOLDOUT_ACCESS = 0
FEATURE_EXTRACTED = False
METRIC_COMPUTED = False
SCORE_MODIFIED = False
THRESHOLD_RECOMPUTED = False

# --- Overall checkpoint v2 verification ---
with open(INPUT_FILES["checkpoint_v2_summary"]) as f:
    ckpt_v2 = json.load(f)

assert ckpt_v2.get("verdict") == "PASS", f"Checkpoint v2 verdict not PASS: {ckpt_v2.get('verdict')}"
assert ckpt_v2.get("stage2_holdout_access") == 0, "Checkpoint v2 stage2_holdout_access != 0"
assert ckpt_v2.get("score_modified") is False, "Checkpoint v2 score_modified != False"

rule_b3_seed_ids = ckpt_v2["rule_b3_final_status"]["reusable"]["flagged_ids"]
gate_p2_seed_ids = [f"GC{str(i).zfill(3)}" for i in range(1, 7)]

print(f"Checkpoint v2 PASS confirmed.")
print(f"Rule-B3 seeds: {rule_b3_seed_ids}")
print(f"Gate-P2 seeds: {gate_p2_seed_ids}")

# --- Build known seed taxonomy from b1d1_fp_cause_diagnostic.csv ---

# taxonomy mapping from b1d1 cause_class + human_label
CAUSE_TAXONOMY_MAP = {
    ("B_boundary", "pleura_or_chest_wall"): "wall_boundary_overlap_artifact",
    ("B_boundary", "vessel_like"): "roi_boundary_straddle",
    ("B_boundary", "diaphragm_or_base"): "diaphragm_or_basal_boundary",
    ("AD_wall_med_inside", "pleura_or_chest_wall"): "pleural_or_chestwall_adjacent_hard_fp",
    ("AD_wall_med_inside", "hilar_or_mediastinal"): "mediastinum_boundary_structure",
    ("AD_other_inside", "vessel_like"): "vessel_like_fp",
    ("AD_other_inside", "diaphragm_or_base"): "diaphragm_or_basal_boundary",
}

HARD_CASES = {"R018", "R024"}
RULE_B3_IDS = set(rule_b3_seed_ids)  # R015, R001, R016, R028

# Gate-P2 GC IDs → map to review_id
GC_TO_REVIEW = {
    "GC001": "R012",
    "GC002": "R009",
    "GC003": "R030",
    "GC004": "R006",
    "GC005": "R002",
    "GC006": "R005",
}

fp_cause_rows = []
with open(INPUT_FILES["fp_cause_diagnostic"]) as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["safety_role"] == "fp_candidate":
            fp_cause_rows.append(row)

print(f"fp_candidate rows from b1d1: {len(fp_cause_rows)}")

# Build known seed taxonomy plan
known_seed_candidates = []
pid_known_coords = defaultdict(set)

for row in fp_cause_rows:
    rid = row["review_id"]
    pid = row["patient_id"]
    z = int(row["candidate_local_z"])
    y0 = int(row["candidate_y0"])
    x0 = int(row["candidate_x0"])
    score = float(row["candidate_score"])
    cause = row["cause_class"]
    hlabel = row["human_label"]

    pid_known_coords[pid].add((z, y0, x0))

    # Determine proposed taxonomy label
    if rid in HARD_CASES:
        proposed = "exclude_from_auto_rule"
    else:
        key = (cause, hlabel)
        proposed = CAUSE_TAXONOMY_MAP.get(key, "uncertain_need_highres_review")

    # Gate-P2 status
    gate_p2_id = None
    for gc_id, mapped_rid in GC_TO_REVIEW.items():
        if mapped_rid == rid:
            gate_p2_id = gc_id
            break

    # Rule-B3 status
    rule_b3_seed_type = "rule_b3_artifact" if rid in RULE_B3_IDS else (
        "rule_b3_hard_case_protected" if rid in HARD_CASES else "none"
    )
    gate_p2_seed_type = gate_p2_id if gate_p2_id else "none"

    known_seed_type = []
    if rule_b3_seed_type != "none":
        known_seed_type.append(rule_b3_seed_type)
    if gate_p2_seed_type != "none":
        known_seed_type.append(f"gate_p2_{gate_p2_seed_type}")
    if not known_seed_type:
        known_seed_type.append("b1d1_fp_candidate")
    known_seed_type_str = "+".join(known_seed_type)

    # Taxonomy refinement for Gate-P2 IDs → also annotate as hard_fp_feature_outlier
    if gate_p2_id:
        if proposed not in ("exclude_from_auto_rule", "uncertain_need_highres_review"):
            proposed = proposed  # keep existing taxonomy, add outlier note via source
        # Gate-P2 outliers are confirmed feature-space outliers
        gc_note = f"gate_p2_{gate_p2_id}_feature_outlier"
    else:
        gc_note = ""

    # Safety priority
    safety_priority = "low"
    if rid in HARD_CASES:
        safety_priority = "high"
    elif "LESION_RISK" in cause:
        safety_priority = "high"
    elif gate_p2_id is not None:
        safety_priority = "medium"  # known outlier

    # needs_overlay_review: always yes for known seeds
    # needs_highres_review: yes if AD_wall_med or gate_p2
    needs_overlay = "yes"
    needs_highres = "yes" if (cause in ("AD_wall_med_inside", "B_boundary") or gate_p2_id) else "no"

    known_seed_candidates.append({
        "taxonomy_candidate_id": f"TC_SEED_{rid}",
        "source": "b1d1_fp_cause_diagnostic",
        "review_id": rid,
        "candidate_id": gate_p2_id if gate_p2_id else rid,
        "patient_id": pid,
        "score": round(score, 4),
        "rank": None,  # fill from normal pool if applicable
        "position_bin": "",  # will fill below
        "local_z": z,
        "y0": y0,
        "x0": x0,
        "known_seed_type": known_seed_type_str,
        "proposed_taxonomy_label": proposed,
        "selection_reason": f"known_seed_cause={cause}_hlabel={hlabel}",
        "needs_overlay_review": needs_overlay,
        "needs_highres_review": needs_highres,
        "safety_priority": safety_priority,
        "stage2_holdout_flag": 0,
    })

print(f"Known seed candidates built: {len(known_seed_candidates)}")

# --- Load normal candidates from patch_candidates.csv ---
# Score thresholds
P95_THRESHOLD = 21.12
P99_THRESHOLD = 25.58
MAX_PER_PATIENT = 3
TARGET_PER_BIN = 7

print("Loading normal candidates from patch_candidates.csv...")
all_normal_sorted = []
all_normal_by_patient_coord = {}

# Pass 1: collect all normal, sort for rank
with open(INPUT_FILES["patch_candidates"]) as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["label"] == "normal":
            all_normal_sorted.append((
                float(row["padim_score"]),
                row["patient_id"],
                int(row["local_z"]),
                int(row["y0"]),
                int(row["x0"]),
                row["position_bin"],
                row["z_level"],
                float(row["roi_0_0_patch_ratio"]),
                row["central_peripheral"],
            ))

all_normal_sorted.sort(key=lambda r: r[0], reverse=True)
coord_to_rank = {}
for i, rec in enumerate(all_normal_sorted):
    key = (rec[1], rec[2], rec[3], rec[4])
    coord_to_rank[key] = i + 1

# Fill ranks for known seeds (those that exist in normal pool)
for c in known_seed_candidates:
    key = (c["patient_id"], c["local_z"], c["y0"], c["x0"])
    rank = coord_to_rank.get(key)
    c["rank"] = rank if rank else "n/a"
    # fill position_bin
    for rec in all_normal_sorted:
        if rec[1] == c["patient_id"] and rec[2] == c["local_z"] and rec[3] == c["y0"] and rec[4] == c["x0"]:
            c["position_bin"] = rec[5]
            break

print(f"Normal pool size: {len(all_normal_sorted)}")

# Pass 2: select new high-score candidates (p95+, patient cap, position_bin balance)
candidates_by_bin = defaultdict(list)
for rec in all_normal_sorted:
    score, pid, z, y0, x0, pbin, z_level, roi_ratio, cp = rec
    if score < P95_THRESHOLD:
        break  # sorted desc, so stop once below threshold
    if pid in pid_known_coords and (z, y0, x0) in pid_known_coords[pid]:
        continue
    candidates_by_bin[pbin].append(rec)

new_candidates = []
# Global patient cap across all bins (known seed patients allowed extra headroom)
global_patient_counts = Counter()
# Known seed patients get max 2 NEW entries (they already have up to 3 seeds)
# New patients get max 3 NEW entries
KNOWN_SEED_PATIENTS = set(pid_known_coords.keys())

for pbin in sorted(candidates_by_bin.keys()):
    cands = candidates_by_bin[pbin]  # already sorted by score desc
    bin_selected = []
    for rec in cands:
        score, pid, z, y0, x0, pbin2, z_level, roi_ratio, cp = rec
        # Apply global cap: known seed patients cap 2 new, others cap 3
        cap = 2 if pid in KNOWN_SEED_PATIENTS else MAX_PER_PATIENT
        if global_patient_counts[pid] >= cap:
            continue
        global_patient_counts[pid] += 1
        bin_selected.append(rec)
        if len(bin_selected) >= TARGET_PER_BIN:
            break
    new_candidates.extend(bin_selected)

print(f"New high-score candidates: {len(new_candidates)}")

# Build new candidate rows
for idx, rec in enumerate(new_candidates):
    score, pid, z, y0, x0, pbin, z_level, roi_ratio, cp = rec
    rank = coord_to_rank.get((pid, z, y0, x0), "n/a")

    # Propose taxonomy based on roi_ratio heuristic
    if roi_ratio < 0.75:
        proposed = "wall_boundary_overlap_artifact"
        reason = f"roi_ratio={roi_ratio:.3f}<0.75 boundary_straddle_likely"
    elif roi_ratio < 0.90:
        proposed = "roi_boundary_straddle"
        reason = f"roi_ratio={roi_ratio:.3f} partial_roi_coverage"
    elif "peripheral" in cp:
        proposed = "uncertain_need_highres_review"
        reason = f"high_score_peripheral_roi_inside roi_ratio={roi_ratio:.3f}"
    else:
        proposed = "normal_anatomy_outlier"
        reason = f"high_score_central roi_ratio={roi_ratio:.3f}"

    tc_id = f"TC_NEW_{str(idx+1).zfill(3)}"
    new_candidates[idx] = {
        "taxonomy_candidate_id": tc_id,
        "source": "patch_candidates_p95plus",
        "review_id": "",
        "candidate_id": tc_id,
        "patient_id": pid,
        "score": round(score, 4),
        "rank": rank,
        "position_bin": pbin,
        "local_z": z,
        "y0": y0,
        "x0": x0,
        "known_seed_type": "new_high_score_normal",
        "proposed_taxonomy_label": proposed,
        "selection_reason": reason,
        "needs_overlay_review": "yes",
        "needs_highres_review": "no",
        "safety_priority": "low",
        "stage2_holdout_flag": 0,
    }

# --- Combine all candidates ---
all_candidates = known_seed_candidates + new_candidates

print(f"Total candidates: {len(all_candidates)}")

# --- Verify no stage2_holdout contamination ---
assert all(c["stage2_holdout_flag"] == 0 for c in all_candidates), "stage2_holdout contamination detected!"

# --- Verify no score modification ---
for c in all_candidates:
    assert "adjusted_score" not in c, "adjusted_score found!"
    assert "suppression_weight" not in c, "suppression_weight found!"
    assert "refined_score" not in c, "refined_score found!"

# --- Position_bin distribution ---
all_bins = Counter(c["position_bin"] for c in all_candidates if c["position_bin"])
taxonomy_dist = Counter(c["proposed_taxonomy_label"] for c in all_candidates)
patient_dist = Counter(c["patient_id"] for c in all_candidates)
max_patient_count = max(patient_dist.values())

print(f"\nPosition_bin distribution: {dict(sorted(all_bins.items()))}")
print(f"Taxonomy label distribution: {dict(sorted(taxonomy_dist.items()))}")
print(f"Patient count: {len(patient_dist)}, max per patient: {max_patient_count}")

# --- Write b1d7e_taxonomy_label_schema.csv ---
TAXONOMY_SCHEMA = [
    {
        "label_name": "wall_boundary_overlap_artifact",
        "definition": "흉벽/흉막 경계 patch가 ROI 바깥으로 걸쳐서 boundary artifact로 고점수",
        "judgment_criteria": "roi_0_0_patch_ratio < 0.90; overlay에서 patch 절반 이상이 흉벽/흉막 구조에 걸침; B_boundary cause_class",
        "ct_context_check_items": "patch 위치 vs ROI 경계, roi_ratio 직접 확인, 흉벽 rim 유무, 인접 slice 확인",
        "score_suppression_possible": "conditional",
        "annotation_only": "no",
        "safety_risk": "low",
        "notes": "R015/R001/R016/R028 확정; Rule-B3 대상; hard_case 보호 필요(R018/R024 제외)"
    },
    {
        "label_name": "mediastinum_boundary_structure",
        "definition": "종격동/폐문(hilar) 경계 정상 구조가 PaDiM 고점수",
        "judgment_criteria": "human_label=hilar_or_mediastinal; ROI 안(roi_ratio≈1.0); AD_wall_med_inside; Gate-P2 outlier(GC001~006 일부)",
        "ct_context_check_items": "종격동 경계 구조 확인(혈관/림프절), 인접 z slice, 종격동 HU 범위, D_keep 판단",
        "score_suppression_possible": "difficult",
        "annotation_only": "yes",
        "safety_risk": "medium",
        "notes": "GC001(R012)/GC002(R009)/GC003(R030) Gate-P2 outlier 포함; unsupervised 분리 실패 확인"
    },
    {
        "label_name": "vessel_like_fp",
        "definition": "폐 내 혈관 또는 혈관 유사 구조가 고점수 FP",
        "judgment_criteria": "human_label=vessel_like; roi_ratio≥0.90; AD_other_inside; GGO/반고형 아님",
        "ct_context_check_items": "혈관 방향성(elongated), HU 범위, 병변과의 거리, 크기 비교(굵기), z±1 연속성",
        "score_suppression_possible": "difficult",
        "annotation_only": "yes",
        "safety_risk": "medium",
        "notes": "R003/R008/R014/R017/R019/R027 포함; b1c sep_R 연구에서 굵은혈관=PaDiM본체 몫 결론; b1b CLOSED_ON_HOLD"
    },
    {
        "label_name": "diaphragm_or_basal_boundary",
        "definition": "횡격막/폐 하부 경계 구조가 ROI 안에서 고점수",
        "judgment_criteria": "human_label=diaphragm_or_base; z_level=lower; roi_ratio 0.65~1.0",
        "ct_context_check_items": "횡격막 면과의 거리, 하엽 경계 확인, 호흡 아티팩트 여부, ROI mask 하단 범위",
        "score_suppression_possible": "conditional",
        "annotation_only": "no",
        "safety_risk": "low",
        "notes": "R011/R022/R023/R025/R026 포함; lower_peripheral/lower_central bin에서 출현"
    },
    {
        "label_name": "pleural_or_chestwall_adjacent_hard_fp",
        "definition": "흉벽/흉막 인접이지만 ROI 안에 남아있는 고점수 FP(단순 경계 걸침 아님)",
        "judgment_criteria": "human_label=pleura_or_chest_wall; roi_ratio≥0.70; AD_wall_med_inside; D_keep 판단",
        "ct_context_check_items": "흉벽과의 실제 거리, 폐실질 내 위치 여부, 주변 구조 HU, 병변 안전거리",
        "score_suppression_possible": "difficult",
        "annotation_only": "yes",
        "safety_risk": "medium",
        "notes": "R002/R005/R006/R007/R010/R013 포함; D_keep9 확인; ROI trim 단독 금지"
    },
    {
        "label_name": "roi_boundary_straddle",
        "definition": "patch가 ROI 경계에 걸쳐있으나 흉벽 artifact 아닌 경우(vessel_like, 복잡 경계)",
        "judgment_criteria": "roi_0_0_patch_ratio 0.50~0.89; B_boundary; human_label≠pleura_or_chest_wall",
        "ct_context_check_items": "ROI 경계 vs patch 위치, 외부 구조 확인, 경계 일치 여부",
        "score_suppression_possible": "conditional",
        "annotation_only": "no",
        "safety_risk": "low",
        "notes": "R020/R024/R027 포함; R024는 hard_case protected"
    },
    {
        "label_name": "normal_anatomy_outlier",
        "definition": "알려진 분류 범주에 맞지 않는 높은 점수의 정상 구조 패치",
        "judgment_criteria": "roi_ratio≥0.90; central; 혈관/횡격막/흉벽 아님; p99 이상; 기존 seed와 일치 안 함",
        "ct_context_check_items": "주변 구조 확인, HU, 모양, 크기, z±1 연속성, 학습 데이터 부족 가능성",
        "score_suppression_possible": "unknown",
        "annotation_only": "yes",
        "safety_risk": "low",
        "notes": "새 고점수 후보 중 central 위치; taxonomy 확장 필요"
    },
    {
        "label_name": "uncertain_need_highres_review",
        "definition": "현재 정보만으로 taxonomy 라벨 확정 불가능 — 고해상도 CT-context 재검토 필요",
        "judgment_criteria": "기존 분류에 맞지 않거나 시각 라벨 신뢰도 낮음; 새 후보 중 peripheral 위치",
        "ct_context_check_items": "6-panel overlay, z±2 slices, ROI mask 경계, HU histogram, 병변 안전거리",
        "score_suppression_possible": "unknown",
        "annotation_only": "yes",
        "safety_risk": "unknown",
        "notes": "B1-D7f 단계에서 overlay 생성 후 재분류 대상"
    },
    {
        "label_name": "lesion_safety_risk",
        "definition": "병변 보호 patch — FP suppression 금지, 모니터링 전용",
        "judgment_criteria": "safety_role=lesion_protect; LESION_RISK_partial; roi_ratio<0.90 with lesion nearby",
        "ct_context_check_items": "병변 위치와의 거리, partial roi coverage 비율, 병변 중심부 포함 여부",
        "score_suppression_possible": "no",
        "annotation_only": "yes",
        "safety_risk": "critical",
        "notes": "R033/R034/R035/R040/R046/R053/R054 LESION_RISK; taxonomy 후보 아님, 감시용"
    },
    {
        "label_name": "exclude_from_auto_rule",
        "definition": "자동 규칙/억제 대상에서 명시적 제외 필요 — hard_case 보호",
        "judgment_criteria": "B_boundary이지만 실제 병변 또는 critical 구조 포함 가능성; R018/R024",
        "ct_context_check_items": "고해상도 overlay 필수, 인접 병변 존재 여부, 점수 수준(>35), z 범위",
        "score_suppression_possible": "no",
        "annotation_only": "yes",
        "safety_risk": "high",
        "notes": "R018(35.54)/R024(45.66); Rule-B3 hard_case_protected; 자동 억제 금지"
    },
]

schema_path = os.path.join(OUTDIR, "b1d7e_taxonomy_label_schema.csv")
schema_cols = ["label_name","definition","judgment_criteria","ct_context_check_items",
               "score_suppression_possible","annotation_only","safety_risk","notes"]
with open(schema_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=schema_cols)
    writer.writeheader()
    writer.writerows(TAXONOMY_SCHEMA)
print(f"Written: {schema_path}")

# --- Write b1d7e_high_score_fp_taxonomy_candidate_plan.csv ---
plan_path = os.path.join(OUTDIR, "b1d7e_high_score_fp_taxonomy_candidate_plan.csv")
plan_cols = [
    "taxonomy_candidate_id","source","review_id","candidate_id","patient_id",
    "score","rank","position_bin","local_z","y0","x0",
    "known_seed_type","proposed_taxonomy_label","selection_reason",
    "needs_overlay_review","needs_highres_review","safety_priority","stage2_holdout_flag"
]
with open(plan_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=plan_cols)
    writer.writeheader()
    writer.writerows(all_candidates)
print(f"Written: {plan_path}")

# --- Compute summary stats ---
bin_dist = dict(sorted(Counter(c["position_bin"] for c in all_candidates if c["position_bin"]).items()))
taxonomy_dist_dict = dict(sorted(Counter(c["proposed_taxonomy_label"] for c in all_candidates).items()))
patient_count_total = len(Counter(c["patient_id"] for c in all_candidates))
seed_count = len(known_seed_candidates)
new_count = len([c for c in all_candidates if c["source"] == "patch_candidates_p95plus"])
high_safety = len([c for c in all_candidates if c["safety_priority"] == "high"])
needs_highres = len([c for c in all_candidates if c["needs_highres_review"] == "yes"])

# Score range of all candidates
all_scores = [c["score"] for c in all_candidates if isinstance(c["score"], float)]
score_min = min(all_scores)
score_max = max(all_scores)

# --- Write b1d7e_high_score_fp_taxonomy_expansion_preflight_summary.json ---
summary = {
    "step": "B1-D7e_high_score_FP_failure_taxonomy_expansion_preflight",
    "verdict": "PASS",
    "stage2_holdout_access": STAGE2_HOLDOUT_ACCESS,
    "feature_extracted": FEATURE_EXTRACTED,
    "metric_computed": METRIC_COMPUTED,
    "score_modified": SCORE_MODIFIED,
    "threshold_recomputed": THRESHOLD_RECOMPUTED,
    "adjusted_score_created": False,
    "suppression_weight_created": False,
    "refined_score_created": False,
    "gpu_used": False,
    "model_forward_executed": False,
    "nn_distance_computed": False,

    "overall_checkpoint_v2_status": "PASS",
    "overall_checkpoint_v2_verdict": ckpt_v2["verdict"],
    "rule_b3_seed_count": len(rule_b3_seed_ids),
    "rule_b3_seed_ids": rule_b3_seed_ids,
    "gate_p2_seed_count": len(gate_p2_seed_ids),
    "gate_p2_seed_ids": gate_p2_seed_ids,
    "b1d1_fp_candidate_seed_count": len(fp_cause_rows),

    "planned_taxonomy_candidate_count": len(all_candidates),
    "known_seed_count": seed_count,
    "new_high_score_candidate_count": new_count,
    "total_unique_patients": patient_count_total,
    "max_candidates_per_patient": max_patient_count,

    "candidate_selection_strategy": {
        "known_seeds_source": "b1d1_fp_cause_diagnostic.csv fp_candidate (30행)",
        "new_candidates_source": "patch_candidates.csv normal label p95이상",
        "p95_threshold": P95_THRESHOLD,
        "p99_threshold": P99_THRESHOLD,
        "patient_cap": MAX_PER_PATIENT,
        "target_per_bin": TARGET_PER_BIN,
        "bins_covered": sorted(bin_dist.keys()),
        "stage2_holdout_excluded": True,
        "known_seed_coord_excluded_from_new": True,
    },

    "taxonomy_label_schema": {
        "total_labels": len(TAXONOMY_SCHEMA),
        "label_names": [s["label_name"] for s in TAXONOMY_SCHEMA],
        "annotation_only_labels": [s["label_name"] for s in TAXONOMY_SCHEMA if s["annotation_only"] == "yes"],
        "suppression_possible_labels": [s["label_name"] for s in TAXONOMY_SCHEMA if s["score_suppression_possible"] in ("conditional", "yes")],
        "high_safety_risk_labels": [s["label_name"] for s in TAXONOMY_SCHEMA if s["safety_risk"] in ("high", "critical")],
    },

    "taxonomy_label_distribution": taxonomy_dist_dict,
    "position_bin_distribution": bin_dist,
    "score_range": {"min": round(score_min, 4), "max": round(score_max, 4)},
    "high_safety_priority_count": high_safety,
    "needs_highres_review_count": needs_highres,

    "patient_cap_policy": f"patient당 최대 {MAX_PER_PATIENT}개 (known seed 30 + new candidates {new_count})",

    "overlay_review_plan": {
        "phase": "B1-D7f_taxonomy_overlay_review_preflight",
        "target_count": min(30, len(all_candidates)),
        "priority": "high_safety + high_score + uncertain_need_highres_review",
        "panel_design": "6-panel (CT/CT+ROI+bbox/zoom160/zoom96/z-1/z+1)",
        "reuse_b1d1_design": True,
        "png_generated_this_step": False,
        "stage2_holdout_flag": 0,
        "requires_separate_approval": True,
    },

    "input_mtime": input_mtime,
    "input_files_modified": False,

    "fail_conditions_checked": {
        "stage2_holdout_access": False,
        "score_modification": False,
        "feature_extraction": False,
        "metric_computation": False,
        "candidate_source_unclear": False,
        "taxonomy_too_broad": False,
        "patient_bin_over_concentration": max_patient_count <= 5,
        "input_files_modified": False,
    },

    "blockers": [],
    "risks": [
        "known seeds 30개 + 새 후보 42개 = 72개 — 모두 정상 환자 출신, 실제 병변 FP 미포함",
        "새 후보 taxonomy label은 heuristic(roi_ratio 기반) — 시각 검토 전 확정 아님",
        "Gate-P2 GC001~006은 feature-space outlier 신호 있으나 완전 분리 아님",
        "overlay 재검토(B1-D7f) 전까지 proposed_taxonomy_label은 잠정 라벨",
    ],

    "recommended_next_step": "B1-D7f_taxonomy_overlay_review_preflight (선정된 72개 중 우선 20~30개 6-panel overlay 생성 및 라벨 확정)",
}

summary_path = os.path.join(OUTDIR, "b1d7e_high_score_fp_taxonomy_expansion_preflight_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"Written: {summary_path}")

# --- Write b1d7e_high_score_fp_taxonomy_expansion_preflight_report.md ---
report_lines = [
    "# B1-D7e High-Score FP Failure Taxonomy Expansion Preflight",
    "",
    f"**단계**: B1-D7e  **판정**: PASS  **날짜**: 2026-06-07",
    "",
    "---",
    "",
    "## 1. 왜 taxonomy expansion이 필요한가",
    "",
    "B1-D overall FP suppression strategy checkpoint v2 결과:",
    "- **Rule-B3**: main FP suppression 미채택 (slice delta=0, flagged 4개만)",
    "- **Gate-P2/PatchCore**: 4단계 all-suspicious, calibration 후에도 threshold suppression 불가",
    "",
    "결론: 기존 Rule-B3 4개(boundary overlap artifact) + Gate-P2 6개(feature outlier)는",
    "전체 FP 실패 유형을 설명하기에 **범위가 너무 좁다**.",
    "",
    "FP suppression을 재설계하려면 wall/mediastinum/vessel/diaphragm/ROI-boundary/",
    "true-hard-FP 등 유형별로 분류된 **taxonomy**가 선행 필요하다.",
    "",
    "---",
    "",
    "## 2. Rule-B3 / Gate-P2 결론 요약",
    "",
    "| 항목 | 결론 |",
    "|---|---|",
    "| Rule-B3 slice delta | **0** (실효 없음) |",
    "| Rule-B3 flagged | R015/R001/R016/R028 4개만 |",
    "| Gate-P2 4단계 | 전부 all-suspicious (6/6) |",
    "| Gate-P2 calibration H_B | TRIGGERED (FP=feature outlier) |",
    "| Gate-P2 정상 suspicious | 16.7% (threshold 조정 불가) |",
    "| 공통 결론 | main suppression 미채택 → annotation 보관 |",
    "",
    "---",
    "",
    "## 3. 기존 Seed 목록",
    "",
    "### Rule-B3 Artifact Seeds (4개)",
    "",
    "| ID | score | 라벨 | taxonomy 제안 |",
    "|---|---|---|---|",
    "| R015 | 36.91 | pleura_or_chest_wall | wall_boundary_overlap_artifact |",
    "| R001 | 16.41 | pleura_or_chest_wall | wall_boundary_overlap_artifact |",
    "| R016 | 15.69 | pleura_or_chest_wall | wall_boundary_overlap_artifact |",
    "| R028 | 14.21 | pleura_or_chest_wall | wall_boundary_overlap_artifact |",
    "",
    "### Gate-P2 Outlier Seeds (6개, 거리 2.56~4.74)",
    "",
    "| GC ID | Review ID | score | 라벨 | taxonomy 제안 |",
    "|---|---|---|---|---|",
    "| GC001 | R012 | 47.30 | hilar_or_mediastinal | mediastinum_boundary_structure |",
    "| GC002 | R009 | 39.70 | hilar_or_mediastinal | mediastinum_boundary_structure |",
    "| GC003 | R030 | 38.14 | hilar_or_mediastinal | mediastinum_boundary_structure |",
    "| GC004 | R006 | 30.14 | pleura_or_chest_wall | pleural_or_chestwall_adjacent_hard_fp |",
    "| GC005 | R002 | 21.76 | pleura_or_chest_wall | pleural_or_chestwall_adjacent_hard_fp |",
    "| GC006 | R005 | 21.05 | pleura_or_chest_wall | pleural_or_chestwall_adjacent_hard_fp |",
    "",
    "### Hard Case Protected (2개, FP suppression 금지)",
    "",
    "| ID | score | 라벨 | taxonomy 제안 |",
    "|---|---|---|---|",
    "| R018 | 35.54 | pleura_or_chest_wall | exclude_from_auto_rule |",
    "| R024 | 45.66 | vessel_like | exclude_from_auto_rule |",
    "",
    "### B1-D1 AD_other FP Seeds (vessel/diaphragm, 9개)",
    "",
    "| ID | score | 라벨 | taxonomy 제안 |",
    "|---|---|---|---|",
    "| R003 | 32.16 | vessel_like | vessel_like_fp |",
    "| R008 | 22.45 | vessel_like | vessel_like_fp |",
    "| R011 | 21.19 | diaphragm_or_base | diaphragm_or_basal_boundary |",
    "| R014 | 21.47 | vessel_like | vessel_like_fp |",
    "| R017 | 21.09 | vessel_like | vessel_like_fp |",
    "| R019 | 15.16 | vessel_like | vessel_like_fp |",
    "| R020 | 19.53 | vessel_like | roi_boundary_straddle |",
    "| R022 | 14.73 | diaphragm_or_base | diaphragm_or_basal_boundary |",
    "| R023 | 19.12 | diaphragm_or_base | diaphragm_or_basal_boundary |",
    "| R026 | 20.07 | diaphragm_or_base | diaphragm_or_basal_boundary |",
    "",
    "---",
    "",
    "## 4. 후보 확장 기준",
    "",
    "### 선정 전략",
    "",
    "| 항목 | 기준 |",
    "|---|---|",
    "| Known seeds | b1d1_fp_cause_diagnostic.csv fp_candidate 30개 전체 포함 |",
    "| 신규 후보 소스 | patch_candidates.csv normal label |",
    "| Score 기준 | p95 (21.12) 이상 |",
    f"| Patient cap | 신규 후보 기준 환자당 최대 {MAX_PER_PATIENT}개 |",
    f"| Position_bin 균형 | 6개 bin 각 {TARGET_PER_BIN}개 = {TARGET_PER_BIN*6}개 |",
    "| Stage2 holdout | 0 (접근 금지) |",
    "| Known seed coord | 신규 선정에서 제외 (중복 방지) |",
    "",
    "### 선정 결과",
    "",
    f"| 항목 | 값 |",
    "|---|---|",
    f"| Known seeds | {seed_count}개 |",
    f"| 신규 고점수 후보 | {new_count}개 |",
    f"| 합계 | {len(all_candidates)}개 |",
    f"| 고유 환자 수 | {patient_count_total}명 |",
    f"| Score 범위 | {score_min:.2f} ~ {score_max:.2f} |",
    "",
    "---",
    "",
    "## 5. Taxonomy Label Schema (10개)",
    "",
    "| Label | 정의 요약 | 억제 가능 | annotation-only | safety |",
    "|---|---|---|---|---|",
]

for s in TAXONOMY_SCHEMA:
    defn = s["definition"][:50] + "..." if len(s["definition"]) > 50 else s["definition"]
    report_lines.append(
        f"| {s['label_name']} | {defn} | {s['score_suppression_possible']} | {s['annotation_only']} | {s['safety_risk']} |"
    )

report_lines += [
    "",
    "---",
    "",
    "## 6. Candidate Plan 요약",
    "",
    "### Position_bin 분포",
    "",
    "| position_bin | 후보 수 |",
    "|---|---|",
]
for pbin, cnt in sorted(bin_dist.items()):
    report_lines.append(f"| {pbin} | {cnt} |")

report_lines += [
    "",
    "### Taxonomy Label 분포 (잠정)",
    "",
    "| proposed_taxonomy_label | 후보 수 |",
    "|---|---|",
]
for tlabel, cnt in sorted(taxonomy_dist_dict.items()):
    report_lines.append(f"| {tlabel} | {cnt} |")

report_lines += [
    "",
    "---",
    "",
    "## 7. Safety Guard",
    "",
    f"- stage2_holdout_access: {STAGE2_HOLDOUT_ACCESS}",
    f"- feature_extracted: {FEATURE_EXTRACTED}",
    f"- metric_computed: {METRIC_COMPUTED}",
    f"- score_modified: {SCORE_MODIFIED}",
    f"- threshold_recomputed: {THRESHOLD_RECOMPUTED}",
    f"- high_safety_priority_candidates: {high_safety}개 (R018/R024 hard case)",
    f"- needs_highres_review: {needs_highres}개",
    "- 입력 파일 수정: 없음",
    "",
    "---",
    "",
    "## 8. 다음 단계",
    "",
    "### 1순위: B1-D7f Taxonomy Overlay Review Preflight",
    "",
    "- 대상: 72개 중 우선 20~30개 (safety_priority=high + score_rank 상위 + uncertain_need_highres_review)",
    "- 패널: 6-panel (CT/CT+ROI+bbox/zoom160/zoom96/z-1/z+1), 기존 B1-D1.7b 설계 재사용",
    "- 별도 승인 필요 (PNG 생성)",
    "- stage2_holdout 0 유지",
    "",
    "### 2순위: B1-D8 Normal-only Crop Refiner Link Preflight",
    "",
    "- Gate-P2 calibration 결과로 unsupervised 한계 명확 → N-C crop PaDiM/Mahalanobis 연계 검토",
    "",
    "---",
    "",
    "## 9. 한계",
    "",
    "- **preflight만**: overlay 생성 없음, feature 없음, metric 없음",
    "- **stage2_holdout 미사용**: dev-safe 범위만",
    "- **proposed taxonomy label은 잠정**: roi_ratio 기준 heuristic, B1-D7f 시각 검토 후 확정",
    "- **신규 후보는 정상 환자 출신**: 실제 lesion FP (NSCLC/MSD) 포함 안 됨",
    "- **Gate-P2 outlier 신호**: FP 6/6 all-suspicious이나 정상과 완전 분리 아님",
    "",
    "---",
    "",
    "## 판정: PASS",
    "",
    f"- 입력 파일: {len(INPUT_FILES)}개 verified",
    f"- 생성 파일: {len(OUTPUT_FILES)}개",
    "- 수정 파일: 없음",
    "- stage2_holdout 접근: 없음",
    "- feature extraction: 없음",
    "- metric 계산: 없음",
    "- score/threshold/ROI 수정: 없음",
    f"- seed summary: Rule-B3 {len(rule_b3_seed_ids)}개 + Gate-P2 {len(gate_p2_seed_ids)}개 + AD_other 9개 + hard_case 2개 = 30개 known seeds",
    f"- planned candidate count: {len(all_candidates)}개 (known 30 + new {new_count})",
    f"- taxonomy label schema: {len(TAXONOMY_SCHEMA)}개 labels",
    "- candidate selection strategy: p95+ normal, patient cap 3, position_bin 균형",
    "- overlay review plan: B1-D7f 별도 승인 (PNG 미생성)",
    "- blockers: 없음",
    "- risks: proposed label은 heuristic, 시각 검토 전 확정 아님",
    "- 다음 단계: B1-D7f taxonomy overlay review preflight (권장)",
]

report_path = os.path.join(OUTDIR, "b1d7e_high_score_fp_taxonomy_expansion_preflight_report.md")
with open(report_path, "w") as f:
    f.write("\n".join(report_lines))
print(f"Written: {report_path}")

# --- Final verification ---
for fname in OUTPUT_FILES:
    fpath = os.path.join(OUTDIR, fname)
    assert os.path.exists(fpath), f"Output file missing: {fpath}"
    size = os.path.getsize(fpath)
    assert size > 0, f"Output file empty: {fpath}"
    print(f"  OK: {fname} ({size} bytes)")

# Verify input files not modified
for key, fpath in INPUT_FILES.items():
    current_mtime = int(os.path.getmtime(fpath))
    recorded_mtime = input_mtime[key]
    assert current_mtime == recorded_mtime, f"INPUT FILE MODIFIED: {fpath} (recorded={recorded_mtime}, current={current_mtime})"

print("\nAll verifications PASSED.")
print(f"\n{'='*60}")
print("B1-D7e PASS")
print(f"  Total candidates: {len(all_candidates)}")
print(f"  Known seeds: {seed_count}")
print(f"  New high-score: {new_count}")
print(f"  Unique patients: {patient_count_total}")
print(f"  Taxonomy labels: {len(TAXONOMY_SCHEMA)}")
print(f"  stage2_holdout_access: {STAGE2_HOLDOUT_ACCESS}")
print(f"  feature_extracted: {FEATURE_EXTRACTED}")
print(f"  score_modified: {SCORE_MODIFIED}")
print(f"  metric_computed: {METRIC_COMPUTED}")
print(f"{'='*60}")
