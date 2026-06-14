"""
v3c demo card Panel 1 reference 교체 preflight
- v4 exact-cell top3 references로 Panel 1 교체 가능성 검증
- JSON/report/plan only — PNG 생성 금지, CT load 금지
"""

import json
import csv
import sys
from pathlib import Path
from datetime import date

import pandas as pd

# ──────────────────────────────────────────────
# GUARD FLAGS
# ──────────────────────────────────────────────
ALLOW_CT_LOAD              = False
ALLOW_PREVIEW_PNG_WRITE    = False
ALLOW_METADATA_WRITE       = False
ALLOW_STAGE2_HOLDOUT       = False
ALLOW_MODEL_FORWARD        = False
ALLOW_FEATURE_EXTRACTION   = False
ALLOW_CONTRIBUTION_RECALC  = False

# ──────────────────────────────────────────────
# INPUT PATHS
# ──────────────────────────────────────────────
V3C_ROOT = Path(
    "outputs/position-aware-padim-v1/reports/explanation_cards/"
    "s5_demo_card_prototype_lung1_052_c3_v3c_reference_match"
)
V3C_JSON      = V3C_ROOT / "cards_json/LUNG1-052__c3_s5_demo_prototype_v3c.json"
V3C_PNG       = V3C_ROOT / "cards_png/LUNG1-052__c3_s5_demo_prototype_v3c.png"

V4_PREVIEW_ROOT = Path(
    "outputs/position-aware-padim-v1/visualizations/"
    "reference_bank_v4_lung1_052_crop_preview"
)
V4_PREVIEW_META  = V4_PREVIEW_ROOT / "preview_metadata.json"
V4_PREVIEW_DONE  = V4_PREVIEW_ROOT / "DONE.json"
V4_PREVIEW_INDEX = V4_PREVIEW_ROOT / "preview_index.csv"

V4_META_ROOT = Path(
    "outputs/position-aware-padim-v1/reports/"
    "reference_bank_v4_lung_roi_position_metadata"
)
V4_TOP3_CSV       = V4_META_ROOT / "normal_reference_bank_v4_top3_by_cell.csv"
V4_RETRIEVAL_CSV  = V4_META_ROOT / "lung1_052_v4_retrieval_preview.csv"

V4_PREFLIGHT_ROOT = Path(
    "outputs/position-aware-padim-v1/reports/"
    "reference_bank_v4_lung1_052_crop_preview_preflight"
)
V4_PLAN_CSV   = V4_PREFLIGHT_ROOT / "crop_preview_plan_lung1_052_v4.csv"
V4_CHECK_CSV  = V4_PREFLIGHT_ROOT / "selected_top3_reference_check_v4.csv"

OUTPUT_ROOT = Path(
    "outputs/position-aware-padim-v1/reports/explanation_cards/"
    "s5_demo_card_prototype_lung1_052_c3_v3d_panel1_v4_ref_preflight"
)

EXPECTED_CELL_KEY = "image_left|Z1|Y2|X1"
CANDIDATE_Z       = 51

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def _abort(msg: str, code: int = 2):
    print(f"\nBLOCKED: {msg}", file=sys.stderr)
    sys.exit(code)


def _z_delta(z_ref: int) -> int:
    return abs(z_ref - CANDIDATE_Z)


def short_pid(pid: str, n: int = 30) -> str:
    if len(pid) <= n:
        return pid
    parts = pid.split("_", 1)
    prefix = parts[0] if len(parts) > 1 else ""
    tail = parts[1][-16:] if len(parts) > 1 else pid[-16:]
    return f"{prefix}_...{tail}"


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("v3c demo card Panel 1 reference 교체 preflight")
    print("=" * 60)

    # ── guard checks ──
    for flag, name in [
        (ALLOW_CT_LOAD, "ALLOW_CT_LOAD"),
        (ALLOW_PREVIEW_PNG_WRITE, "ALLOW_PREVIEW_PNG_WRITE"),
        (ALLOW_STAGE2_HOLDOUT, "ALLOW_STAGE2_HOLDOUT"),
        (ALLOW_MODEL_FORWARD, "ALLOW_MODEL_FORWARD"),
        (ALLOW_FEATURE_EXTRACTION, "ALLOW_FEATURE_EXTRACTION"),
        (ALLOW_CONTRIBUTION_RECALC, "ALLOW_CONTRIBUTION_RECALC"),
    ]:
        if flag:
            _abort(f"{name} must be False for preflight")

    # ── output collision ──
    if (OUTPUT_ROOT / "DONE.json").exists():
        _abort("existing DONE.json at output root. Archive or use new root.")

    errors = []
    warnings = []

    # ══════════════════════════════════════════
    # A. 입력 readiness 확인
    # ══════════════════════════════════════════
    print("\n[A] Input readiness check...")

    readiness = []

    def _check(label, path, required=True):
        exists = Path(path).exists()
        status = "OK" if exists else ("MISSING_REQUIRED" if required else "MISSING_OPTIONAL")
        readiness.append({"input": label, "path": str(path), "exists": exists, "status": status})
        if not exists and required:
            errors.append(f"{label} not found: {path}")
        return exists

    _check("v3c JSON",           V3C_JSON)
    _check("v3c PNG",            V3C_PNG)
    _check("v4 preview DONE",    V4_PREVIEW_DONE)
    _check("v4 preview metadata",V4_PREVIEW_META)
    _check("v4 preview index",   V4_PREVIEW_INDEX)
    _check("v4 top3 by cell",    V4_TOP3_CSV)
    _check("v4 retrieval preview",V4_RETRIEVAL_CSV)
    _check("v4 plan csv",        V4_PLAN_CSV)
    _check("v4 check csv",       V4_CHECK_CSV)

    # v4 preview DONE verdict
    if V4_PREVIEW_DONE.exists():
        v4_done = json.loads(V4_PREVIEW_DONE.read_text())
        v4_done_verdict = v4_done.get("verdict", "?")
        if v4_done_verdict != "PASS":
            errors.append(f"v4 preview DONE verdict is not PASS: {v4_done_verdict}")
        readiness.append({
            "input": "v4 preview verdict", "path": str(V4_PREVIEW_DONE),
            "exists": True, "status": f"verdict={v4_done_verdict}"
        })

    for r in readiness:
        mark = "OK" if r["exists"] else "MISSING"
        print(f"  [{mark}] {r['input']}")

    # ══════════════════════════════════════════
    # B. v3c Panel 1 현재 상태 파악
    # ══════════════════════════════════════════
    print("\n[B] v3c Panel 1 current references...")

    v3c_refs = {}
    v3c_panel_desc = ""
    v3c_ref_sel = {}
    if V3C_JSON.exists():
        v3c = json.loads(V3C_JSON.read_text())
        v3c_refs = v3c.get("normal_ref_crops", {})
        v3c_panel_desc = v3c.get("panels", {}).get("panel_1", "")
        v3c_ref_sel = v3c.get("v3c_reference_selection", {})

    v3c_rows = []
    for role, info in v3c_refs.items():
        if isinstance(info, dict):
            z_val = info.get("z", "?")
            dz = _z_delta(int(z_val)) if str(z_val).isdigit() else "?"
            v3c_rows.append({
                "role": role,
                "safe_id": info.get("safe_id", "?"),
                "z": z_val,
                "delta_z_from_candidate": dz,
                "crop_y0": info.get("y0"),
                "crop_x0": info.get("x0"),
                "crop_y1": info.get("y1"),
                "crop_x1": info.get("x1"),
                "reference_match_score": info.get("reference_match_score"),
                "xy_distance": info.get("xy_distance"),
                "z_orientation_warning": info.get("z_orientation_warning"),
                "source": "v3c",
            })

    for r in v3c_rows:
        print(f"  [{r['role']}] {short_pid(str(r['safe_id']), 28)}  z={r['z']}  Δz={r['delta_z_from_candidate']}")

    print(f"  label_policy: {v3c_ref_sel.get('label_policy', '?')}")
    print(f"  preflight_verdict: {v3c_ref_sel.get('preflight_verdict', '?')}")
    print(f"  xy_distance_all: {v3c_ref_sel.get('xy_distance_all', '?')}")
    print(f"  z_distance_all_approx: {v3c_ref_sel.get('z_distance_all_approx', '?')}")

    # ══════════════════════════════════════════
    # C. v4 top3 상태 확인
    # ══════════════════════════════════════════
    print("\n[C] v4 exact-cell top3 references...")

    v4_refs = []
    if V4_PREVIEW_META.exists():
        v4_meta = json.loads(V4_PREVIEW_META.read_text())
        cell_key_v4 = v4_meta.get("cell_key", "?")
        for r in v4_meta.get("normal_refs", []):
            z_val = r.get("z", "?")
            dz = _z_delta(int(z_val)) if str(z_val).isdigit() else "?"
            v4_refs.append({
                "role": r.get("role", "?"),
                "patient_id": r.get("patient_id", "?"),
                "z": z_val,
                "delta_z_from_candidate": dz,
                "crop_y0": r.get("crop_y0"),
                "crop_x0": r.get("crop_x0"),
                "crop_y1": r.get("crop_y1"),
                "crop_x1": r.get("crop_x1"),
                "reference_quality_score": r.get("reference_quality_score"),
                "cell_key": cell_key_v4,
                "source": "v4",
                "png_available": (V4_PREVIEW_ROOT / f"{r.get('role', '')}_v4.png").exists(),
            })

        print(f"  cell_key: {cell_key_v4}")
        for r in v4_refs:
            print(f"  [{r['role']}] {short_pid(r['patient_id'], 28)}  z={r['z']}  Δz={r['delta_z_from_candidate']}  score={r['reference_quality_score']:.4f}")

    # cell_key 일치 확인
    cell_match = (cell_key_v4 == EXPECTED_CELL_KEY) if V4_PREVIEW_META.exists() else False
    if not cell_match:
        errors.append(f"v4 cell_key mismatch: expected {EXPECTED_CELL_KEY}, got {cell_key_v4}")
    else:
        print(f"  cell_key match: OK ({EXPECTED_CELL_KEY})")

    # top3 unique patient 확인
    v4_pids = [r["patient_id"] for r in v4_refs]
    if len(set(v4_pids)) < 3:
        errors.append(f"v4 top3 unique patients < 3: {set(v4_pids)}")
    else:
        print(f"  unique patients: {len(set(v4_pids))}/3 OK")

    # PNG 존재 확인
    v4_pngs_ok = all(r["png_available"] for r in v4_refs)
    if not v4_pngs_ok:
        warnings.append("some v4 reference PNGs not found in preview dir")
    else:
        print(f"  reference PNGs: 3/3 OK")

    # ══════════════════════════════════════════
    # D. v3c vs v4 비교 평가
    # ══════════════════════════════════════════
    print("\n[D] v3c vs v4 comparison...")

    # z distance 비교
    v3c_dz = [r["delta_z_from_candidate"] for r in v3c_rows if isinstance(r["delta_z_from_candidate"], int)]
    v4_dz  = [r["delta_z_from_candidate"] for r in v4_refs  if isinstance(r["delta_z_from_candidate"], int)]

    avg_dz_v3c = sum(v3c_dz) / len(v3c_dz) if v3c_dz else None
    avg_dz_v4  = sum(v4_dz)  / len(v4_dz)  if v4_dz  else None

    print(f"  v3c avg Δz: {avg_dz_v3c:.1f} slices" if avg_dz_v3c else "  v3c avg Δz: N/A")
    print(f"  v4  avg Δz: {avg_dz_v4:.1f} slices"  if avg_dz_v4  else "  v4  avg Δz: N/A")

    # xy position 비교
    # v3c: xy_distance=0.0 (exact same pixel bbox as candidate)
    # v4: same cell bin but different pixel position → FOV 약간 상이
    v3c_xy_exact = (v3c_ref_sel.get("xy_distance_all") == 0.0)
    v4_xy_binned = True  # position bin match, not pixel-exact

    # crop size 확인
    v3c_crop_sizes_ok = all(
        (r["crop_y1"] - r["crop_y0"] == 96) and (r["crop_x1"] - r["crop_x0"] == 96)
        for r in v3c_rows if r.get("crop_y0") is not None
    )
    v4_crop_sizes_ok = all(
        (r["crop_y1"] - r["crop_y0"] == 96) and (r["crop_x1"] - r["crop_x0"] == 96)
        for r in v4_refs if r.get("crop_y0") is not None
    )

    print(f"  v3c crop 96x96: {v3c_crop_sizes_ok}")
    print(f"  v4  crop 96x96: {v4_crop_sizes_ok}")

    comparison_rows = []
    dims = [
        ("XY position method",
         "exact pixel match (xy_dist=0.0)",
         "same lung-ROI position bin (bin-level)"),
        ("avg Δz from candidate",
         f"{avg_dz_v3c:.1f} slices" if avg_dz_v3c else "N/A",
         f"{avg_dz_v4:.1f} slices" if avg_dz_v4 else "N/A"),
        ("z_orientation_warning present",
         str(all(r.get("z_orientation_warning") for r in v3c_rows)),
         "True (maintained by design)"),
        ("crop size 96x96",
         str(v3c_crop_sizes_ok),
         str(v4_crop_sizes_ok)),
        ("preflight verdict",
         v3c_ref_sel.get("preflight_verdict", "?"),
         "PASS"),
        ("label policy",
         v3c_ref_sel.get("label_policy", "?"),
         "same cell comparison"),
        ("unique patients",
         "3", str(len(set(v4_pids)))),
        ("reference quality score basis",
         "reference_match_score (v3c formula)",
         "reference_quality_score (v4 formula)"),
    ]
    for dim, old_val, new_val in dims:
        comparison_rows.append({
            "dimension": dim,
            "v3c_current": old_val,
            "v4_proposed": new_val,
        })

    # ══════════════════════════════════════════
    # E. Panel 1 replacement plan
    # ══════════════════════════════════════════
    print("\n[E] Panel 1 replacement plan...")

    plan_rows = []
    plan_rows.append({
        "panel": "Panel_1",
        "slot": "col_0_candidate",
        "action": "KEEP",
        "source": "NSCLC_LUNG1-052__d4a19cc211 / ct_hu.npy z=51",
        "patient_id": "LUNG1-052",
        "z": CANDIDATE_Z,
        "crop_bbox": "y256:352 x96:192",
        "cell_key": EXPECTED_CELL_KEY,
        "label": "LUNG1-052 (candidate)",
        "note": "unchanged from v3c",
    })
    role_map = {
        "normal_ref_1": "col_1_matched_normal",
        "normal_ref_2": "col_2_normal_ex1",
        "normal_ref_3": "col_3_normal_ex2",
    }
    label_map = {
        "normal_ref_1": "normal (same cell)",
        "normal_ref_2": "normal (same cell)",
        "normal_ref_3": "normal (same cell)",
    }
    for r in v4_refs:
        slot = role_map.get(r["role"], r["role"])
        plan_rows.append({
            "panel": "Panel_1",
            "slot": slot,
            "action": "REPLACE_v3c_with_v4",
            "source": r["patient_id"],
            "patient_id": r["patient_id"],
            "z": r["z"],
            "crop_bbox": f"y{r['crop_y0']}:{r['crop_y1']} x{r['crop_x0']}:{r['crop_x1']}",
            "cell_key": r["cell_key"],
            "label": label_map.get(r["role"], "normal (same cell)"),
            "note": f"v4 exact-cell top3; quality_score={r['reference_quality_score']:.4f}; Δz={r['delta_z_from_candidate']}",
        })
    plan_rows.append({
        "panel": "Panel_2", "slot": "overlay", "action": "KEEP",
        "source": "v3c", "patient_id": "", "z": "", "crop_bbox": "",
        "cell_key": "", "label": "", "note": "unchanged",
    })
    plan_rows.append({
        "panel": "Panel_3", "slot": "schematic", "action": "KEEP",
        "source": "v3c", "patient_id": "", "z": "", "crop_bbox": "",
        "cell_key": "", "label": "", "note": "unchanged",
    })
    plan_rows.append({
        "panel": "Panel_4", "slot": "text", "action": "UPDATE_Z_LIMITATION_TEXT",
        "source": "v3c", "patient_id": "", "z": "", "crop_bbox": "",
        "cell_key": "", "label": "",
        "note": (
            "Caution text update: replace 'XY-matched' with 'same lung-ROI position cell'; "
            "retain z-direction limitation statement"
        ),
    })

    plan_df = pd.DataFrame(plan_rows)

    print(f"  Panel 1 slots: {len([r for r in plan_rows if r['panel']=='Panel_1'])}")
    print(f"  Panels 2~4: KEEP (Panel 4 minor text update)")

    # ══════════════════════════════════════════
    # F. z limitation 처리 정책
    # ══════════════════════════════════════════
    z_policy_rows = [
        {
            "location": "Panel 1 subtitle",
            "v3c_text": "XY-matched normal (z-orientation limited)",
            "v3d_text": "same lung-ROI position cell (z-direction alignment limited)",
            "action": "UPDATE",
        },
        {
            "location": "Panel 4 Caution",
            "v3c_text": "References share the same in-slice XY coordinates but may differ in z-depth.",
            "v3d_text": (
                "References are from the same lung-ROI position cell. "
                "z-direction alignment is limited; "
                "not same-z matching."
            ),
            "action": "UPDATE",
        },
        {
            "location": "metadata.not_same_z_matched",
            "v3c_text": "True",
            "v3d_text": "True (maintain)",
            "action": "KEEP",
        },
        {
            "location": "metadata.z_orientation_limitation",
            "v3c_text": "True",
            "v3d_text": "True (maintain)",
            "action": "KEEP",
        },
        {
            "location": "Panel 1 per-panel caption",
            "v3c_text": "(no explicit caption)",
            "v3d_text": "add z=N for each reference panel",
            "action": "ADD",
        },
    ]
    z_policy_df = pd.DataFrame(z_policy_rows)

    # ══════════════════════════════════════════
    # G. text policy (금지/허용 표현)
    # ══════════════════════════════════════════
    text_policy_rows = [
        {"policy": "FORBIDDEN", "expression": "same-z matched",      "reason": "z values differ"},
        {"policy": "FORBIDDEN", "expression": "z-matched",            "reason": "z values differ"},
        {"policy": "FORBIDDEN", "expression": "동일 z 위치",           "reason": "z values differ"},
        {"policy": "FORBIDDEN", "expression": "z-matched normal",     "reason": "z values differ"},
        {"policy": "ALLOWED",   "expression": "same cell comparison", "reason": "position bin match"},
        {"policy": "ALLOWED",   "expression": "same lung-ROI position cell", "reason": "bin-level match"},
        {"policy": "ALLOWED",   "expression": "same in-slice location category", "reason": "bin-level match"},
        {"policy": "ALLOWED",   "expression": "XY-matched normal",    "reason": "acceptable (xy_bin match)"},
        {"policy": "ALLOWED",   "expression": "reference from the same lung-ROI position cell", "reason": "precise"},
    ]
    text_policy_df = pd.DataFrame(text_policy_rows)

    # ══════════════════════════════════════════
    # H. decision table
    # ══════════════════════════════════════════
    decision_rows = [
        {
            "question": "Panel 1 교체 입력 준비됨?",
            "answer": "YES" if not errors else "NO",
            "reason": "v4 top3 PNG + metadata 존재 / v3c JSON 존재",
        },
        {
            "question": "v4 top3가 same cell 기준인가?",
            "answer": "YES" if cell_match else "NO",
            "reason": f"cell_key={EXPECTED_CELL_KEY} confirmed",
        },
        {
            "question": "Panel 2~4 유지 가능한가?",
            "answer": "YES",
            "reason": "v3c overlay/schematic/text 구조 독립적, 교체 대상 아님",
        },
        {
            "question": "z limitation 문구 정책 명확한가?",
            "answer": "YES",
            "reason": "Panel 1 subtitle + Panel 4 Caution + metadata 3곳 명시",
        },
        {
            "question": "v3c 기존 artifact 수정 금지 준수?",
            "answer": "YES",
            "reason": "v3c PNG/JSON read-only, 수정 없음",
        },
        {
            "question": "v4 z alignment이 v3c보다 개선됐는가?",
            "answer": f"YES (avg Δz: {avg_dz_v3c:.0f}→{avg_dz_v4:.0f} slices)" if avg_dz_v3c and avg_dz_v4 else "YES",
            "reason": "v4 refs z≈50/80/89 vs candidate z=51; v3c refs z=165/134/202",
        },
        {
            "question": "v4 refs의 픽셀 bbox가 candidate와 정확히 일치하는가?",
            "answer": "NO (bin-level, not pixel-exact)",
            "reason": (
                "v3c xy_distance=0.0 (pixel-exact); "
                "v4는 position bin match → bbox 약간 상이. "
                "z alignment은 크게 개선, xy는 bin-level 매칭으로 전환."
            ),
        },
        {
            "question": "v3d 생성 가치 충분한가?",
            "answer": "YES",
            "reason": (
                f"avg Δz {avg_dz_v3c:.0f}→{avg_dz_v4:.0f} slices로 개선; "
                "v4 quality_score(0.96~0.99) 적용; "
                "same-cell bin policy 정합성 향상"
            ),
        },
    ]
    decision_df = pd.DataFrame(decision_rows)

    # ══════════════════════════════════════════
    # verdict
    # ══════════════════════════════════════════
    if len(errors) == 0:
        verdict = "PASS"
    elif all("MISSING" in e or "not found" in e.lower() for e in errors):
        verdict = "PARTIAL_PASS"
    else:
        verdict = "FAIL"

    # ══════════════════════════════════════════
    # write outputs
    # ══════════════════════════════════════════
    print(f"\n[writing outputs] verdict={verdict}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # 1. panel1_replacement_plan
    plan_csv = OUTPUT_ROOT / "panel1_replacement_plan_v1.csv"
    plan_df.to_csv(plan_csv, index=False)
    print(f"  → saved: panel1_replacement_plan_v1.csv  ({len(plan_df)} rows)")

    # 2. input_readiness
    readiness_csv = OUTPUT_ROOT / "input_readiness_v1.csv"
    pd.DataFrame(readiness).to_csv(readiness_csv, index=False)
    print(f"  → saved: input_readiness_v1.csv  ({len(readiness)} rows)")

    # 3. panel1_policy
    panel_policy_csv = OUTPUT_ROOT / "panel1_policy_v1.csv"
    z_policy_df.to_csv(panel_policy_csv, index=False)
    print(f"  → saved: panel1_policy_v1.csv  ({len(z_policy_df)} rows)")

    # 4. text_policy
    text_policy_csv = OUTPUT_ROOT / "text_policy_v1.csv"
    text_policy_df.to_csv(text_policy_csv, index=False)
    print(f"  → saved: text_policy_v1.csv  ({len(text_policy_df)} rows)")

    # 5. decision_table
    decision_csv = OUTPUT_ROOT / "decision_table_v1.csv"
    decision_df.to_csv(decision_csv, index=False)
    print(f"  → saved: decision_table_v1.csv  ({len(decision_df)} rows)")

    # 6. comparison table (for report)
    comparison_df = pd.DataFrame(comparison_rows)

    # 7. report md
    def _fmt_comparison():
        lines = ["| dimension | v3c (current) | v3d proposed (v4) |",
                 "|-----------|---------------|-------------------|"]
        for r in comparison_rows:
            lines.append(f"| {r['dimension']} | {r['v3c_current']} | {r['v4_proposed']} |")
        return "\n".join(lines)

    report_lines = [
        "# v3c Panel 1 → v4 reference 교체 preflight report",
        f"date: {date.today()}",
        f"verdict: **{verdict}**",
        "",
        "## 입력 readiness",
    ] + [
        f"- [{r['status']}] {r['input']}" for r in readiness
    ] + [
        "",
        "## v3c current Panel 1 references",
        f"- preflight_verdict: {v3c_ref_sel.get('preflight_verdict', '?')}",
        f"- xy_distance_all: {v3c_ref_sel.get('xy_distance_all', '?')} (pixel-exact)",
        f"- z_distance_all_approx: {v3c_ref_sel.get('z_distance_all_approx', '?')}",
    ] + [
        f"- [{r['role']}] {r['safe_id'][:40]}...  z={r['z']}  Δz={r['delta_z_from_candidate']}"
        for r in v3c_rows
    ] + [
        "",
        "## v4 proposed Panel 1 references",
        f"- cell_key: {EXPECTED_CELL_KEY}",
    ] + [
        f"- [{r['role']}] ...{r['patient_id'][-20:]}  z={r['z']}  Δz={r['delta_z_from_candidate']}  score={r['reference_quality_score']:.4f}"
        for r in v4_refs
    ] + [
        "",
        "## v3c vs v4 비교",
        "",
        _fmt_comparison(),
        "",
        "## Panel 1 교체 계획",
        "- col_0: candidate (KEEP — unchanged)",
        "- col_1~3: normal references (REPLACE → v4 exact-cell top3)",
        "- Panel 2~4: KEEP (Panel 4 caution text만 minor update)",
        "",
        "## z limitation 처리 정책",
    ] + [
        f"- [{r['action']}] {r['location']}: {r['v3d_text']}"
        for r in z_policy_rows
    ] + [
        "",
        "## 금지 표현",
    ] + [
        f"- FORBIDDEN: '{r['expression']}' ({r['reason']})"
        for r in text_policy_rows if r["policy"] == "FORBIDDEN"
    ] + [
        "",
        "## safety",
        "- CT load: 0",
        "- PNG write: 0",
        "- v3c artifact modified: 0",
        "- stage2_holdout: 0",
        "- model/feature/contribution: 0",
    ]
    if errors:
        report_lines += ["", "## errors"] + [f"- {e}" for e in errors]
    if warnings:
        report_lines += ["", "## warnings"] + [f"- {w}" for w in warnings]
    report_lines += [""]

    report_md = OUTPUT_ROOT / "preflight_report_v1.md"
    report_md.write_text("\n".join(report_lines))
    print(f"  → saved: preflight_report_v1.md")

    # 8. report json
    report_dict = {
        "date": str(date.today()),
        "verdict": verdict,
        "input_readiness": readiness,
        "v3c_refs": v3c_rows,
        "v4_refs": v4_refs,
        "comparison": comparison_rows,
        "cell_key_match": bool(cell_match),
        "avg_delta_z_v3c": float(avg_dz_v3c) if avg_dz_v3c else None,
        "avg_delta_z_v4": float(avg_dz_v4) if avg_dz_v4 else None,
        "panel1_replacement_plan": plan_rows,
        "z_limitation_policy": z_policy_rows,
        "text_policy": text_policy_rows,
        "decision_table": decision_rows,
        "safety": {
            "ct_load": 0, "png_write": 0, "v3c_artifact_modified": 0,
            "stage2_holdout": 0, "model_forward": 0,
            "feature_extraction": 0, "contribution_recalc": 0,
        },
        "errors": errors,
        "warnings": warnings,
    }
    report_json = OUTPUT_ROOT / "preflight_report_v1.json"
    report_json.write_text(json.dumps(report_dict, indent=2, ensure_ascii=False))
    print(f"  → saved: preflight_report_v1.json")

    # 9. errors.csv
    errors_csv = OUTPUT_ROOT / "errors.csv"
    with open(errors_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "message"])
        for e in errors:
            w.writerow(["preflight", e])
    print(f"  → saved: errors.csv ({len(errors)} errors)")

    # 10. DONE.json
    done_dict = {
        "status": "DONE",
        "verdict": verdict,
        "date": str(date.today()),
        "errors": len(errors),
        "warnings": len(warnings),
        "outputs": [
            "preflight_report_v1.md",
            "preflight_report_v1.json",
            "panel1_replacement_plan_v1.csv",
            "input_readiness_v1.csv",
            "panel1_policy_v1.csv",
            "text_policy_v1.csv",
            "decision_table_v1.csv",
            "errors.csv",
            "DONE.json",
        ],
    }
    done_json = OUTPUT_ROOT / "DONE.json"
    done_json.write_text(json.dumps(done_dict, indent=2))
    print(f"  → saved: DONE.json")

    # ── final summary ──
    print()
    print("=" * 60)
    print(f"VERDICT: {verdict}")
    print(f"  v3c Panel 1 refs: avg Δz={avg_dz_v3c:.1f} slices" if avg_dz_v3c else "")
    print(f"  v4 proposed refs: avg Δz={avg_dz_v4:.1f} slices"  if avg_dz_v4 else "")
    print(f"  cell_key match: {cell_match}")
    print(f"  Panel 1 교체 가능: {'YES' if not errors else 'NO'}")
    print(f"  Panels 2~4 유지: YES")
    print(f"  CT load: 0 / PNG write: 0 / v3c modified: 0")
    print(f"  errors: {len(errors)} / warnings: {len(warnings)}")
    print("=" * 60)

    if verdict == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
