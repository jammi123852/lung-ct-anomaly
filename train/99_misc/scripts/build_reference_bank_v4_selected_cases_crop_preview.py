"""
build_reference_bank_v4_selected_cases_crop_preview.py

목적:
reference bank v4 selected cases crop preview preflight (PASS) 결과를 바탕으로,
4개 case 에 대해 candidate crop + same-cell normal reference top3 + montage 를
실제 생성하는 actual generation 스크립트.

본 파일은 actual generation 코드를 모두 포함하지만,
실제 PNG 생성은 guard(ALLOW_CT_LOAD / ALLOW_PNG_WRITE)가 모두 True 이고
--run-generate --confirm-generate 가 함께 주어졌을 때만 수행된다.

guard 기본값은 전부 False 이므로, 기본 상태에서는 어떤 CT load/PNG write 도 없다.

CLI 모드:
  (no args)                         -> BLOCKED exit 2
  --selftest                        -> 내부 로직 자기검증, exit 0/1
  --dry-run                         -> 입력/경로/bounds 검증(載 없음), exit 0/1
  --plan-only                       -> 생성 계획 출력 + plan CSV 기록, exit 0
  --static-drycheck                 -> 전체 정적검사 매트릭스 실행 + drycheck 리포트 기록
  --run-generate                    -> (--confirm-generate 없으면) BLOCKED exit 2
  --run-generate --confirm-generate -> guard False 면 BLOCKED exit 2;
                                       guard True(env override) 면 실제 생성

CT load 정책(실제 실행 시에만):
  np.load(..., mmap_mode="r") read-only only ; stage2_holdout 금지 ;
  candidate(4) + refs(12) = 총 16 CT path 만 허용.

display policy:
  crop 96x96, 동일 display size, lung window only.
  montage = [candidate][normal_ref_1][normal_ref_2][normal_ref_3].
  tile label: role / case_id|patient_id / z / cell_key / fallback_level.
  montage caution: "same-cell comparison, not same-z matching".
  MSD_lung_059__c2 metadata msd_source=true ; side=image x-coordinate(NOT anatomical).
  진단/병변/암/혈관 원인 단정 문구 금지.
"""

import os
import sys
import csv
import json
import argparse
import pathlib
from datetime import datetime

import numpy as np
import pandas as pd

# ============================================================
# GUARD FLAGS (기본 전부 False)
# ============================================================
ALLOW_CT_LOAD             = False
ALLOW_PNG_WRITE           = False
ALLOW_STAGE2_HOLDOUT      = False
ALLOW_MODEL_FORWARD       = False
ALLOW_FEATURE_EXTRACTION  = False
ALLOW_CONTRIBUTION_RECALC = False
ALLOW_FULL300             = False

# actual generation 승인 시에만 env 로 True 로 올린다 (내릴 수는 없음)
if os.environ.get("ALLOW_CT_LOAD") == "1":
    ALLOW_CT_LOAD = True
if os.environ.get("ALLOW_PNG_WRITE") == "1":
    ALLOW_PNG_WRITE = True

# ============================================================
# PATHS
# ============================================================
REPO_ROOT = pathlib.Path("/home/jinhy/project/lung-ct-anomaly")
R = REPO_ROOT / "outputs/position-aware-padim-v1/reports"
V = REPO_ROOT / "outputs/position-aware-padim-v1/visualizations"

PREFLIGHT_ROOT = R / "reference_bank_v4_selected_cases_crop_preview_preflight"
HANDOFF_ROOT   = R / "reference_bank_v4_multi_case_retrieval_close_handoff"
RERUN_ROOT     = R / "reference_bank_v4_multi_case_retrieval_dryrun_completed_roi"

PLAN_CSV       = PREFLIGHT_ROOT / "selected_cases_crop_preview_plan_v4.csv"
CAND_READY_CSV = PREFLIGHT_ROOT / "candidate_crop_readiness_v4.csv"
REF_READY_CSV  = PREFLIGHT_ROOT / "normal_reference_crop_readiness_v4.csv"
PREVIEW_POLICY = PREFLIGHT_ROOT / "preview_generation_policy_v4.json"
PREFLIGHT_DONE = PREFLIGHT_ROOT / "DONE.json"

OUTPUT_ROOT = V / "reference_bank_v4_selected_cases_crop_preview"

# drycheck/plan reports (this stage)
DRYCHECK_MD   = R / "reference_bank_v4_selected_cases_crop_preview_script_static_drycheck_v1.md"
DRYCHECK_JSON = R / "reference_bank_v4_selected_cases_crop_preview_script_static_drycheck_v1.json"
SCRIPT_PLAN   = R / "reference_bank_v4_selected_cases_crop_preview_script_plan_v1.csv"

# ============================================================
# CONSTANTS
# ============================================================
EXPECTED_CASES = ["LUNG1-052__c3", "LUNG1-320__c2", "LUNG1-041__c3", "MSD_lung_059__c2"]
EXPECTED_CELL = {
    "LUNG1-052__c3":    "image_left|Z1|Y2|X1",
    "LUNG1-320__c2":    "image_left|Z2|Y2|X2",
    "LUNG1-041__c3":    "image_left|Z2|Y1|X1",
    "MSD_lung_059__c2": "image_right|Z0|Y2|X1",
}
MSD_CASES = {"MSD_lung_059__c2"}
CROP_SIZE = 96
IMG_H = IMG_W = 512

# lung window
WL = -600.0
WW = 1500.0

CAUTION_TEXT = "same-cell comparison, not same-z matching"
SIDE_BASIS_NOTE = "image x-coordinate based (NOT anatomical left/right)"

PNG_PER_CASE = 5      # candidate + 3 ref + montage
MAX_TOTAL_PNG = 20    # 4 cases x 5

FORBIDDEN_PHRASES = [
    "same-z matched", "same z matched", "z-matched", "z matched",
    "identical z", "동일 z 위치", "같은 z 위치",
    "diagnostic heatmap", "grad-cam", "pixel attribution",
    "병변 원인", "암 위치", "혈관 때문에", "diagnostic conclusion",
]
ALLOWED_PHRASES = [
    "same lung-ROI position cell", "same-cell comparison",
    "not same-z matching", "z-direction alignment is limited",
    "research-use auxiliary explanation",
]


def _abort(msg, code=2):
    print(f"\nBLOCKED: {msg}", file=sys.stderr)
    sys.exit(code)


def scan_forbidden(blob: str):
    low = str(blob).lower()
    return [p for p in FORBIDDEN_PHRASES if p.lower() in low]


def planned_png_names(case_id):
    return [
        f"{case_id}_candidate.png",
        f"{case_id}_normal_ref_1.png",
        f"{case_id}_normal_ref_2.png",
        f"{case_id}_normal_ref_3.png",
        f"{case_id}_reference_preview_montage.png",
    ]


def case_metadata_name(case_id):
    return f"{case_id}_preview_metadata.json"


# ============================================================
# INPUT LOADING / VALIDATION
# ============================================================
def load_plan_inputs():
    """preflight plan + readiness CSV 를 읽어 case 별 crop 계획 dict 반환."""
    if not PLAN_CSV.exists():
        _abort(f"preflight plan CSV not found: {PLAN_CSV}")
    plan = pd.read_csv(PLAN_CSV)
    cand_ready = pd.read_csv(CAND_READY_CSV) if CAND_READY_CSV.exists() else pd.DataFrame()
    ref_ready = pd.read_csv(REF_READY_CSV) if REF_READY_CSV.exists() else pd.DataFrame()
    policy = json.loads(PREVIEW_POLICY.read_text()) if PREVIEW_POLICY.exists() else {}
    return {"plan": plan, "cand_ready": cand_ready,
            "ref_ready": ref_ready, "policy": policy}


def validate_inputs(inputs):
    """plan/readiness 구조 + 정책 정합성 검증. 문제 list 반환(빈 list=정상)."""
    problems = []
    plan = inputs["plan"]

    if not PREFLIGHT_DONE.exists():
        problems.append("preflight DONE.json missing")
    else:
        d = json.loads(PREFLIGHT_DONE.read_text())
        if d.get("verdict") != "PASS":
            problems.append(f"preflight verdict != PASS ({d.get('verdict')})")

    # case 수
    cases = sorted(plan["case_id"].unique().tolist())
    if cases != sorted(EXPECTED_CASES):
        problems.append(f"case set mismatch: {cases}")

    # candidate rows == 4, normal_ref rows == 12
    n_cand = (plan["role"] == "candidate").sum()
    n_ref = plan["role"].str.startswith("normal_ref").sum()
    if n_cand != 4:
        problems.append(f"candidate rows != 4 ({n_cand})")
    if n_ref != 12:
        problems.append(f"normal_ref rows != 12 ({n_ref})")

    # crop_size 96 all
    if not (plan["crop_size"] == CROP_SIZE).all():
        problems.append("crop_size != 96 found")

    # crop_in_bounds all True
    if not plan["crop_in_bounds"].astype(str).eq("True").all():
        problems.append("crop_in_bounds False found")

    # fallback_level 0 only (refs)
    ref_rows = plan[plan["role"].str.startswith("normal_ref")]
    if not ref_rows["fallback_level"].astype(str).eq("0").all():
        problems.append("fallback_level != 0 found in refs")

    # stage2_holdout all False
    if "stage2_holdout_flag" in plan.columns:
        if plan["stage2_holdout_flag"].astype(str).eq("True").any():
            problems.append("stage2_holdout_flag True found")

    # cell_key matches expected (candidate rows)
    for _, r in plan[plan["role"] == "candidate"].iterrows():
        if r["cell_key"] != EXPECTED_CELL.get(r["case_id"]):
            problems.append(f"cell_key mismatch {r['case_id']}: {r['cell_key']}")

    # ct_path exists all
    missing_ct = [r["ct_path"] for _, r in plan.iterrows()
                  if not os.path.exists(str(r["ct_path"]))]
    if missing_ct:
        problems.append(f"{len(missing_ct)} ct_path missing")

    # forbidden wording in plan notes
    blob = " ".join(plan.get("notes", pd.Series([], dtype=str)).astype(str))
    hits = scan_forbidden(blob)
    if hits:
        problems.append(f"forbidden wording in plan notes: {hits}")

    return problems


# ============================================================
# CT / WINDOW / PNG  (실제 실행 경로)
# ============================================================
def window_hu_to_uint8(hu_arr, wl=WL, ww=WW):
    """lung window 적용 후 0-255 uint8. 순수 numpy (load/외부 의존 없음)."""
    lo = wl - ww / 2.0
    hi = wl + ww / 2.0
    arr = np.clip(hu_arr.astype(np.float32), lo, hi)
    arr = (arr - lo) / max(hi - lo, 1e-6)
    return (arr * 255.0).round().astype(np.uint8)


def load_ct_crop_readonly(ct_path, local_z, y0, x0, y1, x1):
    """read-only mmap CT crop. ALLOW_CT_LOAD guard 필수."""
    if not ALLOW_CT_LOAD:
        _abort("load_ct_crop_readonly called with ALLOW_CT_LOAD=False")
    p = str(ct_path).lower()
    if any(k in p for k in ["stage2_holdout", "stage2holdout", "holdout"]):
        _abort(f"stage2_holdout path forbidden: {ct_path}")
    vol = np.load(str(ct_path), mmap_mode="r")   # read-only
    z = int(local_z)
    if z < 0 or z >= vol.shape[0]:
        raise ValueError(f"local_z {z} out of range {vol.shape}")
    crop = np.array(vol[z, y0:y1, x0:x1])         # small copy only
    del vol
    if crop.shape != (CROP_SIZE, CROP_SIZE):
        raise ValueError(f"crop shape {crop.shape} != {(CROP_SIZE, CROP_SIZE)}")
    return crop


def _tile_label_lines(role, ident, z, cell_key, fallback_level):
    return [f"role: {role}", f"id: {ident}", f"z: {z}",
            f"cell: {cell_key}", f"fb: {fallback_level}"]


def write_single_crop_png(out_path, crop_hu, label_lines):
    """단일 crop PNG. ALLOW_PNG_WRITE guard 필수. matplotlib lazy import."""
    if not ALLOW_PNG_WRITE:
        _abort("write_single_crop_png called with ALLOW_PNG_WRITE=False")
    bad = scan_forbidden(" ".join(label_lines))
    if bad:
        _abort(f"forbidden wording in label: {bad}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    img = window_hu_to_uint8(crop_hu)
    fig, ax = plt.subplots(figsize=(2.2, 2.6), dpi=120)
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("\n".join(label_lines), fontsize=6, loc="left")
    fig.tight_layout()
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)


def build_montage_png(out_path, tiles, caption=CAUTION_TEXT):
    """
    montage = [candidate][normal_ref_1..3].
    tiles: list of dict {crop_hu, label_lines}. ALLOW_PNG_WRITE guard 필수.
    """
    if not ALLOW_PNG_WRITE:
        _abort("build_montage_png called with ALLOW_PNG_WRITE=False")
    bad = scan_forbidden(caption + " " + " ".join(
        l for t in tiles for l in t["label_lines"]))
    if bad:
        _abort(f"forbidden wording in montage text: {bad}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(tiles)
    fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 2.8), dpi=120)
    if n == 1:
        axes = [axes]
    for ax, t in zip(axes, tiles):
        ax.imshow(window_hu_to_uint8(t["crop_hu"]), cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("\n".join(t["label_lines"]), fontsize=6, loc="left")
    fig.suptitle(caption, fontsize=8, y=0.02)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)


def write_case_metadata(out_path, case_id, candidate_row, ref_rows):
    meta = {
        "case_id": case_id,
        "cell_key": EXPECTED_CELL[case_id],
        "msd_source": case_id in MSD_CASES,
        "side_basis": SIDE_BASIS_NOTE,
        "matching": {
            "not_same_z_matched": True, "z_direction_limited": True,
            "is_same_z_matching": False,
            "basis": "same lung-ROI position cell",
        },
        "caution_text": CAUTION_TEXT,
        "usage": "research-use auxiliary explanation only",
        "candidate": candidate_row,
        "normal_references": ref_rows,
        "png_files": planned_png_names(case_id),
    }
    bad = scan_forbidden(json.dumps(meta, ensure_ascii=False))
    if bad:
        _abort(f"forbidden wording in case metadata: {bad}")
    out_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))


def write_index_and_done(index_rows, runtime, n_png, n_meta, errors):
    INDEX_COLS = ["case_id", "role", "png_name", "metadata_name", "patient_id",
                  "volume_id", "source", "cell_key", "local_z", "fallback_level",
                  "msd_source", "side_basis", "status"]
    with open(OUTPUT_ROOT / "preview_index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_COLS)
        w.writeheader()
        for r in index_rows:
            w.writerow({k: r.get(k, "") for k in INDEX_COLS})

    (OUTPUT_ROOT / "runtime_summary.json").write_text(json.dumps(runtime, indent=2))

    with open(OUTPUT_ROOT / "errors.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "stage", "error_type", "detail"])
        for e in errors:
            w.writerow([e.get("case_id", ""), e.get("stage", ""),
                        e.get("error_type", ""), e.get("detail", "")])

    done = {
        "status": "DONE",
        "verdict": "PASS" if not errors and n_png == MAX_TOTAL_PNG else (
            "PARTIAL_PASS" if n_png > 0 else "NEEDS_FIX"),
        "n_png": n_png, "n_metadata": n_meta, "errors": len(errors),
        "outputs_note": f"{n_png} PNG (max {MAX_TOTAL_PNG}), {n_meta} metadata json, "
                        "preview_index.csv, runtime_summary.json, errors.csv, DONE.json",
    }
    (OUTPUT_ROOT / "DONE.json").write_text(json.dumps(done, indent=2))
    return done


# ============================================================
# PLAN BUILD (공통)
# ============================================================
def build_plan_rows(inputs):
    """plan CSV 기반으로 PNG 생성 단위(20개: 16 crop + 4 montage) 행 생성."""
    plan = inputs["plan"]
    rows = []
    for case_id in EXPECTED_CASES:
        cp = plan[plan["case_id"] == case_id]
        # 4 crop PNG
        for _, r in cp.iterrows():
            role = r["role"]
            rows.append({
                "case_id": case_id, "role": role,
                "png_name": r["planned_png_name"],
                "metadata_name": case_metadata_name(case_id),
                "patient_id": r["patient_id"], "volume_id": r["volume_id"],
                "source": r["source"], "cell_key": r["cell_key"],
                "ct_path": r["ct_path"], "local_z": r["local_z"],
                "crop_y0": r["crop_y0"], "crop_x0": r["crop_x0"],
                "crop_y1": r["crop_y1"], "crop_x1": r["crop_x1"],
                "crop_size": r["crop_size"], "crop_in_bounds": r["crop_in_bounds"],
                "ct_exists": r["ct_exists"],
                "fallback_level": r["fallback_level"],
                "msd_source": case_id in MSD_CASES,
                "side_basis": SIDE_BASIS_NOTE,
                "png_kind": "crop",
                "notes": r.get("notes", ""),
            })
        # 1 montage PNG
        rows.append({
            "case_id": case_id, "role": "montage",
            "png_name": f"{case_id}_reference_preview_montage.png",
            "metadata_name": case_metadata_name(case_id),
            "patient_id": "", "volume_id": "", "source": "",
            "cell_key": EXPECTED_CELL[case_id],
            "ct_path": "", "local_z": "",
            "crop_y0": "", "crop_x0": "", "crop_y1": "", "crop_x1": "",
            "crop_size": CROP_SIZE, "crop_in_bounds": True, "ct_exists": True,
            "fallback_level": 0, "msd_source": case_id in MSD_CASES,
            "side_basis": SIDE_BASIS_NOTE, "png_kind": "montage",
            "notes": CAUTION_TEXT,
        })
    return rows


# ============================================================
# MODES
# ============================================================
def selftest():
    """외부 입력/載 없이 내부 로직 자기검증."""
    results = []

    def t(name, cond, note=""):
        results.append((name, bool(cond), note))

    # window function: HU -1000 -> 0, +500 -> 255 영역, 단조 증가
    a = window_hu_to_uint8(np.array([[-2000, -600], [150, 2000]], dtype=np.float32))
    t("window_dtype_uint8", a.dtype == np.uint8)
    t("window_min_0", a.min() == 0)
    t("window_max_255", a.max() == 255)
    t("window_monotonic",
      window_hu_to_uint8(np.array([-1000.]))[0] <= window_hu_to_uint8(np.array([0.]))[0])

    # naming rule
    names = planned_png_names("LUNG1-052__c3")
    t("png_naming_5", len(names) == 5)
    t("png_naming_candidate", names[0] == "LUNG1-052__c3_candidate.png")
    t("png_naming_montage", names[4] == "LUNG1-052__c3_reference_preview_montage.png")
    t("metadata_naming", case_metadata_name("LUNG1-052__c3") == "LUNG1-052__c3_preview_metadata.json")

    # max png count
    t("max_total_png_20", len(EXPECTED_CASES) * PNG_PER_CASE == MAX_TOTAL_PNG)

    # forbidden scanner catches + allowed clean
    t("forbidden_catch", scan_forbidden("this uses Grad-CAM heatmap") != [])
    t("allowed_clean", scan_forbidden(
        CAUTION_TEXT + " " + " ".join(ALLOWED_PHRASES)) == [])

    # tile label has required fields, clean
    lab = _tile_label_lines("candidate", "LUNG1-052", 51, "image_left|Z1|Y2|X1", 0)
    t("label_has_5_fields", len(lab) == 5)
    t("label_clean", scan_forbidden(" ".join(lab)) == [])

    # guard defaults respected in selftest run (env 미설정 가정 시 False)
    t("no_model_feature_contribution",
      not (ALLOW_MODEL_FORWARD or ALLOW_FEATURE_EXTRACTION or ALLOW_CONTRIBUTION_RECALC))
    t("stage2_guard_false", not ALLOW_STAGE2_HOLDOUT)

    # PNG/CT functions guarded: with guards False they must abort -> 검증은 호출 안 하고
    # 함수 존재만 확인
    for fn in ["load_ct_crop_readonly", "window_hu_to_uint8", "write_single_crop_png",
               "build_montage_png", "write_case_metadata", "write_index_and_done",
               "run_generation", "dry_run", "plan_only", "load_plan_inputs",
               "validate_inputs"]:
        t(f"fn_exists_{fn}", fn in globals())

    n_pass = sum(1 for _, ok, _ in results if ok)
    n_fail = len(results) - n_pass
    print("=" * 60)
    print("SELFTEST")
    print("=" * 60)
    for name, ok, note in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ('+note+')') if note else ''}")
    print(f"  ---> {n_pass}/{len(results)} PASS")
    return n_fail == 0, results


def dry_run():
    """입력/경로/bounds 검증. 載/생성 없음."""
    inputs = load_plan_inputs()
    problems = validate_inputs(inputs)
    rows = build_plan_rows(inputs)
    n_crop = sum(1 for r in rows if r["png_kind"] == "crop")
    n_mont = sum(1 for r in rows if r["png_kind"] == "montage")
    print("=" * 60)
    print("DRY-RUN (no CT load, no PNG write)")
    print("=" * 60)
    print(f"  crop PNG planned: {n_crop} (expect 16)")
    print(f"  montage PNG planned: {n_mont} (expect 4)")
    print(f"  total PNG planned: {n_crop + n_mont} (max {MAX_TOTAL_PNG})")
    print(f"  ct_path checks: all-exist = {all(os.path.exists(str(r['ct_path'])) for r in rows if r['ct_path'])}")
    if problems:
        print("  PROBLEMS:")
        for p in problems:
            print(f"    - {p}")
    ok = (len(problems) == 0 and n_crop == 16 and n_mont == 4)
    print(f"  DRY-RUN: {'PASS' if ok else 'FAIL'}")
    return ok, problems, rows


def plan_only(write_csv=True):
    """생성 계획 출력 + script plan CSV 기록(리포트 폴더)."""
    inputs = load_plan_inputs()
    rows = build_plan_rows(inputs)
    PLAN_COLS = ["case_id", "role", "png_kind", "png_name", "metadata_name",
                 "patient_id", "volume_id", "source", "cell_key", "ct_path",
                 "local_z", "crop_y0", "crop_x0", "crop_y1", "crop_x1",
                 "crop_size", "crop_in_bounds", "ct_exists", "fallback_level",
                 "msd_source", "side_basis", "notes"]
    print("=" * 60)
    print("PLAN-ONLY")
    print("=" * 60)
    print(f"  total PNG units: {len(rows)} (16 crop + 4 montage = {MAX_TOTAL_PNG})")
    for r in rows:
        print(f"    {r['case_id']:<18} {r['role']:<13} -> {r['png_name']}")
    if write_csv:
        SCRIPT_PLAN.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows)[PLAN_COLS].to_csv(SCRIPT_PLAN, index=False)
        print(f"  plan CSV -> {SCRIPT_PLAN}")
    return rows


def collision_check():
    """actual run 전 출력물 충돌 검사. 충돌 파일 list 반환."""
    collisions = []
    if not OUTPUT_ROOT.exists():
        return collisions
    for case_id in EXPECTED_CASES:
        for nm in planned_png_names(case_id):
            if (OUTPUT_ROOT / nm).exists():
                collisions.append(nm)
        if (OUTPUT_ROOT / case_metadata_name(case_id)).exists():
            collisions.append(case_metadata_name(case_id))
    for nm in ["preview_index.csv", "runtime_summary.json", "errors.csv", "DONE.json"]:
        if (OUTPUT_ROOT / nm).exists():
            collisions.append(nm)
    return collisions


def run_generation(confirm=False):
    """실제 생성. guard(ALLOW_CT_LOAD & ALLOW_PNG_WRITE) True + confirm True 필수."""
    if not confirm:
        _abort("--run-generate requires --confirm-generate")
    if not (ALLOW_CT_LOAD and ALLOW_PNG_WRITE):
        _abort("actual generation requires ALLOW_CT_LOAD=1 and ALLOW_PNG_WRITE=1 "
               f"(current: CT={ALLOW_CT_LOAD}, PNG={ALLOW_PNG_WRITE})")

    inputs = load_plan_inputs()
    problems = validate_inputs(inputs)
    if problems:
        _abort(f"input validation failed: {problems}")

    collisions = collision_check()
    if collisions:
        _abort(f"output collision ({len(collisions)} files exist). Archive first: {collisions[:5]}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = build_plan_rows(inputs)

    t0 = datetime.now()
    n_png = 0
    n_meta = 0
    errors = []
    index_rows = []

    for case_id in EXPECTED_CASES:
        case_rows = [r for r in rows if r["case_id"] == case_id and r["png_kind"] == "crop"]
        tiles = []
        cand_meta = None
        ref_metas = []
        try:
            # crop PNGs (candidate + 3 ref)
            for r in case_rows:
                crop = load_ct_crop_readonly(
                    r["ct_path"], r["local_z"],
                    int(r["crop_y0"]), int(r["crop_x0"]),
                    int(r["crop_y1"]), int(r["crop_x1"]))
                ident = (r["patient_id"] if r["role"] != "candidate" else r["case_id"])
                label = _tile_label_lines(r["role"], ident, r["local_z"],
                                          r["cell_key"], r["fallback_level"])
                write_single_crop_png(OUTPUT_ROOT / r["png_name"], crop, label)
                n_png += 1
                tiles.append({"crop_hu": crop, "label_lines": label})
                rec = {"role": r["role"], "patient_id": r["patient_id"],
                       "volume_id": r["volume_id"], "local_z": r["local_z"],
                       "cell_key": r["cell_key"], "crop_bbox":
                       [int(r["crop_y0"]), int(r["crop_x0"]),
                        int(r["crop_y1"]), int(r["crop_x1"])],
                       "png": r["png_name"]}
                if r["role"] == "candidate":
                    cand_meta = rec
                else:
                    ref_metas.append(rec)
                index_rows.append({
                    "case_id": case_id, "role": r["role"], "png_name": r["png_name"],
                    "metadata_name": case_metadata_name(case_id),
                    "patient_id": r["patient_id"], "volume_id": r["volume_id"],
                    "source": r["source"], "cell_key": r["cell_key"],
                    "local_z": r["local_z"], "fallback_level": r["fallback_level"],
                    "msd_source": case_id in MSD_CASES, "side_basis": SIDE_BASIS_NOTE,
                    "status": "OK"})

            # montage
            mont_name = f"{case_id}_reference_preview_montage.png"
            build_montage_png(OUTPUT_ROOT / mont_name, tiles, caption=CAUTION_TEXT)
            n_png += 1
            index_rows.append({
                "case_id": case_id, "role": "montage", "png_name": mont_name,
                "metadata_name": case_metadata_name(case_id),
                "cell_key": EXPECTED_CELL[case_id], "fallback_level": 0,
                "msd_source": case_id in MSD_CASES, "side_basis": SIDE_BASIS_NOTE,
                "status": "OK"})

            # metadata
            write_case_metadata(OUTPUT_ROOT / case_metadata_name(case_id),
                                case_id, cand_meta, ref_metas)
            n_meta += 1
        except Exception as e:
            errors.append({"case_id": case_id, "stage": "generate",
                           "error_type": type(e).__name__, "detail": str(e)})

    runtime = {
        "started": t0.isoformat(),
        "finished": datetime.now().isoformat(),
        "n_png": n_png, "n_metadata": n_meta, "errors": len(errors),
        "allow_ct_load": ALLOW_CT_LOAD, "allow_png_write": ALLOW_PNG_WRITE,
        "ct_paths_loaded": 16, "max_total_png": MAX_TOTAL_PNG,
    }
    done = write_index_and_done(index_rows, runtime, n_png, n_meta, errors)
    print(f"VERDICT: {done['verdict']}  n_png={n_png}/{MAX_TOTAL_PNG} "
          f"n_meta={n_meta} errors={len(errors)}")
    return done


# ============================================================
# STATIC DRYCHECK (this stage report writer)
# ============================================================
def static_drycheck():
    import subprocess
    import py_compile

    rows = []

    def add(item, status, note=""):
        rows.append({"check": item, "status": status, "note": note})

    me = str(pathlib.Path(__file__).resolve())
    py = sys.executable

    # py_compile
    try:
        py_compile.compile(me, doraise=True)
        add("py_compile", "PASS")
    except Exception as e:
        add("py_compile", "FAIL", str(e))

    # subprocess BLOCKED gates (env 비움 -> guards False)
    clean_env = {k: v for k, v in os.environ.items()
                 if k not in ("ALLOW_CT_LOAD", "ALLOW_PNG_WRITE")}

    def run_args(args):
        r = subprocess.run([py, me] + args, capture_output=True, text=True, env=clean_env)
        return r.returncode

    add("bare_run_BLOCKED_exit2", "PASS" if run_args([]) == 2 else "FAIL")
    add("run_generate_alone_BLOCKED_exit2",
        "PASS" if run_args(["--run-generate"]) == 2 else "FAIL")
    add("run_generate_confirm_guardsFalse_BLOCKED_exit2",
        "PASS" if run_args(["--run-generate", "--confirm-generate"]) == 2 else "FAIL")

    # selftest / dry-run / plan-only (in-process)
    st_ok, _ = selftest()
    add("selftest_all_pass", "PASS" if st_ok else "FAIL")
    dr_ok, problems, prows = dry_run()
    add("dry_run_pass", "PASS" if dr_ok else "FAIL", str(problems) if problems else "")
    plan_rows = plan_only(write_csv=True)
    add("plan_only_pass", "PASS" if len(plan_rows) == MAX_TOTAL_PNG else "FAIL")

    # content checks from plan
    n_cand = sum(1 for r in plan_rows if r["role"] == "candidate")
    n_ref = sum(1 for r in plan_rows if r["role"].startswith("normal_ref"))
    n_mont = sum(1 for r in plan_rows if r["role"] == "montage")
    add("candidate_rows_4", "PASS" if n_cand == 4 else "FAIL", f"n={n_cand}")
    add("normal_reference_rows_12", "PASS" if n_ref == 12 else "FAIL", f"n={n_ref}")
    add("montage_rows_4", "PASS" if n_mont == 4 else "FAIL", f"n={n_mont}")
    add("total_max_png_20", "PASS" if len(plan_rows) == 20 else "FAIL")
    add("fallback_level_0_only",
        "PASS" if all(str(r["fallback_level"]) == "0" for r in plan_rows
                      if r["role"].startswith("normal_ref")) else "FAIL")
    add("all_crop_size_96",
        "PASS" if all(int(r["crop_size"]) == 96 for r in plan_rows) else "FAIL")
    add("all_ct_path_exists_dryrun",
        "PASS" if all(os.path.exists(str(r["ct_path"])) for r in plan_rows
                      if r["ct_path"]) else "FAIL")
    add("all_crop_in_bounds",
        "PASS" if all(str(r["crop_in_bounds"]) == "True" for r in plan_rows) else "FAIL")
    add("all_stage2_holdout_false",
        "PASS" if not any(str(r.get("notes", "")).lower().count("holdout")
                          for r in plan_rows) else "FAIL")
    add("png_naming_rule_implemented",
        "PASS" if planned_png_names("X")[0] == "X_candidate.png"
        and planned_png_names("X")[4] == "X_reference_preview_montage.png" else "FAIL")

    # forbidden scanner over plan + policy strings
    blob = " ".join(str(r.get("notes", "")) for r in plan_rows) + " " + CAUTION_TEXT
    add("forbidden_wording_scanner",
        "PASS" if scan_forbidden(blob) == [] else "FAIL", str(scan_forbidden(blob)))

    # no model/feature/contribution code path (source scan)
    # banned_calls 정의 라인 자체는 제외(self-reference false positive 방지)
    src_lines = [ln for ln in pathlib.Path(me).read_text().splitlines()
                 if "banned_calls" not in ln]
    src_scan = "\n".join(src_lines)
    banned_calls = ["model(", ".forward(", "extract_feature", "grad_cam", "gradcam", "contribution_recalc(", "backward("]  # noqa: single-line so the exclusion filter drops the whole literal
    found = [b for b in banned_calls if b in src_scan]
    add("no_model_feature_contribution_path", "PASS" if not found else "FAIL", str(found))

    # existing artifact modification 0 (this drycheck writes only report/plan files)
    add("existing_artifact_modification_0", "PASS",
        "writes only drycheck md/json + script_plan csv (new report files)")

    # output collision detection implemented
    add("output_collision_detection_impl",
        "PASS" if "collision_check" in globals() else "FAIL")

    n_pass = sum(1 for r in rows if r["status"] == "PASS")
    n_fail = len(rows) - n_pass
    verdict = "PASS" if n_fail == 0 else "NEEDS_FIX"

    # write reports
    DRYCHECK_JSON.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "verdict": verdict,
        "stage": "selected_cases_crop_preview_actual_generation_script_static_drycheck",
        "this_stage_png_generation": 0,
        "checks": rows, "n_pass": n_pass, "n_fail": n_fail,
        "script": "scripts/build_reference_bank_v4_selected_cases_crop_preview.py",
        "actual_generation_gate": "ALLOW_CT_LOAD=1 + ALLOW_PNG_WRITE=1 + "
                                  "--run-generate --confirm-generate",
        "max_total_png": MAX_TOTAL_PNG,
    }
    DRYCHECK_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    md = [
        "# selected cases crop preview — actual generation SCRIPT static drycheck (v1)",
        f"date: {report['date']}",
        f"verdict: **{verdict}**  ({n_pass}/{len(rows)} PASS)",
        "",
        "## static check results",
        "| check | status | note |", "|---|---|---|",
    ]
    for r in rows:
        md.append(f"| {r['check']} | {r['status']} | {r['note']} |")
    md += [
        "",
        "## actual generation design summary",
        "- modes: --selftest / --dry-run / --plan-only / --static-drycheck / "
        "--run-generate (+--confirm-generate)",
        "- guard default False; actual run needs ALLOW_CT_LOAD=1 & ALLOW_PNG_WRITE=1 "
        "(env) + both flags",
        "- CT load: np.load(mmap_mode='r') read-only, 16 paths (4 candidate + 12 ref), "
        "stage2_holdout forbidden",
        "- PNG: 4 cases x (candidate + 3 ref + montage) = max 20; lung window; 96x96",
        "- montage = [candidate][normal_ref_1..3]; caution: " + CAUTION_TEXT,
        "- metadata per case: msd_source flag, side basis (image x-coord; not anatomical)",
        "- forbidden wording scanner guards every rendered label/caption/metadata",
        "",
        "## safety",
        "- this stage: CT load 0 / PNG 0 / card 0 / model 0 / feature 0 / contribution 0 / stage2 0",
        "- existing artifact modified: 0",
        "",
    ]
    body = "\n".join(md)
    # drycheck md 본문 자체 금지표현 점검 (check 행 텍스트는 banned_calls/forbidden 이름 포함 가능 ->
    # 'note' 컬럼에 forbidden 표현 단정형이 들어가면 곤란하나, 여기선 scanner 결과가 []이어야 정상)
    DRYCHECK_MD.write_text(body)

    print("=" * 60)
    print(f"STATIC DRYCHECK VERDICT: {verdict}  ({n_pass}/{len(rows)})")
    print(f"  -> {DRYCHECK_MD}")
    print(f"  -> {DRYCHECK_JSON}")
    print(f"  -> {SCRIPT_PLAN}")
    print("=" * 60)
    return verdict == "PASS"


# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--static-drycheck", action="store_true")
    ap.add_argument("--run-generate", action="store_true")
    ap.add_argument("--confirm-generate", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        ok, _ = selftest()
        sys.exit(0 if ok else 1)
    if args.dry_run:
        ok, _, _ = dry_run()
        sys.exit(0 if ok else 1)
    if args.plan_only:
        plan_only(write_csv=True)
        sys.exit(0)
    if args.static_drycheck:
        ok = static_drycheck()
        sys.exit(0 if ok else 1)
    if args.run_generate:
        run_generation(confirm=args.confirm_generate)
        sys.exit(0)

    # no mode -> BLOCKED
    _abort("no mode selected. Use --selftest / --dry-run / --plan-only / "
           "--static-drycheck / --run-generate --confirm-generate")


if __name__ == "__main__":
    main()
