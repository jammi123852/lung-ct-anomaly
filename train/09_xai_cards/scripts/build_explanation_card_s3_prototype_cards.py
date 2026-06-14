#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Explanation Card S3 : prototype explanation card 생성기 (Tier1, 4-panel)

기준:
  - reports/explanation_cards/s3_prototype_card_preflight_v1.md (PASS)
  - manifest: candidates/s3_prototype_manifest_v1/s3_prototype_candidate_manifest_v1.csv (24행/8명)
  - S1 reference bank full(reference_crop_manifest.csv) read-only

설계 결정:
  - Tier1 only. 4-panel: A whole-slice+bbox / B candidate crop+contour / C 같은 bin 정상 reference / D z-1,z,z+1.
  - 권위 키 = safe_id. role 로 normal/lesion CT·mask root 분기.
  - score 는 manifest 값만 사용(원본 score CSV/모델 재호출 없음). lung window center -600/width 1500.
  - 진단명/악성·양성 추정 금지. 금지어 guard(텍스트·JSON 스캔) 적용. lesion GT mask 미사용.
  - ★lesion CT root 는 preflight 에서 실재 확인된 'usable_only' 버전 사용
    (작업지시 표기 _no_dilate_v1 -> 실제 _no_dilate_usable_only_v1).

가드:
  - 플래그 없으면 BLOCKED. --selftest/--dry-run/--plan-only 는 read-only(npy 미열람).
  - --run-cards 는 --confirm-generate 동반 필요. DONE/잔여 산출물 있으면 BLOCKED. --overwrite 없음.
  - manifest 24행만 대상(full/all 옵션 없음). holdout 교집합 0 assert.
  - CT/mask 는 run-cards 에서만 np.load(mmap_mode="r"). 본 단계 --run-cards --confirm-generate 미실행.
"""

import argparse
import csv
import inspect
import json
import os
import sys
from datetime import datetime

import numpy as np

csv.field_size_limit(10 ** 9)

# ----------------------------------------------------------------------------
# 상수
# ----------------------------------------------------------------------------
THRESHOLD_P95 = 14.0921
THRESHOLD_TYPE = "p95"
N_EXPECTED_ROWS = 24
N_EXPECTED_PATIENTS = 8
LUNG_WINDOW_CENTER = -600.0
LUNG_WINDOW_WIDTH = 1500.0
CROP_MARGIN = 16
REF_CROP_MAX = 3
CONTOUR_RGB = (0, 255, 0)

ROLE_NORMAL = "normal_control"
ROLE_LESION = "lesion_candidate"

POSITION_BINS = ("upper_central", "upper_peripheral", "middle_central",
                 "middle_peripheral", "lower_central", "lower_peripheral")

FORBIDDEN_PATH_TOKENS = ("stage2_holdout", "holdout")

# 카드 텍스트/JSON 진단명 금지어 (소문자 비교)
FORBIDDEN_TERMS = (
    "cancer", "malignancy", "malignant", "benign", "adenocarcinoma", "carcinoma",
    "tumor", "tumour", "nodule 확정", "pulmonary nodule 확정", "ground-glass nodule 확정",
    "ggn 확정", "폐암", "악성", "양성", "선암", "종양", "결절로 진단", "유리결절로 진단",
)

CARD_JSON_FIELDS = (
    "prototype_case_id", "prototype_role", "patient_id", "safe_id", "component_id",
    "rank_in_patient", "position_bin", "slice_index_min", "slice_index_max",
    "max_score_slice_index", "z_span", "bbox", "patch_count", "max_padim_score",
    "mean_padim_score", "threshold", "threshold_type", "roi_0_0_patch_ratio_mean",
    "central_peripheral", "left_right_metadata", "normal_reference_crops",
    "card_png_path", "explanation_text", "fp_caution_text", "source_manifest",
    "stage_split_safety_flag", "diagnostic_terms_blocked",
)

INDEX_FIELDS = (
    "prototype_case_id", "prototype_role", "patient_id", "safe_id", "component_id",
    "rank_in_patient", "position_bin", "max_padim_score", "threshold", "z_span",
    "patch_count", "card_png_path", "card_json_path", "status", "error",
)

PLANNED_ARTIFACTS = (
    "cards_png/<prototype_case_id>.png", "cards_json/<prototype_case_id>.json",
    "index_cards.csv", "runtime_summary.json", "errors.csv", "DONE.json",
)

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_BLOCKED = 2

# ----------------------------------------------------------------------------
# 경로
# ----------------------------------------------------------------------------
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_CSV = os.path.join(
    REPO, "outputs/position-aware-padim-v1/candidates/s3_prototype_manifest_v1/s3_prototype_candidate_manifest_v1.csv")
PATIENT_CSV = os.path.join(
    REPO, "outputs/position-aware-padim-v1/candidates/s3_prototype_manifest_v1/s3_prototype_patient_summary_v1.csv")
REF_BANK_FULL = os.path.join(
    REPO, "outputs/position-aware-padim-v1/visualizations/candidate_cards/reference_bank_v1/full")
REF_STATS_CSV = os.path.join(REF_BANK_FULL, "reference_stats_by_position_bin.csv")
REF_CROP_MANIFEST = os.path.join(REF_BANK_FULL, "reference_crop_manifest.csv")
SPLIT_CSV = os.path.join(
    REPO, "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1.csv")

NORMAL_CT_ROOT = (
    "/mnt/c/Users/jinhy/Desktop/"
    "Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1/volumes_npy")
# ★ preflight 에서 실재 확인된 lesion CT root (usable_only)
LESION_CT_ROOT = (
    "/mnt/c/Users/jinhy/Desktop/"
    "NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1/volumes_npy")
MASK_ROOT = os.path.join(
    REPO, "outputs/mip-postprocess-research-v1/masks/refined_roi_v4_20_modeB_all_v1")

OUT_DIR = os.path.join(
    REPO, "outputs/position-aware-padim-v1/visualizations/candidate_cards/s3_prototype_cards_v1")


# ----------------------------------------------------------------------------
# 가드 / 경로 해결 (순수, selftest 대상)
# ----------------------------------------------------------------------------
def safe_path(path):
    low = str(path).replace("\\", "/").lower()
    for tok in FORBIDDEN_PATH_TOKENS:
        if tok in low:
            raise RuntimeError("FORBIDDEN path token '%s' in: %s" % (tok, path))
    return path


def resolve_ct_path(safe_id, role):
    root = NORMAL_CT_ROOT if role == ROLE_NORMAL else LESION_CT_ROOT
    return os.path.join(root, safe_id, "ct_hu.npy")


def resolve_mask_path(safe_id, role):
    sub = "normal" if role == ROLE_NORMAL else "lesion"
    return os.path.join(MASK_ROOT, sub, safe_id, "refined_roi.npy")


def assert_no_holdout(pids, sids, hp, hs):
    ip, is_ = set(pids) & set(hp), set(sids) & set(hs)
    if ip or is_:
        raise RuntimeError("HOLDOUT LEAK -> BLOCKED pid=%s sid=%s" % (sorted(ip)[:5], sorted(is_)[:5]))
    return True


# ----------------------------------------------------------------------------
# 텍스트 / 금지어 (순수, selftest 대상)
# ----------------------------------------------------------------------------
def scan_forbidden_terms(text):
    low = str(text).lower()
    return [t for t in FORBIDDEN_TERMS if t.lower() in low]


def build_explanation_text(row):
    rd = row["prototype_role"]
    if rd == ROLE_NORMAL:
        head = ("Normal control / FP review case. 정상 control FP 검토용. "
                "This card is for reviewing false-positive-like high PaDiM response in a normal case.")
    else:
        head = "Stage1-dev candidate for explanation-card prototype."
    contrast = ("같은 위치 bin의 정상 reference와 비교했을 때, 이 후보는 PaDiM 이상 점수가 높게 나타난 영역입니다. "
                "(Compared with normal reference crops from the same position_bin, "
                "this candidate shows a higher PaDiM anomaly score.)")
    meas = ("position_bin=%s, rank=%s, slice=%s~%s, z_span=%s, patch_count=%s, "
            "max_padim_score=%.4f (threshold=%s p95), roi_ratio_mean=%s, %s, %s." % (
                row["position_bin"], row["rank_in_patient"], row["slice_index_min"],
                row["slice_index_max"], row["z_span"], row["patch_count"],
                float(row["max_padim_score"]), THRESHOLD_P95, row["roi_0_0_patch_ratio_mean"],
                row["central_peripheral"], row["left_right_metadata"]))
    return "%s %s %s" % (head, contrast, meas)


def build_fp_caution_text(row):
    parts = []
    if row["prototype_role"] == ROLE_NORMAL:
        parts.append("정상 control / structural FP 검토용 (not an abnormality).")
    if str(row["central_peripheral"]).strip().endswith("peripheral") or \
            str(row["position_bin"]).endswith("peripheral"):
        parts.append("경계/흉막 인접 가능성 (boundary/pleura-adjacent possible).")
    return " ".join(parts) if parts else "(no specific caution)"


# ----------------------------------------------------------------------------
# 영상 헬퍼 (순수, selftest 대상; npy 미열람 — 배열만 받음)
# ----------------------------------------------------------------------------
def window_to_uint8(hu, center=LUNG_WINDOW_CENTER, width=LUNG_WINDOW_WIDTH):
    lo, hi = center - width / 2.0, center + width / 2.0
    x = (np.asarray(hu, dtype=np.float32) - lo) / (hi - lo)
    return (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)


def clip_bbox(y0, x0, y1, x1, H, W, margin=CROP_MARGIN):
    y0 = max(0, int(y0) - margin); x0 = max(0, int(x0) - margin)
    y1 = min(int(H), int(y1) + margin); x1 = min(int(W), int(x1) + margin)
    return y0, x0, y1, x1


def z_context_slices(z, depth):
    return [s for s in (int(z) - 1, int(z), int(z) + 1) if 0 <= s < int(depth)]


def mask_contour(mask_bool):
    m = np.asarray(mask_bool, dtype=bool)
    er = np.zeros_like(m)
    if m.shape[0] >= 3 and m.shape[1] >= 3:
        er[1:-1, 1:-1] = (m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1]
                          & m[1:-1, :-2] & m[1:-1, 2:])
    return m & (~er)


def select_reference_crops(ref_rows, position_bin, k=REF_CROP_MAX):
    same = [r for r in ref_rows if r.get("position_bin") == position_bin]
    return same[:k]


# ----------------------------------------------------------------------------
# read-only 로더
# ----------------------------------------------------------------------------
def load_holdout_denylist():
    hp, hs = set(), set()
    with open(safe_path(SPLIT_CSV), "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f); r.fieldnames = [c.strip() for c in r.fieldnames]
        for row in r:
            if (row.get("stage_split") or "").strip() == "stage2_holdout":
                if row.get("patient_id"): hp.add(row["patient_id"].strip())
                if row.get("safe_id"): hs.add(row["safe_id"].strip())
    return hp, hs


def load_manifest():
    with open(safe_path(MANIFEST_CSV), "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_reference_crop_rows():
    with open(safe_path(REF_CROP_MANIFEST), "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


# ----------------------------------------------------------------------------
# 실제 카드 생성 (--run-cards --confirm-generate; 본 단계 미실행)
# ----------------------------------------------------------------------------
def _generate_cards(out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    png_dir = os.path.join(out_dir, "cards_png")
    json_dir = os.path.join(out_dir, "cards_json")
    index_path = os.path.join(out_dir, "index_cards.csv")
    runtime_path = os.path.join(out_dir, "runtime_summary.json")
    errors_path = os.path.join(out_dir, "errors.csv")
    done_path = os.path.join(out_dir, "DONE.json")

    if os.path.exists(safe_path(done_path)):
        sys.stderr.write("[BLOCKED] DONE.json 존재: %s\n" % done_path); return EXIT_BLOCKED
    if os.path.isdir(safe_path(out_dir)):
        leftovers = [p for p in (index_path, runtime_path, errors_path) if os.path.exists(p)]
        for d in (png_dir, json_dir):
            if os.path.isdir(d) and any(os.scandir(safe_path(d))):
                leftovers.append(d)
        if leftovers:
            sys.stderr.write("[BLOCKED] 잔여 산출물 존재: %s\n" % leftovers); return EXIT_BLOCKED

    hp, hs = load_holdout_denylist()
    rows = load_manifest()
    ref_rows = load_reference_crop_rows()

    # 대상 무결성 + holdout assert
    pids = set(r["patient_id"].strip() for r in rows)
    sids = set(r["safe_id"].strip() for r in rows)
    assert_no_holdout(pids, sids, hp, hs)

    os.makedirs(safe_path(png_dir), exist_ok=True)
    os.makedirs(safe_path(json_dir), exist_ok=True)

    errors, index_rows = [], []
    started = datetime.now()
    vol_cache = {}  # safe_id -> (ct_mmap, mask_mmap)

    def _load_vol(safe_id, role):
        if safe_id not in vol_cache:
            ct = np.load(safe_path(resolve_ct_path(safe_id, role)), mmap_mode="r")
            mask = np.load(safe_path(resolve_mask_path(safe_id, role)), mmap_mode="r")
            vol_cache[safe_id] = (ct, mask)
        return vol_cache[safe_id]

    n_ok = 0
    for row in rows:
        cid = row["prototype_case_id"]
        role = row["prototype_role"]
        safe_id = row["safe_id"].strip()
        png_rel = os.path.join("cards_png", "%s.png" % cid)
        json_rel = os.path.join("cards_json", "%s.json" % cid)
        png_abs = os.path.join(out_dir, png_rel)
        json_abs = os.path.join(out_dir, json_rel)
        try:
            ct, mask = _load_vol(safe_id, role)
            depth, H, W = ct.shape[0], ct.shape[1], ct.shape[2]
            zc = int(row["max_score_slice_index"])
            zc = min(max(zc, 0), depth - 1)
            y0, x0, y1, x1 = int(row["y0"]), int(row["x0"]), int(row["y1"]), int(row["x1"])
            cy0, cx0, cy1, cx1 = clip_bbox(y0, x0, y1, x1, H, W)
            zs = z_context_slices(zc, depth)

            ref_sel = select_reference_crops(ref_rows, row["position_bin"])
            expl = build_explanation_text(row)
            caution = build_fp_caution_text(row)
            bad = scan_forbidden_terms(expl) + scan_forbidden_terms(caution)
            if bad:
                raise RuntimeError("forbidden diagnostic term in text: %s" % bad)

            # ---- 4-panel ----
            fig, axes = plt.subplots(2, 2, figsize=(11, 11))
            base = window_to_uint8(np.asarray(ct[zc]))
            # A whole-slice + bbox
            axA = axes[0, 0]; axA.imshow(base, cmap="gray"); axA.set_title(
                "A. slice z=%d  bbox  %s  rank=%s" % (zc, row["position_bin"], row["rank_in_patient"]))
            axA.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="red", lw=1.5))
            axA.axis("off")
            # B candidate crop + v4 contour
            crop_hu = np.asarray(ct[zc, cy0:cy1, cx0:cx1])
            crop_rgb = np.stack([window_to_uint8(crop_hu)] * 3, axis=-1)
            cont = mask_contour(np.asarray(mask[zc, cy0:cy1, cx0:cx1]).astype(bool))
            for ch, v in enumerate(CONTOUR_RGB):
                crop_rgb[..., ch][cont] = v
            axB = axes[0, 1]; axB.imshow(crop_rgb); axB.axis("off")
            axB.set_title("B. candidate crop  score=%.3f (thr=%s)" % (
                float(row["max_padim_score"]), THRESHOLD_P95))
            # C normal reference (같은 bin)
            axC = axes[1, 0]; axC.axis("off")
            axC.set_title("C. normal reference (%s)  n=%d" % (row["position_bin"], len(ref_sel)))
            ref_imgs = []
            for rr in ref_sel:
                rp = os.path.join(REF_BANK_FULL, rr.get("crop_png_path", ""))
                if os.path.isfile(rp):
                    ref_imgs.append(plt.imread(rp))
            if ref_imgs:
                concat = np.concatenate([np.asarray(im)[:, :, :3] if np.asarray(im).ndim == 3
                                         else np.stack([im] * 3, -1) for im in ref_imgs], axis=1)
                axC.imshow(concat)
            # D z-context
            axD = axes[1, 1]; axD.axis("off")
            axD.set_title("D. z-context %s" % zs)
            if zs:
                zimgs = [window_to_uint8(np.asarray(ct[s, cy0:cy1, cx0:cx1])) for s in zs]
                axD.imshow(np.concatenate(zimgs, axis=1), cmap="gray")
            fig.suptitle("%s | %s | %s" % (cid, role, safe_id), fontsize=10)
            fig.tight_layout()
            fig.savefig(safe_path(png_abs), dpi=110); plt.close(fig)

            card = {
                "prototype_case_id": cid, "prototype_role": role,
                "patient_id": row["patient_id"], "safe_id": safe_id,
                "component_id": row["component_id"], "rank_in_patient": row["rank_in_patient"],
                "position_bin": row["position_bin"], "slice_index_min": row["slice_index_min"],
                "slice_index_max": row["slice_index_max"], "max_score_slice_index": row["max_score_slice_index"],
                "z_span": row["z_span"], "bbox": [y0, x0, y1, x1], "patch_count": row["patch_count"],
                "max_padim_score": float(row["max_padim_score"]),
                "mean_padim_score": float(row["mean_padim_score"]),
                "threshold": THRESHOLD_P95, "threshold_type": THRESHOLD_TYPE,
                "roi_0_0_patch_ratio_mean": row["roi_0_0_patch_ratio_mean"],
                "central_peripheral": row["central_peripheral"],
                "left_right_metadata": row["left_right_metadata"],
                "normal_reference_crops": [rr.get("crop_png_path", "") for rr in ref_sel],
                "card_png_path": png_rel, "explanation_text": expl, "fp_caution_text": caution,
                "source_manifest": os.path.relpath(MANIFEST_CSV, REPO),
                "stage_split_safety_flag": row["stage_split_safety_flag"],
                "diagnostic_terms_blocked": list(FORBIDDEN_TERMS),
            }
            with open(safe_path(json_abs), "w", encoding="utf-8") as jf:
                json.dump(card, jf, ensure_ascii=False, indent=2)

            index_rows.append({
                "prototype_case_id": cid, "prototype_role": role, "patient_id": row["patient_id"],
                "safe_id": safe_id, "component_id": row["component_id"],
                "rank_in_patient": row["rank_in_patient"], "position_bin": row["position_bin"],
                "max_padim_score": row["max_padim_score"], "threshold": THRESHOLD_P95,
                "z_span": row["z_span"], "patch_count": row["patch_count"],
                "card_png_path": png_rel, "card_json_path": json_rel, "status": "ok", "error": ""})
            n_ok += 1
        except Exception as e:
            errors.append({"prototype_case_id": cid, "safe_id": safe_id, "stage": "card", "detail": str(e)[:300]})
            index_rows.append({
                "prototype_case_id": cid, "prototype_role": role, "patient_id": row.get("patient_id", ""),
                "safe_id": safe_id, "component_id": row.get("component_id", ""),
                "rank_in_patient": row.get("rank_in_patient", ""), "position_bin": row.get("position_bin", ""),
                "max_padim_score": row.get("max_padim_score", ""), "threshold": THRESHOLD_P95,
                "z_span": row.get("z_span", ""), "patch_count": row.get("patch_count", ""),
                "card_png_path": "", "card_json_path": "", "status": "error", "error": str(e)[:200]})

    with open(safe_path(index_path), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(INDEX_FIELDS)); w.writeheader()
        for r in index_rows: w.writerow(r)
    with open(safe_path(errors_path), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["prototype_case_id", "safe_id", "stage", "detail"]); w.writeheader()
        for e in errors: w.writerow(e)

    summary = {
        "mode": "s3_prototype_cards", "n_manifest_rows": len(rows), "n_cards_ok": n_ok,
        "n_errors": len(errors), "threshold": THRESHOLD_P95, "threshold_type": THRESHOLD_TYPE,
        "holdout_intersection": 0, "unique_volumes": len(vol_cache),
        "source_manifest": os.path.relpath(MANIFEST_CSV, REPO),
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"), "done": True,
    }
    with open(safe_path(runtime_path), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(safe_path(done_path), "w", encoding="utf-8") as f:
        json.dump({"done": True, "summary": summary}, f, ensure_ascii=False, indent=2)

    print("[run-cards] 완료. cards_ok=%d errors=%d -> %s" % (n_ok, len(errors), out_dir))
    return EXIT_OK


# ----------------------------------------------------------------------------
# 모드
# ----------------------------------------------------------------------------
def mode_dry_run():
    print("[MODE] --dry-run (입력 read-only + 출력 계획; npy 미열람)")
    ok = True
    def chk(n, c, e=""):
        nonlocal ok; ok = ok and bool(c); print("  [%s] %s %s" % ("OK" if c else "MISS", n, e))
    chk("manifest", os.path.isfile(safe_path(MANIFEST_CSV)))
    chk("reference crop manifest", os.path.isfile(safe_path(REF_CROP_MANIFEST)))
    chk("split", os.path.isfile(safe_path(SPLIT_CSV)))
    chk("출력 DONE 부재", not os.path.exists(os.path.join(OUT_DIR, "DONE.json")), "(존재시 BLOCKED)")
    print("  [PLAN] 출력:", os.path.relpath(OUT_DIR, REPO))
    for a in PLANNED_ARTIFACTS: print("     -", a)
    return EXIT_OK if ok else EXIT_FAIL


def mode_plan_only():
    print("[MODE] --plan-only (manifest 24 + CT/mask path 존재만; npy 미열람)")
    rows = load_manifest()
    hp, hs = load_holdout_denylist()
    pids = set(r["patient_id"].strip() for r in rows)
    sids = set(r["safe_id"].strip() for r in rows)
    n_pat = len(sids)
    print("  manifest rows:", len(rows), "(기대 %d)" % N_EXPECTED_ROWS)
    print("  patients:", n_pat, "(기대 %d)" % N_EXPECTED_PATIENTS)
    try:
        assert_no_holdout(pids, sids, hp, hs); inter = 0
    except RuntimeError as e:
        print("  [BLOCKED]", e); return EXIT_BLOCKED
    ct_ok = mask_ok = 0
    for r in rows:
        sid, role = r["safe_id"].strip(), r["prototype_role"]
        if os.path.isfile(safe_path(resolve_ct_path(sid, role))): ct_ok += 1
        if os.path.isfile(safe_path(resolve_mask_path(sid, role))): mask_ok += 1
    print("  CT path 존재:", ct_ok, "/", len(rows))
    print("  mask path 존재:", mask_ok, "/", len(rows))
    print("  holdout 교집합:", inter)
    ok = (len(rows) == N_EXPECTED_ROWS and n_pat == N_EXPECTED_PATIENTS
          and ct_ok == len(rows) and mask_ok == len(rows) and inter == 0)
    print("  [%s] plan readiness" % ("PASS" if ok else "CHECK"))
    return EXIT_OK if ok else EXIT_FAIL


def mode_selftest():
    print("[MODE] --selftest")
    results = []
    def expect(n, c):
        results.append(bool(c)); print("  [%s] %s" % ("PASS" if c else "FAIL", n))

    # 1 forbidden path guard
    g = True
    for p in ("a/stage2_holdout/x", "b/holdout.csv"):
        try: safe_path(p); g = False
        except RuntimeError: pass
    expect("forbidden path guard(holdout)", g)
    expect("threshold 14.0921", THRESHOLD_P95 == 14.0921)

    # 4,5 manifest/patient 기대 상수
    expect("기대 row=24 / patient=8", N_EXPECTED_ROWS == 24 and N_EXPECTED_PATIENTS == 8)

    # 6 holdout 교집합 차단
    blk = False
    try: assert_no_holdout({"H"}, set(), {"H"}, set())
    except RuntimeError: blk = True
    expect("holdout 교집합 차단", blk)
    expect("holdout 무교집합 통과", assert_no_holdout({"P"}, {"S"}, {"H"}, {"HS"}) is True)

    # 7,18 금지어 guard
    expect("forbidden term 검출", "adenocarcinoma" in scan_forbidden_terms("this is adenocarcinoma here"))
    expect("forbidden term 한국어 검출", "악성" in scan_forbidden_terms("악성 의심"))
    expect("clean 텍스트 통과", scan_forbidden_terms("higher PaDiM anomaly score") == [])

    # 8,9 role 문구
    nrow = {"prototype_role": ROLE_NORMAL, "position_bin": "upper_central", "rank_in_patient": "1",
            "slice_index_min": "10", "slice_index_max": "12", "z_span": "2", "patch_count": "5",
            "max_padim_score": "30.0", "roi_0_0_patch_ratio_mean": "1.0",
            "central_peripheral": "central", "left_right_metadata": "L"}
    lrow = dict(nrow, prototype_role=ROLE_LESION, position_bin="lower_peripheral", central_peripheral="peripheral")
    ntext = build_explanation_text(nrow); ltext = build_explanation_text(lrow)
    expect("normal_control 문구", "Normal control" in ntext and "정상 control FP" in ntext)
    expect("lesion_candidate 문구", "Stage1-dev candidate" in ltext)
    expect("대조 문장 포함", "PaDiM" in ntext and "정상 reference" in ntext)
    expect("생성 문구 금지어 없음", scan_forbidden_terms(ntext) == [] and scan_forbidden_terms(ltext) == [])
    expect("normal_control fp_caution", "정상 control" in build_fp_caution_text(nrow))
    expect("peripheral fp_caution", "경계" in build_fp_caution_text(lrow))

    # 10 bbox clipping
    expect("bbox clip 범위 안", clip_bbox(5, 5, 40, 40, 50, 50, 16) == (0, 0, 50, 50))
    expect("bbox clip 정상", clip_bbox(20, 20, 30, 30, 100, 100, 5) == (15, 15, 35, 35))

    # 11 z-context clipping
    expect("z-context 경계 클립", z_context_slices(0, 5) == [0, 1])
    expect("z-context 상단 클립", z_context_slices(4, 5) == [3, 4])
    expect("z-context 중앙", z_context_slices(10, 50) == [9, 10, 11])

    # 12 lung window uint8
    w = window_to_uint8(np.array([[-1350.0, 150.0, -600.0]]))
    expect("lung window uint8", w.dtype == np.uint8 and w[0, 0] == 0 and w[0, 1] == 255)

    # 13 mask contour
    m = np.ones((4, 4), bool); cont = mask_contour(m)
    expect("mask contour 경계만", bool(cont[0, 0]) and (not bool(cont[1, 1])) and bool(cont.any()))

    # 14 reference crop selection same bin
    refs = [{"position_bin": "upper_central", "crop_png_path": "a"},
            {"position_bin": "upper_central", "crop_png_path": "b"},
            {"position_bin": "lower_central", "crop_png_path": "c"},
            {"position_bin": "upper_central", "crop_png_path": "d"},
            {"position_bin": "upper_central", "crop_png_path": "e"}]
    sel = select_reference_crops(refs, "upper_central")
    expect("reference 같은 bin만/최대3", len(sel) == 3 and all(r["position_bin"] == "upper_central" for r in sel))

    # schema
    expect("CARD_JSON schema 27열", len(CARD_JSON_FIELDS) == 27)
    expect("INDEX schema 15열", len(INDEX_FIELDS) == 15)

    # 소스 정적
    src_gen = inspect.getsource(_generate_cards)
    src_run = inspect.getsource(mode_run_cards)
    src_dry = inspect.getsource(mode_dry_run)
    src_plan = inspect.getsource(mode_plan_only)
    src_self = inspect.getsource(mode_selftest)
    # 15 np.load mmap 은 생성부에만 (selftest 소스는 정적검사 문자열 리터럴이 있어 제외)
    expect("np.load mmap 생성부에만(dry/plan 없음)", 'np.load(' in src_gen
           and 'np.load(' not in src_dry and 'np.load(' not in src_plan)
    expect("생성부 mmap_mode='r'", 'mmap_mode="r"' in src_gen)
    # 16 run-cards 실제 연결
    expect("run-cards 실제 생성 연결", "_generate_cards(OUT_DIR)" in src_run)
    expect("run-cards confirm 가드", "confirm_generate" in src_run and "EXIT_BLOCKED" in src_run)
    expect("placeholder 아님", not any(b in src_gen for b in ("placeholder", "TODO", "구현 자리")))
    # 17 dry/plan/selftest 에 np.load/value 접근 없음 (위 15 와 함께)
    expect("dry/plan value 접근 없음", "ct[" not in src_dry and "ct[" not in src_plan)
    # 가드/산출물
    expect("DONE/잔여 가드", "DONE.json 존재" in src_gen and "잔여 산출물" in src_gen)
    expect("holdout assert 호출", "assert_no_holdout(pids" in src_gen)
    expect("금지어 텍스트 스캔", "scan_forbidden_terms(" in src_gen)
    expect("lesion GT mask 미사용", "lesion_mask" not in src_gen)
    expect("전체 산출물 기록", all(k in src_gen for k in ("cards_png", "cards_json", "index_cards.csv",
            "runtime_summary.json", "errors.csv", "DONE.json")))

    n = sum(1 for x in results if x)
    print("\n[SELFTEST] %d/%d PASS" % (n, len(results)))
    return EXIT_OK if n == len(results) else EXIT_FAIL


def mode_run_cards(confirm_generate):
    if not confirm_generate:
        sys.stderr.write("[BLOCKED] --run-cards 은 --confirm-generate 동반 + 사용자 승인 필요.\n")
        return EXIT_BLOCKED
    return _generate_cards(OUT_DIR)


def build_parser():
    p = argparse.ArgumentParser(description="Explanation Card S3 prototype card 생성기 (가드 필수).")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--run-cards", action="store_true")
    p.add_argument("--confirm-generate", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.selftest: return mode_selftest()
    if args.dry_run: return mode_dry_run()
    if args.plan_only: return mode_plan_only()
    if args.run_cards: return mode_run_cards(args.confirm_generate)
    sys.stderr.write("[BLOCKED] 가드 플래그 필요: --selftest | --dry-run | --plan-only "
                     "| (--run-cards --confirm-generate)\n")
    return EXIT_BLOCKED


if __name__ == "__main__":
    sys.exit(main())
