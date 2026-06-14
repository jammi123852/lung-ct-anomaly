"""
retrieve_dynamic_normal_refs_three_patients.py

목적 (STEP 2):
새 환자 이상 후보(candidate) patch 의 폐 내부 상대 위치가 주어졌을 때,
정상 환자 3명 각각에서 가장 비슷한 폐 내부 상대 위치의 normal slice/patch 를 1개씩 찾아
normal reference 3개 crop PNG 를 생성한다.

핵심 원칙:
- z 는 절대 local_z 가 아니라 lung_z_pct 로 비교한다.
- y/x 는 whole image 좌표가 아니라 lung bbox 상대좌표(y_pct, x_pct)로 비교한다.
- 정상 환자 3명 각각에서 best 1개만 선택한다 (한 환자에서 여러 장 금지).
- same-z matching 아님. diagnostic 아님.
- model forward / feature extraction / score recompute / contribution recalc 금지.
- reference bank PNG 에서 crop 하므로 정상 환자 raw CT load 불필요.
- candidate_patch 는 candidate CT/PNG 가 제공된 경우에만 별도 guard 로 생성.

guard:
- no-args -> BLOCKED. candidate metadata 없음 -> BLOCKED.
- 실제 retrieval 예시 생성은 `--run-retrieve --confirm` (+ 필요한 입력) 일 때만.
- --selftest / --static-drycheck 는 입력/생성 없이 동작.
"""

import os
import sys
import csv
import json
import argparse

CROP_SIZE = 96
HALF = CROP_SIZE // 2  # 48

# 기본 distance 가중치
WZ = 2.0
WY = 1.0
WX = 1.0
SIDE_PENALTY = 0.5      # side mismatch 이고 양쪽 side 가 모두 known 일 때만
QUALITY_PENALTY = 0.2   # ref slice_quality == low

REQUIRED_CANDIDATE_KEYS = [
    "case_id", "candidate_id", "local_z",
    "candidate_bbox_y0", "candidate_bbox_x0", "candidate_bbox_y1", "candidate_bbox_x1",
    "candidate_center_y", "candidate_center_x",
    "candidate_lung_bbox_y0", "candidate_lung_bbox_x0",
    "candidate_lung_bbox_y1", "candidate_lung_bbox_x1",
    "candidate_lung_z_min", "candidate_lung_z_max", "candidate_lung_z_pct",
    "candidate_side", "crop_size",
]

REF_INDEX_REQUIRED_COLUMNS = [
    "patient_alias", "patient_id", "volume_id", "local_z", "png_path",
    "lung_bbox_y0", "lung_bbox_x0", "lung_bbox_y1", "lung_bbox_x1",
    "lung_center_y", "lung_center_x", "lung_z_min", "lung_z_max", "lung_z_pct",
    "valid_lung_slice", "slice_quality", "image_lung_side_available",
]


# --------------------------------------------------------------------------------------
# 순수 함수 (selftest 대상)
# --------------------------------------------------------------------------------------
def safe_pct(val, lo, hi):
    return (val - lo) / max(hi - lo, 1)


def candidate_position(cand):
    """candidate metadata dict -> (z_pct, y_pct, x_pct)."""
    z_pct = float(cand.get("candidate_lung_z_pct",
                           safe_pct(float(cand["local_z"]),
                                    float(cand["candidate_lung_z_min"]),
                                    float(cand["candidate_lung_z_max"]))))
    y_pct = safe_pct(float(cand["candidate_center_y"]),
                     float(cand["candidate_lung_bbox_y0"]),
                     float(cand["candidate_lung_bbox_y1"]))
    x_pct = safe_pct(float(cand["candidate_center_x"]),
                     float(cand["candidate_lung_bbox_x0"]),
                     float(cand["candidate_lung_bbox_x1"]))
    return z_pct, y_pct, x_pct


def side_known(side):
    return side in ("left", "right")


def distance(cand_pos, cand_side, ref_row,
             wz=WZ, wy=WY, wx=WX, side_penalty=SIDE_PENALTY, quality_penalty=QUALITY_PENALTY):
    """cand_pos=(z_pct,y_pct,x_pct). ref_row=dict. 낮을수록 비슷."""
    cz, cy, cx = cand_pos
    rz = float(ref_row["lung_z_pct"])
    ry = safe_pct(float(ref_row["lung_center_y"]),
                  float(ref_row["lung_bbox_y0"]), float(ref_row["lung_bbox_y1"]))
    rx = safe_pct(float(ref_row["lung_center_x"]),
                  float(ref_row["lung_bbox_x0"]), float(ref_row["lung_bbox_x1"]))
    d = wz * abs(cz - rz) + wy * abs(cy - ry) + wx * abs(cx - rx)
    # side: 양쪽 모두 known 이고 mismatch 일 때만 penalty. 불확실하면 penalty 없음.
    rside = str(ref_row.get("image_lung_side_available", ""))
    if side_known(cand_side) and side_known(rside) and cand_side != rside:
        d += side_penalty
    if str(ref_row.get("slice_quality", "ok")) == "low":
        d += quality_penalty
    return d


def select_best_per_patient(cand_pos, cand_side, ref_rows):
    """ref_rows(여러 환자 섞임) -> 환자별 best 1개. valid_lung_slice 만."""
    best = {}  # alias -> (dist, row)
    for r in ref_rows:
        if str(r.get("valid_lung_slice", "True")).lower() in ("false", "0"):
            continue
        a = r["patient_alias"]
        d = distance(cand_pos, cand_side, r)
        if a not in best or d < best[a][0]:
            best[a] = (d, r)
    out = [{"patient_alias": a, "distance": round(dst, 6), "row": row}
           for a, (dst, row) in best.items()]
    out.sort(key=lambda x: x["patient_alias"])
    return out


def crop_bbox_from_pct(y_pct, x_pct, ref_row, H=512, W=512, size=CROP_SIZE):
    """candidate 의 lung-bbox 상대 위치를 ref lung bbox 에 매핑 -> 96x96 crop bbox.
    경계 밖이면 clamp(shift) 처리하고 정책 기록. (512>=96 이므로 pad 불필요, 정책은 남김)"""
    ry0 = float(ref_row["lung_bbox_y0"]); ry1 = float(ref_row["lung_bbox_y1"])
    rx0 = float(ref_row["lung_bbox_x0"]); rx1 = float(ref_row["lung_bbox_x1"])
    ref_cy = ry0 + y_pct * (ry1 - ry0)
    ref_cx = rx0 + x_pct * (rx1 - rx0)
    y0 = int(round(ref_cy)) - size // 2
    x0 = int(round(ref_cx)) - size // 2
    y0_c = max(0, min(y0, H - size))
    x0_c = max(0, min(x0, W - size))
    shift_y = y0_c - y0
    shift_x = x0_c - x0
    pad = (H < size) or (W < size)
    policy = "shift_clamp" if (shift_y or shift_x) else "none"
    if pad:
        policy = "pad"
    return {
        "ref_center_y": round(ref_cy, 2), "ref_center_x": round(ref_cx, 2),
        "crop_y0": y0_c, "crop_x0": x0_c, "crop_y1": y0_c + size, "crop_x1": x0_c + size,
        "shift_y": shift_y, "shift_x": shift_x, "edge_policy": policy,
    }


# --------------------------------------------------------------------------------------
# 입력 로드 / 검증
# --------------------------------------------------------------------------------------
def load_candidate(path):
    if path.endswith(".json"):
        return json.load(open(path))
    with open(path) as f:
        r = list(csv.DictReader(f))
    if not r:
        raise ValueError("candidate csv empty")
    return r[0]


def validate_candidate(cand):
    missing = [k for k in REQUIRED_CANDIDATE_KEYS if k not in cand]
    return missing


def load_ref_index(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    cols = rows[0].keys() if rows else []
    missing = [c for c in REF_INDEX_REQUIRED_COLUMNS if c not in cols]
    return rows, missing


# --------------------------------------------------------------------------------------
# 모드: selftest / static-drycheck
# --------------------------------------------------------------------------------------
def run_selftest():
    checks = []

    def ck(n, c):
        checks.append((n, bool(c)))

    # candidate_position
    cand = {
        "local_z": 100, "candidate_lung_z_min": 80, "candidate_lung_z_max": 180,
        "candidate_lung_z_pct": 0.2,
        "candidate_center_y": 300, "candidate_lung_bbox_y0": 200, "candidate_lung_bbox_y1": 400,
        "candidate_center_x": 150, "candidate_lung_bbox_x0": 100, "candidate_lung_bbox_x1": 300,
        "candidate_side": "left",
    }
    z, y, x = candidate_position(cand)
    ck("cand_z_pct", abs(z - 0.2) < 1e-9)
    ck("cand_y_pct", abs(y - 0.5) < 1e-9)
    ck("cand_x_pct", abs(x - 0.25) < 1e-9)

    # distance: z 가중치 2.0 반영
    ref_same = {"lung_z_pct": 0.2, "lung_center_y": 300, "lung_bbox_y0": 200, "lung_bbox_y1": 400,
                "lung_center_x": 150, "lung_bbox_x0": 100, "lung_bbox_x1": 300,
                "image_lung_side_available": "both", "slice_quality": "ok"}
    ck("dist_zero_when_same", abs(distance((z, y, x), "left", ref_same)) < 1e-9)
    ref_zoff = dict(ref_same); ref_zoff["lung_z_pct"] = 0.3
    ck("dist_z_weight2", abs(distance((z, y, x), "left", ref_zoff) - 2.0 * 0.1) < 1e-9)
    ref_low = dict(ref_same); ref_low["slice_quality"] = "low"
    ck("dist_quality_penalty", abs(distance((z, y, x), "left", ref_low) - 0.2) < 1e-9)
    ref_rs = dict(ref_same); ref_rs["image_lung_side_available"] = "right"
    ck("dist_side_penalty_known", abs(distance((z, y, x), "left", ref_rs) - 0.5) < 1e-9)
    ck("dist_no_side_penalty_when_both", abs(distance((z, y, x), "left", ref_same)) < 1e-9)

    # select best per patient: 환자 3명, 각 2장 -> 결과 3개(환자별 best 1)
    rows = []
    for a, zp, q in [("normal_patient_1", 0.2, "ok"), ("normal_patient_1", 0.9, "ok"),
                     ("normal_patient_2", 0.25, "ok"), ("normal_patient_2", 0.8, "ok"),
                     ("normal_patient_3", 0.5, "ok"), ("normal_patient_3", 0.21, "ok")]:
        rows.append({"patient_alias": a, "lung_z_pct": zp,
                     "lung_center_y": 300, "lung_bbox_y0": 200, "lung_bbox_y1": 400,
                     "lung_center_x": 150, "lung_bbox_x0": 100, "lung_bbox_x1": 300,
                     "image_lung_side_available": "both", "slice_quality": q,
                     "valid_lung_slice": "True"})
    best = select_best_per_patient((z, y, x), "left", rows)
    ck("best_three_patients", len(best) == 3)
    ck("best_one_per_patient", len({b["patient_alias"] for b in best}) == 3)
    # patient_1 best 는 zp=0.2 (dist 0) 이어야
    b1 = [b for b in best if b["patient_alias"] == "normal_patient_1"][0]
    ck("best_picks_closest_z", abs(b1["distance"]) < 1e-9)

    # invalid slice 제외
    rows2 = [{"patient_alias": "p", "lung_z_pct": 0.2, "lung_center_y": 300,
              "lung_bbox_y0": 200, "lung_bbox_y1": 400, "lung_center_x": 150,
              "lung_bbox_x0": 100, "lung_bbox_x1": 300, "image_lung_side_available": "both",
              "slice_quality": "ok", "valid_lung_slice": "False"}]
    ck("invalid_slice_excluded", len(select_best_per_patient((z, y, x), "left", rows2)) == 0)

    # crop 96x96 + edge
    refrow = {"lung_bbox_y0": 200, "lung_bbox_y1": 400, "lung_bbox_x0": 100, "lung_bbox_x1": 300}
    c = crop_bbox_from_pct(0.5, 0.5, refrow)
    ck("crop_h96", c["crop_y1"] - c["crop_y0"] == 96)
    ck("crop_w96", c["crop_x1"] - c["crop_x0"] == 96)
    ck("crop_center_mid", c["crop_y0"] == int(round(300)) - 48)
    # edge: center near 0 -> clamp + shift
    refedge = {"lung_bbox_y0": 0, "lung_bbox_y1": 10, "lung_bbox_x0": 0, "lung_bbox_x1": 10}
    ce = crop_bbox_from_pct(0.0, 0.0, refedge)
    ck("crop_edge_clamp_y0", ce["crop_y0"] == 0)
    ck("crop_edge_size96", ce["crop_y1"] - ce["crop_y0"] == 96 and ce["crop_x1"] - ce["crop_x0"] == 96)
    ck("crop_edge_shift_recorded", ce["shift_y"] != 0 or ce["shift_x"] != 0 or ce["edge_policy"] != "none")
    # far edge clamp
    reffar = {"lung_bbox_y0": 500, "lung_bbox_y1": 512, "lung_bbox_x0": 500, "lung_bbox_x1": 512}
    cf = crop_bbox_from_pct(1.0, 1.0, reffar, H=512, W=512)
    ck("crop_far_clamp", cf["crop_y1"] <= 512 and cf["crop_x1"] <= 512)

    # candidate validation
    full = dict.fromkeys(REQUIRED_CANDIDATE_KEYS, 0)
    ck("validate_full_ok", validate_candidate(full) == [])
    ck("validate_missing", "case_id" in validate_candidate({k: 0 for k in REQUIRED_CANDIDATE_KEYS if k != "case_id"}))

    npass = sum(1 for _, c in checks if c)
    for n, c in checks:
        print(f"  [{'PASS' if c else 'FAIL'}] {n}")
    print(f"SELFTEST: {npass}/{len(checks)} PASS")
    return npass == len(checks)


def run_static_drycheck():
    print("=== STATIC-DRYCHECK ===")
    full = open(__file__, encoding="utf-8").read()
    # 모듈 docstring(금지문구 prose) + 자기 검사 함수 본문은 forbidden 토큰 리터럴을 포함하므로 제외
    d0 = full.find('"""'); d1 = full.find('"""', d0 + 3)
    body = full[d1 + 3:] if (d0 != -1 and d1 != -1) else full
    start = body.find("def run_static_drycheck")
    end = body.find("\ndef run_retrieve", start)
    src = body[:start] + (body[end:] if end != -1 else "")
    checks = []

    def ck(n, c):
        checks.append((n, bool(c)))

    ck("no_model_forward", '.forward(' not in src and 'model(' not in src)
    ck("no_feature_extract", 'extract_feature' not in src and 'featuremap' not in src)
    ck("no_score_recompute", 'mahalanobis' not in src.lower() and 'cov_inv' not in src.lower())
    ck("no_contribution_recalc", 'contribution' not in src.lower())
    ck("no_stage2_holdout", 'stage2_holdout' not in src.lower() or 'stage2_holdout 접근 금지' in src)
    ck("z_uses_lung_z_pct", 'lung_z_pct' in src and 'absolute' not in src.lower())
    ck("best_one_per_patient", 'select_best_per_patient' in src)
    ck("crop_96", 'CROP_SIZE = 96' in src)
    ck("edge_policy", 'edge_policy' in src and 'shift_clamp' in src)
    ck("weights_default", 'WZ = 2.0' in src and 'WY = 1.0' in src and 'WX = 1.0' in src)
    ck("guard_no_args_blocked", 'BLOCKED' in src)
    npass = sum(1 for _, c in checks if c)
    for n, c in checks:
        print(f"  [{'PASS' if c else 'FAIL'}] {n}")
    print(f"STATIC-DRYCHECK: {npass}/{len(checks)} PASS")
    return npass == len(checks)


# --------------------------------------------------------------------------------------
# 실제 retrieval (guarded) — 이번 단계에서는 사용하지 않음 (static/selftest only)
# --------------------------------------------------------------------------------------
def run_retrieve(ref_index_path, candidate_path, out_root, confirm):
    if not confirm:
        print("BLOCKED: 실제 retrieval 예시 생성에는 --confirm 필요.")
        return False
    if not ref_index_path or not os.path.exists(ref_index_path):
        print("BLOCKED: dynamic_reference_slice_index.csv 경로 필요/존재 안 함.")
        return False
    if not candidate_path or not os.path.exists(candidate_path):
        print("BLOCKED: candidate metadata (json/csv) 경로 필요/존재 안 함.")
        return False
    from PIL import Image
    cand = load_candidate(candidate_path)
    missing = validate_candidate(cand)
    if missing:
        print(f"BLOCKED: candidate metadata 필수 컬럼 누락: {missing}")
        return False
    rows, miss_cols = load_ref_index(ref_index_path)
    if miss_cols:
        print(f"BLOCKED: reference index 필수 컬럼 누락: {miss_cols}")
        return False

    pos = candidate_position(cand)
    cand_side = str(cand.get("candidate_side", "unknown"))
    best = select_best_per_patient(pos, cand_side, rows)
    if len(best) == 0:
        print("BLOCKED: 매칭 가능한 정상 slice 없음.")
        return False

    case_id = str(cand["case_id"]); cid = str(cand["candidate_id"])
    ex_dir = os.path.join(out_root, f"{case_id}__{cid}")
    os.makedirs(ex_dir, exist_ok=True)
    ref_bank_root = os.path.dirname(os.path.abspath(ref_index_path))

    from PIL import ImageDraw
    errors = []
    results = []
    ref_tiles = []  # (label, PIL.Image L 96x96)

    for i, b in enumerate(best, 1):
        r = b["row"]
        cb = crop_bbox_from_pct(pos[1], pos[2], r)
        src_png = os.path.join(ref_bank_root, r["png_path"])
        out_png = os.path.join(ex_dir, f"normal_ref_patient{i}_patch.png")
        tile = None
        if os.path.exists(src_png):
            try:
                im = Image.open(src_png).convert("L")
                tile = im.crop((cb["crop_x0"], cb["crop_y0"], cb["crop_x1"], cb["crop_y1"]))
                tile.save(out_png)
            except Exception as e:
                errors.append({"stage": f"crop_patient{i}", "error": repr(e)})
        else:
            errors.append({"stage": f"crop_patient{i}", "error": f"ref png missing: {src_png}"})
        ref_tiles.append((f"p{i}:{b['patient_alias'].split('_')[-1]} z{r['local_z']} d{b['distance']:.3f}", tile))
        results.append({**{"rank": i, "patient_alias": b["patient_alias"],
                           "distance": b["distance"], "ref_local_z": r["local_z"],
                           "ref_lung_z_pct": r["lung_z_pct"],
                           "ref_png_path": r["png_path"], "out_patch": os.path.basename(out_png)}, **cb})

    # candidate_patch: candidate CT/PNG 가 제공된 경우에만 (CT load 금지 -> 미제공 시 skip)
    cand_png = cand.get("candidate_png_path", "")
    candidate_tile = None
    candidate_generated = False
    if cand_png and os.path.exists(cand_png):
        try:
            cim = Image.open(cand_png).convert("L")
            cb0 = cand["candidate_bbox_y0"]; cb1 = cand["candidate_bbox_x0"]
            candidate_tile = cim  # 제공 PNG 그대로(또는 호출측에서 crop)
            candidate_tile.save(os.path.join(ex_dir, "candidate_patch.png"))
            candidate_generated = True
        except Exception as e:
            errors.append({"stage": "candidate_patch", "error": repr(e)})

    # montage (candidate slot + 3 normal refs). ASCII label (폰트 의존 최소화).
    try:
        T = 96; PAD = 8; LBL = 16
        cols_n = 4
        cw = T + PAD
        W = PAD + cols_n * cw
        H = PAD + T + LBL + PAD
        mont = Image.new("RGB", (W, H), (18, 18, 18))
        draw = ImageDraw.Draw(mont)
        slots = [("candidate" + ("" if candidate_generated else "(CT n/a)"), candidate_tile)] + ref_tiles
        for ci, (lbl, tile) in enumerate(slots[:cols_n]):
            x0 = PAD + ci * cw
            y0 = PAD
            if tile is not None:
                mont.paste(tile.convert("L").convert("RGB"), (x0, y0))
            else:
                draw.rectangle([x0, y0, x0 + T, y0 + T], outline=(90, 90, 90), fill=(30, 30, 30))
            draw.text((x0 + 1, y0 + T + 2), lbl[:18], fill=(230, 200, 70))
        mont.save(os.path.join(ex_dir, "dynamic_ref_montage.png"))
    except Exception as e:
        errors.append({"stage": "montage", "error": repr(e)})

    retrieval_json = {
        "case_id": case_id, "candidate_id": cid,
        "candidate_position": {"z_pct": round(pos[0], 6), "y_pct": round(pos[1], 6), "x_pct": round(pos[2], 6)},
        "candidate_side": cand_side, "candidate_patch_generated": candidate_generated,
        "results": results,
        "matching": "lung_z_pct + lung-bbox relative y/x (NOT same-z, NOT whole-image, NOT abs slice index)",
        "weights": {"wz": WZ, "wy": WY, "wx": WX, "side_penalty": SIDE_PENALTY, "quality_penalty": QUALITY_PENALTY},
        "ct_load": False, "model_forward": False, "feature_extraction": False,
        "score_recompute": False, "contribution_recalc": False, "stage2_holdout_access": False,
    }
    with open(os.path.join(ex_dir, "retrieval_result.json"), "w") as f:
        json.dump(retrieval_json, f, indent=2)
    cols = list(results[0].keys()) if results else []
    with open(os.path.join(ex_dir, "retrieval_result.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(results)
    with open(os.path.join(ex_dir, "safety_check.json"), "w") as f:
        json.dump({"ct_load": False, "model_forward": False, "feature_extraction": False,
                   "score_recompute": False, "contribution_recalc": False,
                   "stage2_holdout_access": False, "raw_ct_copied": False,
                   "crop_source": "dynamic reference bank PNG only"}, f, indent=2)
    with open(os.path.join(ex_dir, "errors.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stage", "error"]); w.writeheader(); w.writerows(errors)
    n_patch = sum(1 for r in results if os.path.exists(os.path.join(ex_dir, r["out_patch"])))
    with open(os.path.join(ex_dir, "DONE.json"), "w") as f:
        json.dump({"conditions_ok": len(results) > 0 and len(errors) == 0,
                   "n_refs": len(results), "n_patch_png": n_patch,
                   "montage": os.path.exists(os.path.join(ex_dir, "dynamic_ref_montage.png")),
                   "errors": len(errors)}, f, indent=2)
    print(f"retrieval done: {ex_dir} (refs={len(results)} patch={n_patch} errors={len(errors)})")
    return len(results) > 0 and len(errors) == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--static-drycheck", action="store_true")
    ap.add_argument("--run-retrieve", action="store_true")
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--ref-index", default="")
    ap.add_argument("--candidate", default="")
    ap.add_argument("--out-root", default="")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if run_selftest() else 1)
    if args.static_drycheck:
        sys.exit(0 if run_static_drycheck() else 1)
    if args.run_retrieve:
        ok = run_retrieve(args.ref_index, args.candidate, args.out_root or ".", args.confirm)
        sys.exit(0 if ok else 2)

    print("BLOCKED: 모드 미지정. --selftest / --static-drycheck / "
          "--run-retrieve --confirm --ref-index ... --candidate ... 중 하나 필요.")
    sys.exit(2)


if __name__ == "__main__":
    main()
