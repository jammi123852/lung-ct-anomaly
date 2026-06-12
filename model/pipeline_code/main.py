from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import List
import io
import asyncio
import tempfile
import shutil
import os
import sys
import hashlib
import time
from datetime import datetime
import numpy as np
import pydicom
import torch
import boto3
import json
from botocore.client import Config

# pipeline_code/ 를 sys.path 맨 앞에 추가 → models.*, position_aware_padim.* import 보장
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# TotalSegmentator 마스크 디스크 캐시 (7일 유효)
_TS_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
_TS_CACHE_TTL = 7 * 24 * 3600  # 7일 (초)
os.makedirs(_TS_CACHE_DIR, exist_ok=True)
from models.padim import run_padim, PADIM_THRESHOLD_P90
from models.rd4ad import run_rd4ad, build_lung3ch_crop, _CROP_SIZE, run_rd4ad_e2_spatial_map_base64
from models.card_generator import generate_card_data
from models.nsclc_classifier import run_nsclc_batch
from models.gradcam import compute_gradcam_base64

# ── 정식 전처리(노트북 [1]+[2]) 스위치 ────────────────────────────────────────
# USE_FULL_PREPROCESS=1 일 때만 활성. 업로드 DICOM → orient + z 1mm 리샘플 + TS +
# refined 폐마스크 + roi_0_0 → ct_hu/roi_0_0 (학습과 동일 전처리) → 모델.
# 기본 0 → 기존 inline 전처리 경로 그대로 (원본 보존).
USE_FULL_PREPROCESS = os.getenv("USE_FULL_PREPROCESS", "0").strip().lower() in ("1", "true", "on", "yes")
_PREPROCESS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "cache", "preprocessed")

# 최근 분석된 HU volume 캐시 (단일 사용자 로컬 앱)
_cached_hu_volume: np.ndarray | None = None
_cached_ts_guard_vol: np.ndarray | None = None   # TotalSegmentator 폐 마스크 (bool, Z×H×W)
_cached_lung_zmin: int = 0                        # 폐 마스크 유효 슬라이스 시작
_cached_lung_zmax: int = 0                        # 폐 마스크 유효 슬라이스 끝
_cached_spacing_xy: float = 1.0                   # pixel spacing mm/px (x=y 기준)

app = FastAPI()

# GPU 상태 출력
_cuda_ok = torch.cuda.is_available()
print(f"[LUNAR] device = {'cuda (' + torch.cuda.get_device_name(0) + ')' if _cuda_ok else 'cpu (CUDA 미사용)'}")
print(f"[LUNAR] torch {torch.__version__}")
print(f"[LUNAR] USE_FULL_PREPROCESS = {USE_FULL_PREPROCESS}  (정식 전처리 {'ON' if USE_FULL_PREPROCESS else 'OFF'})")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    dicom_bytes = await file.read()
    padim_result = run_padim(dicom_bytes)
    final_result = run_rd4ad(padim_result)
    return {
        "score": final_result["score"],
        "risk": final_result["risk"],
        "anomaly_patches": final_result["patches"]
    }

@app.post("/analyze_volume")
async def analyze_volume(files: List[UploadFile] = File(...)):
    """
    SSE 스트리밍: 슬라이스별로 결과를 즉시 전송해서 브라우저 타임아웃 방지
    """
    try:
        all_bytes = [await f.read() for f in files]
        print(f"[INFO] analyze_volume: {len(all_bytes)} slices received")

        hu_slices = []
        spacing_x, spacing_y, spacing_z = 1.0, 1.0, 1.0
        for i, b in enumerate(all_bytes):
            ds        = pydicom.dcmread(io.BytesIO(b))
            pixel     = ds.pixel_array.astype(np.float32)
            slope     = float(getattr(ds, "RescaleSlope", 1))
            intercept = float(getattr(ds, "RescaleIntercept", -1024))
            hu_slices.append(pixel * slope + intercept)
            if i == 0:
                ps = getattr(ds, "PixelSpacing", None)
                if ps is not None:
                    spacing_y, spacing_x = float(ps[0]), float(ps[1])
                spacing_z = float(getattr(ds, "SliceThickness",
                                  getattr(ds, "SpacingBetweenSlices", 1.0)))

        hu_volume = np.stack(hu_slices, axis=0)
        del hu_slices

        # 캐시 키: 첫 슬라이스 + 마지막 슬라이스 raw bytes 해시 (빠른 식별)
        _hash_src = all_bytes[0] + all_bytes[-1] if len(all_bytes) > 1 else all_bytes[0]
        ts_cache_key = hashlib.sha256(_hash_src).hexdigest()[:16]
        _preproc_dir = None
        if USE_FULL_PREPROCESS:
            # 정식 전처리는 DICOM 폴더 입력이 필요 → 업로드 바이트를 임시 폴더에 기록
            _preproc_dir = tempfile.mkdtemp(prefix="lunar_pp_upload_")
            for _i, _b in enumerate(all_bytes):
                with open(os.path.join(_preproc_dir, f"{_i:05d}.dcm"), "wb") as _fp:
                    _fp.write(_b)
        del all_bytes

        global _cached_hu_volume, _cached_ts_guard_vol, _cached_lung_zmin, _cached_lung_zmax, _cached_spacing_xy
        loop = asyncio.get_running_loop()

        if USE_FULL_PREPROCESS and _preproc_dir is not None:
            # 정식 전처리: 업로드 DICOM → ct_hu/roi_0_0 (학습과 동일 전처리) → 모델
            ct_hu, ts_guard, organ_exc = await loop.run_in_executor(
                None, _full_preprocess_dicom_dir, _preproc_dir, ts_cache_key, "Upload", "unknown")
            hu_volume = ct_hu
            use_ts = True
            Z, H, W = hu_volume.shape
            _cached_hu_volume = hu_volume
            _cached_ts_guard_vol = ts_guard
            # 폐 범위 = roi_0_0(ts_guard) 존재 슬라이스 (크롭 안 한 전체 볼륨일 수 있음)
            _areas = np.array([int(ts_guard[zz].sum()) for zz in range(Z)])
            _zs = np.where(_areas > 0)[0]
            _cached_lung_zmin = int(_zs.min()) if _zs.size > 0 else 0
            _cached_lung_zmax = int(_zs.max()) if _zs.size > 0 else Z - 1
            _cached_spacing_xy = float(spacing_x)
            print(f"[INFO] FULL_PREPROCESS volume: ({Z},{H},{W}), lung_z=[{_cached_lung_zmin},{_cached_lung_zmax}], cache_key={ts_cache_key}")
        else:
            Z, H, W = hu_volume.shape
            _cached_hu_volume = hu_volume
            _cached_ts_guard_vol = None
            _cached_lung_zmin = 0
            _cached_lung_zmax = Z - 1
            _cached_spacing_xy = float(spacing_x)
            print(f"[INFO] HU volume: ({Z},{H},{W}), spacing: ({spacing_x:.2f},{spacing_y:.2f},{spacing_z:.2f}), cache_key={ts_cache_key}")

            ts_guard, organ_exc, use_ts = await loop.run_in_executor(
                None, _prepare_ts_context, hu_volume, spacing_x, spacing_y, spacing_z, ts_cache_key)
            if use_ts and ts_guard is not None:
                _cached_ts_guard_vol = ts_guard
                # 폐 마스크 유효 슬라이스 범위 계산 (원본 roi_0_0.npy 기준과 동일)
                areas = np.array([int(ts_guard[zz].sum()) for zz in range(Z)])
                zs = np.where(areas > 0)[0]
                if zs.size > 0:
                    _cached_lung_zmin = int(zs.min())
                    _cached_lung_zmax = int(zs.max())

        async def stream_slices():
            z_track: dict = {}   # {(y0,x0): run_len} — 슬라이스 간 연속 추적
            track_sink: list = []
            try:
                for z in range(Z):
                    result = await loop.run_in_executor(
                        None, _process_single_slice,
                        hu_volume, z, Z, H, W, ts_guard, organ_exc, use_ts, z_track,
                        _cached_lung_zmin, _cached_lung_zmax, track_sink
                    )
                    try:
                        payload = json.dumps({'z': z, 'total': Z, 'result': result})
                    except (TypeError, ValueError) as je:
                        print(f"[ERROR] JSON 직렬화 실패 z={z}: {je}")
                        payload = json.dumps({'z': z, 'total': Z, 'result': {'score': 0.0, 'risk': 'Low', 'anomaly_patches': []}})
                    yield f"data: {payload}\n\n"
                # 트랙 집계 top10 + 절약률 (분모 = 폐 z-range 슬라이스)
                _denom = _cached_lung_zmax - _cached_lung_zmin + 1
                track_summary = _compute_track_summary(track_sink, _denom)
                print(f"[TRACKS] n_tracks={track_summary['n_tracks']} "
                      f"top10_covered={track_summary['reduction']['covered_slices']}/"
                      f"{track_summary['reduction']['total_slices']} "
                      f"reduction={track_summary['reduction']['reduction_rate']}")
                yield f"data: {json.dumps({'tracks': track_summary})}\n\n"
                yield 'data: {"done": true}\n\n'
            except asyncio.CancelledError:
                print("[WARN] 스트림 취소됨 (클라이언트 연결 종료)")
                raise
            except Exception as e:
                import traceback
                traceback.print_exc()
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            finally:
                if _preproc_dir is not None:
                    shutil.rmtree(_preproc_dir, ignore_errors=True)

        return StreamingResponse(
            stream_slices(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analyze/status/{task_id}")
async def analyze_status(task_id: str):
    """TotalSegmentator 진행 상태 확인 (향후 비동기 처리용)"""
    return {"task_id": task_id, "status": "processing"}


@app.get("/ct_slice/{z}")
async def get_ct_slice(z: int):
    """캐시된 HU volume에서 lung window(WL=-600,WW=1500) 적용 PNG 반환"""
    from PIL import Image as PILImage
    if _cached_hu_volume is None:
        raise HTTPException(status_code=404, detail="CT volume not cached yet")
    Z, H, W = _cached_hu_volume.shape
    z_clamped = max(0, min(z, Z - 1))
    hu_slice = _cached_hu_volume[z_clamped].astype(np.float32)
    lo, hi = -1350.0, 150.0   # WL=-600, WW=1500 → lo=WL-WW/2, hi=WL+WW/2
    img_arr = np.clip((hu_slice - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    PILImage.fromarray(img_arr, "L").save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png",
                             headers={"Cache-Control": "no-store"})

# ── Backblaze B2 설정 ──────────────────────────────
# 키 우선순위: 환경변수 > 로컬 b2_secrets.json > (없으면 빈 값).
# 평문 하드코딩 제거(#3). 로컬 앱은 b2_secrets.json(깃 제외) 또는 start.bat env 사용.
def _load_b2_secrets() -> dict:
    sec = {}
    _sec_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "b2_secrets.json")
    if os.path.exists(_sec_path):
        try:
            with open(_sec_path, "r", encoding="utf-8") as f:
                sec = json.load(f)
        except Exception as e:
            print(f"[WARN] b2_secrets.json 읽기 실패: {e}")
    return sec

_b2_sec = _load_b2_secrets()
B2_KEY_ID   = os.getenv("B2_KEY_ID",   _b2_sec.get("B2_KEY_ID", ""))
B2_APP_KEY  = os.getenv("B2_APP_KEY",  _b2_sec.get("B2_APP_KEY", ""))
B2_BUCKET   = os.getenv("B2_BUCKET",   _b2_sec.get("B2_BUCKET", "lunar-dicom-storage"))
B2_ENDPOINT = os.getenv("B2_ENDPOINT", _b2_sec.get("B2_ENDPOINT", "https://s3.us-west-004.backblazeb2.com"))
if not B2_KEY_ID or not B2_APP_KEY:
    print("[WARN] B2 키 미설정 — 환경변수(B2_KEY_ID/B2_APP_KEY) 또는 pipeline_code/b2_secrets.json 필요. B2 기능 비활성.")

b2 = boto3.client(
    "s3",
    endpoint_url=B2_ENDPOINT,
    aws_access_key_id=B2_KEY_ID,
    aws_secret_access_key=B2_APP_KEY,
    config=Config(signature_version="s3v4"),
)

# 환자별 분석 상태 (in-memory)
# status: "pending" | "downloading" | "analyzing" | "done" | "error"
_analysis_status: dict = {}
_pending_timers: dict = {}   # patient_id → asyncio.Task


# ── 코어 분석 로직 ────────────────────────────────────────────────────────────

def _ts_cache_path(cache_key: str) -> str:
    return os.path.join(_TS_CACHE_DIR, f"ts_masks_{cache_key}.npz")


def _load_ts_cache(cache_key: str):
    """캐시 히트 → (ts_guard, organ_exc, True). 미스/만료 → None."""
    path = _ts_cache_path(cache_key)
    if not os.path.exists(path):
        return None
    if time.time() - os.path.getmtime(path) > _TS_CACHE_TTL:
        os.remove(path)
        print(f"[TS_CACHE] 만료 삭제: {path}")
        return None
    try:
        data = np.load(path)
        print(f"[TS_CACHE] 캐시 히트: {path}")
        return data["ts_guard"], data["organ_exc"], True
    except Exception as e:
        print(f"[TS_CACHE] 읽기 실패 → 재계산: {e}")
        return None


def _save_ts_cache(cache_key: str, ts_guard: np.ndarray, organ_exc: np.ndarray):
    path = _ts_cache_path(cache_key)
    try:
        np.savez_compressed(path, ts_guard=ts_guard, organ_exc=organ_exc)
        sz_mb = os.path.getsize(path) / 1024 / 1024
        print(f"[TS_CACHE] 저장 완료: {path} ({sz_mb:.1f} MB)")
    except Exception as e:
        print(f"[TS_CACHE] 저장 실패: {e}")


def _prepare_ts_context(hu_volume: np.ndarray,
                        spacing_x: float, spacing_y: float, spacing_z: float,
                        cache_key: str = ""):
    """TotalSegmentator 전체 볼륨 실행 → (ts_guard_vol, organ_exc_vol, use_ts)
    cache_key 있으면 디스크 캐시 히트 시 TotalSegmentator 생략.
    """
    # 캐시 확인
    if cache_key:
        cached = _load_ts_cache(cache_key)
        if cached is not None:
            return cached

    import SimpleITK as sitk
    from models.padim import _ts_lung_guard, _organ_excl, _run_ts
    Z, H, W = hu_volume.shape
    tmp_dir = tempfile.mkdtemp(prefix="lunar_ts_")
    try:
        # HU 전체 볼륨(Z×H×W) → SimpleITK 변환 후 TotalSegmentator 실행
        sitk_img = sitk.GetImageFromArray(hu_volume.astype(np.float32))
        sitk_img.SetSpacing((spacing_x, spacing_y, spacing_z))
        masks = _run_ts(sitk_img, tmp_dir)
        ts_guard = _ts_lung_guard(masks, (Z, H, W))
        organ_exc = _organ_excl(masks, (Z, H, W))
        print(f"[INFO] TotalSegmentator OK — 볼륨 ({Z},{H},{W})")
        if cache_key:
            _save_ts_cache(cache_key, ts_guard, organ_exc)
        return ts_guard, organ_exc, True
    except Exception as e:
        print(f"[WARN] TotalSegmentator 실패 → HU 폴백: {e}")
        return None, None, False
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _process_single_slice(hu_volume: np.ndarray, z: int, Z: int, H: int, W: int,
                          ts_guard_vol, organ_exc_vol, use_ts: bool,
                          z_track_state: dict | None = None,
                          lung_zmin: int = 0, lung_zmax: int | None = None,
                          track_sink: list | None = None) -> dict:
    """슬라이스 1장 처리 → result dict
    z_track_state: {(y0,x0): run_len} — 호출 측에서 유지하며 in-place 수정됨
    track_sink: 주어지면 RD4AD 통과 패치의 트랙 원천값(z,위치,cosine,roi_ratio,P1)을 누적
                (호출 측에서 compute_tracks 로 트랙 집계용)
    """
    from models.padim import (
        _hu_lung_mask, assign_position_bin, _padim_model, _feature_extractor,
    )
    from position_aware_padim.preprocessing import preprocess_ct_slice
    from scipy.ndimage import binary_dilation

    hu_slice = hu_volume[z]
    hu_lung = _hu_lung_mask(hu_slice)
    if use_ts:
        pure_lung = (hu_lung & ts_guard_vol[z]) & ~organ_exc_vol[z]
    else:
        med = np.zeros((H, W), dtype=bool)
        med[int(H*0.2):int(H*0.8), int(W*0.35):int(W*0.65)] = True
        org = binary_dilation(med & (hu_slice > -100), iterations=3)
        pure_lung = hu_lung & ~org

    if pure_lung.sum() < 100:
        return {"score": 0.0, "risk": "Low", "anomaly_patches": []}

    preprocessed = preprocess_ct_slice(hu_slice)
    patch_size, stride = 32, 16
    patch_coords = []
    for y0 in range(0, H - patch_size + 1, stride):
        for x0 in range(0, W - patch_size + 1, stride):
            cy = (y0 * 2 + patch_size) // 2
            cx = (x0 * 2 + patch_size) // 2
            if not pure_lung[cy, cx]:
                continue
            if pure_lung[y0:y0+patch_size, x0:x0+patch_size].sum() >= patch_size * patch_size * 0.5:
                patch_coords.append((y0, x0, y0+patch_size, x0+patch_size))

    if not patch_coords:
        return {"score": 0.0, "risk": "Low", "anomaly_patches": []}

    features_448 = _feature_extractor.extract_patch_features(preprocessed, patch_coords)
    features_100 = features_448[:, _padim_model.selected_feature_indices]

    # ── PaDiM P90 후보 수집 ──────────────────────────────────────────────────────
    padim_hits = {}   # (y0,x0) → {"score", "y1", "x1", "pb", "idx"}
    for i, (y0, x0, y1, x1) in enumerate(patch_coords):
        cy = (y0 + y1) / 2; cx = (x0 + x1) / 2
        pb = assign_position_bin(cy, cx, H, W)
        try:
            score = _padim_model.score_patch(features_100[i], pb)
        except Exception:
            continue
        if score > PADIM_THRESHOLD_P90:
            padim_hits[(y0, x0)] = {"score": float(score), "y1": y1, "x1": x1, "pb": pb}

    # ── z-track 필터: 동일 (y0,x0) 위치가 직전 슬라이스에도 P90 초과인 것만 통과 ──
    # z_track_state: {(y0,x0): run_len} — 슬라이스 간 연속 카운트
    if z_track_state is None:
        z_track_state = {}
    new_state: dict = {}
    for key, hit in padim_hits.items():
        prev_run = z_track_state.get(key, 0)
        new_run = prev_run + 1
        # 50슬라이스 초과 시 리셋 → 100슬라이스 연속이면 50+50 그룹으로 분리
        if new_run > 50:
            new_run = 0
        new_state[key] = new_run
    z_track_state.clear()
    z_track_state.update(new_state)

    # run_len >= 2 인 것만 RD4AD로 넘김
    candidates = []
    for (y0, x0), hit in padim_hits.items():
        run_len = z_track_state.get((y0, x0), 1)
        if run_len < 2:
            continue
        y1, x1 = hit["y1"], hit["x1"]
        ccy = (y0 + y1) // 2; ccx = (x0 + x1) // 2
        cy0_96 = ccy - _CROP_SIZE // 2; cx0_96 = ccx - _CROP_SIZE // 2
        candidates.append({
            "position": {"y0": y0, "x0": x0, "y1": y1, "x1": x1},
            "padim_score": hit["score"],
            "position_bin": hit["pb"],
            "track_len": run_len,
            "image": build_lung3ch_crop(
                hu_volume, z,
                cy0_96, cx0_96, cy0_96 + _CROP_SIZE, cx0_96 + _CROP_SIZE,
            ).tolist(),
        })

    if not candidates:
        return {"score": 0.0, "risk": "Low", "anomaly_patches": []}

    final_result = run_rd4ad({"candidate_patches": candidates})

    # ── P5 스코어링: P1(RD4AD cosine × roi_ratio) × min(track_len,3)/3 ────────
    # 프로젝트 정의: P1 = rd4ad_score × roi_0_0_patch_ratio
    track_by_pos = {(c["position"]["y0"], c["position"]["x0"]): c["track_len"]
                    for c in candidates}
    for rd_patch in final_result["patches"]:
        pk = (rd_patch["position"]["y0"], rd_patch["position"]["x0"])
        tlen = track_by_pos.get(pk, 2)
        py0, px0 = pk
        ry0 = max(0, py0);  ry1 = min(H, py0 + 32)
        rx0 = max(0, px0);  rx1 = min(W, px0 + 32)
        roi_ratio = float(pure_lung[ry0:ry1, rx0:rx1].mean()) if (ry1 > ry0 and rx1 > rx0) else 0.5
        cosine = float(rd_patch["score"])   # RD4AD cosine (roi/tlen 곱하기 전 원천값 = P1 재료)
        if track_sink is not None:
            _pos = rd_patch["position"]
            track_sink.append({"z": z, "y0": _pos["y0"], "x0": _pos["x0"],
                               "y1": _pos["y1"], "x1": _pos["x1"],
                               "cosine": cosine, "roi_ratio": roi_ratio,
                               "P1": cosine * roi_ratio})
        rd_patch["score"] = cosine * roi_ratio * (tlen / 3)

    max_p5 = max((p["score"] for p in final_result["patches"]), default=0.0)
    padim_risk = ("Critical" if max_p5 > 0.45 else
                  "High"     if max_p5 > 0.35 else
                  "Medium"   if max_p5 > 0.20 else "Low")

    _lung_zmax = lung_zmax if lung_zmax is not None else Z - 1
    lz_pct = round((z - lung_zmin) / max(_lung_zmax - lung_zmin, 1), 4)
    # crop_lung_roi_ratio: 학습 시 기준 = roi_0_0.npy (TS 폐 마스크) 비율
    # ts_guard_vol[z]가 가장 가까운 근사 (organ_exc 제외 없는 순수 폐 마스크)
    # TS 실패 시 HU > -950 폴백

    # NSCLC crop은 32px PaDiM 패치 중심 기준 96px 정중앙으로 맞춤
    _NSCLC_HALF = _CROP_SIZE // 2  # 48
    nsclc_inputs = []
    for c in candidates:
        py0, px0 = c["position"]["y0"], c["position"]["x0"]
        py1, px1 = c["position"]["y1"], c["position"]["x1"]
        patch_cy = (py0 + py1) // 2
        patch_cx = (px0 + px1) // 2
        nsclc_y0 = patch_cy - _NSCLC_HALF
        nsclc_x0 = patch_cx - _NSCLC_HALF
        ry0 = max(0, nsclc_y0); ry1 = min(H, nsclc_y0 + _CROP_SIZE)
        rx0 = max(0, nsclc_x0); rx1 = min(W, nsclc_x0 + _CROP_SIZE)
        if use_ts and ts_guard_vol is not None and (ry1 > ry0 and rx1 > rx0):
            roi = float(ts_guard_vol[z, ry0:ry1, rx0:rx1].mean())
        elif ry1 > ry0 and rx1 > rx0:
            roi = float((hu_volume[z, ry0:ry1, rx0:rx1] > -950).mean())
        else:
            roi = 0.5
        nsclc_inputs.append({"z": z, "y0": nsclc_y0, "x0": nsclc_x0,
                              "lung_z_pct": lz_pct, "crop_lung_roi_ratio": roi})
    try:
        nsclc_results = run_nsclc_batch(hu_volume, nsclc_inputs)
    except Exception as e:
        print(f"[WARN] NSCLC z={z}: {e}")
        nsclc_results = [{"prob": None, "label": "unavailable"}] * len(candidates)

    nsclc_by_pos = {(c["position"]["y0"], c["position"]["x0"]): n
                    for c, n in zip(candidates, nsclc_results)}

    patches_out = []
    for rd_patch in final_result["patches"]:
        matched = next((c for c in candidates if c["position"] == rd_patch["position"]), None)
        pk = (rd_patch["position"]["y0"], rd_patch["position"]["x0"])
        nsclc = nsclc_by_pos.get(pk, {"prob": None, "label": "unavailable"})
        patches_out.append({
            "score":       rd_patch["score"],
            "padim_score": matched["padim_score"] if matched else 0.0,
            "position":    rd_patch["position"],
            "nsclc_prob":  nsclc["prob"],
            "nsclc_label": nsclc["label"],
        })

    print(f"[DEBUG] z={z}: risk={padim_risk}, patches={len(patches_out)}")
    return {"score": round(max_p5, 4), "risk": padim_risk, "anomaly_patches": patches_out}


# ── 트랙 집계 + top10 + 절약률 (eval_single_patient.compute_tracks 정본 이식) ──

def compute_tracks(all_raw: list) -> list:
    """RD4AD 통과 패치(run_len>=2)를 트랙으로 집계.
    (y0,x0,y1,x1)로 그룹 → 연속 z run 분할(gap이면 끊음) → len>=2 run만 트랙.
    p5 = (top3 P1 평균) × (track_len/3). p5 내림차순 정렬.
    eval_single_patient.py:compute_tracks 와 동일 로직.
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for r in all_raw:
        groups[(r["y0"], r["x0"], r["y1"], r["x1"])].append(r)
    tracks = []
    for key, grp in groups.items():
        grp.sort(key=lambda x: x["z"])
        runs = []
        cur = [grp[0]]
        for i in range(1, len(grp)):
            if grp[i]["z"] - grp[i - 1]["z"] == 1:
                cur.append(grp[i])
            else:
                runs.append(cur); cur = [grp[i]]
        runs.append(cur)
        for run in runs:
            if len(run) < 2:          # MIN_RUN = 2 (main z-track 필터와 동일)
                continue
            tlen = len(run)
            p1_vals = sorted([r["P1"] for r in run], reverse=True)
            top3_p1 = sum(p1_vals[:3]) / min(3, tlen)
            p5 = top3_p1 * (tlen / 3.0)
            best = max(run, key=lambda x: x["P1"])
            tracks.append({
                "y0": key[0], "x0": key[1], "y1": key[2], "x1": key[3],
                "z_start": run[0]["z"], "z_end": run[-1]["z"],
                "z": best["z"], "track_len": tlen,
                "p5_score": round(float(p5), 6), "cosine": round(float(best["cosine"]), 6),
                "all_z": [r["z"] for r in run],
            })
    tracks.sort(key=lambda x: x["p5_score"], reverse=True)
    return tracks


def _compute_track_summary(track_sink: list, total_slices: int, top_k: int = 10) -> dict:
    """track_sink → 트랙 top_k + 절약률.
    절약률: top_k 트랙이 덮는 슬라이스 합집합 / total_slices.
      read_ratio    = 덮는 슬라이스 수 / 전체 슬라이스   (판독해야 할 비율)
      reduction_rate= 1 - read_ratio                    (줄어든 비율)
    """
    tracks = compute_tracks(track_sink)
    top = tracks[:top_k]
    covered = set()
    for t in top:
        covered.update(t["all_z"])
    n_cov = len(covered)
    total = int(total_slices) if total_slices else 0
    read_ratio = (n_cov / total) if total > 0 else 0.0
    # tracks_ranked: 프론트 topK 셀렉터(1~50)가 동적으로 절약률을 계산할 수 있도록
    # all_z 포함한 상위 50개 트랙을 함께 전송. (top10 모놀로지 방지 = 트랙당 1행)
    ranked = [
        {"rank": i + 1, "p5_score": t["p5_score"], "track_len": t["track_len"],
         "z": t["z"], "z_start": t["z_start"], "z_end": t["z_end"],
         "all_z": t["all_z"],
         "position": {"y0": t["y0"], "x0": t["x0"], "y1": t["y1"], "x1": t["x1"]}}
        for i, t in enumerate(tracks[:50])
    ]
    return {
        "n_tracks": len(tracks),
        "tracks_top10": [
            {k: v for k, v in r.items() if k != "all_z"} for r in ranked[:top_k]
        ],
        "tracks_ranked": ranked,
        "reduction": {
            "covered_slices": n_cov,
            "total_slices": total,
            "read_ratio": round(read_ratio, 4),
            "reduction_rate": round(1.0 - read_ratio, 4),
        },
    }


def _analyze_hu_volume(hu_volume: np.ndarray,
                       spacing_x: float, spacing_y: float, spacing_z: float,
                       cache_key: str = ""):
    """B2 백그라운드 태스크용 동기 래퍼 → (slices, track_summary)"""
    Z, H, W = hu_volume.shape
    ts_guard_vol, organ_exc_vol, use_ts = _prepare_ts_context(
        hu_volume, spacing_x, spacing_y, spacing_z, cache_key)
    lung_zmin, lung_zmax = 0, Z - 1
    if use_ts and ts_guard_vol is not None:
        areas = np.array([int(ts_guard_vol[zz].sum()) for zz in range(Z)])
        zs = np.where(areas > 0)[0]
        if zs.size > 0:
            lung_zmin, lung_zmax = int(zs.min()), int(zs.max())
    z_track: dict = {}
    track_sink: list = []
    slices = [_process_single_slice(hu_volume, z, Z, H, W, ts_guard_vol, organ_exc_vol, use_ts, z_track,
                                    lung_zmin, lung_zmax, track_sink)
              for z in range(Z)]
    # 절약률 분모 = 폐 z-range 슬라이스 수
    denom = (lung_zmax - lung_zmin + 1) if (lung_zmax is not None) else Z
    return slices, _compute_track_summary(track_sink, denom)


# ── 정식 전처리 경로 (USE_FULL_PREPROCESS=1) ──────────────────────────────────

def _full_preprocess_dicom_dir(dicom_dir: str, patient_id: str,
                               group: str = "Upload", label: str = "unknown"):
    """업로드 DICOM 폴더 → 노트북 [1]+[2] 정식 전처리 →
    (ct_hu float32, ts_guard_vol bool, organ_exc_vol bool).

    - ct_hu: orient + z 1mm 리샘플 + 폐 z-range crop 된 HU 볼륨 (학습 ct_hu.npy 동일).
    - roi_0_0: 학습 roi_0_0.npy 동일. 스코어링 루프의 ts_guard 로 주입하고
      organ_exc 는 0 (organ exclusion 이 이미 roi_0_0 에 반영됨).
    - ct_hu/roi_0_0/meta 는 cache/preprocessed/<patient_id>/ 에 저장(재현·디버그용).
    """
    from preprocessing_full.run import preprocess_to_arrays
    ct_hu, roi_0_0, meta = preprocess_to_arrays(
        input_path=dicom_dir, patient_id=patient_id, group=group, label=label,
    )
    try:
        out = os.path.join(_PREPROCESS_DIR, str(patient_id))
        os.makedirs(out, exist_ok=True)
        np.save(os.path.join(out, "ct_hu.npy"), ct_hu)
        np.save(os.path.join(out, "roi_0_0.npy"), roi_0_0)
        with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[PREPROCESS] saved npy → {out}  ct_hu={ct_hu.shape}")
    except Exception as e:
        print(f"[WARN] preprocessed npy 저장 실패: {e}")

    ts_guard_vol = roi_0_0.astype(bool)
    organ_exc_vol = np.zeros_like(ts_guard_vol)
    return ct_hu.astype(np.float32), ts_guard_vol, organ_exc_vol


def _analyze_preprocessed(ct_hu: np.ndarray, ts_guard_vol: np.ndarray,
                          organ_exc_vol: np.ndarray):
    """정식 전처리 결과(ct_hu + roi_0_0=ts_guard)로 슬라이스 분석 → (slices, track_summary).
    LUNG_CROP_ENABLED=False면 ct_hu는 폐 crop 안 된 전체 1mm 볼륨 → 폐 범위를
    roi_0_0(ts_guard) 존재 슬라이스로 계산해 lung_z_pct/절약률 분모에 사용.
    비폐 슬라이스는 _process_single_slice에서 pure_lung.sum()<100으로 자동 skip(Low).
    """
    Z, H, W = ct_hu.shape
    # 실제 폐 범위 = roi_0_0(ts_guard)가 있는 슬라이스 구간
    areas = np.array([int(ts_guard_vol[zz].sum()) for zz in range(Z)])
    zs = np.where(areas > 0)[0]
    lung_zmin, lung_zmax = (int(zs.min()), int(zs.max())) if zs.size > 0 else (0, Z - 1)
    z_track: dict = {}
    track_sink: list = []
    slices = [_process_single_slice(ct_hu, z, Z, H, W, ts_guard_vol, organ_exc_vol, True,
                                    z_track, lung_zmin, lung_zmax, track_sink)
              for z in range(Z)]
    # 절약률 분모 = 실제 폐 범위 슬라이스 수
    denom = lung_zmax - lung_zmin + 1
    return slices, _compute_track_summary(track_sink, denom)


# ── 백그라운드 분석 태스크 ─────────────────────────────────────────────────────

async def _run_analysis_for_patient(patient_id: str):
    """B2에서 DICOM 다운로드 → 분석 → 결과 저장"""
    _analysis_status[patient_id] = {
        "status": "downloading",
        "started_at": datetime.now().isoformat(),
    }
    tmp_dir = tempfile.mkdtemp(prefix=f"lunar_dcm_{patient_id}_")
    try:
        # 1. B2에서 DICOM 파일 목록 수집
        paginator = b2.get_paginator("list_objects_v2")
        dcm_keys = []
        for page in paginator.paginate(Bucket=B2_BUCKET, Prefix=f"dicom/{patient_id}/"):
            for obj in page.get("Contents", []):
                if obj["Key"].lower().endswith(".dcm"):
                    dcm_keys.append(obj["Key"])

        if not dcm_keys:
            _analysis_status[patient_id] = {"status": "error", "error": "DICOM 파일 없음"}
            return

        dcm_keys.sort()

        # 2. 다운로드 (스레드 풀에서 실행)
        def _download():
            local_paths = []
            for key in dcm_keys:
                local = os.path.join(tmp_dir, os.path.basename(key))
                b2.download_file(B2_BUCKET, key, local)
                local_paths.append(local)
            return local_paths

        local_paths = await asyncio.to_thread(_download)

        # 3. HU 볼륨 생성
        _analysis_status[patient_id]["status"] = "analyzing"
        hu_slices = []
        spacing_x, spacing_y, spacing_z = 1.0, 1.0, 1.0
        dicom_name, dicom_dob, dicom_sex = patient_id, "", ""
        for i, path in enumerate(local_paths):
            ds = pydicom.dcmread(path)
            pixel = ds.pixel_array.astype(np.float32)
            slope = float(getattr(ds, "RescaleSlope", 1))
            intercept = float(getattr(ds, "RescaleIntercept", -1024))
            hu_slices.append(pixel * slope + intercept)
            if i == 0:
                ps = getattr(ds, "PixelSpacing", None)
                if ps:
                    spacing_y, spacing_x = float(ps[0]), float(ps[1])
                spacing_z = float(getattr(ds, "SliceThickness",
                                          getattr(ds, "SpacingBetweenSlices", 1.0)))
                dicom_name = str(getattr(ds, "PatientName", patient_id))
                dicom_dob  = str(getattr(ds, "PatientBirthDate", ""))
                dicom_sex  = str(getattr(ds, "PatientSex", ""))

        hu_volume = np.stack(hu_slices, axis=0)
        del hu_slices

        # 4. 분석 (스레드 풀에서 실행 — 무거운 연산이라 event loop 블로킹 방지)
        # patient_id 기반 캐시 키 (B2 태스크는 patient_id가 고정 식별자)
        b2_cache_key = hashlib.sha256(patient_id.encode()).hexdigest()[:16]
        if USE_FULL_PREPROCESS:
            # 업로드 DICOM 폴더(tmp_dir)를 정식 전처리 → ct_hu/roi_0_0 → 모델
            ct_hu, ts_guard_vol, organ_exc_vol = await asyncio.to_thread(
                _full_preprocess_dicom_dir, tmp_dir, patient_id, "Upload",
                dicom_name or "unknown")
            slices, track_summary = await asyncio.to_thread(
                _analyze_preprocessed, ct_hu, ts_guard_vol, organ_exc_vol)
            del ct_hu, ts_guard_vol, organ_exc_vol
        else:
            slices, track_summary = await asyncio.to_thread(
                _analyze_hu_volume, hu_volume, spacing_x, spacing_y, spacing_z, b2_cache_key
            )
        del hu_volume
        print(f"[TRACKS] n_tracks={track_summary['n_tracks']} "
              f"top10_covered={track_summary['reduction']['covered_slices']}/"
              f"{track_summary['reduction']['total_slices']} "
              f"reduction={track_summary['reduction']['reduction_rate']}")

        # 5. 전체 위험도 집계
        risk_order = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
        overall_risk = max((s["risk"] for s in slices), key=lambda r: risk_order.get(r, 0),
                           default="Low")

        # 6. B2에 저장
        record = {
            "patientId":   patient_id,
            "name":        dicom_name,
            "birthdate":   dicom_dob,
            "gender":      dicom_sex,
            "date":        datetime.now().isoformat(),
            "status":      "Completed",
            "source":      "auto_webhook",
            "risk":        overall_risk,
            "opinion":     "",
            "slices":      slices,
            "tracks":      track_summary,
        }
        b2.put_object(
            Bucket=B2_BUCKET,
            Key=f"records/{patient_id}.json",
            Body=json.dumps(record, ensure_ascii=False),
            ContentType="application/json",
        )

        # MD 보고서 자동 저장 (#4: 슬라이스 번호는 정렬 전 원래 z 인덱스를 보존)
        _indexed_slices = list(enumerate(slices))  # (z, slice_dict)
        top_slices = sorted(_indexed_slices, key=lambda zs: zs[1].get("score", 0), reverse=True)[:20]
        risk_emoji = {"Critical": "🟣 위험", "High": "🔴 고위험", "Medium": "🟡 중위험", "Low": "🟢 저위험"}
        md_lines = [
            "# LUNAR 폐 이상탐지 AI 보고서 (자동 분석)",
            "",
            "## 환자 정보",
            f"| 항목 | 내용 |",
            f"|------|------|",
            f"| 환자 ID | {patient_id} |",
            f"| 이름 | {dicom_name} |",
            f"| 생년월일 | {dicom_dob or '-'} |",
            f"| 성별 | {dicom_sex or '-'} |",
            f"| 분석 일시 | {record['date']} |",
            "",
            "## AI 분석 결과",
            f"- **종합 위험도**: {risk_emoji.get(overall_risk, overall_risk)}",
            f"- **총 슬라이스 수**: {len(slices)}",
            "",
            "## 고위험 구역 (상위 20개)",
            "",
            "| 순위 | 슬라이스 | 이상 점수 | 위험도 |",
            "|------|---------|---------|--------|",
        ]
        for i, (z, s) in enumerate(top_slices):
            r = "위험" if s.get("score", 0) > 0.45 else "고위험" if s.get("score", 0) > 0.35 else "중위험" if s.get("score", 0) > 0.20 else "저위험"
            md_lines.append(f"| {i+1} | {z+1} | {s.get('score', 0):.2f} | {r} |")
        md_lines += ["", "---", "*본 보고서는 LUNAR AI 자동 분석 결과입니다. 최종 진단은 전문의가 확인하십시오.*"]
        b2.put_object(
            Bucket=B2_BUCKET,
            Key=f"records/{patient_id}.md",
            Body="\n".join(md_lines).encode("utf-8"),
            ContentType="text/markdown; charset=utf-8",
        )

        _analysis_status[patient_id] = {
            "status":      "done",
            "started_at":  _analysis_status[patient_id].get("started_at"),
            "finished_at": datetime.now().isoformat(),
            "risk":        overall_risk,
        }
        print(f"[AUTO] {patient_id} 분석 완료 — risk={overall_risk}")

    except Exception as e:
        _analysis_status[patient_id] = {"status": "error", "error": str(e)}
        print(f"[AUTO] {patient_id} 분석 오류: {e}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        _pending_timers.pop(patient_id, None)


async def _debounce_and_run(patient_id: str, delay: int = 60):
    """delay초 동안 새 이벤트 없으면 분석 시작 (디바운스)"""
    await asyncio.sleep(delay)
    await _run_analysis_for_patient(patient_id)


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/b2/health")
def b2_health():
    try:
        b2.head_bucket(Bucket=B2_BUCKET)
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/b2/save-record")
async def save_record(request: Request):
    data = await request.json()
    patient_id = data.get("patientId")
    b2.put_object(
        Bucket=B2_BUCKET,
        Key=f"records/{patient_id}.json",
        Body=json.dumps(data, ensure_ascii=False, indent=2),
        ContentType="application/json",
    )
    return {"status": "ok"}

@app.delete("/b2/delete-record/{patient_id}")
def delete_record(patient_id: str):
    # 환자 JSON 삭제
    b2.delete_object(Bucket=B2_BUCKET, Key=f"records/{patient_id}.json")
    # DICOM 파일 전체 삭제
    try:
        paginator = b2.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=B2_BUCKET, Prefix=f"dicom/{patient_id}/"):
            for obj in page.get("Contents", []):
                b2.delete_object(Bucket=B2_BUCKET, Key=obj["Key"])
    except Exception as e:
        print(f"[WARN] DICOM 삭제 실패: {e}")
    return {"status": "ok"}

@app.get("/b2/download-record/{patient_id}")
def download_record(patient_id: str):
    obj = b2.get_object(Bucket=B2_BUCKET, Key=f"records/{patient_id}.json")
    content = obj["Body"].read().decode("utf-8")
    return json.loads(content)

@app.post("/b2/save-report-md")
async def save_report_md(request: Request):
    data = await request.json()
    patient_id = data.get("patientId")
    markdown = data.get("markdown", "")
    b2.put_object(
        Bucket=B2_BUCKET,
        Key=f"records/{patient_id}.md",
        Body=markdown.encode("utf-8"),
        ContentType="text/markdown; charset=utf-8",
    )
    return {"status": "ok"}

@app.post("/b2/save-dicom")
async def save_dicom(file: UploadFile = File(...), patient_id: str = ""):
    contents = await file.read()
    b2.put_object(
        Bucket=B2_BUCKET,
        Key=f"dicom/{patient_id}/{file.filename}",
        Body=contents,
        ContentType="application/octet-stream",
    )
    return {"status": "ok"}

@app.post("/validate")
async def validate_dicom(files: List[UploadFile] = File(...)):
    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "total": len(files),
        "valid_count": 0,
        "patient_ids": [],
        "series_uids": [],
    }
    patient_ids = set()
    series_uids = set()

    for file in files:
        contents = await file.read()
        try:
            ds = pydicom.dcmread(io.BytesIO(contents))
            _ = ds.pixel_array
            patient_ids.add(str(getattr(ds, "PatientID", "Unknown")))
            series_uids.add(str(getattr(ds, "SeriesInstanceUID", "Unknown")))
            results["valid_count"] += 1
        except Exception as e:
            results["errors"].append({"filename": file.filename, "message": str(e)})

    if len(patient_ids) > 1:
        results["warnings"].append(f"여러 환자 ID 감지: {list(patient_ids)}")
        results["valid"] = False
    if len(series_uids) > 1:
        results["warnings"].append(f"여러 시리즈 감지: {len(series_uids)}개")
    if results["errors"]:
        results["valid"] = False

    results["patient_ids"] = list(patient_ids)
    results["series_uids"] = list(series_uids)
    return results

@app.post("/card/generate")
async def generate_card(request: Request):
    """
    슬라이스 분석 결과로 설명 카드 데이터 생성
    body: {
        "slice_index": 133,
        "total_slices": 287,
        "anomaly_patches": [...]   ← /analyze_volume 결과 그대로
    }
    CT crop은 _cached_hu_volume에서 직접 생성 (lung window 보장)
    """
    data = await request.json()
    slice_index   = data.get("slice_index", 0)
    total_slices  = data.get("total_slices", 1)
    patches       = data.get("anomaly_patches", [])

    if not patches:
        return {"error": "anomaly_patches가 비어있습니다"}

    top_patch = max(patches, key=lambda p: p["score"])

    # ── 히트맵 + CT crop (백엔드 HU volume 직접 사용) ──────────────────────────
    gradcam_b64   = None
    heatmap_type  = None
    ct_crop_b64   = None   # Panel1 후보 crop (폐 비율 정규화, lung window, 256×256)
    heatmap_ct_crop_b64 = None  # 히트맵 오버레이용 CT (Grad-CAM과 동일 96px FOV, B)

    if _cached_hu_volume is not None:
        Z_v, H_v, W_v = _cached_hu_volume.shape
        z_clamped = max(0, min(slice_index, Z_v - 1))

        pos = top_patch.get("position", {})
        y0_p = pos.get("y0", 0); x0_p = pos.get("x0", 0)
        y1_p = pos.get("y1", y0_p + 32); x1_p = pos.get("x1", x0_p + 32)
        cy_p = (y0_p + y1_p) // 2; cx_p = (x0_p + x1_p) // 2

        from PIL import Image as PILImage
        import base64 as _b64
        full_hu = _cached_hu_volume[z_clamped].astype(np.float32)
        lo, hi = -1350.0, 150.0
        full_arr = np.clip((full_hu - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
        full_img = PILImage.fromarray(full_arr, "L")

        # 폐 마스크 bbox (폐 크기 비율 정규화 + y/x_pct 계산에 공통 사용)
        _mask_src = _cached_ts_guard_vol if _cached_ts_guard_vol is not None else None
        if _mask_src is not None:
            _mslice = _mask_src[z_clamped]
            _ys = np.where(_mslice.any(axis=1))[0]
            _xs = np.where(_mslice.any(axis=0))[0]
            if len(_ys) > 0 and len(_xs) > 0:
                _ly0, _ly1 = int(_ys[0]), int(_ys[-1])
                _lx0, _lx1 = int(_xs[0]), int(_xs[-1])
            else:
                _ly0, _ly1, _lx0, _lx1 = 0, H_v - 1, 0, W_v - 1
        else:
            _ly0, _ly1, _lx0, _lx1 = 0, H_v - 1, 0, W_v - 1
        card_y_pct = round((cy_p - _ly0) / max(_ly1 - _ly0, 1), 4)
        card_x_pct = round((cx_p - _lx0) / max(_lx1 - _lx0, 1), 4)

        # ── Panel1 후보 crop: 폐 bbox 높이의 일정 비율을 FOV로 사용 (A) ──────────
        # 스캔별 mm/px가 달라도 "폐 대비 줌"으로 정규화 → 정상 ref와 해상도/줌 일치.
        # 정상 ref(card_generator)도 동일한 LUNG_CROP_FRAC을 쓰고, 둘 다 256으로 리사이즈.
        OUT_SIZE = 256
        LUNG_CROP_FRAC = 0.70
        half = max(48, int(LUNG_CROP_FRAC * max(_ly1 - _ly0, 1) / 2))
        cy0c = max(0, cy_p - half); cy1c = min(H_v, cy_p + half)
        cx0c = max(0, cx_p - half); cx1c = min(W_v, cx_p + half)
        crop = full_img.crop((cx0c, cy0c, cx1c, cy1c))
        side_px = 2 * half
        if crop.size != (side_px, side_px):
            padded = PILImage.new("L", (side_px, side_px), 0)
            padded.paste(crop, (half - (cx_p - cx0c), half - (cy_p - cy0c)))
            crop = padded
        crop = crop.resize((OUT_SIZE, OUT_SIZE), PILImage.BILINEAR)
        buf_ct = io.BytesIO()
        crop.convert("RGB").save(buf_ct, format="PNG")
        ct_crop_b64 = _b64.b64encode(buf_ct.getvalue()).decode()
        print(f"[DBG-CANDIDATE] HU=({Z_v},{H_v},{W_v}) lung_h={_ly1-_ly0} half={half} fov={side_px}px->256 cy_p={cy_p} cx_p={cx_p}", flush=True)

        # NSCLC crop 좌표 (96×96, patch 중심 기준)
        csize = 48
        nsclc_y0 = cy_p - csize; nsclc_x0 = cx_p - csize
        lung_z_pct = round((slice_index - _cached_lung_zmin) / max(_cached_lung_zmax - _cached_lung_zmin, 1), 4)

        ry0 = max(0, nsclc_y0); rx0 = max(0, nsclc_x0)
        ry1 = min(H_v, nsclc_y0 + 96); rx1 = min(W_v, nsclc_x0 + 96)

        # ── 히트맵 오버레이용 CT: Grad-CAM과 동일한 96px FOV (B) ────────────────
        # 후보 crop(폐비율 256)과 달리 히트맵은 96px이므로, 같은 96px CT를 따로 만들어
        # 오버레이가 정확히 정합되게 함. full_arr은 이미 lung window 적용됨.
        _hm = full_arr[ry0:ry1, rx0:rx1]
        _hm_img = PILImage.fromarray(_hm, "L")
        if _hm_img.size != (96, 96):
            # Grad-CAM 마스크와 동일하게 우/하단 패딩(np.pad (0,ph),(0,pw))으로 top-left 정렬
            _hm_pad = PILImage.new("L", (96, 96), 0)
            _hm_pad.paste(_hm_img, (0, 0))
            _hm_img = _hm_pad
        _buf_hm = io.BytesIO()
        _hm_img.convert("RGB").save(_buf_hm, format="PNG")
        heatmap_ct_crop_b64 = _b64.b64encode(_buf_hm.getvalue()).decode()
        if _cached_ts_guard_vol is not None:
            roi_patch = _cached_ts_guard_vol[z_clamped, ry0:ry1, rx0:rx1].astype(np.float32)
        else:
            roi_patch_hu = _cached_hu_volume[z_clamped, ry0:ry1, rx0:rx1]
            roi_patch = (roi_patch_hu > -950).astype(np.float32)
        crop_roi_ratio = float(roi_patch.mean()) if roi_patch.size > 0 else 0.5

        nsclc_prob = top_patch.get("nsclc_prob")
        if nsclc_prob is not None and nsclc_prob >= 0.5:
            # NSCLC-like → Grad-CAM (P-C-NORMAL30b)
            gradcam_b64  = compute_gradcam_base64(
                _cached_hu_volume, slice_index,
                nsclc_y0, nsclc_x0,
                lung_z_pct, crop_roi_ratio,
                ts_guard_vol=_cached_ts_guard_vol,
            )
            heatmap_type = "gradcam"
        else:
            # Normal-like or unknown → RD4AD E2 spatial map
            gradcam_b64  = run_rd4ad_e2_spatial_map_base64(
                _cached_hu_volume, slice_index,
                nsclc_y0, nsclc_x0,
                ts_guard_vol=_cached_ts_guard_vol,
            )
            heatmap_type = "rd4ad"

    card_data = generate_card_data(
        slice_index=slice_index,
        total_slices=total_slices,
        top_patch=top_patch,
        all_patches=patches,
        ct_crop_b64=ct_crop_b64,
        heatmap_ct_crop_b64=heatmap_ct_crop_b64,
        nsclc_prob=top_patch.get("nsclc_prob"),
        nsclc_label=top_patch.get("nsclc_label"),
        gradcam_base64=gradcam_b64,
        heatmap_type=heatmap_type,
        lung_z_pct=lung_z_pct,
        y_pct=card_y_pct,
        x_pct=card_x_pct,
    )
    return card_data


# ── B2 자동화 엔드포인트 ──────────────────────────────────────────────────────

@app.post("/b2/webhook")
async def b2_webhook(request: Request):
    """B2 Event Notification 수신 → 60초 디바운스 후 자동 분석"""
    try:
        body = await request.json()
    except Exception:
        return {"status": "ignored", "reason": "invalid json"}

    events = body if isinstance(body, list) else body.get("events", [body])

    triggered = []
    for event in events:
        event_type = event.get("eventType", event.get("event_type", ""))
        if "ObjectCreated" not in event_type and "object:created" not in event_type:
            continue
        file_name = event.get("fileName", event.get("objectName", event.get("object_name", "")))
        parts = file_name.split("/")
        if len(parts) < 2 or parts[0] != "dicom":
            continue
        patient_id = parts[1]
        if not patient_id:
            continue

        # 기존 타이머 취소 후 새 60초 디바운스 시작
        old_task = _pending_timers.get(patient_id)
        if old_task and not old_task.done():
            old_task.cancel()

        _analysis_status[patient_id] = {
            "status": "pending",
            "queued_at": datetime.now().isoformat(),
        }
        task = asyncio.create_task(_debounce_and_run(patient_id, delay=60))
        _pending_timers[patient_id] = task
        triggered.append(patient_id)
        print(f"[WEBHOOK] {patient_id} 대기 시작 (60초 디바운스)")

    return {"status": "ok", "triggered": triggered}


@app.get("/b2/analysis-status/{patient_id}")
def get_analysis_status(patient_id: str):
    """환자 분석 진행 상태 반환"""
    status = _analysis_status.get(patient_id)
    if status is None:
        # B2에 완료된 record가 있는지 확인
        try:
            b2.head_object(Bucket=B2_BUCKET, Key=f"records/{patient_id}.json")
            return {"patient_id": patient_id, "status": "done"}
        except Exception:
            return {"patient_id": patient_id, "status": "unknown"}
    return {"patient_id": patient_id, **status}


@app.get("/b2/list-records")
def list_records():
    """B2에 저장된 모든 환자 기록 목록 반환"""
    try:
        paginator = b2.get_paginator("list_objects_v2")
        records = []
        for page in paginator.paginate(Bucket=B2_BUCKET, Prefix="records/"):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                patient_id = key.replace("records/", "").replace(".json", "")
                if not patient_id:
                    continue
                try:
                    resp = b2.get_object(Bucket=B2_BUCKET, Key=key)
                    data = json.loads(resp["Body"].read().decode("utf-8"))
                    # 프론트엔드에 필요한 필드만 반환 (slices는 무겁기 때문에 제외)
                    records.append({
                        "patientId":  data.get("patientId", patient_id),
                        "name":       data.get("name", patient_id),
                        "birthdate":  data.get("birthdate", ""),
                        "gender":     data.get("gender", ""),
                        "date":       data.get("date", ""),
                        "status":     data.get("status", "Completed"),
                        "risk":       data.get("risk", "Low"),
                        "opinion":    data.get("opinion", ""),
                        "source":     data.get("source", "manual"),
                        "hasSlices":  "slices" in data,
                    })
                except Exception as e:
                    print(f"[WARN] record 로드 실패 {key}: {e}")
        return {"records": records}
    except Exception as e:
        return {"records": [], "error": str(e)}


@app.get("/b2/record-slices/{patient_id}")
def get_record_slices(patient_id: str):
    """환자의 분석 결과(slices) 전체 반환 — 뷰어 로드 시 호출"""
    try:
        resp = b2.get_object(Bucket=B2_BUCKET, Key=f"records/{patient_id}.json")
        data = json.loads(resp["Body"].read().decode("utf-8"))
        return {
            "patientId": data.get("patientId", patient_id),
            "risk":      data.get("risk", "Low"),
            "slices":    data.get("slices", []),
            "tracks":    data.get("tracks"),
            # 수동 저장 기록은 slices 대신 highRiskPatches 스키마 → 재오픈 시 복원용
            "highRiskPatches": data.get("highRiskPatches", []),
            "totalSlices":     data.get("totalSlices", 0),
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/load-volume-cache/{patient_id}")
def load_volume_cache(patient_id: str):
    """저장된 전처리 npy(cache/preprocessed/<id>/)를 _cached_hu_volume 전역에 로드.
    B2 저장본을 다시 열 때 /card/generate, /ct_slice 가 올바른 환자 볼륨을 쓰도록 함.
    (#9 수정: 전역 캐시는 /analyze_volume 만 채웠어서 저장본 카드가 엉뚱한 볼륨 사용)
    """
    global _cached_hu_volume, _cached_ts_guard_vol, _cached_lung_zmin, _cached_lung_zmax
    d = os.path.join(_PREPROCESS_DIR, str(patient_id))
    ct_path = os.path.join(d, "ct_hu.npy")
    roi_path = os.path.join(d, "roi_0_0.npy")
    if not os.path.exists(ct_path):
        return {"status": "missing", "patient_id": patient_id}
    try:
        ct = np.load(ct_path).astype(np.float32)
        roi = np.load(roi_path).astype(bool) if os.path.exists(roi_path) else None
        _cached_hu_volume = ct
        _cached_ts_guard_vol = roi
        if roi is not None:
            _areas = np.array([int(roi[zz].sum()) for zz in range(roi.shape[0])])
            _zs = np.where(_areas > 0)[0]
            _cached_lung_zmin = int(_zs.min()) if _zs.size > 0 else 0
            _cached_lung_zmax = int(_zs.max()) if _zs.size > 0 else int(ct.shape[0]) - 1
        else:
            _cached_lung_zmin = 0
            _cached_lung_zmax = int(ct.shape[0]) - 1
        print(f"[CACHE] load-volume-cache {patient_id}: ct_hu={ct.shape} lung_z=[{_cached_lung_zmin},{_cached_lung_zmax}]")
        return {"status": "ok", "patient_id": patient_id, "z": int(ct.shape[0])}
    except Exception as e:
        return {"status": "error", "patient_id": patient_id, "message": str(e)}


@app.post("/save-volume-cache/{patient_id}")
def save_volume_cache(patient_id: str):
    """현재 _cached_hu_volume(방금 분석한 볼륨)을 cache/preprocessed/<patient_id>/ 에 저장.
    수동 저장(SSE 분석) 기록은 npy가 DICOM해시 키로 저장돼 patient_id로 재로드 못 하는 문제(#B)
    → 저장 시 patient_id로도 복사해 재오픈 시 카드 CT crop이 복원되게 함.
    """
    if _cached_hu_volume is None:
        return {"status": "no_volume"}
    try:
        out = os.path.join(_PREPROCESS_DIR, str(patient_id))
        os.makedirs(out, exist_ok=True)
        np.save(os.path.join(out, "ct_hu.npy"), _cached_hu_volume.astype(np.int16))
        if _cached_ts_guard_vol is not None:
            np.save(os.path.join(out, "roi_0_0.npy"), _cached_ts_guard_vol.astype(np.uint8))
        print(f"[CACHE] save-volume-cache {patient_id}: ct_hu={_cached_hu_volume.shape}")
        return {"status": "ok", "patient_id": patient_id, "z": int(_cached_hu_volume.shape[0])}
    except Exception as e:
        return {"status": "error", "patient_id": patient_id, "message": str(e)}