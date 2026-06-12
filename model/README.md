# LUNAR — 폐 CT 이상탐지 AI (판독 보조)

정상 폐 CT의 **위치별 정상 feature 분포**를 학습해, 새 CT에서 정상과 다른 patch/슬라이스를
**이상 후보로 찾아 판독을 보조**하는 웹 애플리케이션입니다. 진단 도구가 아니며 연구·보조 용도입니다.

- **백엔드**: FastAPI (`pipeline_code/`) — Position-aware PaDiM + RD4AD + NSCLC 분류 + Grad-CAM
- **프론트엔드**: Next.js (`lunar_web/`)
- **런타임**: Windows 임베디드 Python (`lunar_env/`), 실행 런처 `start.bat`
- **저장(선택)**: Backblaze B2

---

## 파이프라인 개요

1. **전처리** (`preprocessing_full/`): DICOM → LPS 정렬 → z축 1mm 리샘플 → TotalSegmentator 폐엽/장기 분할
   → refined 폐 마스크 → `ct_hu` + `roi_0_0` 생성 (학습과 동일 전처리)
   - 폐 z-range 크롭은 기본 **비활성**(`LUNG_CROP_ENABLED=False`) → 전체 볼륨 사용(폐 손실 방지), 비폐 슬라이스는 스코어링에서 자동 skip
2. **PaDiM**: 위치 bin별 정상 분포 기준 patch 이상 점수(P90 후보 수집)
3. **z-track 필터**: 동일 위치가 연속 슬라이스(≥2)에서 후보일 때만 통과(연속성)
4. **RD4AD**: 통과 후보를 Teacher-Student 재구성 오차로 재검증
5. **스코어링**: P5 = RD4AD cosine × roi_ratio × (track_len/3)
6. **XAI 카드**: 후보 vs 정상 reference 비교(폐 비율 정규화), NSCLC 확률, Grad-CAM / RD4AD 히트맵
7. **고위험 top-K**: 트랙 단위로 정렬, top-K 트랙이 덮는 슬라이스로 판독 절약률 산출

---

## 저장소 구조

```
model/
├── pipeline_code/            # FastAPI 백엔드
│   ├── main.py               # API 서버 (분석/카드/B2/webhook)
│   ├── preprocessing_full/   # 정식 전처리 모듈 (DICOM → ct_hu/roi_0_0)
│   ├── models/               # padim / rd4ad / nsclc / gradcam / card_generator
│   ├── position_aware_padim/ # 위치 인식 PaDiM
│   ├── reference_bank/       # 정상 3인 동적 reference (카드용)
│   ├── weights/              # 모델 가중치 (저장소 포함)
│   └── b2_secrets.json.example
├── lunar_web/                # Next.js 프론트엔드
├── runtime_libs/DOWNLOAD.md  # 대용량 파일(Google Drive) 안내
├── start.bat                 # 백엔드+프론트 실행 런처
└── .gitignore
```

> `lunar_env/`(Python 환경 ≈2.6GB)와 `pipeline_code/totalseg_data/`(TS 가중치 ≈1.5GB)는
> 용량 한도 때문에 저장소에 없습니다 → **[runtime_libs/DOWNLOAD.md](runtime_libs/DOWNLOAD.md)** 참고.

---

## 설치 & 실행 (Windows)

1. 저장소 clone 또는 다운로드
2. **대용량 파일 받기** — [runtime_libs/DOWNLOAD.md](runtime_libs/DOWNLOAD.md)의 Google Drive 링크에서
   `lunar_env.zip`, `totalseg_data.zip` 받아 지정 위치에 압축 해제
3. (선택) B2 저장 사용 시: `pipeline_code/b2_secrets.json.example` → `b2_secrets.json` 복사 후 키 입력
4. **`start.bat` 실행** — 백엔드(`:8000`) + 프론트(`:3000`) 기동, 브라우저 자동 오픈

```bat
:: start.bat 주요 설정
set "USE_FULL_PREPROCESS=1"   :: 정식 전처리 ON
```

---

## 사용 흐름

- **업로드 → 분석**: DICOM 폴더 업로드 → 전처리 + 분석(슬라이스별 스트리밍)
- **All 탭**: 전처리된 전체 슬라이스 브라우징 (XAI 미실행)
- **고위험 탭**: 트랙 단위 top-K 후보 + 클릭 시 XAI 카드(후보/정상 비교, 히트맵)
- **저장/불러오기**: 결과를 B2에 저장, "B2에서 불러오기"로 기록 목록 로드

> **자동분석(webhook)**: `POST /b2/webhook`로 B2 업로드 이벤트를 받아 자동 분석하는 기능이 있으나,
> 공개 도달 가능한 URL + B2 Event Notification 설정이 있어야 동작합니다(로컬 단독 실행에선 미동작).

---

## 주의

- 본 시스템은 **판독 보조용**이며 최종 진단은 전문의가 확인해야 합니다.
- 자격증명(B2 키 등)을 코드에 하드코딩하지 마세요. `b2_secrets.json`(gitignore) 또는 환경변수를 사용합니다.
