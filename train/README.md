# lung-ct-anomaly — 모델 학습/추론/평가 스크립트 (버전관리)

정상 폐 CT로 **위치별 정상 feature 분포**를 만들고, 새 CT에서 그 분포와 다른
patch/slice를 **이상 후보**로 의사에게 제안하는 **판독 보조 도구** (자동 진단 아님).

파이프라인:
`1차 PaDiM(이상 후보) → 전처리/억제 → 2차 RD4AD(병변 vs FP) → 후보 정렬 + heatmap + 설명카드`

> 이 `train/` 폴더는 프로젝트 전 여정에서 **시도한 모델들의 .py 스크립트**를
> journey 트랙별로 정리한 것입니다. 데이터/결과/가중치 파일은 포함하지 않습니다(.gitignore).
> 전체 여정 서술은 [`docs/project-journey-summary.md`](docs/project-journey-summary.md)
> (6/1 이후 본편)와 [`docs/project-journey-pre-0601-detail.md`](docs/project-journey-pre-0601-detail.md)
> (6/1 이전 기반) 참조.

---

## 폴더 구조 (journey 트랙별)

| 폴더 | 내용 |
|------|------|
| `00_core_src/` | 공통 핵심 모듈 (PaDiMModel / FeatureExtractor / DataLoader / Evaluator 등) |
| `01_first_stage_padim/` | 1차 PaDiM + backbone 비교 (ResNet18/50, RadImageNet, rand224, **EfficientNet-B0**) |
| `02_evaluation/` | 평가 (patient AUROC 폐기 → per-scan FROC / z-track hit rate / metric) |
| `03_preprocess_rim_wall/` | 전처리 A — 흉벽 rim/wall/diaphragm 제거 (phase2_23, b1a/b1g/b1k) |
| `04_preprocess_vessel/` | 전처리 B — 혈관 FP 억제 (b1b/b1c/b1d, vessel) |
| `05_roi_mask_v4/` | 표준 폐 ROI 마스크 확정 (`refined_roi_v4_20_modeB`) |
| `06_second_stage_rd4ad/` | 2차 RD4AD (rd_b/d/e 시리즈, z-track, second-stage refiner) |
| `07_pc_normal_classifier/` | P-C-NORMAL supervised classifier + vessel softmask |
| `08_gradcam/` | Grad-CAM 시각화 (P-C-NORMAL37) |
| `09_xai_cards/` | XAI 설명카드 (S5 reference-bank, dynamic refbank) |
| `99_misc/` | 기타 보조/분석/파이프라인/임시 스크립트 (단일 트랙 미해당) |

각 파일의 **원본 경로 ↔ train 경로 ↔ 트랙** 매핑은 [`MANIFEST.csv`](MANIFEST.csv) 참조.

---

## 핵심 결과 요약

| 트랙 | 상태 | 핵심 결과 |
|------|------|-----------|
| 1차 PaDiM backbone | ✅ | **EfficientNet-B0 + v4_20 = patch AUROC 0.7555** (ResNet 전부 능가) |
| 평가 지표 | ✅ 재정의 | patient AUROC 폐기(도메인 아티팩트) → per-scan FROC / z-track hit rate |
| 2차 RD4AD | ✅ | stage2 최강 = **E2 (lung3ch + EfficientNet teacher) P5 top10 = 0.9216**, PD top20 = 0.9477 |
| Grad-CAM / XAI 카드 | ✅ | 37g 최종 / dynamic-reference full 4-panel 카드 완성 |

---

## 실행 환경

- WSL2(Ubuntu) + Python 3.12 venv (`~/ai_env`)
- GPU: NVIDIA RTX 4060 Ti, torch 2.5.1+cu121
- 주요: numpy / pandas / opencv / scikit-image / scipy (※ scikit-learn 미설치 — AUROC 등 numpy 직접 구현)
- 실행: 터미널 `.py` 스크립트, 긴 작업은 `nohup` 백그라운드 + 로그

## 주의

- 스크립트 내부에 **로컬 절대경로**(`/home/jinhy/...`)와 데이터셋명이 하드코딩돼 있습니다.
  그대로 실행되지 않으며, 경로/데이터는 각 환경에 맞게 조정해야 합니다.
- 사용 데이터셋: LUNA16 / NSCLC-Radiomics / MSD_Lung — 전부 **공개(TCIA 등) 익명 데이터셋**.
- 본 코드는 **연구·판독 보조** 목적이며 **자동 진단용이 아닙니다**.
