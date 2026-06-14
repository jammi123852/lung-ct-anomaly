# 폐 CT 이상탐지 — 2026-06-01 이전 작업 상세 (기반/연구 시대)

> 작성일: 2026-06-11
> 범위: **2026-06-01 이전**에 진행한 작업만. (6/1 이후 = `project-journey-summary.md` 본편)
> 출처: `docs/phase_1_to_8_summary.md`(2026-05-18), `decision-log.md`(~항목9), `current_change.md`,
>       `project-progress-summary.md`, P-A 메모리, mip-postprocess-research 연구 폴더
> 성격: 이 시기는 **"실험 결과"보다 "토대 구축 + 대량 read-only 분석/QA"**가 핵심.
>       스스로 "분석 마비"라 진단한 구간(아래 9장).

---

## 0. 규모 한눈에

| 구분 | 양 | 비고 |
|------|----|------|
| 메인 Phase 1~8 | 8개 | 구조 → 핵심 PaDiM → 시각화 → 병변평가(보류) |
| P-A backbone 시리즈 | ~39단계 (P-A1~A38) | ResNet18/50, rand224, RadImageNet 비교 |
| phase2 전처리 (rim/vessel) | 18종 (~2.23) | 흉벽 rim cut, 혈관 softmask 설계 |
| phase5 (weak 3D merge/QA) | 30종 (~5.81) | z축 병합, visual QA 다수 |
| mip-postprocess 연구 스크립트 | **430개 .py** | 전처리 연구 workspace |
| scripts/ .py (6/1 이전 mtime) | 110개 | 메인 파이프라인 |
| 6/1 이전 outputs 결과 폴더 | 3개 | mixed_cohort_auroc / localization_froc_v1 / raw_slice_view |
| 초기 2차 모델 | 1개 | "RD4AD"라 불렀으나 실제 ConvAutoencoder |

문서화된 sub-experiment만 약 **90단계 이상**. 단 다수가 결론 없는 분석/QA.

---

## 1. Phase 1 — 프로젝트 구조와 설정 (2026-05-18)

**목표:** 산출물/설정/코드 경로가 분리된 표준 폴더 트리 + yaml 단일 설정 로더.

- Task 1.1 outputs 9개 하위 폴더(분포 npz/score CSV/feature cache/PNG/reports) 생성.
- Task 1.2 configs (`model.yaml`/`scoring.yaml`/`output.yaml`) — backbone ResNet18, position_bins 6개, top_n 등 외부화.
- Task 1.3 `ConfigManager` — 저장소 루트 configs/만 load source, 필수 누락 시 에러.
- Task 1.4 `.gitignore` 권장만 문서화(자동 수정 금지).

**결과:** 통과. 이후 모든 단계의 토대.

---

## 2. Phase 2 — 데이터 검증

**목표:** manifest 기반 안전 경로 해석 + 정상 데이터 무결성 검증.

- Task 2.0 `PathResolver` — manifest(utf-8-sig) + `normal_training_ready` base join. 컬럼 7개 고정. `{patient_id}` 직접 조립 금지.
- Task 2.1 `DataValidator` — ct_hu/pure_lung/meta 존재·shape 일치.
- Task 2.2 `CSVValidator` — patch 좌표 범위(0≤x0<x1≤512), position_bin 6값 검증.
- Task 2.3 `validate_data.py --limit 5` → normal001~005 전부 True.
- Task 2.3.5 환경: `which python=/home/jinhy/ai_env/bin/python`, torch 2.5.1+cu121, **scipy/scikit-learn 미설치**(과금/환경 변경 금지로 보류).
- Task 2.4 lesion 경로 검증 = **미실행**(`nsclc_msd_usable_only` 빈 문자열 게이트).

**결과:** 정상 데이터 5명 통과. 병변 경로는 시작부터 게이트로 막힘.

---

## 3. Phase 3 — Split과 Loader

**목표:** 환자 단위 split(슬라이스 누설 방지) + mmap 스트리밍 loader.

- `PatientSplitter` — 기존 `train_val_test_split.csv`를 source of truth로, 중복 patient_id면 ValueError.
- `DataLoader` — patient_id → PathResolver → ct_hu/pure_lung mmap 로드, 1ch→3ch는 slice 단계.
- `create_split.py` → **n_train=290 / n_val=36 / n_test=36** (total 362), 중복 없음, 기존 json 유지.

**결과:** 통과. 이 290/36/36 split이 이후 전체 학습의 고정 기준.

---

## 4. Phase 4 — HU Stat Baseline

**목표:** 설명 가능한 baseline 먼저 (decision-log 6 "설명 가능한 baseline 우선").

- `HUStatBaseline` — position_bin별 mean/std를 mask 내부만 계산, z-score를 anomaly score로.
- `train_hu_stat.py --limit 5` → 9.07초. `score_hu_stat.py --limit 5` → resume(skip) 동작 확인.

**결과:** PaDiM과 같은 split·위치 정의 공유 → 직접 비교 가능한 baseline 확보.

---

## 5. Phase 5 — FeatureExtractor (PaDiM 핵심 부품)

**목표:** patch를 직접 CNN에 안 넣고, slice feature map을 패치 좌표로 indexing.

- `preprocess_ct_slice` — HU clip[-1000,200] → 0-1 normalize → 1ch→3ch → ImageNet normalize.
- `FeatureExtractor` — ResNet18(ImageNet1K V1), stem+layer1/2/3 feature map 추출, patch 중심 좌표를 stride로 나눠 indexing → **patch당 448차원 concat**.
- 차원 축소: **random seed=42로 448 → 100차원**, `selected_feature_indices.npy` 저장(이후 반드시 동일 인덱스).

**결과:** 통과. *(이 "100차원 random 축소"가 나중에 ResNet50 retention 5.6% 문제의 뿌리 — 본편 1장)*

---

## 6. Phase 6 — Position-aware PaDiM (★핵심)

**목표:** 정상 폐 feature 공간의 위치별 분포를 메모리 안전 streaming으로 학습.

- `PaDiMModel` — **6 position_bin + 3 z_level + 1 global = 10키** 누적. patch feature를 list/array에 append 금지(메모리 안전).
- 누적 = `{sum_vec(100), sum_outer(100,100), count}`. finalize에서 mean=sum/count, cov=sum_outer/count − outer(mean,mean) + eps·I.
- score_patch = Mahalanobis, inv 실패 시 pseudo-inverse, sample 부족 시 fallback chain(position_bin → z_level → global).
- `train_padim.py` — `--limit N` 또는 `--full-run` 강제. 환자별 `del`+`gc.collect()`+`empty_cache()`. 개별 실패가 전체 중단 안 함.
- 실행: `--limit 5` → 145,679 patch, 15.86초. score `--limit 1`(normal023) → 16.06초.

**결과:** 통과. **위치별 정상 분포 + fallback 구조가 프로젝트의 심장.** *(단 이때 position bin이 6개뿐 = 좌·우 폐 구분 없음 → 나중에 hilum FP 원인으로 지목, 9장)*

---

## 7. Phase 7 — Score Aggregation과 시각화

**목표:** patch score를 사람이 눈으로 검증 가능한 결과물로.

- `ScoreAggregator`(patch→slice mean/max/p95→patient), `HeatmapGenerator`(grid 매핑+colormap+mask 외부 제외+overlay), `CandidateRanker`(top N), `CandidateCardGenerator`(카드 PNG/JSON).
- `generate_visualizations.py --limit 1` → normal023 top1 카드(score=35.91, middle_central, local_z=101).

**결과:** 통과. **후보 카드 구조가 나중 S5 XAI 카드의 원형**(본편 10장).

---

## 8. Phase 8 — 병변 Subset 평가 (부분 통과)

**목표:** 정상 기반 모델이 실제 병변을 얼마나 잡는지 평가.

- **게이트 미통과:** `nsclc_msd_usable_only` 빈 문자열 → `evaluate_lesion_subset.py` 미생성.
- 미리 구현만: `evaluator.py`(compute_patch/slice/patient_labels, compute_auroc/auprc numpy 직접구현, dice/iou, compare_models).

**결과:** 코드만 준비, 실제 병변 평가는 데이터 미준비로 보류. → 이후 P-A 시리즈로 별도 진행됨.

---

## 9. P-A backbone 비교 시리즈 (P-A1~A38, ~39단계)

**목표:** 어떤 CNN backbone이 최선인지 동일 stage1_dev·동일 patch-label로 비교.

**진행:**
- P-A1~A6: 데이터/경로 확인, 정상 학습 기반.
- P-A7~A11: ResNet50 ImageNet 병변 scoring + ResNet18 비교 → **ResNet50 < ResNet18 확인**(patch AUROC 0.662 vs 0.702).
- P-A12~A22: rand224(224차원) 설계·index 생성·full train(~24분)·metrics.
- P-A23~A36: RadImageNet(의료 pretrain) weight 다운로드·학습·평가 → **not_recovered**(격차 회복 못함) → ON HOLD.
- P-A37~A38: ResNet18 기반 개선 계획, FP/FN 패턴 분석 preflight.

**벽:** ResNet50은 1792→100 = retention **5.6%**만 사용(ResNet18 22.3%) → 큰 backbone에 동일 100-cap 불리. 차원 224로 늘려도, 의료 pretrain도 ResNet18 못 넘음.

**결과:** **ResNet18 v2/v2 = ResNet-era 최강**(patch AUROC 0.702). rand224/RadImageNet = ON HOLD.
*(이 결론은 6/1 이후 EfficientNet 0.7555가 갈아치움 — 본편 7장)*

---

## 10. 전처리 연구 — phase2(rim/vessel) + phase5(weak 3D merge)

> mip-postprocess-research workspace, 스크립트 430개. 6/1 이전 가장 양 많은 구간.

**phase2 (흉벽 rim cut + 혈관 softmask, ~2.23):**
- 흉벽 흰 rim이 FP 주원인 → ROI에서 깎는 component-protected smoothing 설계.
- **벽:** rim을 검게 만들면 모델이 더 이상하게 봄(OOD), 재학습 없인 FP 못 줄임. *(b1a2 재스코어 — 본편 3장)*
- 혈관 softmask: Adaptive MIP 두께별 projection + vessel shape score 설계. **벽:** 2D 단면에서 혈관(관형)과 결절(구형) 구분 불가.

**phase5 (weak 3D merge, ~5.81):**
- z축 연속성으로 파편 patch 후보를 3D 병변 단위로 병합.
- visual pack preflight 다수. **벽:** score만 z축으로 묶을 뿐 feature는 여전히 2D → 구분력 안 생김.

**결과:** 설계·QA는 풍부하나 대부분 read-only 상태로 정체. 실제 적용/재학습은 보류.

---

## 11. 초기 2차 모델 — "RD4AD"(실제 ConvAutoencoder)

**목표:** 1차 후보 중 진짜 병변 vs FP를 거르는 2차.

**시도:** 2.5D crop으로 학습.

**벽 (current_change 비판에서 발각):**
- 구현체가 teacher-student/distillation 없는 **단순 ConvAutoencoder(L1 복원)** — config 주석에 "minimal reconstruction baseline, not full RD4AD" 명시.
- anomaly score가 whole-crop 96×96 평균 → **폐 밖(흉벽·종격동)이 점수 지배**(outside_roi_mean > roi_mean 149/150) → 2차가 1차의 흉벽 FP를 그대로 재현.
- 풍부한 supervised label(positive 43,553 + hard_negative 87,106)을 안 쓰고 비지도 AE로 우회.

**결과:** 보류. *(6/1 이후 진짜 teacher-student RD-D1s로 교체 — 본편 8장)*

---

## 12. ★ 평가 재정의 + "분석 마비" 변곡점 (6/1 직전)

**목표:** 1차 성능을 신뢰 가능한 지표로 측정.

**벽 (결정적):**
- patient-level AUROC에서 **병변 patch를 통째 제거해도 0.9995 → 0.9995**(하락 0.0001) → 모델이 "병변"이 아니라 "LUNA 데이터셋이 아님"을 감지하는 **도메인 아티팩트**.
- 메타 문제: 2차·혈관·ResNet50 **세 갈래를 분모(평가셋) 없이 동시에** 끌고 가다 전부 같은 벽(평가셋 없음·승인 게이트)에서 정체 = "분석 마비". handoff 문서 18절·38KB 비대.

**결정/결과:**
- patient AUROC **폐기** → **per-scan FROC** 채택 (스캔 내부 modified-z(MAD)로 도메인 offset 상쇄 → z>4 + 3D CC → top-K).
- 산출 폴더: `outputs/mixed_cohort_auroc`, `outputs/localization_froc_v1`, `outputs/raw_slice_view`.
- FP 정체 분석: **흉막/흉벽 67%**, 폐 내부 기타 24%, 혈관 0.8%.
- localization: top-5 sensitivity 0.46 / top-10 0.62. 흉막 억제는 저-K만 이득.
- 결정 원칙: "병렬을 버리는 게 아니라 공통 분모(평가셋)부터 깔고 한 번에 한 변수씩 비교한다."

---

## 13. 6/1 이전에 확정된 핵심 설계 결정 (decision-log 1~9)

| # | 결정 |
|---|------|
| 1 | TotalSegmentator = 폐 mask용 아님, 장기 위치용. patch 기준은 별도 HU mask |
| 2 | patch 후보 기준 = HU 기반 pure lung mask, overlay 눈검증 필수 |
| 3 | 출력 3종 분리(핵심입력 / 디버깅 PNG / 보존용) |
| 4 | 전체 실행 시 PNG·verbose 끄고 runtime/error csv만 |
| 5 | patch는 환자별 CSV + done marker + resume (메모리에 안 쌓음) |
| 6 | 설명 가능한 baseline(HU stat + 위치) 우선, CNN feature는 이후 비교 |
| 7 | HU window 다중증강/다축 slice/multi-scale/clustering = baseline 안정화 전 보류 |
| 8 | 노트북 코드 → `python -m src.xxx --config` CLI 하네스로 정리 |
| 9 | 흉벽 rim 안 극소병변(1.5~3.5mm) 미탐 = ROI 설계상 한계로 수용 (재학습 없인 해결 불가) |

---

## 14. 6/1 시점 종합 판정

- **된 것:** 정상 학습 파이프라인 전체(290명 PaDiM), ResNet18 최강 확정, baseline, 시각화/카드 원형, 평가 재정의(FROC), FP 정체 진단.
- **안 된 것:** 전체 데이터 full-run(승인 게이트), 병변 평가 본실행, 2차 모델 검증, 혈관/rim 억제 실제 적용 — 전부 보류/게이트.
- **핵심 병목:** "무엇이 개선인지" 판정할 평가셋이 늦게 잡혀, 그 전까지 수십 단계 read-only QA만 반복(분석 마비).
- **6/1 이후 전환:** EfficientNet 백본 돌파 + 진짜 RD4AD + P-C/S5 XAI로 "실행이 결론 나는" 시대로 이동(본편 참조).
