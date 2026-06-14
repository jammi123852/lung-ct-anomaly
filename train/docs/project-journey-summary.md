# 폐 CT 이상탐지 프로젝트 — 전체 여정 정리

> 작성일: 2026-06-11
> 출처: `.claude/memory/` 28개 + `docs/context-handoff/` + `docs/project-progress-summary.md`
> 형식: 각 트랙별 **목표 → 시도 → 벽 → 결과**
> 범위: 실제 진행한 작업만. 폐기/보류된 시도도 "왜 막혔는지"가 의미 있으면 포함.

---

## 0. 프로젝트 한 줄 정의

정상 폐 CT로 **위치별 정상 feature 분포**를 만들고, 새 CT에서 그 분포와 다른 patch/slice를
**이상 후보**로 의사에게 제안하는 **판독 보조 도구**(자동 진단 아님).

파이프라인: `1차 PaDiM(이상 후보) → 전처리/억제 → 2차 RD4AD(병변 vs FP) → 후보 정렬 + heatmap + 설명카드`

데이터: 정상 학습 LUNA16 290명(train)/72명(val·test), 병변 테스트 NSCLC 249 + MSD_Lung 59 = 308명.
**stage1_dev 154명(실험 허용) / stage2_holdout 154명(봉인, 접근 금지).**

---

## 0.5 전처리 기반 설계 결정 (프로젝트 시작점, decision-log 1~8)

> 가장 처음 정한 토대. 이후 모든 트랙이 이 위에서 돌아감.

- **TotalSegmentator 역할 제한** — 폐 mask 자체가 아니라 "폐 이외 장기 위치" 얻는 용도. patch 기준은 별도 HU 기반 pure lung mask.
- **HU 기반 pure lung mask** — patch는 실제 폐 내부에서만 추출, overlay/patch box 눈검증 필수.
- **3종 출력 분리** — 핵심 입력(ct_1mm_lung_range 등) / 디버깅용(overlay·mask PNG) / 보존용(native_lps 등)을 섞지 않음.
- **전체 실행 시 PNG·verbose 제한** — runtime_summary/error csv/진행률만 유지.
- **patch는 환자별 CSV + done marker + resume** — 전체를 메모리에 안 쌓음(MemoryError 방지).
- **설명 가능한 baseline 우선** — HU 통계 + 위치별 정상 기준 먼저, 복잡한 CNN feature는 이후 비교 실험으로.
- **보류 기능** — HU window 다중증강 / 다축 slice / multi-scale patch / clustering 세분화는 baseline 안정화 전까지 보류.
- **하네스 목표** — 노트북 성공 코드를 `python -m src.xxx --config` CLI로 정리(원본 로직·함수명 유지, resume/error/runtime).

---

## 0.6 ★ 변곡점 — "분석 마비" 비판과 평가 우선 전환 (current_change, evaluation 재정의 직전)

> 한 세션의 자기비판이 이후 방향(평가 재정의 + 진짜 RD4AD)으로 이어진 계기. 기록으로 남길 가치 있음.

**당시 진단(비판):**
- **2D feature로는 혈관(관형)과 결절(구형)을 원리적으로 구분 불가** — 혈관 FP의 진짜 원인. weak 3D merge는 score만 묶을 뿐 feature는 여전히 2D.
- **초기 "RD4AD"가 사실 RD4AD가 아니었음** — teacher-student/distillation 없는 단순 ConvAutoencoder(L1 복원). score가 whole-crop 96×96 평균이라 **폐 밖(흉벽·종격동)이 점수를 지배**(outside_roi_mean > roi_mean이 149/150) → 2차가 1차의 흉벽 FP를 그대로 재현. *(→ 이후 진짜 teacher-student RD-D1s로 교체됨, 8장)*
- **position model이 너무 거칢** — bin 6개(상/중/하 × central/peripheral), 좌·우 폐 구분조차 없음 → hilum/종격동 고대비 정상이 "정상"으로 안 묻혀 high-score.
- **풍부한 supervised label(positive 43,553 + hard_negative 87,106)을 안 쓰고 비지도 AE로 우회.**
- **메타 문제 = "분석 마비"** — 2차·혈관·ResNet50 **세 갈래를 분모(평가셋) 없이 동시에** 끌고 가다 전부 같은 벽(평가셋 없음·승인 게이트)에서 정체. CLAUDE.md "변수 동시 변경 금지" 위반.

**결정:** "병렬을 버리는 게 아니라 **공통 분모(평가셋)부터 깔고** 한 번에 한 변수씩 비교한다." → **mixed cohort 평가 우선** → 2장의 per-scan FROC 평가 재정의로 이어짐.

---

## 1. 기반 구축 — 1차 PaDiM + Backbone 비교 (P-A 시리즈)

**목표:** 위치별 정상 분포 기반 PaDiM 1차 모델을 세우고, 어떤 CNN backbone이 최선인지 확정.

**시도:**
- ResNet18/ResNet50(ImageNet), ResNet50(RadImageNet) 비교. feature를 layer1+2+3 concat 후 seed42로 차원 축소.
- ResNet50이 약한 원인 가설 H1(낮은 retention) 검증을 위해 rand224(224차원) 실험까지 설계·학습.
- 속도: Mahalanobis CPU 루프가 36명에 ~90분 → **GPU + cov_inv bin당 1회 캐시**로 전환.

**벽:**
- ResNet50은 1792→100 = retention **5.6%**만 사용 → 큰 backbone에 동일 100-cap이 불리.
- 차원을 224로 늘려도(rand224), RadImageNet 의료 pretrain도 ResNet18을 못 따라잡음.

**결과:**
- **ResNet18 v2/v2 = ResNet-era 최강** (patch AUROC 0.702, slice 0.637).
- 비교 순위: ResNet18 0.702 > rand224 0.690 > RadImageNet 0.681 > ResNet50 random100 0.662.
- rand224/RadImageNet branch = ON HOLD. → 나중에 EfficientNet이 이 결론을 갈아치움(7장).
- 관련: `p-a7-lesion-data-source`, `rand224-experiment-progress`, `gpu-always-for-training`.

---

## 2. 평가 방식 재정의 — patient AUROC 폐기

**목표:** 1차 성능을 신뢰할 수 있는 지표로 측정.

**시도:** patient-level AUROC로 병변 환자 vs 정상 환자 구분 측정.

**벽 (결정적):**
- 병변 patch를 **통째로 제거해도 AUROC 0.9995 → 0.9995** (하락 0.0001).
- 즉 모델이 "병변"이 아니라 "LUNA 데이터셋이 아님"을 감지하는 **도메인 아티팩트**.
- cross-source(LUNA vs NSCLC/MSD) 비교는 신뢰 불가.

**결과:**
- patient AUROC **폐기** → **per-scan FROC** 채택 (스캔 내부 modified-z(MAD)로 도메인 offset 상쇄 → z>4 + 3D connected-component → top-K).
- FP 정체 분석: **흉막/흉벽 경계 패치 67%**(주범), 폐 내부 기타 24%, 혈관 중심 0.8%.
- 흉막 억제는 **저-K(top-3/5)에서만 이득**(+0.05~0.09), top-10에선 병변도 깎임(−0.05).
- 병목 구조: 병변은 잡히지만(p95 hit 99.4%) 흉막·혈관 region에 **순위에서 밀림** → patch AUROC 0.7 정체.
- 관련: `evaluation-reframe-localization-froc`.

---

## 3. 전처리 A — 흉벽 흰선(rim) 제거 (phase2_23 → b1a → b1g → b1k)

**목표:** 흉벽 흰 rim이 FP의 주원인 → ROI에서 깎아 경계 FP를 줄인다.

**시도 & 벽 (시간순):**

1. **phase2_23 rim cut + b1a2 재스코어링:** rim을 검게 만들고 재스코어.
   - **벽:** FP를 8.7%밖에 못 줄이고 오히려 평균 점수 **상승**(OOD 효과). 모델이 "밝은 rim=정상"으로 학습해서, 검게 만들면 더 이상해 보임.
   - **결과:** **재학습 없이는 rim cut으로 FP 못 줄임** → CLOSED/ON HOLD.

2. **b1a4 병변 손실 검증(154명, lesion slice 6,132개):** rim 깎으면 병변도 깎이나?
   - safe_cut 순기여 손실 ~662px(2.2%), 나머지 97.8%는 GaussianBlur σ=3 수축.
   - 3D 완전소멸 병변 27개 = 전부 3~18 voxel(직경 1.5~3.5mm), 큰 병변 소멸 0건.
   - 한때 1088px loss 우려 → **2조건만 재계산한 재구현 버그**로 판명, 정식 5조건은 정상.
   - **결과:** 흉벽 극소병변(1.5~3.5mm) 미탐은 **한계로 수용**. `b1a4-lesion-loss-resolved`.

3. **b1g 둘레/circularity 기반 흉벽 보호제외:** 흉벽만 골라 깎으려 시도(v1~v9).
   - **벽:** 흉벽선이 혈관·종격동과 **한 거대 연결 component로 융합** → 병변보호 가드가 흉벽도 막음.
   - per-pixel thickness(v3), notch 보호(v6), 스무딩 제외(v6-B) 등으로 두꺼운/둥근 병변은 공존 성공.
   - **그러나 얇고 납작한 흉막밑 병변은 흉벽과 두께가 동일 → 같은 픽셀이라 원리적으로 택일**(irreducible).
   - **결과:** 사용자 "보류" 선택 → **CLOSED_ON_HOLD, 병변 안전 우선**. `b1g-perimeter-unprotect-finding`.

4. **b1k 상단/바닥 intrusion 제거:** 마스크 내부 잔존 흉벽/횡격막 돌출만 제거.
   - v1 3D grow 폭주(좌폐 92% 번짐) → 폐기. v3 단순 기하(apex=큰z) → dry-run PASS.
   - 방향 전환: bottom-diaphragm 연부조직 제거(V9 + 둥근 mass 보호) → 11명 PASS10/FAIL1.
   - **결과:** 로직 확정(b1k5), 최종 일반화 코드 작성. **batch actual은 보류**(승인 게이트). `b1k-top-protrusion-remove-v3`.

**A 트랙 종합:** rim cut은 FP 해결 못 함(재학습 필요). 병변 안전을 위해 깎기는 보수적으로 유지.

---

## 4. 전처리 B — 혈관 FP 억제 (b1b / b1c)

**목표:** 혈관 단면이 병변과 외관이 유사 → MIP으로 혈관 흐름 파악 후 억제.

**시도:** PCA elongation, HU threshold, negative-selection(B1-C2), sep_R granulometry, MIP argmax + object Tier 분류.

**벽:**
- HU 밀도로 구별 불가(혈관 0.071 ≈ 병변 0.079), PCA elongation **역전**(병변이 더 길쭉).
- 근본 한계: **병변에 붙은 굵은 혈관(7/12명)은 접합부 HU 연속이라 분리 불가.**
- sep_R granulometry로 "가는 혈관 제거 + 병변 보존" 가능성은 확인(sep_R2.0: 병변 0.90 / 혈관 0.43)하나 robust rule 아님.

**결과:**
- **CLOSED_ON_HOLD** — robust rule 없음. suppression/score/softmask 적용 **금지** 유지.
- 억제는 patch 격자가 아니라 **raw CT voxel 단위(Frangi/3면)**에서 해야 한다는 설계 결론.
- 관련: `vessel-fp-branch-b1b-progress`, `b1d-wall-mediastinum-fp-cause-diagnostic`(흉벽/종격동 FP 원인진단도 Rule-B3/Gate-P2 모두 미채택).

---

## 5. 표준 폐 ROI 마스크 확정 — refined_roi_v4_20_modeB

**목표:** 전처리 A/B의 학습을 모아 흉벽 ~8% 제거하는 **표준 마스크**를 확정.

**시도:** v4 modeB = `v4_A ∪ (roi0 & thick_protect)`. T_THICK 값(2.5/2.0/1.4) 스윕, spike/center-peaked 보호로직 측정.

**벽:**
- T_THICK 2.5는 blocked까지 깎아 병변 손실 큼(MSD_004 40.5%). 1.0은 흉벽 붕괴.
- 흉막융합 병변 자동분리 사실상 불가(병변이 별도 chunk를 안 만듦). 유일한 미검증 단서 = bright_inward(AUC 0.84, n=3).
- **데이터 무결성 발견:** 테스트셋 병변 마스크는 이미 `roi_0_0 ∩ 병변`으로 **폐 내부만 truncated**(TotalSegmentator가 상류에서 흉벽쪽 제거). 실질 잘림 >20%는 22명(7%)뿐.

**결과:**
- **정상/병변 모두 동일 T_THICK=2.0으로 통일** (마스크 차이가 가짜 domain artifact 되는 것 차단).
- **v4@2.0 전체 670개(정상362+병변308) 빌드 완료** (흉벽 제거 ~6%, 병변 손실 중앙 0.12%).
- 저장: `masks/refined_roi_v4_20_modeB_all_v1/`. **이후 EfficientNet 학습의 핵심 입력이 됨**(7장).
- 관련: `refined-roi-v4-2p5-modeb-standard`.

---

## 6. (생략 가능) FP suppression dry-run

**목표:** score-CSV proxy rule(roi<1.0/position lower/central)로 FP 억제 가능한지.

**벽:** normal FP와 lesion이 광범위하게 겹쳐 **직접 suppression 차단**. lesion 24/24 전부 vessel/pleura 인접.

**결과:** proxy 억제 금지 확정. preflight/counterfactual table만 남김. `fp-suppression-dry-rule-state`.

---

## 7. 백본 업그레이드 — EfficientNet-B0 (현재 최강)

**목표:** ResNet 한계(0.70 정체) 돌파.

**시도:** EfficientNet-B0 PaDiM + 5장의 v4_20 흉벽제거 마스크 결합. 동일 stage1_dev·동일 patch-label로 공정 비교.

**결과 (돌파):**
- **EfficientNet-B0 + v4_20 = patch AUROC 0.7555 / slice 0.7048 → 모든 ResNet 능가.**
- backbone 단독 효과도 확인: EfficientNet roi_0_0 0.7385 > ResNet18 0.7018.
- 단 **stage1_dev 한정**. main 승격·holdout sealed eval은 **중단**(검증 충분하다 판단) → S5 설명카드 제작으로 전환.
- 관련: `efficientnet-padim-v4-20-best`, `efficientnet-stage2-main-promotion-stopped`.

---

## 8. 2차 모델 — RD4AD (병변 vs FP 분류)

**목표:** 1차 후보 중 진짜 병변 vs 정상 FP를 거르는 verifier.

**시도 & 결과:**

1. **초기 학습(ResNet18 teacher-student, 2.5D crop):** v1/v2 = patch AUROC 0.7523. S6-A 6ch crop 130,659개 생성.

2. **z-continuity group-level full scoring:** 후보 113,447 → 20,216 group(82.2% 감축).
   - 무결성 PASS(scalar_repro ~3.4e-6)이나 **성능 FAIL**: group top-k < patch baseline(저-K에서 patch 우세, top50만 역전).
   - smoke가 positive 우선 샘플링으로 부풀려진 것 → full로 일반화 불가. **후보 삭제/진단 사용 금지.** `rd4ad-group-full-scoring-v1`.

3. **RD-E1 입력조건 비교(A/B/C/D1s + 변형):** window/MIP/teacher 영향 분리.
   - patch AUROC: D1s(medi+스택) 0.7505 > B 0.7068 > A 0.5267 > C 0.4778 (AUROC ≠ hit rate).
   - ⚠️ **정정:** 초기 stage1_dev에서 "C(lung+MIP3ch) P5 top10=0.9276이 최강"이라 했으나 **그 값은 오류**.
     제대로 된 **stage2-holdout 재평가(2026-06-11)에서 C는 0.9085(≈0.90)로 내려감.**
   - ROI 픽셀 마스킹(C2/A2)은 여전히 효과 없음(positive crop ROI coverage < negative).
   - **★ stage2-holdout 최종 최강 = E2 (lung3ch + EfficientNet-B0 teacher) P5 top10 = 0.9216.**
     E1(lung+MIP3ch+EffNet)=0.9150, E2z(E2+z_pct)=0.9150으로 둘 다 E2보다 낮음.
     → 초기 stage1_dev의 "EfficientNet teacher FAIL" 결론도 **뒤집힘**(stage2에선 E2가 이김).
   - 비교표(stage2-holdout P5 top10): E2 **0.9216** > E1=E2z 0.9150 > A=B=C 0.9085 > C2=A2 0.8889.

4. **최종 평가(z-track min_run=2, patient hit rate):**
   - PaDiM(EffNet v4_20) stage1 top20 = **0.9156**, stage2-holdout top20 = 0.8442.
   - RD4AD PD stage2-holdout top20 = **0.9477**, top30 = 0.9673, top50 = 0.9935.
   - RD4AD P5 stage2-holdout top20 = 0.9673(D1s/E2 동률), top50 = 1.0000.
   - PaDiM 환자 AUROC(threshold 기반): stage1+normal = 0.9136, stage2+normal = 0.9223.
   - PD top20 기준 **슬라이스 56.9% 절약.** `rd4ad-padim-final-eval-summary`.
   - 남은 TODO: crop에 v4_20 마스크 픽셀 적용 실험(`rd4ad-roi-mask-pixel-todo`).

---

## 9. P-C-NORMAL branch — normal vs NSCLC classifier + vessel softmask + Grad-CAM

**목표:** EfficientNet supervised로 정상 vs 병변(NSCLC) 직접 분류 + 시각화.

**시도 & 벽:**

1. **P-C-AUX (side branch):** 의도와 달리 **NSCLC(TCIA) vs MSD_Lung(Decathlon) source 구분 모델**이 됨(val AUC 0.9956, best epoch=1).
   - shortcut risk OPEN(데이터셋 도메인/스캐너 차이 학습 의심). **진단 표현 금지, side branch 보관.** `p-c-aux-nsclc-msd-side-branch`.

2. **P-C-NORMAL24 vessel softmask 좌표 버그:** `slice_index`로 마스크 생성 → 잘못된 slice.
   - **벽:** ct_hu.npy는 resample/crop돼 원본 idx와 z축 어긋남. split×label마다 정답 컬럼 제각각.
   - **결과:** crop↔volume MAE=0 매칭으로 91,250 crop 재확정(99.93% resolved). `canonical_volume_z`만 사용. `p-c-normal24-zindex-coordinate-bug`.

3. **P-C-NORMAL24e vessel-병변 오염:** 생성한 vessel softmask가 **NSCLC 병변을 중앙값 94.7% 덮음**.
   - **벽:** `compute_vessel_mask_s3`는 "ROI 내 밝은/관상 구조"를 잡아 고형 종괴도 포함. b1c4의 object-level Tier/위험도 게이팅을 통째로 누락한 게 원인.
   - within-component split(thick_core 제거)까지 시도했으나 **병변제거와 혈관보존이 coupled**(R 따라 같이 이동) → clean 분리 불가.
   - **결과:** **vessel feature 비viable로 결론, 보류.** z/ROI scalar만(`24f-zroi-only`) 권장. `p-c-normal24-vessel-lesion-contamination`.

4. **P-C-NORMAL37 Grad-CAM:** masked-input 모델(P-C-NORMAL30b)의 시각화 확정.
   - 초기 버그: HU clip→[0,1] normalize 누락으로 logit 수십만.
   - **결과:** **37g 최종 확정**(YlOrRd, mask 내부만). 추론 전용 standalone 코드 패키징 완료. `p-c-normal37-gradcam-final`.

---

## 10. XAI 설명카드 (S5) — reference-bank v4 + dynamic refbank

**목표:** 1차 후보를 의사에게 설명하는 카드. "정상 환자 같은 위치"와 비교해 보여줌.

**시도 & 결과:**

1. **S5 reference-bank v4 (정적, same-cell):**
   - 후보의 폐 내부 상대 위치(lung_z_pct + y/x bin → 90-cell)로 정상 bank top3 retrieval.
   - 4 case 검증: LUNG1-052 GOOD_FOR_DEMO / 320·059 USABLE_WITH_CAUTION / 041 NOT_RECOMMENDED(ref가 혈관·junction 위주).
   - Panel1 비율 왜곡(matplotlib baked title) → **clean 96×96 CT re-crop + 280px contain-fit**으로 해결.
   - **결과:** `v4_ref_update_aspectfix` = internal-demo 최종본. CLOSE/PASS. `s5-candidate-roi-position-metadata-v4`, `s5-lung1-052-card-chain-close`.

2. **Dynamic normal-reference bank (동적, 정상 3명):**
   - 절대 slice index가 아니라 **lung_z_pct + lung-bbox 상대 위치**로 정상 3명에서 best slice/patch 동적 선택.
   - 정상 3명 폐 slice **781 PNG** bank → retrieval.
   - **벽:** candidate(0.977mm) vs 정상(0.49~0.64mm) **물리 배율 불일치** → A안(80mm physical-scale crop) + B안(bilateral frame 통일)로 해결.
   - 저-z lung base 후보는 80mm crop이 횡격막/종격동(흰색 비폐) 포함 → **context-crop(160mm) + 위치 bounding box**로 표시 종결.
   - **결과:** **dynamic-reference 통합 full 4-panel XAI 카드 완성** + 범용 CLI(`build_dynamic_ref_card_any_candidate_v1.py`, 후보 id만 주면 자동 생성). `dynamic-normal-refbank-three-patients`.

**공통 원칙:** saved 1차 score만 읽음(재계산 0), raw CT copy 0, same-z matching 아님, branch-specific score 절대비교 금지, **진단 아님**.

---

## 11. 현재 상태 한눈에

| 트랙 | 상태 | 핵심 결과 |
|------|------|-----------|
| 1차 PaDiM backbone | ✅ 확정 | **EfficientNet-B0 + v4_20 = patch AUROC 0.7555** (ResNet 전부 능가) |
| 평가 지표 | ✅ 재정의 | patient AUROC 폐기 → per-scan FROC / z-track hit rate |
| 전처리 A (rim cut) | ⏸ CLOSED/HOLD | 재학습 없이 FP 못 줄임. 극소병변 미탐 한계 수용 |
| 전처리 B (vessel) | ⏸ CLOSED_ON_HOLD | robust rule 없음. voxel 단위(Frangi) 필요 |
| 표준 마스크 | ✅ 확정 | v4_20_modeB 전체 670개 빌드 완료 |
| 2차 RD4AD | ✅ eval 완료 | stage2 최강=E2(lung3ch+EffNet) P5 top10=0.9216, PD top20=0.9477, 슬라이스 56.9% 절약 |
| P-C vessel feature | ⏸ 비viable | 병변 94.7% 오염, z/ROI scalar만 권장 |
| Grad-CAM | ✅ CLOSED | 37g 최종, 추론 전용 코드 완료 |
| S5 정적 카드 | ✅ CLOSED | v4_ref_update_aspectfix internal-demo |
| Dynamic 카드 | ✅ FINAL | full XAI 카드 + 범용 CLI 완성 |

**봉인 유지:** stage2_holdout는 eval-only로만 접근, 재평가/방법변경 금지.

---

## 12. 반복된 근본 벽 (프로젝트 전반의 교훈)

1. **도메인 아티팩트** — 모델이 병변이 아니라 "데이터셋 출처"를 감지. patient AUROC·P-C-AUX classifier 모두 이 함정.
2. **흉막융합 병변 = irreducible** — 얇고 납작한 흉막밑 병변은 흉벽과 같은 픽셀 → rim cut/vessel 분리 모두 자동분리 불가(b1g, vessel, v4 마스크 공통 결론).
3. **혈관 vs 병변 robust rule 없음** — HU/PCA/sep_R/morphology 다 시도했으나 굵은 구조에서 coupled.
4. **smoke ≠ full** — positive 우선 샘플링 편향으로 smoke가 부풀려짐(RD4AD group scoring).
5. **OOD 효과** — 입력을 바꾸면(rim을 검게) 모델이 더 이상하게 봄. 재학습 없는 후처리 억제는 한계.
