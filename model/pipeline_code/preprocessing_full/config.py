# ============================================================
# config.py
# ------------------------------------------------------------
# 고정 파라미터 (학습 계약 — 절대 변경 금지)
# ref_01_preprocess.py CONFIG 기준
# ============================================================

# --------------------------------------------------------
# z 리샘플 target (학습 계약 고정값)
# --------------------------------------------------------
TARGET_Z = 1.0  # mm, 절대 변경 금지

# --------------------------------------------------------
# CT orientation
# --------------------------------------------------------
ORIENTATION = "LPS"

# --------------------------------------------------------
# CT HU window (학습 계약 고정값)
# --------------------------------------------------------
HU_MIN = -1000
HU_MAX = 400

# --------------------------------------------------------
# body guard HU threshold (학습 계약 고정값)
# --------------------------------------------------------
BODY_GUARD_HU_THRESHOLD = -500

# --------------------------------------------------------
# TotalSegmentator 설정 (학습 계약 고정값)
# --------------------------------------------------------
USE_FAST_TOTALSEG = False  # non-fast 정밀 모드
OVERWRITE_TOTALSEG = False

# TS 폐엽 ROI 이름 (guard용)
TS_LUNG_ROI_NAMES = [
    "lung_upper_lobe_left",
    "lung_lower_lobe_left",
    "lung_upper_lobe_right",
    "lung_middle_lobe_right",
    "lung_lower_lobe_right",
]

# pure_lung에서 제외할 장기
ORGAN_EXCLUSION_ROI_NAMES = [
    "heart",
    "aorta",
    "trachea",
    "esophagus",
    "liver",
    "stomach",
    "spleen",
    "pancreas",
]

# TotalSegmentator에 요청할 전체 ROI subset (13개)
ORGAN_ROI_SUBSET = [
    # TS 폐 guard용 5개
    "lung_upper_lobe_left",
    "lung_lower_lobe_left",
    "lung_upper_lobe_right",
    "lung_middle_lobe_right",
    "lung_lower_lobe_right",
    # pure_lung 제외용 장기 8개
    "heart",
    "aorta",
    "trachea",
    "esophagus",
    "liver",
    "stomach",
    "spleen",
    "pancreas",
]

# organ_exclusion 3D dilation iterations (학습 계약 고정값)
ORGAN_EXCLUSION_DILATE_ITER = 1

# --------------------------------------------------------
# TS lung guard 설정 (학습 계약 고정값)
# --------------------------------------------------------
USE_TS_LUNG_GUARD = True
TS_LUNG_GUARD_DILATE_ITER = 2  # 학습 계약 고정값
STRICT_TS_LUNG_GUARD = True

# --------------------------------------------------------
# 폐 z-range 파라미터 (학습 계약 고정값)
# --------------------------------------------------------
LUNG_RANGE_MIN_PURE_LUNG_AREA_RATIO = 0.01   # 학습 계약 고정값
LUNG_RANGE_MARGIN_SLICES = 5                  # 학습 계약 고정값
LUNG_RANGE_MAX_GAP_SLICES = 5                 # 학습 계약 고정값
LUNG_RANGE_MIN_SEGMENT_SLICES = 10            # 학습 계약 고정값

# 폐 구간 선택 방식:
#   "full_span"       = 유효 폐 구간이 여러 개로 쪼개져도 첫 구간 시작 ~ 마지막 구간 끝까지 모두 포함
#                       (폐 중간 dip으로 폐가 잘리는 과다 crop 방지. 단일 구간이면 largest와 동일)
#   "largest_segment" = 가장 큰 단일 구간만 (노트북 원본 동작)
LUNG_RANGE_SELECT_MODE = "full_span"

# 폐 z-range 크롭 적용 여부 (사용자 선택: 크롭 안 함).
#   False = 1mm 리샘플 전체 볼륨을 그대로 사용 → 폐 손실 0. 비폐 슬라이스는
#           스코어링 루프에서 pure_lung.sum()<100 으로 자동 skip(Low) 되므로 안전.
#   True  = 위 SELECT_MODE 기준으로 폐 z-range만 잘라 사용(노트북 원본).
# pure_lung 1%-of-frame 기준이 큰 FOV 스캔에서 폐 끝을 잘라내는 문제 때문에 기본 False.
LUNG_CROP_ENABLED = False

# --------------------------------------------------------
# build_roi_0_0 파라미터 (학습 계약 고정값, "no_dilate" 버전 기준)
# ref_02_roi_0_0.py의 0/0 설정:
#   TS_LUNG_DILATE_ITER = 0  → 폐엽 dilation 없음
#   ROI_EXTRA_DILATE_ITER = 0
#   USE_BODY_GUARD = False
#   USE_ORGAN_EXCLUSION = True
#   ORGAN_EXCLUSION_DILATE_ITER = 1
# --------------------------------------------------------
ROI_TS_LUNG_DILATE_ITER = 0      # 학습 계약 고정값 — no_dilate 버전
ROI_EXTRA_DILATE_ITER = 0        # 학습 계약 고정값
ROI_USE_BODY_GUARD = False        # 학습 계약 고정값
ROI_USE_ORGAN_EXCLUSION = True    # 학습 계약 고정값
ROI_ORGAN_EXCLUSION_DILATE_ITER = 1  # 학습 계약 고정값

# --------------------------------------------------------
# lung crop 통계 (선택, run에서 사용 안 함)
# --------------------------------------------------------
LUNG_CROP_MARGIN_PIXELS = 16
LUNG_CROP_SIZE_MULTIPLE = 32

# --------------------------------------------------------
# verbose 기본값
# --------------------------------------------------------
VERBOSE = False
MAX_MISSING_RAW_WARNINGS = 10
