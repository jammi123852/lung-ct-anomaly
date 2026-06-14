"""
Task 8.1~8.3 병변 평가 로직 구현.
실제 병변 파일 접근 없이 DataFrame/ndarray 기반 계산만 수행한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class Evaluator:
    """patch/slice/patient 레벨 레이블 계산 및 메트릭 계산."""

    # ------------------------------------------------------------------ #
    # 입력 검증
    # ------------------------------------------------------------------ #

    def _validate_patch_df(self, patch_df: pd.DataFrame) -> None:
        required = ["patient_id", "safe_id", "local_z", "y0", "x0", "y1", "x1"]
        missing = [c for c in required if c not in patch_df.columns]
        if missing:
            raise ValueError(f"[Evaluator] patch_df 필수 컬럼 누락: {missing}")

    def _validate_score_df(
        self, df: pd.DataFrame, score_col: str, label_col: str
    ) -> None:
        for col in [score_col, label_col]:
            if col not in df.columns:
                raise ValueError(f"[Evaluator] score_df에 컬럼 없음: {col}")
        if df[score_col].isnull().any():
            raise ValueError(f"[Evaluator] score_col '{score_col}'에 NaN 포함")
        if np.isinf(df[score_col].values).any():
            raise ValueError(f"[Evaluator] score_col '{score_col}'에 inf 포함")

    def _validate_lesion_mask(
        self, mask: np.ndarray, patch_df: pd.DataFrame
    ) -> None:
        if mask.ndim not in (2, 3):
            raise ValueError(
                f"[Evaluator] lesion_mask는 2D 또는 3D여야 한다. 현재: {mask.ndim}D"
            )
        if mask.ndim == 3:
            max_z = int(patch_df["local_z"].max())
            if max_z >= mask.shape[0]:
                raise ValueError(
                    f"[Evaluator] 3D mask의 depth({mask.shape[0]})가 "
                    f"최대 local_z({max_z})보다 작다."
                )

    # ------------------------------------------------------------------ #
    # Task 8.1 레이블 계산
    # ------------------------------------------------------------------ #

    def compute_patch_labels(
        self,
        patch_df: pd.DataFrame,
        lesion_mask: np.ndarray,
        min_lesion_pixels: int = 5,
    ) -> pd.DataFrame:
        """각 patch에 lesion_pixel_count / lesion_overlap_ratio / patch_label 을 추가한다."""
        self._validate_patch_df(patch_df)
        self._validate_lesion_mask(lesion_mask, patch_df)

        result = patch_df.copy()
        pixel_counts: list[int] = []
        overlap_ratios: list[float] = []
        labels: list[int] = []

        is_3d = lesion_mask.ndim == 3

        for _, row in result.iterrows():
            z = int(row["local_z"])
            y0, x0, y1, x1 = int(row["y0"]), int(row["x0"]), int(row["y1"]), int(row["x1"])

            if is_3d:
                region = lesion_mask[z, y0:y1, x0:x1]
            else:
                region = lesion_mask[y0:y1, x0:x1]

            patch_area = max((y1 - y0) * (x1 - x0), 1)
            lpc = int((region > 0).sum())
            lor = lpc / patch_area
            lbl = 1 if (lor >= 0.01 or lpc >= min_lesion_pixels) else 0

            pixel_counts.append(lpc)
            overlap_ratios.append(lor)
            labels.append(lbl)

        result["lesion_pixel_count"] = pixel_counts
        result["lesion_overlap_ratio"] = overlap_ratios
        result["patch_label"] = labels
        return result

    def compute_slice_labels(self, labeled_patch_df: pd.DataFrame) -> pd.DataFrame:
        """local_z 기준 slice-level 레이블을 집계한다."""
        if "patch_label" not in labeled_patch_df.columns:
            raise ValueError("[Evaluator] compute_patch_labels 결과 DataFrame이 필요하다.")

        records = []
        for z, group in labeled_patch_df.groupby("local_z", sort=True):
            pos = int((group["patch_label"] == 1).sum())
            total = len(group)
            records.append(
                {
                    "local_z": z,
                    "slice_label": 1 if pos >= 1 else 0,
                    "positive_patch_count": pos,
                    "total_patch_count": total,
                }
            )
        return pd.DataFrame(records)

    def compute_patient_labels(self, labeled_patch_df: pd.DataFrame) -> pd.DataFrame:
        """patient_id 기준 patient-level 레이블을 집계한다."""
        if "patch_label" not in labeled_patch_df.columns:
            raise ValueError("[Evaluator] compute_patch_labels 결과 DataFrame이 필요하다.")

        records = []
        for pid, group in labeled_patch_df.groupby("patient_id", sort=True):
            pos = int((group["patch_label"] == 1).sum())
            total = len(group)
            records.append(
                {
                    "patient_id": pid,
                    "patient_label": 1 if pos >= 1 else 0,
                    "positive_patch_count": pos,
                    "total_patch_count": total,
                }
            )
        return pd.DataFrame(records)

    # ------------------------------------------------------------------ #
    # Task 8.2 메트릭 계산 (numpy 기반, sklearn 미사용)
    # ------------------------------------------------------------------ #

    def compute_auroc(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """ROC 곡선 아래 면적 (numpy trapz 적분)."""
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)

        n_pos = int((y_true == 1).sum())
        n_neg = int((y_true == 0).sum())

        if n_pos == 0 or n_neg == 0:
            print(
                f"[Evaluator] compute_auroc: 양성({n_pos})과 음성({n_neg}) 중 "
                "한쪽이 없어 AUROC를 계산할 수 없다. np.nan 반환."
            )
            return float(np.nan)

        thresholds = np.sort(np.unique(y_score))[::-1]
        tprs = [0.0]
        fprs = [0.0]

        for thr in thresholds:
            y_pred = (y_score >= thr).astype(int)
            tp = int(((y_pred == 1) & (y_true == 1)).sum())
            fp = int(((y_pred == 1) & (y_true == 0)).sum())
            tprs.append(tp / n_pos)
            fprs.append(fp / n_neg)

        tprs.append(1.0)
        fprs.append(1.0)

        return float(np.trapz(tprs, fprs))

    def compute_auprc(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Precision-Recall 곡선 아래 면적 (numpy trapz 적분)."""
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)

        n_pos = int((y_true == 1).sum())

        if n_pos == 0:
            print(
                "[Evaluator] compute_auprc: 양성 샘플이 없어 AUPRC를 계산할 수 없다. np.nan 반환."
            )
            return float(np.nan)

        thresholds = np.sort(np.unique(y_score))[::-1]
        precisions = []
        recalls = []

        for thr in thresholds:
            y_pred = (y_score >= thr).astype(int)
            tp = int(((y_pred == 1) & (y_true == 1)).sum())
            fp = int(((y_pred == 1) & (y_true == 0)).sum())
            pred_pos = tp + fp
            prec = tp / pred_pos if pred_pos > 0 else 0.0
            rec = tp / n_pos
            precisions.append(prec)
            recalls.append(rec)

        recalls_arr = np.array([0.0] + recalls + [recalls[-1]])
        precisions_arr = np.array([1.0] + precisions + [0.0])

        return float(np.trapz(precisions_arr, recalls_arr))

    def compute_dice(
        self, y_true_binary: np.ndarray, y_pred_binary: np.ndarray
    ) -> float:
        """Dice 계수."""
        y_true_binary = np.asarray(y_true_binary)
        y_pred_binary = np.asarray(y_pred_binary)
        tp = int(((y_pred_binary == 1) & (y_true_binary == 1)).sum())
        denom = int(y_true_binary.sum()) + int(y_pred_binary.sum())
        if denom == 0:
            return float(np.nan)
        return 2 * tp / denom

    def compute_iou(
        self, y_true_binary: np.ndarray, y_pred_binary: np.ndarray
    ) -> float:
        """Intersection over Union."""
        y_true_binary = np.asarray(y_true_binary)
        y_pred_binary = np.asarray(y_pred_binary)
        intersection = int(((y_pred_binary == 1) & (y_true_binary == 1)).sum())
        union = int(((y_pred_binary == 1) | (y_true_binary == 1)).sum())
        if union == 0:
            return float(np.nan)
        return intersection / union

    def apply_threshold(
        self, y_score: np.ndarray, threshold: float
    ) -> np.ndarray:
        """score >= threshold 이면 1, 아니면 0."""
        return (np.asarray(y_score) >= threshold).astype(int)

    # ------------------------------------------------------------------ #
    # Task 8.3 DataFrame 기반 비교 로직 (실제 파일 접근 없음)
    # ------------------------------------------------------------------ #

    def compare_models(
        self,
        score_df: pd.DataFrame,
        label_col: str = "patch_label",
        score_cols: list[str] | None = None,
        subset_col: str | None = None,
        model_name: str | None = None,
    ) -> dict:
        """PaDiM / HU Stat score DataFrame을 받아 모델별 AUROC / AUPRC를 계산한다.

        model_name: score_cols가 단일 컬럼일 때 결과 dict key를 이 이름으로 대체한다.
                    score_cols가 여러 개이면 무시하고 기존처럼 score_col 이름을 key로 사용한다.
        """
        if score_cols is None:
            score_cols = ["padim_score", "hu_z_score"]

        available_cols = [c for c in score_cols if c in score_df.columns]
        if not available_cols:
            raise ValueError(
                f"[Evaluator] score_df에 score_cols({score_cols}) 중 존재하는 컬럼이 없다."
            )

        use_model_name = model_name is not None and len(available_cols) == 1

        result: dict = {}

        def _calc(df: pd.DataFrame, col: str) -> dict:
            self._validate_score_df(df, score_col=col, label_col=label_col)
            y_true = df[label_col].values
            y_score = df[col].values
            return {
                "auroc": self.compute_auroc(y_true, y_score),
                "auprc": self.compute_auprc(y_true, y_score),
            }

        for col in available_cols:
            top_key = model_name if use_model_name else col
            result[top_key] = _calc(score_df, col)

            if subset_col and subset_col in score_df.columns:
                for subset_val, sub_df in score_df.groupby(subset_col):
                    key = f"{top_key}__{subset_col}={subset_val}"
                    result[key] = _calc(sub_df, col)

        return result
