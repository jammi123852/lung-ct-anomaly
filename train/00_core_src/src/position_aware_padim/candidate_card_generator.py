"""
CandidateCardGenerator: 후보 patch 정보와 overlay를 받아 후보 카드 dict를 생성한다.

설계 원칙:
- cv2 / matplotlib 의존 없음. PIL + numpy만 사용한다.
- 파일 저장은 이 클래스에서 수행하지 않는다. 호출부에서 저장한다.
- lesion_mask 처리는 optional (None이면 건너뜀).
- 원본 overlay를 수정하지 않는다. 복사본에 박스를 그린다.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


REQUIRED_CANDIDATE_COLUMNS: List[str] = [
    "patient_id",
    "safe_id",
    "local_z",
    "y0",
    "x0",
    "y1",
    "x1",
    "position_bin",
]


class CandidateCardGenerator:
    """후보 patch 정보 + overlay ndarray → 후보 카드 dict 생성기."""

    def __init__(self, score_col: str) -> None:
        """
        Parameters
        ----------
        score_col : str
            카드에 기록할 점수 컬럼명. 예: 'padim_score', 'hu_z_score'.
        """
        self.score_col = score_col

    # ------------------------------------------------------------------
    # 입력 검증
    # ------------------------------------------------------------------

    def _validate_candidate(self, candidate: dict) -> None:
        """필수 후보 컬럼과 score_col 키 존재, 좌표 유효성을 검증한다."""
        required = REQUIRED_CANDIDATE_COLUMNS + [self.score_col]
        missing = [k for k in required if k not in candidate]
        if missing:
            raise ValueError(f"candidate에 필수 키가 없습니다: {missing}")

        y0 = int(candidate["y0"])
        y1 = int(candidate["y1"])
        x0 = int(candidate["x0"])
        x1 = int(candidate["x1"])

        if not (y0 < y1):
            raise ValueError(f"y0({y0}) < y1({y1}) 조건이 맞지 않습니다.")
        if not (x0 < x1):
            raise ValueError(f"x0({x0}) < x1({x1}) 조건이 맞지 않습니다.")

    def _validate_overlay(self, overlay: np.ndarray) -> None:
        """overlay가 (H, W, 3) uint8인지 확인한다."""
        if overlay.ndim != 3 or overlay.shape[2] != 3:
            raise ValueError(
                f"overlay는 (H, W, 3)이어야 합니다. 실제 shape: {overlay.shape}"
            )
        if overlay.dtype != np.uint8:
            raise ValueError(
                f"overlay dtype은 uint8이어야 합니다. 실제 dtype: {overlay.dtype}"
            )

    def _validate_coords_in_bounds(
        self, candidate: dict, overlay: np.ndarray
    ) -> None:
        """patch 좌표가 overlay 범위 내인지 확인한다."""
        h, w = overlay.shape[:2]
        y0 = int(candidate["y0"])
        x0 = int(candidate["x0"])
        y1 = int(candidate["y1"])
        x1 = int(candidate["x1"])

        if y0 < 0 or x0 < 0:
            raise ValueError(
                f"y0({y0}), x0({x0})는 0 이상이어야 합니다."
            )
        if y1 > h:
            raise ValueError(
                f"y1({y1})이 overlay 높이({h})를 초과합니다."
            )
        if x1 > w:
            raise ValueError(
                f"x1({x1})이 overlay 너비({w})를 초과합니다."
            )

    # ------------------------------------------------------------------
    # 박스 그리기
    # ------------------------------------------------------------------

    def draw_patch_box(
        self,
        overlay: np.ndarray,
        y0: int,
        x0: int,
        y1: int,
        x1: int,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        overlay 복사본에 PIL ImageDraw로 patch 좌표 사각형 박스를 그린다.

        Parameters
        ----------
        overlay : np.ndarray
            shape (H, W, 3) uint8. 원본은 수정하지 않는다.
        y0, x0, y1, x1 : int
            patch 좌표.
        color : tuple
            RGB 박스 색상 (기본: 녹색).
        thickness : int
            박스 선 굵기 (기본: 2).

        Returns
        -------
        np.ndarray
            shape (H, W, 3) uint8, 박스가 그려진 복사본.
        """
        img = Image.fromarray(overlay.copy())
        draw = ImageDraw.Draw(img)
        for t in range(thickness):
            draw.rectangle(
                [x0 + t, y0 + t, x1 - 1 - t, y1 - 1 - t],
                outline=color,
            )
        return np.array(img, dtype=np.uint8)

    # ------------------------------------------------------------------
    # 단일 카드 생성 (파일 저장 없음)
    # ------------------------------------------------------------------

    def make_card(
        self,
        candidate: dict,
        overlay: np.ndarray,
        rank: int,
        lesion_mask: Optional[np.ndarray] = None,
    ) -> dict:
        """
        후보 1건에 대한 카드 dict를 생성한다. 파일 저장은 하지 않는다.

        Parameters
        ----------
        candidate : dict
            REQUIRED_CANDIDATE_COLUMNS + score_col 키를 포함하는 후보 정보 dict.
        overlay : np.ndarray
            shape (H, W, 3) uint8. HeatmapGenerator.create_overlay() 결과.
        rank : int
            후보 순위 (1-based).
        lesion_mask : np.ndarray | None
            병변 마스크. None이면 건너뜀.

        Returns
        -------
        dict
            키: patient_id, safe_id, local_z, y0, x0, y1, x1,
                position_bin, score_col, score, rank,
                thumbnail, overlay_with_box, lesion_mask_provided.
        """
        self._validate_candidate(candidate)
        self._validate_overlay(overlay)

        h, w = overlay.shape[:2]

        # 좌표 clamp (범위 초과 시 경고 없이 자름)
        y0 = int(max(0, int(candidate["y0"])))
        x0 = int(max(0, int(candidate["x0"])))
        y1 = int(min(h, int(candidate["y1"])))
        x1 = int(min(w, int(candidate["x1"])))

        # thumbnail: overlay에서 patch 영역 추출
        if y1 > y0 and x1 > x0:
            thumbnail = overlay[y0:y1, x0:x1].copy()
        else:
            thumbnail = np.zeros((0, 0, 3), dtype=np.uint8)

        # overlay 복사본에 박스 그리기 (원본 수정 없음)
        overlay_with_box = self.draw_patch_box(overlay, y0, x0, y1, x1)

        return {
            "patient_id": str(candidate["patient_id"]),
            "safe_id": str(candidate["safe_id"]),
            "local_z": int(candidate["local_z"]),
            "y0": y0,
            "x0": x0,
            "y1": y1,
            "x1": x1,
            "position_bin": str(candidate["position_bin"]),
            "score_col": self.score_col,
            "score": float(candidate[self.score_col]),
            "rank": int(rank),
            "thumbnail": thumbnail,
            "overlay_with_box": overlay_with_box,
            "lesion_mask_provided": lesion_mask is not None,
        }

    # ------------------------------------------------------------------
    # 다건 카드 생성 (파일 저장 없음)
    # ------------------------------------------------------------------

    def make_cards(
        self,
        candidates: pd.DataFrame,
        overlay_map: Dict[int, np.ndarray],
        top_k: int = 5,
    ) -> List[dict]:
        """
        CandidateRanker.rank_patches() 결과 DataFrame에서 top_k 후보 카드를 생성한다.

        Parameters
        ----------
        candidates : pd.DataFrame
            rank_patches() 반환 DataFrame. score_col 내림차순 정렬 상태.
        overlay_map : dict
            {local_z (int): overlay ndarray (H, W, 3) uint8}.
            해당 local_z가 없으면 해당 후보는 skip한다.
        top_k : int
            처리할 상위 후보 수 (기본: 5).

        Returns
        -------
        list of dict
            make_card() 결과 list. 파일 저장 없음.
        """
        cards: List[dict] = []
        for rank, (_, row) in enumerate(
            candidates.head(top_k).iterrows(), start=1
        ):
            local_z = int(row["local_z"])
            if local_z not in overlay_map:
                continue
            candidate = row.to_dict()
            overlay = overlay_map[local_z]
            card = self.make_card(candidate, overlay, rank=rank)
            cards.append(card)
        return cards
