from __future__ import annotations

from typing import Optional, Tuple

import cv2
import mediapipe as mp


class HandDetector:
    """Wrap MediaPipe hand tracking for consistent landmark processing."""

    def __init__(
        self,
        max_hands: int = 1,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame_bgr: cv2.typing.MatLike) -> Tuple[Optional[object], cv2.typing.MatLike, str]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        annotated = frame_bgr.copy()
        if result.multi_hand_landmarks is None:
            return None, annotated, "No hand detected"

        if len(result.multi_hand_landmarks) > 1:
            return None, annotated, "Multiple hands detected"

        hand_landmarks = result.multi_hand_landmarks[0]
        self.mp_drawing.draw_landmarks(annotated, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return hand_landmarks, annotated, "Hand detected"

    def release(self) -> None:
        self.hands.close()
