from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional, Tuple

import numpy as np

from src.utils.config import (
    DIRECTION_CONSISTENCY_THRESHOLD,
    HAND_MISSING_GRACE_PERIOD,
    MAX_SWIPE_DURATION,
    MIN_HORIZONTAL_DISPLACEMENT,
    MIN_SWIPE_DURATION,
    SWIPE_COOLDOWN_SECONDS,
    SWIPE_DISPLAY_DURATION_SECONDS,
    SWIPE_LEFT_ZONE,
    SWIPE_RIGHT_ZONE,
    SWIPE_SMOOTHING_WINDOW,
    SWIPE_VERTICAL_LIMIT,
)


class TemporalSwipeDetector:
    """Simple reliable swipe detector using mirrored screen-space trajectory history."""

    @staticmethod
    def to_screen_x(mediapipe_x: float) -> float:
        return 1.0 - float(mediapipe_x)

    def __init__(
        self,
        min_swipe_duration: float = MIN_SWIPE_DURATION,
        max_swipe_duration: float = MAX_SWIPE_DURATION,
        min_horizontal_displacement: float = MIN_HORIZONTAL_DISPLACEMENT,
        direction_consistency_threshold: float = DIRECTION_CONSISTENCY_THRESHOLD,
        smoothing_window_size: int = SWIPE_SMOOTHING_WINDOW,
        cooldown_seconds: float = SWIPE_COOLDOWN_SECONDS,
        missing_hand_grace_period: float = HAND_MISSING_GRACE_PERIOD,
        vertical_limit: float = SWIPE_VERTICAL_LIMIT,
    ) -> None:
        self.min_swipe_duration = min_swipe_duration
        self.max_swipe_duration = max_swipe_duration
        self.min_horizontal_displacement = min_horizontal_displacement
        self.direction_consistency_threshold = direction_consistency_threshold
        self.smoothing_window_size = max(3, int(smoothing_window_size))
        self.cooldown_seconds = cooldown_seconds
        self.missing_hand_grace_period = missing_hand_grace_period
        self.vertical_limit = vertical_limit

        self.state = "idle"
        self.cooldown_until = 0.0
        self.last_seen_at = 0.0
        self.current_x: Optional[float] = None
        self.current_y: Optional[float] = None
        self.trajectory: Deque[Tuple[float, float, float]] = deque(maxlen=30)
        self.reason = "Waiting for enough horizontal motion"

    def reset(self) -> None:
        self.state = "idle"
        self.trajectory.clear()
        self.current_x = None
        self.current_y = None
        self.reason = "Waiting for enough horizontal motion"

    def _add_sample(self, raw_x_value: float, y_value: float, timestamp: float) -> None:
        if self.state == "cooldown":
            return
        screen_x = self.to_screen_x(float(raw_x_value))
        self.trajectory.append((float(timestamp), float(screen_x), float(y_value)))
        if len(self.trajectory) > 30:
            self.trajectory.popleft()

    def _smoothed_x_values(self) -> list[float]:
        if not self.trajectory:
            return []
        values = [float(item[1]) for item in self.trajectory]
        if len(values) <= 2:
            return values
        smoothed: list[float] = []
        window = min(5, len(values))
        for index in range(len(values)):
            start = max(0, index - window + 1)
            smoothed.append(float(np.mean(values[start : index + 1])))
        return smoothed

    def _compute_summary(self) -> Dict[str, float | None]:
        if not self.trajectory:
            return {
                "raw_start_x": None,
                "raw_current_x": None,
                "screen_start_x": None,
                "screen_current_x": None,
                "screen_delta_x": 0.0,
                "vertical_displacement": 0.0,
                "elapsed_time": 0.0,
                "displacement_x": 0.0,
                "start_x": None,
                "current_x": None,
            }

        smoothed = self._smoothed_x_values()
        oldest = self.trajectory[0]
        newest = self.trajectory[-1]
        start_x = float(smoothed[0]) if smoothed else float(oldest[1])
        current_x = float(smoothed[-1]) if smoothed else float(newest[1])
        delta_x = current_x - start_x
        start_y = float(oldest[2])
        current_y = float(newest[2])
        vertical = abs(current_y - start_y)
        elapsed = float(newest[0] - oldest[0])

        return {
            "raw_start_x": float(oldest[1]),
            "raw_current_x": float(newest[1]),
            "screen_start_x": start_x,
            "screen_current_x": current_x,
            "screen_delta_x": delta_x,
            "vertical_displacement": vertical,
            "elapsed_time": elapsed,
            "displacement_x": delta_x,
            "start_x": start_x,
            "current_x": current_x,
        }

    def _direction_consistency(self, delta_x: float) -> float:
        if len(self.trajectory) < 2:
            return 0.0
        smoothed = self._smoothed_x_values()
        if len(smoothed) < 2:
            return 0.0
        signs = []
        for i in range(1, len(smoothed)):
            step = smoothed[i] - smoothed[i - 1]
            if abs(step) > 1e-4:
                signs.append(1 if step > 0 else -1)
        if not signs:
            return 0.0
        target = 1 if delta_x >= 0 else -1
        consistency = float(np.mean(np.asarray(signs) == target))
        if len(signs) > 2:
            sign_changes = 0
            for i in range(1, len(signs)):
                if signs[i] != signs[i - 1]:
                    sign_changes += 1
            reversal_ratio = sign_changes / (len(signs) - 1)
            if reversal_ratio > 0.60:
                return 0.0
        return consistency

    def _evaluate_swipe(self) -> Optional[str]:
        if len(self.trajectory) < 3:
            return None

        summary = self._compute_summary()
        delta_x = float(summary["screen_delta_x"])
        elapsed = float(summary["elapsed_time"])
        vertical = float(summary["vertical_displacement"])
        direction_ratio = self._direction_consistency(delta_x)

        if not (0.30 <= elapsed <= 2.5):
            self.reason = "Movement duration out of range"
            return None

        if abs(delta_x) < self.min_horizontal_displacement:
            self.reason = "Movement too small"
            return None

        if direction_ratio < self.direction_consistency_threshold:
            self.reason = "Direction not consistent enough"
            return None

        if vertical > 0.0 and abs(delta_x) < 0.18 and vertical > abs(delta_x) * 1.5:
            self.reason = "Mostly vertical movement"
            return None

        if delta_x >= 0.10:
            self.reason = "Swipe Right detected"
            return "swipe_right"
        if delta_x <= -0.10:
            self.reason = "Swipe Left detected"
            return "swipe_left"

        self.reason = "Waiting for enough horizontal motion"
        return None

    def update(
        self,
        x_value: Optional[float],
        timestamp: float,
        hand_present: bool = True,
        y_value: Optional[float] = None,
    ) -> Dict[str, object]:
        if timestamp < self.cooldown_until:
            self.state = "cooldown"
            return {
                "gesture": None,
                "state": "cooldown",
                "hand_detected": hand_present,
                "raw_start_x": None,
                "raw_current_x": None,
                "screen_start_x": None,
                "screen_current_x": None,
                "screen_delta_x": 0.0,
                "current_x": self.current_x,
                "start_x": None,
                "displacement": 0.0,
                "vertical_displacement": 0.0,
                "direction_consistency": 0.0,
                "direction": "idle",
                "detected_direction": "none",
                "elapsed": 0.0,
                "cooldown": float(max(0.0, self.cooldown_until - timestamp)),
                "reason": "Cooldown active",
            }

        if timestamp >= self.cooldown_until and self.state == "cooldown":
            self.reset()

        if x_value is None or not hand_present:
            if self.state != "idle" and (timestamp - self.last_seen_at) > self.missing_hand_grace_period:
                self.reset()
            return {
                "gesture": None,
                "state": self.state,
                "hand_detected": False,
                "raw_start_x": None,
                "raw_current_x": None,
                "screen_start_x": None,
                "screen_current_x": None,
                "screen_delta_x": 0.0,
                "current_x": self.current_x,
                "start_x": None,
                "displacement": 0.0,
                "vertical_displacement": 0.0,
                "direction_consistency": 0.0,
                "direction": "idle",
                "detected_direction": "none",
                "elapsed": 0.0,
                "cooldown": float(max(0.0, self.cooldown_until - timestamp)) if self.cooldown_until > timestamp else 0.0,
                "reason": self.reason,
            }

        self.current_x = float(x_value)
        self.current_y = float(y_value) if y_value is not None else 0.0
        self.last_seen_at = timestamp

        if self.state == "idle":
            self.state = "tracking"

        self._add_sample(self.current_x, self.current_y, timestamp)

        summary = self._compute_summary()
        direction_consistency = self._direction_consistency(float(summary["screen_delta_x"]))
        result = self._evaluate_swipe()
        detected_direction = "right" if float(summary["screen_delta_x"]) > 0 else "left" if float(summary["screen_delta_x"]) < 0 else "none"

        if result is not None:
            self.state = "cooldown"
            self.cooldown_until = timestamp + self.cooldown_seconds
            return {
                "gesture": result,
                "state": "cooldown",
                "hand_detected": True,
                "raw_start_x": summary.get("raw_start_x"),
                "raw_current_x": summary.get("raw_current_x"),
                "screen_start_x": summary.get("screen_start_x"),
                "screen_current_x": summary.get("screen_current_x"),
                "screen_delta_x": float(summary.get("screen_delta_x", 0.0)),
                "current_x": summary.get("current_x"),
                "start_x": summary.get("start_x"),
                "displacement": float(summary.get("displacement_x", 0.0)),
                "vertical_displacement": float(summary.get("vertical_displacement", 0.0)),
                "direction_consistency": float(direction_consistency),
                "direction": "right" if result == "swipe_right" else "left",
                "detected_direction": detected_direction,
                "elapsed": float(summary.get("elapsed_time", 0.0)),
                "cooldown": float(self.cooldown_seconds),
                "reason": self.reason,
            }

        return {
            "gesture": None,
            "state": self.state,
            "hand_detected": True,
            "raw_start_x": summary.get("raw_start_x"),
            "raw_current_x": summary.get("raw_current_x"),
            "screen_start_x": summary.get("screen_start_x"),
            "screen_current_x": summary.get("screen_current_x"),
            "screen_delta_x": float(summary.get("screen_delta_x", 0.0)),
            "current_x": summary.get("current_x"),
            "start_x": summary.get("start_x"),
            "displacement": float(summary.get("displacement_x", 0.0)),
            "vertical_displacement": float(summary.get("vertical_displacement", 0.0)),
            "direction_consistency": float(direction_consistency),
            "direction": "right" if float(summary.get("screen_delta_x", 0.0)) > 0 else "left" if float(summary.get("screen_delta_x", 0.0)) < 0 else "idle",
            "detected_direction": detected_direction,
            "elapsed": float(summary.get("elapsed_time", 0.0)),
            "cooldown": float(max(0.0, self.cooldown_until - timestamp)) if self.cooldown_until > timestamp else 0.0,
            "reason": self.reason,
        }

