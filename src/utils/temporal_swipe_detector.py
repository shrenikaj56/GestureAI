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
    """Natural-usage temporal swipe detector using relative hand trajectory and timestamps."""

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
        self.smoothing_window_size = max(8, smoothing_window_size)
        self.cooldown_seconds = cooldown_seconds
        self.missing_hand_grace_period = missing_hand_grace_period
        self.vertical_limit = vertical_limit

        self.state = "IDLE"
        self.cooldown_until = 0.0
        self.last_seen_at = 0.0
        self.current_x: Optional[float] = None
        self.current_y: Optional[float] = None
        self.trajectory: Deque[Tuple[float, float]] = deque(maxlen=self.smoothing_window_size)
        self.reason = "Waiting for more horizontal movement"

    def reset(self) -> None:
        self.state = "IDLE"
        self.trajectory.clear()
        self.current_x = None
        self.current_y = None
        self.reason = "Waiting for more horizontal movement"

    def _add_sample(self, x_value: float, y_value: float, timestamp: float) -> None:
        if self.state == "COOLDOWN":
            return
        self.trajectory.append((timestamp, float(x_value), float(y_value)))
        while len(self.trajectory) > self.smoothing_window_size:
            self.trajectory.popleft()

    def _compute_summary(self) -> Dict[str, Optional[float]]:
        if not self.trajectory:
            return {"start_x": None, "current_x": None, "displacement_x": 0.0, "start_y": None, "end_y": None, "vertical_displacement": 0.0, "elapsed_time": 0.0}
        start = self.trajectory[0]
        end = self.trajectory[-1]
        start_x = float(start[1])
        end_x = float(end[1])
        start_y = float(start[2])
        end_y = float(end[2])
        return {
            "start_x": start_x,
            "current_x": end_x,
            "displacement_x": float(end_x - start_x),
            "start_y": start_y,
            "end_y": end_y,
            "vertical_displacement": float(abs(end_y - start_y)),
            "elapsed_time": float(end[0] - start[0]),
        }

    def _direction_consistency(self, delta_x: float) -> float:
        if len(self.trajectory) < 2:
            return 0.0
        step_signs = []
        for i in range(1, len(self.trajectory)):
            step = float(self.trajectory[i][1] - self.trajectory[i - 1][1])
            if abs(step) > 1e-4:
                step_signs.append(1 if step > 0 else -1)
        if not step_signs:
            return 0.0
        target_sign = 1 if delta_x > 0 else -1
        direction_ratio = float(np.mean(np.asarray(step_signs) == target_sign))

        if len(step_signs) > 1:
            sign_changes = 0
            for i in range(1, len(step_signs)):
                if step_signs[i] != step_signs[i - 1]:
                    sign_changes += 1
            reversal_ratio = sign_changes / (len(step_signs) - 1)
            if reversal_ratio > 0.45:
                return 0.0

        return direction_ratio

    def _evaluate_swipe(self, timestamp: float) -> Optional[str]:
        if len(self.trajectory) < 2:
            return None
        summary = self._compute_summary()
        delta_x = float(summary["displacement_x"])
        elapsed = float(summary["elapsed_time"])
        displacement_mag = abs(delta_x)
        vertical = float(summary["vertical_displacement"])
        if not (self.min_swipe_duration <= elapsed <= self.max_swipe_duration):
            self.reason = "Movement duration out of range"
            return None
        if displacement_mag < self.min_horizontal_displacement:
            self.reason = "Movement too small"
            return None
        if vertical > self.vertical_limit and displacement_mag < self.min_horizontal_displacement * 1.5:
            self.reason = "Mostly vertical movement"
            return None
        direction_ratio = self._direction_consistency(delta_x)
        if direction_ratio < self.direction_consistency_threshold:
            self.reason = "Direction not consistent enough"
            return None
        if abs(delta_x) < abs(self.min_horizontal_displacement * 1.5) and len(self.trajectory) >= 4:
            step_magnitudes = [
                abs(float(self.trajectory[i][1] - self.trajectory[i - 1][1]))
                for i in range(1, len(self.trajectory))
            ]
            total_step_magnitude = sum(step_magnitudes)
            if total_step_magnitude > 0 and abs(delta_x) / total_step_magnitude < 0.72:
                self.reason = "Trajectory too irregular"
                return None
        if delta_x > 0:
            self.reason = "Swipe Right detected"
            return "swipe_right"
        if delta_x < 0:
            self.reason = "Swipe Left detected"
            return "swipe_left"
        self.reason = "Waiting for more horizontal movement"
        return None

    def update(
        self,
        x_value: Optional[float],
        timestamp: float,
        hand_present: bool = True,
        y_value: Optional[float] = None,
    ) -> Dict[str, object]:
        if timestamp < self.cooldown_until:
            self.state = "COOLDOWN"
            return {
                "gesture": None,
                "state": "COOLDOWN",
                "hand_detected": hand_present,
                "current_x": self.current_x,
                "start_x": self.trajectory[0][1] if self.trajectory else None,
                "displacement": float((self.trajectory[-1][1] - self.trajectory[0][1]) if len(self.trajectory) >= 2 else 0.0),
                "direction": "idle",
                "elapsed": float((timestamp - self.trajectory[0][0]) if len(self.trajectory) >= 2 else 0.0),
                "cooldown": float(max(0.0, self.cooldown_until - timestamp)),
                "reason": "Cooldown active",
            }

        if x_value is None or not hand_present:
            if self.state != "IDLE" and (timestamp - self.last_seen_at) > self.missing_hand_grace_period:
                self.reset()
            self.current_x = None
            self.current_y = None
            return {
                "gesture": None,
                "state": self.state,
                "hand_detected": False,
                "current_x": None,
                "start_x": self.trajectory[0][1] if self.trajectory else None,
                "displacement": float((self.trajectory[-1][1] - self.trajectory[0][1]) if len(self.trajectory) >= 2 else 0.0),
                "direction": "idle",
                "elapsed": float((timestamp - self.trajectory[0][0]) if len(self.trajectory) >= 2 else 0.0),
                "cooldown": float(max(0.0, self.cooldown_until - timestamp)) if self.cooldown_until > timestamp else 0.0,
                "reason": self.reason,
            }

        self.current_x = float(x_value)
        self.current_y = float(y_value) if y_value is not None else self.current_y if self.current_y is not None else 0.0
        self.last_seen_at = timestamp
        if self.state == "IDLE":
            self.state = "TRACKING"
        self._add_sample(self.current_x, self.current_y, timestamp)

        result = self._evaluate_swipe(timestamp)
        if result is not None:
            self.state = "COOLDOWN"
            self.cooldown_until = timestamp + self.cooldown_seconds
            summary = self._compute_summary()
            return {
                "gesture": result,
                "state": "COOLDOWN",
                "hand_detected": True,
                "current_x": float(summary["current_x"]),
                "start_x": float(summary["start_x"]),
                "displacement": float(summary["displacement_x"]),
                "direction": "right" if result == "swipe_right" else "left",
                "elapsed": float(summary["elapsed_time"]),
                "cooldown": float(self.cooldown_seconds),
                "reason": self.reason,
            }

        summary = self._compute_summary()
        return {
            "gesture": None,
            "state": self.state,
            "hand_detected": True,
            "current_x": float(summary["current_x"]),
            "start_x": float(summary["start_x"]) if summary["start_x"] is not None else None,
            "displacement": float(summary["displacement_x"]),
            "direction": "right" if float(summary["displacement_x"]) > 0 else "left" if float(summary["displacement_x"]) < 0 else "idle",
            "elapsed": float(summary["elapsed_time"]),
            "cooldown": float(max(0.0, self.cooldown_until - timestamp)) if self.cooldown_until > timestamp else 0.0,
            "reason": self.reason,
        }
