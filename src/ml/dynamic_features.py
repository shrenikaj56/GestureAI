from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from src.utils.config import DYNAMIC_SEQUENCE_LENGTH

LANDMARK_COUNT = 21
COORDINATE_COUNT = 3


def _as_array(sequence: Iterable[object]) -> np.ndarray:
    frames = []
    for frame in sequence:
        if hasattr(frame, "landmark"):
            frames.append([[point.x, point.y, point.z] for point in frame.landmark])
        else:
            frames.append(np.asarray(frame, dtype=np.float32))
    array = np.asarray(frames, dtype=np.float32)
    if array.ndim != 3 or array.shape[1:] != (LANDMARK_COUNT, COORDINATE_COUNT):
        raise ValueError("A dynamic sequence must contain frames shaped (21, 3).")
    return array


def resample_sequence(sequence: Iterable[object], target_length: int = DYNAMIC_SEQUENCE_LENGTH) -> np.ndarray:
    """Uniformly interpolate a sequence to the fixed model length."""
    array = _as_array(sequence)
    if len(array) < 2:
        raise ValueError("A dynamic sequence needs at least two valid hand frames.")
    if len(array) == target_length:
        return array

    old_axis = np.linspace(0.0, 1.0, len(array))
    new_axis = np.linspace(0.0, 1.0, target_length)
    flattened = array.reshape(len(array), -1)
    interpolated = np.stack(
        [np.interp(new_axis, old_axis, flattened[:, index]) for index in range(flattened.shape[1])],
        axis=1,
    )
    return interpolated.reshape(target_length, LANDMARK_COUNT, COORDINATE_COUNT).astype(np.float32)


def normalize_sequence(sequence: Iterable[object]) -> np.ndarray:
    """Translate every frame to its wrist and scale it by wrist-to-middle MCP."""
    resampled = resample_sequence(sequence)
    normalized = np.empty_like(resampled)
    for frame_index, frame in enumerate(resampled):
        wrist = frame[0]
        scale = float(np.linalg.norm(frame[9, :2] - wrist[:2]))
        scale = max(scale, 1e-3)
        normalized[frame_index] = (frame - wrist) / scale
    return normalized


def extract_dynamic_features(sequence: Iterable[object]) -> np.ndarray:
    """Create one fixed-size vector containing shape and movement information."""
    resampled = resample_sequence(sequence)
    normalized = normalize_sequence(resampled)
    deltas = np.diff(normalized, axis=0, prepend=normalized[[0]])
    frame_scales = np.maximum(np.linalg.norm(resampled[:, 9, :2] - resampled[:, 0, :2], axis=1), 1e-3)
    wrist_trajectory = (resampled[:, 0, :] - resampled[0, 0, :]) / frame_scales[:, None]
    raw_centers = resampled.mean(axis=1)
    hand_centers = (raw_centers - raw_centers[0]) / frame_scales[:, None]
    overall_motion = np.asarray(
        [
            wrist_trajectory[-1, 0],
            wrist_trajectory[-1, 1],
            wrist_trajectory[-1, 2],
        ],
        dtype=np.float32,
    )
    features = np.concatenate(
        [normalized.reshape(-1), deltas.reshape(-1), wrist_trajectory.reshape(-1), hand_centers.reshape(-1), overall_motion]
    )
    return features.astype(np.float32)


def sequence_movement(sequence: Iterable[object]) -> tuple[float, float]:
    """Return normalized wrist displacement for unknown/no-gesture rejection."""
    resampled = resample_sequence(sequence)
    scale = max(float(np.mean(np.linalg.norm(resampled[:, 9, :2] - resampled[:, 0, :2], axis=1))), 1e-3)
    displacement = (resampled[-1, 0] - resampled[0, 0]) / scale
    return float(displacement[0]), float(displacement[1])
