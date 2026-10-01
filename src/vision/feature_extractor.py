from __future__ import annotations

import math
from typing import Iterable, List, Sequence

import numpy as np


def landmark_to_array(landmarks: object) -> np.ndarray:
    """Convert MediaPipe landmarks to a flat NumPy array of x, y, z values."""
    coords = []
    for landmark in landmarks.landmark:
        coords.append([landmark.x, landmark.y, landmark.z])
    return np.asarray(coords, dtype=np.float32)


def extract_feature_vector(landmarks: object, image_shape: Sequence[int]) -> np.ndarray:
    """Build normalized landmark and distance features for ML classification."""
    if landmarks is None:
        raise ValueError("Landmarks were not provided.")

    arr = landmark_to_array(landmarks)
    wrist = arr[0]
    normalized = []
    for x, y, z in arr:
        normalized.extend([x - wrist[0], y - wrist[1], z - wrist[2]])

    height, width = image_shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be valid positive integers.")

    landmark_pairs = [
        (0, 5), (0, 9), (0, 13), (0, 17),
        (5, 8), (9, 12), (13, 16), (17, 20),
        (8, 12), (12, 16), (16, 20), (4, 8),
        (8, 5), (5, 9)
    ]

    distances: List[float] = []
    for a_idx, b_idx in landmark_pairs:
        a_xyz = arr[a_idx]
        b_xyz = arr[b_idx]
        diff = a_xyz - b_xyz
        distance = float(np.linalg.norm(diff))
        distances.append(distance)

    finger_span = float(np.linalg.norm(arr[8] - arr[4]))
    palm_width = float(np.linalg.norm(arr[5] - arr[17]))
    distances.extend([finger_span, palm_width])

    normalized_features = np.asarray(normalized, dtype=np.float32)
    distance_features = np.asarray(distances, dtype=np.float32)
    feature_vector = np.concatenate([normalized_features, distance_features])
    return feature_vector.astype(np.float32)


def feature_vector_to_dict(vector: Iterable[float]) -> dict:
    return {f"feature_{idx}": float(value) for idx, value in enumerate(vector)}
