from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np

from src.ml.dynamic_features import extract_dynamic_features, sequence_movement
from src.utils.config import DYNAMIC_CONFIDENCE_THRESHOLD, DYNAMIC_MODEL_PATH


class DynamicGesturePredictor:
    """Predict dynamic events with confidence and movement rejection."""

    def __init__(self, model_path: Path = DYNAMIC_MODEL_PATH, confidence_threshold: float = DYNAMIC_CONFIDENCE_THRESHOLD) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError("Dynamic model is not trained yet.")
        self.model = joblib.load(self.model_path)

    def predict(self, sequence: np.ndarray) -> Dict[str, Any]:
        if self.model is None:
            self.load()
        horizontal, vertical = sequence_movement(sequence)
        movement = abs(horizontal) >= 0.5 and abs(horizontal) >= abs(vertical) * 1.15
        features = extract_dynamic_features(sequence)
        probabilities = self.model.predict_proba([features])[0]
        index = int(np.argmax(probabilities))
        label = str(self.model.classes_[index])
        confidence = float(probabilities[index])
        if confidence < self.confidence_threshold or not movement:
            return {"label": None, "confidence": confidence, "movement": movement, "raw_label": label}
        return {"label": label, "confidence": confidence, "movement": movement, "raw_label": label}
