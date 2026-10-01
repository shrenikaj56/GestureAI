from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np

from src.utils.config import LABELS_PATH, MODEL_PATH


class GesturePredictor:
    """Load a trained model and predict gesture labels for new feature vectors."""

    def __init__(self, model_path: Path = MODEL_PATH, labels_path: Path = LABELS_PATH) -> None:
        self.model_path = model_path
        self.labels_path = labels_path
        self.model = None
        self.label_map = {}

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError("Model file not found. Please train a model first.")
        self.model = joblib.load(self.model_path)
        if self.labels_path.exists():
            with open(self.labels_path, "r", encoding="utf-8") as f:
                self.label_map = json.load(f)

    def predict(self, feature_vector: np.ndarray) -> Dict[str, Any]:
        if self.model is None:
            self.load()
        features = np.asarray([feature_vector], dtype=np.float32)
        prediction_index = int(self.model.predict(features)[0])
        probabilities = self.model.predict_proba(features)[0]
        label = self.label_map.get(str(prediction_index), self.label_map.get(prediction_index, "unknown"))
        return {"label": label, "confidence": float(np.max(probabilities))}
