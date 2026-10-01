from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.utils.config import EVALUATION_PATH


class GestureEvaluator:
    """Read stored evaluation metrics and provide summary information."""

    def __init__(self, evaluation_path: Path = EVALUATION_PATH) -> None:
        self.evaluation_path = evaluation_path

    def load(self) -> Dict[str, Any]:
        if not self.evaluation_path.exists():
            return {}
        with open(self.evaluation_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def summary(self) -> Dict[str, Any]:
        evaluation = self.load()
        if not evaluation:
            return {
                "status": "no_model",
                "message": "No model evaluation available yet. Train the model to view metrics.",
            }
        return {
            "status": "ready",
            "accuracy": evaluation.get("accuracy", 0.0),
            "precision": evaluation.get("precision", 0.0),
            "recall": evaluation.get("recall", 0.0),
            "f1_score": evaluation.get("f1_score", 0.0),
            "n_training_samples": evaluation.get("n_training_samples", 0),
            "n_classes": evaluation.get("n_classes", 0),
            "confusion_matrix": evaluation.get("confusion_matrix", []),
        }
