from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.ml.dynamic_features import extract_dynamic_features
from src.utils.config import (
    DYNAMIC_EVALUATION_PATH,
    DYNAMIC_LABELS_PATH,
    DYNAMIC_META_PATH,
    DYNAMIC_MODEL_PATH,
    DYNAMIC_SEQUENCE_LENGTH,
)


class DynamicGestureTrainer:
    """Train the separate two-class sequence model."""

    def __init__(self, model_path: Path = DYNAMIC_MODEL_PATH) -> None:
        self.model_path = model_path

    def train(self, sequences: list[np.ndarray], labels: list[str]) -> Dict[str, Any]:
        if len(set(labels)) < 2:
            raise ValueError("At least swipe_left and swipe_right sequences are required.")
        if len(sequences) != len(labels) or len(sequences) < 4:
            raise ValueError("Dynamic dataset is too small for a held-out evaluation.")

        features = np.stack([extract_dynamic_features(sequence) for sequence in sequences])
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.25, random_state=42, stratify=labels
        )
        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        classes = sorted(set(labels))
        evaluation = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, predictions, average="weighted", zero_division=0)),
            "f1_score": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, predictions, labels=classes).tolist(),
            "classes": classes,
            "n_training_sequences": len(sequences),
            "sequence_length": DYNAMIC_SEQUENCE_LENGTH,
            "test_sequences": len(y_test),
        }

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self.model_path)
        with open(DYNAMIC_LABELS_PATH, "w", encoding="utf-8") as file:
            json.dump({str(index): label for index, label in enumerate(model.classes_)}, file, indent=2)
        with open(DYNAMIC_META_PATH, "w", encoding="utf-8") as file:
            json.dump({"sequence_length": DYNAMIC_SEQUENCE_LENGTH, "feature_count": int(features.shape[1]), "classes": classes}, file, indent=2)
        with open(DYNAMIC_EVALUATION_PATH, "w", encoding="utf-8") as file:
            json.dump(evaluation, file, indent=2)
        return evaluation
