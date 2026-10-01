from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.utils.config import (
    DATASET_PATH,
    EVALUATION_PATH,
    FEATURE_META_PATH,
    LABELS_PATH,
    MODEL_PATH,
)


class GestureTrainer:
    """Train and evaluate the gesture recognition model."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        labels_path: Path = LABELS_PATH,
        feature_meta_path: Path = FEATURE_META_PATH,
        evaluation_path: Path = EVALUATION_PATH,
    ) -> None:
        self.model_path = model_path
        self.labels_path = labels_path
        self.feature_meta_path = feature_meta_path
        self.evaluation_path = evaluation_path
        self.model = None
        self.label_map = {}

    def _prepare_data(self, dataset: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        if dataset.empty:
            raise ValueError("Dataset is empty. Collect more gesture samples before training.")

        feature_columns = [col for col in dataset.columns if col.startswith("feature_")]
        if not feature_columns:
            raise ValueError("No feature columns found in dataset.")

        if "gesture" not in dataset.columns:
            raise ValueError("Dataset does not contain a gesture label column.")

        labels = dataset["gesture"].astype(str).tolist()
        features = dataset[feature_columns].values.astype(np.float32)
        unique_labels = sorted(dataset["gesture"].unique().tolist())
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        encoded = np.asarray([label_map[label] for label in labels], dtype=np.int64)
        return features, encoded, unique_labels, feature_columns

    def train(self, dataset: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        features, labels, unique_labels, feature_columns = self._prepare_data(dataset)

        if len(unique_labels) < 2:
            raise ValueError("At least two gesture classes are required for training.")

        class_counts = pd.Series(labels).value_counts()
        if class_counts.min() < 2:
            raise ValueError("Each gesture class needs at least two samples for train/test splitting.")

        X_train, X_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
        )

        model = RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            class_weight="balanced",
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        precision = precision_score(y_test, predictions, average="weighted", zero_division=0)
        recall = recall_score(y_test, predictions, average="weighted", zero_division=0)
        f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, predictions, labels=np.arange(len(unique_labels)))

        self.model = model
        self.label_map = {idx: label for idx, label in enumerate(unique_labels)}

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self.model_path)
        with open(self.labels_path, "w", encoding="utf-8") as labels_file:
            json.dump(self.label_map, labels_file, indent=2)

        meta = {
            "feature_columns": feature_columns,
            "n_features": len(feature_columns),
            "n_classes": len(unique_labels),
            "labels": unique_labels,
        }
        with open(self.feature_meta_path, "w", encoding="utf-8") as meta_file:
            json.dump(meta, meta_file, indent=2)

        evaluation = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confusion_matrix": cm.tolist(),
            "n_training_samples": int(len(dataset)),
            "n_classes": int(len(unique_labels)),
            "classes": unique_labels,
        }
        with open(self.evaluation_path, "w", encoding="utf-8") as eval_file:
            json.dump(evaluation, eval_file, indent=2)

        return evaluation

    def load(self) -> Any:
        if not self.model_path.exists():
            raise FileNotFoundError("No trained model was found. Please train the model first.")
        self.model = joblib.load(self.model_path)
        with open(self.labels_path, "r", encoding="utf-8") as labels_file:
            self.label_map = json.load(labels_file)
        return self.model

    def predict(self, feature_vector: np.ndarray) -> Dict[str, Any]:
        if self.model is None:
            self.load()
        prediction_index = int(self.model.predict(np.asarray([feature_vector], dtype=np.float32))[0])
        probabilities = self.model.predict_proba(np.asarray([feature_vector], dtype=np.float32))[0]
        prediction_label = self.label_map.get(str(prediction_index), self.label_map.get(prediction_index))
        confidence = float(np.max(probabilities))
        return {"label": prediction_label, "confidence": confidence}
