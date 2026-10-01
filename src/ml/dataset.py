from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.utils.config import DATASET_PATH, MIN_SAMPLES_PER_CLASS


class GestureDataset:
    """Simple dataset manager for collected gesture samples."""

    def __init__(self, dataset_path: Path = DATASET_PATH) -> None:
        self.dataset_path = dataset_path
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, rows: List[Dict[str, object]]) -> None:
        if not rows:
            raise ValueError("No samples available to save.")
        df = pd.DataFrame(rows)
        df.to_csv(self.dataset_path, index=False)

    def load(self) -> pd.DataFrame:
        if not self.dataset_path.exists():
            return pd.DataFrame()
        return pd.read_csv(self.dataset_path)

    def append_sample(self, gesture_name: str, feature_vector: np.ndarray) -> None:
        normalized_name = gesture_name.strip().lower().replace(" ", "_")
        if not normalized_name:
            raise ValueError("Gesture name cannot be empty.")
        row = {"gesture": normalized_name}
        for idx, value in enumerate(feature_vector):
            row[f"feature_{idx}"] = float(value)
        df = self.load()
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(self.dataset_path, index=False)

    def validate(self, min_samples_per_class: int = MIN_SAMPLES_PER_CLASS) -> List[str]:
        """Return actionable validation errors without creating or altering data."""
        if not self.dataset_path.exists():
            return ["Dataset file does not exist. Collect gesture samples first."]

        try:
            df = self.load()
        except Exception as exc:
            return [f"Dataset could not be read: {exc}"]

        if df.empty:
            return ["Dataset is empty. Collect gesture samples first."]
        if "gesture" not in df.columns:
            return ["Dataset is missing the gesture label column."]

        feature_columns = [column for column in df.columns if column.startswith("feature_")]
        if not feature_columns:
            return ["Dataset has no feature columns. Recollect samples through the app."]

        errors = []
        class_counts = df["gesture"].dropna().astype(str).value_counts()
        if len(class_counts) < 2:
            errors.append("Training cannot start because the dataset contains only one gesture class.")
        for label, count in class_counts.items():
            if count < min_samples_per_class:
                errors.append(
                    f"{label.replace('_', ' ').title()} has only {count} samples. "
                    f"Collect at least {min_samples_per_class} samples before training."
                )
        if len(df) < max(2 * min_samples_per_class, 20):
            errors.append("Collect more total samples before training a meaningful model.")
        return errors

    def get_feature_columns(self) -> List[str]:
        df = self.load()
        if df.empty:
            return []
        return [col for col in df.columns if col.startswith("feature_")]

    def get_labels(self) -> List[str]:
        df = self.load()
        if df.empty:
            return []
        return sorted(df["gesture"].dropna().unique().tolist())

    def clear(self) -> None:
        if self.dataset_path.exists():
            self.dataset_path.unlink()
