from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.ml.dataset import GestureDataset


class GestureCollector:
    """Collect and export gesture samples from landmarks."""

    def __init__(self, dataset_path: Path | None = None) -> None:
        self.dataset = GestureDataset(dataset_path=dataset_path) if dataset_path else GestureDataset()
        self.samples: List[Dict[str, object]] = []

    def add_sample(self, gesture_name: str, feature_vector: np.ndarray) -> None:
        row = {"gesture": gesture_name}
        for idx, value in enumerate(feature_vector):
            row[f"feature_{idx}"] = float(value)
        self.samples.append(row)

    def save_samples(self) -> pd.DataFrame:
        if not self.samples:
            raise ValueError("No gesture samples were collected.")

        existing = self.dataset.load()
        if not existing.empty:
            combined = pd.concat([existing, pd.DataFrame(self.samples)], ignore_index=True)
            combined.to_csv(self.dataset.dataset_path, index=False)
        else:
            pd.DataFrame(self.samples).to_csv(self.dataset.dataset_path, index=False)

        saved = self.dataset.load()
        self.samples = []
        return saved
