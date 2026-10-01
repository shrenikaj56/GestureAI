from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

from src.utils.config import DYNAMIC_CLASSES, DYNAMIC_DATA_DIR, DYNAMIC_SEQUENCE_LENGTH


class DynamicSequenceDataset:
    """Store one real webcam sequence per .npy file, separate from static data."""

    def __init__(self, root: Path = DYNAMIC_DATA_DIR) -> None:
        self.root = root
        for label in DYNAMIC_CLASSES:
            (root / label).mkdir(parents=True, exist_ok=True)

    def add_sequence(self, label: str, sequence: np.ndarray) -> Path:
        normalized_label = label.strip().lower()
        if normalized_label not in DYNAMIC_CLASSES:
            raise ValueError(f"Unsupported dynamic label: {label}")
        array = np.asarray(sequence, dtype=np.float32)
        if array.ndim != 3 or array.shape[1:] != (21, 3) or len(array) < 2:
            raise ValueError("Dynamic sequences must contain valid (frames, 21, 3) landmarks.")
        target_dir = self.root / normalized_label
        existing = sorted(target_dir.glob("sequence_*.npy"))
        path = target_dir / f"sequence_{len(existing) + 1:04d}.npy"
        np.save(path, array)
        return path

    def list_sequences(self, label: str) -> List[Path]:
        return sorted((self.root / label).glob("sequence_*.npy"))

    def load_sequences(self) -> Tuple[List[np.ndarray], List[str]]:
        sequences: List[np.ndarray] = []
        labels: List[str] = []
        for label in DYNAMIC_CLASSES:
            for path in self.list_sequences(label):
                try:
                    sequence = np.load(path, allow_pickle=False)
                    if sequence.ndim == 3 and sequence.shape[1:] == (21, 3) and len(sequence) >= 2:
                        sequences.append(sequence.astype(np.float32))
                        labels.append(label)
                except (OSError, ValueError):
                    continue
        return sequences, labels

    def counts(self) -> dict[str, int]:
        return {label: len(self.list_sequences(label)) for label in DYNAMIC_CLASSES}

    def validate(self, minimum_sequences: int = 2) -> List[str]:
        counts = self.counts()
        errors = []
        for label in DYNAMIC_CLASSES:
            if counts[label] < minimum_sequences:
                errors.append(f"{label} has {counts[label]} valid sequences; collect at least {minimum_sequences}.")
        sequences, _ = self.load_sequences()
        if sequences and any(len(sequence) < 2 for sequence in sequences):
            errors.append("Some dynamic sequences contain insufficient frames.")
        return errors
