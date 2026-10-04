from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class KnowledgeService:
    """
    Lightweight local knowledge retrieval system for GestureAI.

    It searches the educational knowledge base using TF-IDF
    and cosine similarity.
    """

    def __init__(
        self,
        knowledge_path: Optional[Path] = None,
        similarity_threshold: float = 0.20,
    ) -> None:

        if knowledge_path is None:
            # Project root is three levels above this file:
            # src/services/knowledge_service.py
            project_root = Path(__file__).resolve().parents[2]
            knowledge_path = project_root / "knowledge" / "knowledge_base.csv"

        self.knowledge_path = Path(knowledge_path)
        self.similarity_threshold = similarity_threshold

        self.data: Optional[pd.DataFrame] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.question_vectors = None

        self._load_knowledge_base()

    def _load_knowledge_base(self) -> None:
        """Load and index the local knowledge base."""

        if not self.knowledge_path.exists():
            raise FileNotFoundError(
                f"Knowledge base not found: {self.knowledge_path}"
            )

        data = pd.read_csv(self.knowledge_path)

        required_columns = {"question", "answer", "topic"}

        missing_columns = required_columns - set(data.columns)

        if missing_columns:
            raise ValueError(
                "Knowledge base is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )

        # Remove incomplete rows.
        data = data.dropna(
            subset=["question", "answer", "topic"]
        ).copy()

        # Convert everything to strings.
        data["question"] = data["question"].astype(str)
        data["answer"] = data["answer"].astype(str)
        data["topic"] = data["topic"].astype(str)

        self.data = data.reset_index(drop=True)

        # Convert knowledge-base questions into TF-IDF vectors.
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
        )

        self.question_vectors = self.vectorizer.fit_transform(
            self.data["question"]
        )

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Search the knowledge base for the most relevant question.

        Returns:
            {
                "found": bool,
                "question": str | None,
                "answer": str | None,
                "topic": str | None,
                "similarity": float,
                "matches": [...]
            }
        """

        if not query or not query.strip():
            return self._empty_result()

        if self.data is None or self.vectorizer is None:
            return self._empty_result()

        query = query.strip()

        query_vector = self.vectorizer.transform([query])

        similarities = cosine_similarity(
            query_vector,
            self.question_vectors,
        )[0]

        ranked_indices = similarities.argsort()[::-1][:top_k]

        matches = []

        for index in ranked_indices:
            matches.append(
                {
                    "question": self.data.iloc[index]["question"],
                    "answer": self.data.iloc[index]["answer"],
                    "topic": self.data.iloc[index]["topic"],
                    "similarity": float(similarities[index]),
                }
            )

        best_index = int(ranked_indices[0])
        best_similarity = float(similarities[best_index])

        if best_similarity < self.similarity_threshold:
            return {
                "found": False,
                "question": None,
                "answer": None,
                "topic": None,
                "similarity": best_similarity,
                "matches": matches,
            }

        best_row = self.data.iloc[best_index]

        return {
            "found": True,
            "question": str(best_row["question"]),
            "answer": str(best_row["answer"]),
            "topic": str(best_row["topic"]),
            "similarity": best_similarity,
            "matches": matches,
        }

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        """Return a consistent empty search result."""

        return {
            "found": False,
            "question": None,
            "answer": None,
            "topic": None,
            "similarity": 0.0,
            "matches": [],
        }