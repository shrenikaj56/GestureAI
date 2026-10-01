from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from src.utils.config import ACTIONS_PATH


class ContextManager:
    """Map a gesture and current context to a safe action description."""

    def __init__(self, actions_path: Path = ACTIONS_PATH) -> None:
        self.actions_path = actions_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if not self.actions_path.exists():
            return {"gesture_actions": {}, "contexts": {}, "safe_actions": []}
        with open(self.actions_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def resolve_action(self, gesture_label: str, context: str = "default") -> str:
        if not gesture_label:
            return "noop"

        normalized = gesture_label.strip().lower().replace(" ", "_")

        if context in self.config.get("contexts", {}):
            context_map = self.config["contexts"][context]
            if normalized in context_map:
                return context_map[normalized]

        default_map = self.config.get("gesture_actions", {})
        if normalized in default_map:
            return default_map[normalized]
        return "noop"
