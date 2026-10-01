from __future__ import annotations

import json
from typing import Dict, List

import pyautogui

from src.utils.config import ACTIONS_PATH


class ActionManager:
    """Handle safe, configurable computer actions for recognized gestures."""

    def __init__(self, actions_path: ACTIONS_PATH = ACTIONS_PATH) -> None:
        self.actions_path = actions_path
        self.allowed = self._load_allowed_actions()

    def _load_allowed_actions(self) -> List[str]:
        try:
            with open(self.actions_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            return config.get("safe_actions", [])
        except Exception:
            return [
                "noop",
                "next",
                "previous",
                "forward",
                "back",
                "confirm",
                "cancel",
                "toggle_focus",
                "scroll_up",
                "next_slide",
                "previous_slide",
                "toggle_highlight",
            ]

    def execute(self, action: str) -> Dict[str, str]:
        normalized = action.strip().lower()
        if normalized not in self.allowed:
            return {"status": "blocked", "message": f"Action '{action}' is not in the whitelist."}

        try:
            if normalized == "noop":
                return {"status": "success", "message": "No action emitted."}
            if normalized == "next":
                pyautogui.hotkey("alt", "right")
            elif normalized == "previous":
                pyautogui.hotkey("alt", "left")
            elif normalized == "forward":
                pyautogui.hotkey("ctrl", "right")
            elif normalized == "back":
                pyautogui.hotkey("ctrl", "left")
            elif normalized == "confirm":
                pyautogui.press("enter")
            elif normalized == "cancel":
                pyautogui.press("esc")
            elif normalized == "toggle_focus":
                pyautogui.press("tab")
            elif normalized == "scroll_up":
                pyautogui.scroll(200)
            elif normalized == "next_slide":
                pyautogui.hotkey("pagedown")
            elif normalized == "previous_slide":
                pyautogui.hotkey("pageup")
            elif normalized == "toggle_highlight":
                pyautogui.press("h")
            return {"status": "success", "message": f"Executed action '{normalized}' safely."}
        except Exception as exc:  # pragma: no cover - runtime dependency path
            return {"status": "error", "message": f"Action failed: {exc}"}
