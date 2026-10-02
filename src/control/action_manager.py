from __future__ import annotations

import json
import time
from typing import Dict, List

import pyautogui
import pygetwindow

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

    def _find_powerpoint_window(self):
        windows = [
            window
            for window in pygetwindow.getAllWindows()
            if any(
                marker in str(window.title or "").casefold()
                for marker in ("powerpoint", "slide show")
            )
        ]
        windows.sort(key=lambda window: "slide show" not in str(window.title or "").casefold())
        return windows[0] if windows else None

    def _send_powerpoint_arrow(self, key: str, controller: str) -> Dict[str, str]:
        details = {
            "controller": controller,
            "powerpoint": "NOT FOUND",
            "focus": "NOT ATTEMPTED",
            "key": key.upper(),
            "result": "NOT SENT",
        }
        try:
            window = self._find_powerpoint_window()
            if window is None:
                return {"status": "error", "message": "PowerPoint window not found.", **details}

            details["powerpoint"] = "FOUND"
            if getattr(window, "isMinimized", False):
                window.restore()
            window.activate()
            time.sleep(0.15)
            active_window = pygetwindow.getActiveWindow()
            window_handle = getattr(window, "_hWnd", None)
            active_handle = getattr(active_window, "_hWnd", None)
            focused = active_window is not None and (
                active_handle == window_handle
                if window_handle is not None
                else active_window.title == window.title
            )
            if not focused:
                details["focus"] = "FAILED"
                return {"status": "error", "message": "PowerPoint was found but could not be focused.", **details}

            details["focus"] = "SUCCESS"
            pyautogui.press(key)
            details["result"] = "SENT"
            return {"status": "success", "message": f"{key.upper()} ARROW SENT.", **details}
        except Exception as exc:  # pragma: no cover - Windows desktop integration path
            return {"status": "error", "message": f"PowerPoint action failed: {exc}", **details}

    def next_slide(self) -> Dict[str, str]:
        return self._send_powerpoint_arrow("right", "next_slide()")

    def previous_slide(self) -> Dict[str, str]:
        return self._send_powerpoint_arrow("left", "previous_slide()")

    def execute(self, action: str) -> Dict[str, str]:
        normalized = action.strip().lower()
        if normalized not in self.allowed:
            return {"status": "blocked", "message": f"Action '{action}' is not in the whitelist."}

        try:
            if normalized == "noop":
                return {"status": "success", "message": "No action emitted."}
            if normalized == "next_slide":
                return self.next_slide()
            if normalized == "previous_slide":
                return self.previous_slide()
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
            elif normalized == "toggle_highlight":
                pyautogui.press("h")
            elif normalized == "space":
                pyautogui.press("space")
            elif normalized == "left_click":
                pyautogui.click()
            elif normalized == "right_arrow":
                pyautogui.press("right")
            elif normalized == "left_arrow":
                pyautogui.press("left")
            elif normalized == "page_down":
                pyautogui.press("pagedown")
            elif normalized == "page_up":
                pyautogui.press("pageup")
            elif normalized == "disable_control":
                return {"status": "success", "message": "Control disabled by gesture."}
            return {"status": "success", "message": f"Executed action '{normalized}' safely."}
        except Exception as exc:  # pragma: no cover - runtime dependency path
            return {"status": "error", "message": f"Action failed: {exc}"}
