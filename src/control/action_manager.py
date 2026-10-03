from __future__ import annotations

import json
import time
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List

import pyautogui

try:
    pythoncom = import_module("pythoncom")
    win32api = import_module("win32api")
    win32con = import_module("win32con")
    win32gui = import_module("win32gui")
    win32process = import_module("win32process")
    win32com_client = import_module("win32com.client")
except ImportError:  # pragma: no cover - unavailable outside Windows installs
    pythoncom = None
    win32api = None
    win32con = None
    win32gui = None
    win32process = None
    win32com_client = None

from src.utils.config import ACTIONS_PATH


class ActionManager:
    """Handle safe, configurable computer actions for recognized gestures."""

    REQUIRED_ACTIONS = ("next_slide", "previous_slide", "space", "left_click", "disable_control")

    def __init__(self, actions_path: Path = ACTIONS_PATH) -> None:
        self.actions_path = actions_path
        self.allowed = list(dict.fromkeys([*self._load_allowed_actions(), *self.REQUIRED_ACTIONS]))
        self._cached_powerpoint_hwnd: int | None = None
        self._cached_powerpoint_title = ""

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
                "space",
                "left_click",
                "right_arrow",
                "left_arrow",
                "page_down",
                "page_up",
            ]

    def _window_matches(self, title: str, window_class: str, target_app: str) -> bool:
        normalized_title = title.casefold()
        normalized_class = window_class.casefold()
        if target_app == "PowerPoint":
            return "slide show" in normalized_title and (
                "powerpoint" in normalized_title or normalized_class == "screenclass"
            )
        if target_app == "Chrome":
            return "google chrome" in normalized_title or normalized_class == "chrome_widgetwin_1"
        if target_app == "PDF":
            return any(token in normalized_title for token in (".pdf", "acrobat", "foxit", "pdf viewer"))
        return False

    def _get_powerpoint_application(self) -> Any | None:
        if win32com_client is None:
            return None
        try:
            return win32com_client.GetActiveObject("PowerPoint.Application")
        except Exception:
            return None

    @contextmanager
    def _com_apartment(self):
        initialized = False
        try:
            if pythoncom is not None:
                pythoncom.CoInitialize()
                initialized = True
            yield
        finally:
            if initialized:
                pythoncom.CoUninitialize()

    @staticmethod
    def _get_active_slideshow(application: Any) -> tuple[Any | None, int]:
        try:
            slide_show_windows = application.SlideShowWindows
            count = int(slide_show_windows.Count)
            if count == 0:
                return None, 0

            windows = [slide_show_windows.Item(index) for index in range(1, count + 1)]
            if win32gui is not None:
                foreground = win32gui.GetForegroundWindow()
                for window in windows:
                    if int(window.HWND) == foreground:
                        return window, count
            return windows[0], count
        except Exception:
            return None, 0

    def _powerpoint_process_id(self, application: Any | None) -> int | None:
        if application is None or win32gui is None or win32process is None:
            return None
        try:
            _thread_id, process_id = win32process.GetWindowThreadProcessId(int(application.HWND))
            return int(process_id)
        except Exception:
            return None

    @staticmethod
    def _is_foreground(hwnd: int) -> bool:
        return win32gui is not None and win32gui.GetForegroundWindow() == hwnd

    def _focus_powerpoint_hwnd(self, hwnd: int, title: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "found": True,
            "focused": False,
            "title": title,
            "hwnd": hwnd,
            "error": "",
            "already_foreground": False,
            "powerpoint_running": True,
        }
        if win32gui is None or win32con is None:
            result["error"] = "Windows window-focus APIs are unavailable. Install pywin32."
            return result

        try:
            if not win32gui.IsWindow(hwnd) or not win32gui.IsWindowVisible(hwnd):
                result.update(found=False, error="Cached PowerPoint Slide Show window is no longer valid.")
                return result
            if self._is_foreground(hwnd):
                result.update(focused=True, already_foreground=True)
                return result

            result["title"] = win32gui.GetWindowText(hwnd).strip() or title
            if win32gui.IsIconic(hwnd):
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            win32gui.BringWindowToTop(hwnd)
            win32gui.SetForegroundWindow(hwnd)
            result["focused"] = self._is_foreground(hwnd)
            if not result["focused"]:
                result["error"] = "PowerPoint Slide Show could not be controlled."
        except Exception as exc:  # pragma: no cover - Windows desktop integration path
            result["error"] = f"PowerPoint Slide Show could not be controlled: {exc}"
        return result

    @staticmethod
    def _log_powerpoint_target_action(action: str, focus_required: bool, success: bool) -> None:
        print(f"Action requested: {action}")
        print("Target: PowerPoint")
        print(f"Focus required: {'YES' if focus_required else 'NO'}")
        print(f"Action result: {'SUCCESS' if success else 'FAILURE'}")

    def _find_powerpoint_slideshow_window(self, application: Any | None) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "found": False,
            "focused": False,
            "title": "",
            "hwnd": None,
            "error": "",
            "powerpoint_running": application is not None,
        }
        if any(api is None for api in (win32gui, win32con, win32process, win32api)):
            result["error"] = "Windows window-focus APIs are unavailable. Install pywin32."
            return result

        application_pid = self._powerpoint_process_id(application)
        process_matches: dict[int, bool] = {}
        slideshow_windows: list[tuple[int, str]] = []
        powerpoint_windows: list[tuple[int, str]] = []
        powerpoint_running = application is not None

        def is_powerpoint_process(process_id: int) -> bool:
            if application_pid is not None:
                return process_id == application_pid
            if process_id not in process_matches:
                handle = None
                try:
                    access = win32con.PROCESS_QUERY_INFORMATION | win32con.PROCESS_VM_READ
                    handle = win32api.OpenProcess(access, False, process_id)
                    executable = win32process.GetModuleFileNameEx(handle, 0)
                    process_matches[process_id] = executable.casefold().endswith("\\powerpnt.exe")
                except Exception:
                    process_matches[process_id] = False
                finally:
                    if handle is not None:
                        win32api.CloseHandle(handle)
            return process_matches[process_id]

        def collect_window(hwnd: int, _extra: Any) -> bool:
            nonlocal powerpoint_running
            if not win32gui.IsWindowVisible(hwnd):
                return True
            try:
                _thread_id, process_id = win32process.GetWindowThreadProcessId(hwnd)
                if not is_powerpoint_process(int(process_id)):
                    return True
                powerpoint_running = True
                title = win32gui.GetWindowText(hwnd).strip()
                window_class = win32gui.GetClassName(hwnd).casefold()
                if title:
                    powerpoint_windows.append((hwnd, title))
                if "slide show" in title.casefold() or window_class == "screenclass":
                    slideshow_windows.append((hwnd, title))
            except Exception:
                return True
            return True

        try:
            win32gui.EnumWindows(collect_window, None)
            result["powerpoint_running"] = powerpoint_running
            if not slideshow_windows:
                if powerpoint_windows:
                    hwnd, title = powerpoint_windows[0]
                    result.update(title=title, hwnd=hwnd)
                result["error"] = (
                    "PowerPoint is running, but Slide Show is not active."
                    if powerpoint_running
                    else "PowerPoint is not running."
                )
                return result

            slideshow_windows.sort(key=lambda item: "slide show" not in item[1].casefold())
            hwnd, title = slideshow_windows[0]
            result.update(found=True, title=title, hwnd=hwnd)
            result["already_foreground"] = self._is_foreground(hwnd)
            if result["already_foreground"]:
                result["focused"] = True
                return result
            if win32gui.IsIconic(hwnd):
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            win32gui.BringWindowToTop(hwnd)
            win32gui.SetForegroundWindow(hwnd)
            time.sleep(0.1)
            result["focused"] = self._is_foreground(hwnd)
            if not result["focused"]:
                result["error"] = "PowerPoint Slide Show could not be controlled."
        except Exception as exc:  # pragma: no cover - Windows desktop integration path
            result["powerpoint_running"] = powerpoint_running
            result["error"] = f"PowerPoint Slide Show could not be controlled: {exc}"
        return result

    @staticmethod
    def _log_powerpoint_action(
        com_available: bool,
        application_found: bool,
        slide_show_count: int,
        active_slide_show_found: bool,
        action: str,
        method: str,
        success: bool,
        error: str,
    ) -> None:
        print(
            "\n".join(
                (
                    f"PowerPoint COM available: {'YES' if com_available else 'NO'}",
                    f"PowerPoint application found: {'YES' if application_found else 'NO'}",
                    f"Slide Show windows count: {slide_show_count}",
                    f"Active Slide Show found: {'YES' if active_slide_show_found else 'NO'}",
                    f"Action requested: {action}",
                    f"Execution method: {method}",
                    f"Action result: {'SUCCESS' if success else 'FAILURE'}",
                    f"Error: {error or 'NONE'}",
                )
            )
        )

    def _execute_powerpoint_slide(self, action: str) -> Dict[str, Any]:
        requested = "NEXT" if action == "next_slide" else "PREVIOUS"
        com_available = win32com_client is not None
        application = self._get_powerpoint_application()
        application_found = application is not None
        slide_show_count = 0
        slide_show = None
        view = None
        before = None
        target = None
        after = None
        error = ""
        method = "COM GOTOSLIDE"
        key = "COM View.GotoSlide(target)"
        command_attempted = False
        print(f"PowerPoint COM available: {'YES' if com_available else 'NO'}")
        print(f"COM APPLICATION FOUND: {'YES' if application_found else 'NO'}")

        if application is None:
            print("SLIDE SHOW COUNT: 0")
            error = "PowerPoint is not running."
        else:
            try:
                slide_show_windows = application.SlideShowWindows
                slide_show_count = int(slide_show_windows.Count)
            except Exception as exc:
                error = str(exc)
            print(f"SLIDE SHOW COUNT: {slide_show_count}")
            if not error and slide_show_count == 0:
                error = "PowerPoint is running, but Slide Show is not active."
            elif not error and slide_show_count > 0:
                try:
                    slide_show = slide_show_windows.Item(1)
                    view = slide_show.View
                    before = int(view.CurrentShowPosition)
                    target = before + 1 if action == "next_slide" else before - 1
                    slide_count = int(slide_show.Presentation.Slides.Count)
                    if target < 1 or target > slide_count:
                        error = f"Target slide {target} is outside the presentation (1-{slide_count})."
                    else:
                        try:
                            view.GotoSlide(target)
                            command_attempted = True
                        except Exception as exc:
                            error = str(exc)
                            try:
                                if int(view.CurrentShowPosition) == target:
                                    command_attempted = True
                                    error = ""
                                else:
                                    method = "WINDOW FALLBACK"
                                    fallback = self._find_powerpoint_slideshow_window(application)
                                    if not fallback["focused"]:
                                        raise RuntimeError(
                                            fallback["error"] or "PowerPoint Slide Show could not be controlled."
                                        )
                                    pyautogui.press("right" if action == "next_slide" else "left")
                                    command_attempted = True
                                    error = ""
                            except Exception as fallback_exc:
                                error = f"COM failed: {exc}; keyboard fallback failed: {fallback_exc}"

                        if slide_show is not None and win32gui is not None and win32con is not None:
                            try:
                                hwnd = int(getattr(slide_show, "HWND", 0))
                                if hwnd:
                                    if win32gui.IsIconic(hwnd):
                                        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                                    win32gui.BringWindowToTop(hwnd)
                                    win32gui.SetForegroundWindow(hwnd)
                            except Exception:
                                pass

                        time.sleep(0.5)
                        after = int(view.CurrentShowPosition)
                        if after != target:
                            error = "PowerPoint received the command but the slide position did not change to the target."
                except Exception as exc:
                    error = str(exc)

        title = ""
        hwnd = None
        if slide_show is not None:
            title = str(getattr(slide_show, "Caption", "PowerPoint Slide Show"))
            try:
                hwnd = int(getattr(slide_show, "HWND", 0)) or None
            except (TypeError, ValueError):
                hwnd = None

        success = not error and target is not None and after == target
        print(f"Action requested: {requested}")
        print(f"BEFORE: {before if before is not None else 'UNAVAILABLE'}")
        print(f"TARGET: {target if target is not None else 'UNAVAILABLE'}")
        print(f"AFTER: {after if after is not None else 'UNAVAILABLE'}")
        print(f"Execution method: {method}")
        print(f"Action result: {'SUCCESS' if success else 'FAILURE'}")
        if error:
            print(f"Error: {error}")
        return {
            "status": "success" if success else "error",
            "message": (
                f"{requested.title()} slide moved to position {target}."
                if success
                else error or "PowerPoint Slide Show could not be controlled."
            ),
            "powerpoint": "FOUND" if application_found else "NOT FOUND",
            "window": title,
            "hwnd": hwnd,
            "focus": "NOT REQUIRED",
            "key": key if method == "COM GOTOSLIDE" else ("RIGHT" if action == "next_slide" else "LEFT"),
            "result": "SENT" if command_attempted else "NOT SENT",
            "method": method,
            "error": error,
            "slide_before": before,
            "slide_target": target,
            "slide_after": after,
        }

    def focus_target_window(self, target_app: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "found": False,
            "focused": False,
            "title": "",
            "hwnd": None,
            "error": "",
        }
        if target_app == "PowerPoint":
            cached_hwnd = self._cached_powerpoint_hwnd
            if cached_hwnd is not None:
                cached_focus = self._focus_powerpoint_hwnd(
                    cached_hwnd, self._cached_powerpoint_title
                )
                if cached_focus["found"]:
                    return cached_focus
                self._cached_powerpoint_hwnd = None
                self._cached_powerpoint_title = ""

            with self._com_apartment():
                application = self._get_powerpoint_application()
                focus = self._find_powerpoint_slideshow_window(application)
            if focus["found"] and focus["hwnd"] is not None:
                self._cached_powerpoint_hwnd = int(focus["hwnd"])
                self._cached_powerpoint_title = focus["title"]
            return focus
        if win32gui is None or win32con is None:
            result["error"] = "Windows window-focus APIs are unavailable. Install pywin32."
            return result

        windows: list[tuple[int, str, str]] = []
        powerpoint_windows: list[tuple[int, str, str]] = []

        def collect_window(hwnd: int, _extra: Any) -> bool:
            if not win32gui.IsWindowVisible(hwnd):
                return True
            title = win32gui.GetWindowText(hwnd).strip()
            if not title:
                return True
            try:
                window_class = win32gui.GetClassName(hwnd)
            except Exception:
                window_class = ""
            if target_app == "PowerPoint" and "powerpoint" in title.casefold():
                powerpoint_windows.append((hwnd, title, window_class))
            if self._window_matches(title, window_class, target_app):
                windows.append((hwnd, title, window_class))
            return True

        try:
            win32gui.EnumWindows(collect_window, None)
            if target_app == "PowerPoint":
                windows.sort(
                    key=lambda item: "powerpoint slide show" not in item[1].casefold()
                )
            if not windows:
                if target_app == "PowerPoint" and powerpoint_windows:
                    hwnd, title, _window_class = powerpoint_windows[0]
                    result.update(found=True, title=title, hwnd=hwnd)
                result["error"] = (
                    "PowerPoint Slide Show not detected. Start the presentation in Slide Show mode."
                    if target_app == "PowerPoint"
                    else f"{target_app} window not found."
                )
                return result

            hwnd, title, _window_class = windows[0]
            result.update(found=True, title=title, hwnd=hwnd)
            if win32gui.IsIconic(hwnd):
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            win32gui.BringWindowToTop(hwnd)
            win32gui.SetForegroundWindow(hwnd)
            time.sleep(0.1)
            result["focused"] = win32gui.GetForegroundWindow() == hwnd
            if not result["focused"]:
                result["error"] = f"{target_app} window was found but could not be focused."
        except Exception as exc:  # pragma: no cover - Windows desktop integration path
            result["error"] = f"Could not focus {target_app}: {exc}"
        return result

    def focus_powerpoint(self) -> Dict[str, Any]:
        return self.focus_target_window("PowerPoint")

    def execute(self, action: str, target_app: str = "PowerPoint") -> Dict[str, Any]:
        if target_app != "PowerPoint":
            return self._execute_action(action, target_app)
        if action.strip().lower() not in {"next_slide", "previous_slide"}:
            return self._execute_action(action, target_app)
        try:
            with self._com_apartment():
                return self._execute_action(action, target_app)
        except Exception as exc:  # pragma: no cover - COM initialization failure
            return {
                "status": "error",
                "message": f"PowerPoint Slide Show could not be controlled: {exc}",
                "powerpoint": "UNKNOWN",
                "focus": "FAILED",
                "result": "NOT SENT",
                "error": str(exc),
            }

    def _execute_action(self, action: str, target_app: str) -> Dict[str, Any]:
        normalized = action.strip().lower()
        if normalized not in self.allowed:
            return {"status": "blocked", "message": f"Action '{action}' is not in the whitelist."}
        if normalized == "noop":
            return {"status": "success", "message": "No action emitted.", "result": "NOT SENT"}
        if normalized == "disable_control":
            return {"status": "success", "message": "Control disabled by gesture.", "result": "NOT SENT"}
        if target_app == "PowerPoint" and normalized in {"next_slide", "previous_slide"}:
            return self._execute_powerpoint_slide(normalized)

        try:
            focus = self.focus_target_window(target_app)
            target_found = focus["found"]
            target_focused = focus["focused"]
            focus_required = not focus.get("already_foreground", False)
            key = {
                "next_slide": "RIGHT",
                "previous_slide": "LEFT",
                "next": "ALT+RIGHT",
                "previous": "ALT+LEFT",
                "forward": "CTRL+RIGHT",
                "back": "CTRL+LEFT",
                "confirm": "ENTER",
                "cancel": "ESC",
                "toggle_focus": "TAB",
                "toggle_highlight": "H",
                "space": "SPACE",
                "right_arrow": "RIGHT",
                "left_arrow": "LEFT",
                "page_down": "PAGEDOWN",
                "page_up": "PAGEUP",
                "scroll_up": "SCROLL UP",
                "left_click": "LEFT CLICK",
            }.get(normalized, normalized.upper())
            powerpoint = "FOUND" if target_found else "NOT FOUND"
            details = {
                "found": target_found,
                "focused": target_focused,
                "target": target_app,
                "powerpoint": powerpoint if target_app == "PowerPoint" else "N/A",
                "window": focus["title"],
                "hwnd": focus["hwnd"],
                "focus": "SUCCESS" if target_focused else "FAILED",
                "key": key,
                "result": "NOT SENT",
            }
            if not target_focused:
                if target_app == "PowerPoint" and normalized in {"space", "left_click"}:
                    self._log_powerpoint_target_action(
                        "SPACE" if normalized == "space" else "LEFT_CLICK",
                        focus_required,
                        False,
                    )
                return {
                    "status": "error",
                    "message": focus["error"],
                    **details,
                }

            if normalized in {"next_slide", "right_arrow"}:
                pyautogui.press("right")
            elif normalized in {"previous_slide", "left_arrow"}:
                pyautogui.press("left")
            elif normalized == "next":
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
            else:
                return {"status": "blocked", "message": f"Unsupported action '{normalized}'.", **details}

            details["result"] = "SENT"
            if target_app == "PowerPoint" and normalized in {"space", "left_click"}:
                self._log_powerpoint_target_action(
                    "SPACE" if normalized == "space" else "LEFT_CLICK",
                    focus_required,
                    True,
                )
            return {
                "status": "success",
                "message": f"{key} sent to {target_app} window '{focus['title']}'.",
                **details,
            }
        except Exception as exc:  # pragma: no cover - runtime dependency path
            if target_app == "PowerPoint" and normalized in {"space", "left_click"}:
                self._log_powerpoint_target_action(
                    "SPACE" if normalized == "space" else "LEFT_CLICK",
                    not locals().get("focus", {}).get("already_foreground", False),
                    False,
                )
            return {
                "status": "error",
                "message": f"Action failed: {exc}",
                **locals().get("details", {}),
                "focus": "SUCCESS" if locals().get("target_focused") else "FAILED",
                "key": locals().get("key", normalized.upper()),
                "result": "NOT SENT",
            }
