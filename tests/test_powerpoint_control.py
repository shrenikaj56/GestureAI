import json
import pytest
from types import SimpleNamespace

import app as app_module
from app import get_mode_actions, is_new_stable_gesture, normalize_gesture
import src.control.action_manager as action_manager_module
from src.control.action_manager import ActionManager


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        ("Thumbs Up", "THUMBS_UP"),
        ("THUMBS_UP", "THUMBS_UP"),
        ("thumbs-up", "THUMBS_UP"),
        ("thumbs down", "THUMBS_DOWN"),
        ("Open_Palm", "OPEN_PALM"),
        ("fist", "FIST"),
        ("Pinch", "PINCH"),
    ],
)
def test_normalize_gesture_labels(raw, canonical):
    assert normalize_gesture(raw) == canonical


def test_unknown_gesture_is_not_fuzzily_mapped():
    assert normalize_gesture("thumbs maybe") is None


def test_stable_gesture_event_is_one_shot_until_state_changes():
    last_processed = None
    events = []
    for _ in range(8):
        stable = "THUMBS_UP"
        if is_new_stable_gesture(stable, last_processed):
            events.append(stable)
        last_processed = stable

    assert events == ["THUMBS_UP"]
    assert is_new_stable_gesture(None, last_processed) is False
    assert is_new_stable_gesture("THUMBS_UP", None) is True


class FakeWindow:
    def __init__(self, title, window_class="", minimized=False):
        self.title = title
        self.window_class = window_class
        self.minimized = minimized


class FakeWin32:
    def __init__(self, windows, focus_succeeds=True):
        self.windows = windows
        self.focus_succeeds = focus_succeeds
        self.foreground = None
        self.restored = []
        self.brought_to_top = []

    def EnumWindows(self, callback, extra):
        for hwnd, window in self.windows.items():
            callback(hwnd, extra)

    def IsWindowVisible(self, hwnd):
        return hwnd in self.windows

    def GetWindowText(self, hwnd):
        return self.windows[hwnd].title

    def GetClassName(self, hwnd):
        return self.windows[hwnd].window_class

    def IsIconic(self, hwnd):
        return self.windows[hwnd].minimized

    def ShowWindow(self, hwnd, _command):
        self.restored.append(hwnd)

    def BringWindowToTop(self, hwnd):
        self.brought_to_top.append(hwnd)

    def SetForegroundWindow(self, hwnd):
        if self.focus_succeeds:
            self.foreground = hwnd

    def GetForegroundWindow(self):
        return self.foreground


class FakeWin32Process:
    @staticmethod
    def GetWindowThreadProcessId(_hwnd):
        return 1, 1234

    @staticmethod
    def GetModuleFileNameEx(_handle, _module):
        return r"C:\Program Files\Microsoft Office\POWERPNT.EXE"


class FakeWin32Api:
    @staticmethod
    def OpenProcess(_access, _inherit, _process_id):
        return 5678

    @staticmethod
    def CloseHandle(_handle):
        pass


class FakeSlideShowView:
    def __init__(self):
        self.calls = []

    def Next(self):
        self.calls.append("Next")

    def Previous(self):
        self.calls.append("Previous")


class FakeSlideShowWindow:
    def __init__(self, hwnd=42):
        self.HWND = hwnd
        self.Caption = "PowerPoint Slide Show - Presentation.pptx"
        self.View = FakeSlideShowView()


class FakeSlideShowWindows:
    def __init__(self, windows):
        self.windows = windows
        self.Count = len(windows)

    def Item(self, index):
        return self.windows[index - 1]


class FakePowerPointApplication:
    HWND = 41

    def __init__(self, slide_show_windows):
        self.SlideShowWindows = FakeSlideShowWindows(slide_show_windows)


def disable_powerpoint_com(monkeypatch):
    def no_running_application(_program_id):
        raise RuntimeError("PowerPoint COM object is not registered")

    monkeypatch.setattr(
        action_manager_module,
        "win32com_client",
        SimpleNamespace(GetActiveObject=no_running_application),
    )
    monkeypatch.setattr(action_manager_module, "win32process", FakeWin32Process)
    monkeypatch.setattr(action_manager_module, "win32api", FakeWin32Api)


@pytest.mark.parametrize(
    ("action", "expected_key"),
    [
        ("next_slide", "right"),
        ("previous_slide", "left"),
    ],
)
def test_slide_action_uses_powerpoint_com_view(monkeypatch, action, expected_key):
    window = FakeSlideShowWindow()
    application = FakePowerPointApplication([window])
    monkeypatch.setattr(
        action_manager_module,
        "win32com_client",
        SimpleNamespace(GetActiveObject=lambda _program_id: application),
    )
    monkeypatch.setattr(action_manager_module.win32gui, "GetForegroundWindow", lambda: 0)
    pressed = []
    monkeypatch.setattr(action_manager_module.pyautogui, "press", pressed.append)

    result = ActionManager().execute(action)

    expected_method = "Next" if action == "next_slide" else "Previous"
    assert result["status"] == "success"
    assert result["method"] == "COM"
    assert result["focus"] == "NOT REQUIRED"
    assert result["key"] == f"COM View.{expected_method}()"
    assert result["result"] == "SENT"
    assert window.View.calls == [expected_method]
    assert pressed == []


@pytest.mark.parametrize(
    ("action", "expected_key"),
    [
        ("next_slide", "right"),
        ("previous_slide", "left"),
    ],
)
def test_slide_action_falls_back_to_focused_slideshow_window(
    monkeypatch, action, expected_key
):
    win32 = FakeWin32({
        41: FakeWindow("Presentation.pptx - PowerPoint"),
        42: FakeWindow("PowerPoint Slide Show - Presentation.pptx", "screenClass", minimized=True),
    })
    pressed = []
    disable_powerpoint_com(monkeypatch)
    monkeypatch.setattr(action_manager_module, "win32gui", win32)
    monkeypatch.setattr(action_manager_module, "win32process", FakeWin32Process)
    monkeypatch.setattr(action_manager_module, "win32api", FakeWin32Api)
    monkeypatch.setattr(action_manager_module, "win32con", SimpleNamespace(
        SW_RESTORE=9, PROCESS_QUERY_INFORMATION=0x400, PROCESS_VM_READ=0x10
    ))
    monkeypatch.setattr(action_manager_module.time, "sleep", lambda _: None)
    monkeypatch.setattr(action_manager_module.pyautogui, "press", pressed.append)

    result = ActionManager().execute(action)

    assert result["status"] == "success"
    assert result["window"] == "PowerPoint Slide Show - Presentation.pptx"
    assert result["hwnd"] == 42
    assert result["powerpoint"] == "FOUND"
    assert result["method"] == "WINDOW FALLBACK"
    assert result["focus"] == "SUCCESS"
    assert result["key"] == expected_key.upper()
    assert result["result"] == "SENT"
    assert pressed == [expected_key]
    assert win32.restored == [42]


def test_powerpoint_document_without_slideshow_is_not_targeted(monkeypatch):
    win32 = FakeWin32({41: FakeWindow("Presentation.pptx - PowerPoint")})
    pressed = []
    disable_powerpoint_com(monkeypatch)
    monkeypatch.setattr(action_manager_module, "win32gui", win32)
    monkeypatch.setattr(action_manager_module, "win32process", FakeWin32Process)
    monkeypatch.setattr(action_manager_module, "win32api", FakeWin32Api)
    monkeypatch.setattr(action_manager_module, "win32con", SimpleNamespace(
        SW_RESTORE=9, PROCESS_QUERY_INFORMATION=0x400, PROCESS_VM_READ=0x10
    ))
    monkeypatch.setattr(action_manager_module.pyautogui, "press", pressed.append)

    result = ActionManager().execute("next_slide")

    assert result["status"] == "error"
    assert result["powerpoint"] == "FOUND"
    assert result["focus"] == "FAILED"
    assert result["window"] == "Presentation.pptx - PowerPoint"
    assert result["hwnd"] == 41
    assert result["result"] == "NOT SENT"
    assert result["message"] == "PowerPoint is running, but Slide Show is not active."
    assert pressed == []


def test_powerpoint_focus_failure_never_sends_key(monkeypatch):
    win32 = FakeWin32({42: FakeWindow("PowerPoint Slide Show - Presentation.pptx", "screenClass")}, focus_succeeds=False)
    pressed = []
    disable_powerpoint_com(monkeypatch)
    monkeypatch.setattr(action_manager_module, "win32gui", win32)
    monkeypatch.setattr(action_manager_module, "win32process", FakeWin32Process)
    monkeypatch.setattr(action_manager_module, "win32api", FakeWin32Api)
    monkeypatch.setattr(action_manager_module, "win32con", SimpleNamespace(
        SW_RESTORE=9, PROCESS_QUERY_INFORMATION=0x400, PROCESS_VM_READ=0x10
    ))
    monkeypatch.setattr(action_manager_module, "pyautogui", SimpleNamespace(press=pressed.append))

    result = ActionManager().execute("next_slide")

    assert result["status"] == "error"
    assert result["powerpoint"] == "FOUND"
    assert result["focus"] == "FAILED"
    assert result["result"] == "NOT SENT"
    assert result["message"] == "PowerPoint Slide Show could not be controlled."
    assert pressed == []


def test_powerpoint_not_running_has_clear_message(monkeypatch):
    disable_powerpoint_com(monkeypatch)
    monkeypatch.setattr(action_manager_module, "win32gui", FakeWin32({}))
    monkeypatch.setattr(action_manager_module, "win32con", SimpleNamespace(
        SW_RESTORE=9, PROCESS_QUERY_INFORMATION=0x400, PROCESS_VM_READ=0x10
    ))

    result = ActionManager().execute("next_slide")

    assert result["status"] == "error"
    assert result["message"] == "PowerPoint is not running."
    assert result["result"] == "NOT SENT"


def test_com_running_without_slideshow_has_clear_message(monkeypatch):
    application = FakePowerPointApplication([])
    monkeypatch.setattr(
        action_manager_module,
        "win32com_client",
        SimpleNamespace(GetActiveObject=lambda _program_id: application),
    )
    monkeypatch.setattr(action_manager_module, "win32gui", FakeWin32({}))

    result = ActionManager().execute("next_slide")

    assert result["status"] == "error"
    assert result["message"] == "PowerPoint is running, but Slide Show is not active."
    assert result["result"] == "NOT SENT"


@pytest.mark.parametrize(
    ("action", "target", "title", "expected_call"),
    [
        ("space", "PowerPoint", "PowerPoint Slide Show - Presentation.pptx", ("press", "space")),
        ("left_click", "PowerPoint", "PowerPoint Slide Show - Presentation.pptx", ("click",)),
        ("page_down", "PDF", "manual.pdf - Chrome", ("press", "pagedown")),
        ("next", "Chrome", "GestureAI - Google Chrome", ("hotkey", "alt", "right")),
        ("previous", "Chrome", "GestureAI - Google Chrome", ("hotkey", "alt", "left")),
    ],
)
def test_actions_focus_selected_target_before_input(monkeypatch, action, target, title, expected_call):
    window_class = "screenClass" if target == "PowerPoint" else "Chrome_WidgetWin_1"
    win32 = FakeWin32({42: FakeWindow(title, window_class)})
    calls = []
    if target == "PowerPoint":
        disable_powerpoint_com(monkeypatch)
        monkeypatch.setattr(action_manager_module, "win32process", FakeWin32Process)
        monkeypatch.setattr(action_manager_module, "win32api", FakeWin32Api)
    monkeypatch.setattr(action_manager_module, "win32gui", win32)
    monkeypatch.setattr(action_manager_module, "win32con", SimpleNamespace(
        SW_RESTORE=9, PROCESS_QUERY_INFORMATION=0x400, PROCESS_VM_READ=0x10
    ))
    monkeypatch.setattr(action_manager_module.time, "sleep", lambda _: None)
    monkeypatch.setattr(action_manager_module.pyautogui, "press", lambda *args: calls.append(("press", *args)))
    monkeypatch.setattr(action_manager_module.pyautogui, "hotkey", lambda *args: calls.append(("hotkey", *args)))
    monkeypatch.setattr(action_manager_module.pyautogui, "click", lambda: calls.append(("click",)))

    result = ActionManager().execute(action, target_app=target)

    assert result["status"] == "success"
    assert result["focused"] is True
    assert result["result"] == "SENT"
    assert calls == [expected_call]


def test_requested_application_mappings():
    assert get_mode_actions("PowerPoint") == {
        "OPEN_PALM": "space",
        "THUMBS_UP": "next_slide",
        "THUMBS_DOWN": "previous_slide",
        "PINCH": "left_click",
        "FIST": "disable_control",
    }
    assert get_mode_actions("PDF") == {
        "OPEN_PALM": "page_down",
        "THUMBS_UP": "page_down",
        "THUMBS_DOWN": "page_up",
        "PINCH": "left_click",
        "FIST": "disable_control",
    }
    assert get_mode_actions("Chrome") == {
        "OPEN_PALM": "space",
        "THUMBS_UP": "next",
        "THUMBS_DOWN": "previous",
        "PINCH": "left_click",
        "FIST": "disable_control",
    }


def test_required_actions_are_added_when_config_omits_them(tmp_path):
    actions_path = tmp_path / "actions.json"
    actions_path.write_text(json.dumps({"safe_actions": ["noop"]}), encoding="utf-8")

    manager = ActionManager(actions_path)

    assert {
        "next_slide",
        "previous_slide",
        "space",
        "left_click",
        "disable_control",
    }.issubset(manager.allowed)


def test_fist_blocks_actions_and_next_valid_gesture_recovers(monkeypatch):
    state = {
        "gesture_event_debug": "",
        "runtime_control_blocked": False,
        "computer_control": True,
        "last_action_time": 0.0,
    }
    actions = []

    class FakeActionManager:
        def execute(self, action, target_app):
            actions.append((action, target_app))
            return {
                "status": "success",
                "message": "Action sent",
                "powerpoint": "FOUND",
                "window": "PowerPoint Slide Show",
                "hwnd": 42,
                "focus": "SUCCESS",
                "key": "RIGHT",
                "result": "SENT",
            }

    monkeypatch.setattr(app_module.st, "session_state", state)
    monkeypatch.setattr(app_module, "get_action_manager", FakeActionManager)

    app_module.execute_stable_gesture("FIST", "PowerPoint")
    assert state["runtime_control_blocked"] is True
    assert actions == []

    app_module.execute_stable_gesture("THUMBS_UP", "PowerPoint")
    assert state["runtime_control_blocked"] is False
    assert actions == [("next_slide", "PowerPoint")]
