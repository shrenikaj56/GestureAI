import pytest

from app import is_new_stable_gesture, normalize_gesture
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
    title = "Presentation.pptx - PowerPoint"
    _hWnd = 42
    isMinimized = False

    def activate(self):
        pass

    def restore(self):
        pass


@pytest.mark.parametrize(
    ("action", "expected_key", "expected_controller"),
    [
        ("next_slide", "right", "next_slide()"),
        ("previous_slide", "left", "previous_slide()"),
    ],
)
def test_slide_action_focuses_powerpoint_then_sends_one_arrow(
    monkeypatch, action, expected_key, expected_controller
):
    window = FakeWindow()
    pressed = []
    monkeypatch.setattr("src.control.action_manager.pygetwindow.getAllWindows", lambda: [window])
    monkeypatch.setattr("src.control.action_manager.pygetwindow.getActiveWindow", lambda: window)
    monkeypatch.setattr("src.control.action_manager.time.sleep", lambda _: None)
    monkeypatch.setattr("src.control.action_manager.pyautogui.press", pressed.append)

    result = ActionManager().execute(action)

    assert result["status"] == "success"
    assert result["controller"] == expected_controller
    assert result["powerpoint"] == "FOUND"
    assert result["focus"] == "SUCCESS"
    assert result["key"] == expected_key.upper()
    assert result["result"] == "SENT"
    assert pressed == [expected_key]


def test_slide_action_does_not_send_key_when_powerpoint_is_missing(monkeypatch):
    pressed = []
    monkeypatch.setattr("src.control.action_manager.pygetwindow.getAllWindows", lambda: [])
    monkeypatch.setattr("src.control.action_manager.pyautogui.press", pressed.append)

    result = ActionManager().execute("next_slide")

    assert result["status"] == "error"
    assert result["powerpoint"] == "NOT FOUND"
    assert result["result"] == "NOT SENT"
    assert pressed == []
