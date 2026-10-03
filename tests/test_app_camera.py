import numpy as np

from app import normalize_gesture, open_camera_capture


class FakeCapture:
    def __init__(self, opened: bool = True, readable: bool = True) -> None:
        self.opened = opened
        self.readable = readable
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def set(self, *_args: object) -> None:
        pass

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self.readable:
            return False, None
        return True, np.zeros((24, 32, 3), dtype=np.uint8)

    def release(self) -> None:
        self.released = True


def test_normalize_gesture_handles_expected_labels() -> None:
    assert normalize_gesture("thumbs_up") == "THUMBS_UP"
    assert normalize_gesture("thumbs_down") == "THUMBS_DOWN"
    assert normalize_gesture("open_palm") == "OPEN_PALM"


def test_open_camera_capture_validates_one_capture(monkeypatch) -> None:
    capture = FakeCapture()
    created = []

    def open_capture(index: int) -> FakeCapture:
        created.append(index)
        return capture

    monkeypatch.setattr("app.cv2.VideoCapture", open_capture)
    result, message = open_camera_capture()

    assert result is capture
    assert created == [0]
    assert "frame received" in message


def test_open_camera_capture_releases_capture_when_first_frame_fails(monkeypatch) -> None:
    capture = FakeCapture(readable=False)
    monkeypatch.setattr("app.cv2.VideoCapture", lambda _index: capture)

    result, message = open_camera_capture()

    assert result is None
    assert capture.released
    assert "first frame could not be read" in message