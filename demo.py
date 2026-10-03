from __future__ import annotations

import ctypes
import re
import time
from collections import Counter, deque
from typing import Any

import cv2
import pyautogui

from src.control.action_manager import ActionManager
from src.ml.predictor import GesturePredictor
from src.vision.feature_extractor import extract_feature_vector
from src.vision.hand_detector import HandDetector

WINDOW_NAME = "GestureAI"
CONFIDENCE_THRESHOLD = 0.60
ACTION_COOLDOWN_SECONDS = 0.35
STABILIZATION_WINDOW = 3
STABILIZATION_VOTES = 2

GESTURE_ALIASES = {
    "OPENPALM": "OPEN_PALM",
    "FIST": "FIST",
    "THUMBSUP": "THUMBS_UP",
    "THUMBSDOWN": "THUMBS_DOWN",
    "PINCH": "PINCH",
}
POWERPOINT_ACTIONS = {
    "THUMBS_UP": "next_slide",
    "THUMBS_DOWN": "previous_slide",
    "OPEN_PALM": "space",
    "PINCH": "left_click",
}
ACTION_LABELS = {
    "next_slide": "NEXT SLIDE",
    "previous_slide": "PREVIOUS SLIDE",
    "space": "SPACE",
    "left_click": "LEFT CLICK",
}


def normalize_gesture(label: Any) -> str | None:
    token = re.sub(r"[^A-Z0-9]", "", str(label).upper())
    return GESTURE_ALIASES.get(token)


def stable_gesture(history: deque[tuple[str | None, float]]) -> str | None:
    eligible = [
        label
        for label, confidence in history
        if label is not None and confidence >= CONFIDENCE_THRESHOLD
    ]
    if not eligible:
        return None
    label, votes = Counter(eligible).most_common(1)[0]
    return label if votes >= STABILIZATION_VOTES else None


def draw_dashboard(
    frame: Any,
    gesture: str | None,
    confidence: float,
    control_enabled: bool,
    last_action: str,
    status: str,
) -> Any:
    height, width = frame.shape[:2]
    panel_height = min(205, height)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, panel_height), (12, 20, 31), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    lines = [
        ("GestureAI", (255, 255, 255), 0.82),
        ("Move naturally, control digitally.", (190, 215, 235), 0.54),
        (f"Gesture: {gesture.replace('_', ' ') if gesture else 'WAITING'}", (255, 255, 255), 0.62),
        (f"Confidence: {confidence:.0%}", (255, 255, 255), 0.55),
        (f"Control: {'ENABLED' if control_enabled else 'DISABLED'}", (70, 230, 120) if control_enabled else (50, 80, 255), 0.55),
        (f"Last Action: {last_action}", (255, 255, 255), 0.55),
        (f"Status: {status}", (190, 215, 235), 0.48),
    ]
    y = 30
    for index, (text, color, scale) in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            2 if index == 0 else 1,
            cv2.LINE_AA,
        )
        y += 25
    return frame


def keyboard_edges(user32: Any, previous: dict[str, bool]) -> set[str]:
    keys: set[str] = set()
    cv_key = cv2.waitKey(1) & 0xFF
    if cv_key in (ord("q"), ord("Q")):
        keys.add("q")
    if cv_key in (ord("r"), ord("R")):
        keys.add("r")

    if user32 is not None:
        for key, virtual_key in (("q", 0x51), ("r", 0x52)):
            is_down = bool(user32.GetAsyncKeyState(virtual_key) & 0x8000)
            if is_down and not previous[key]:
                keys.add(key)
            previous[key] = is_down
    return keys


def perform_powerpoint_action(
    action_manager: ActionManager,
    gesture: str,
) -> tuple[bool, str]:
    action = POWERPOINT_ACTIONS[gesture]
    action_label = ACTION_LABELS[action]

    if action in {"next_slide", "previous_slide"}:
        result = action_manager.execute(action, target_app="PowerPoint")
        before = result.get("slide_before")
        after = result.get("slide_after")
        position = f"{before} -> {after}" if before is not None and after is not None else "unavailable"
        print(f"Gesture detected: {gesture}")
        print(f"Action: {action_label}")
        print(f"PowerPoint position: {position}")
        message = result["message"]
        if message in {
            "PowerPoint is not running.",
            "PowerPoint is running, but Slide Show is not active.",
        }:
            message = "PowerPoint Slide Show not found. Open PowerPoint and start Slide Show."
        print(f"Result: {'SUCCESS' if result['status'] == 'success' else message}")
        if result["status"] == "success":
            return True, action_label
        return False, message

    focus = action_manager.focus_powerpoint()
    if not focus["found"] or not focus["focused"]:
        message = "PowerPoint Slide Show not found. Open PowerPoint and start Slide Show."
        print(f"Gesture detected: {gesture}")
        print(f"Action: {action_label}")
        print("Result: FAILURE")
        print(message)
        return False, message

    try:
        if action == "space":
            pyautogui.press("space")
        else:
            pyautogui.click(button="left")
    except Exception as exc:
        print(f"Gesture detected: {gesture}")
        print(f"Action: {action_label}")
        print(f"Result: FAILURE ({exc})")
        return False, f"PowerPoint action failed: {exc}"

    print(f"Gesture detected: {gesture}")
    print(f"Action: {action_label}")
    print("Result: SUCCESS")
    return True, action_label


def main() -> int:
    predictor = GesturePredictor()
    try:
        predictor.load()
    except Exception as exc:
        print(f"Gesture model could not be loaded: {exc}")
        return 1

    try:
        detector = HandDetector()
    except Exception as exc:
        print(f"Hand detector could not be initialized: {exc}")
        return 1

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera.release()
        detector.release()
        print("Camera could not be opened.")
        return 1

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    action_manager = ActionManager()
    history: deque[tuple[str | None, float]] = deque(maxlen=STABILIZATION_WINDOW)
    control_enabled = True
    current_stable: str | None = None
    pinch_active = False
    last_action_at = 0.0
    last_action = "NONE"
    status = "READY"
    previous_keys = {"q": False, "r": False}
    try:
        user32 = ctypes.windll.user32
    except AttributeError:
        user32 = None

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 960, 540)
    print("GestureAI demo ready. Q quits; R re-enables control after Fist.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None or frame.size == 0:
                status = "Camera frame unavailable"
                print("Camera could not provide a frame.")
                break

            hand_landmarks, annotated, detection_status = detector.detect(frame)
            gesture = None
            confidence = 0.0
            if hand_landmarks is not None:
                try:
                    features = extract_feature_vector(hand_landmarks, frame.shape)
                    prediction = predictor.predict(features)
                    gesture = normalize_gesture(prediction["label"])
                    confidence = float(prediction["confidence"])
                except Exception as exc:
                    status = f"Prediction unavailable: {exc}"
                    gesture = None
                    confidence = 0.0
            else:
                status = detection_status.upper()

            if hand_landmarks is None:
                history.clear()
                current_stable = None
                pinch_active = False
                stable = None
            else:
                history.append((gesture, confidence))
                stable = stable_gesture(history)

            if stable is not None and stable != current_stable:
                current_stable = stable
                if stable != "PINCH":
                    pinch_active = False
                if stable == "FIST":
                    if control_enabled:
                        control_enabled = False
                        status = "CONTROL DISABLED"
                        print("Gesture detected: FIST")
                        print("Control: DISABLED")
                elif control_enabled and time.monotonic() - last_action_at >= ACTION_COOLDOWN_SECONDS:
                    if stable == "PINCH":
                        if not pinch_active:
                            pinch_active = True
                            success, action_status = perform_powerpoint_action(action_manager, stable)
                            last_action = ACTION_LABELS[POWERPOINT_ACTIONS[stable]]
                            status = "READY" if success else action_status
                            last_action_at = time.monotonic()
                    elif stable in POWERPOINT_ACTIONS:
                        success, action_status = perform_powerpoint_action(action_manager, stable)
                        last_action = ACTION_LABELS[POWERPOINT_ACTIONS[stable]]
                        status = "READY" if success else action_status
                        last_action_at = time.monotonic()

            for key in keyboard_edges(user32, previous_keys):
                if key == "q":
                    return 0
                if key == "r" and not control_enabled:
                    control_enabled = True
                    status = "CONTROL ENABLED"
                    print("Control: ENABLED")

            confidence_display = confidence
            if stable is not None:
                confidence_display = max(
                    (value for label, value in history if label == stable),
                    default=confidence,
                )
            display_frame = draw_dashboard(
                annotated,
                stable,
                confidence_display,
                control_enabled,
                last_action,
                status,
            )
            cv2.imshow(WINDOW_NAME, display_frame)
    except KeyboardInterrupt:
        return 0
    finally:
        camera.release()
        detector.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
