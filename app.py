from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path
from time import monotonic
from typing import Any, List

import cv2
import numpy as np
import streamlit as st

from src.control.action_manager import ActionManager
from src.control.powerpoint_launcher import launch_powerpoint
from src.ml.dataset import GestureDataset
from src.ml.evaluator import GestureEvaluator
from src.ml.predictor import GesturePredictor
from src.ml.trainer import GestureTrainer
from src.utils.config import DATASET_PATH, DEFAULT_GESTURES, EVALUATION_PATH, MODEL_PATH, MODEL_VERSION
from src.vision.feature_extractor import extract_feature_vector
from src.vision.hand_detector import HandDetector

st.set_page_config(page_title="GestureAI", page_icon="🤖", layout="wide")


@st.cache_resource
def get_hand_detector() -> HandDetector:
    return HandDetector()


@st.cache_resource
def get_predictor() -> GesturePredictor:
    return GesturePredictor()


@st.cache_resource
def get_trainer() -> GestureTrainer:
    return GestureTrainer()


@st.cache_resource
def get_action_manager() -> ActionManager:
    return ActionManager()


def load_or_create_dataset() -> GestureDataset:
    return GestureDataset(DATASET_PATH)


def parse_dataset_for_ui() -> List[str]:
    dataset = load_or_create_dataset().load()
    if dataset.empty:
        return []
    return sorted(dataset["gesture"].dropna().unique().tolist())


def ensure_session_state() -> None:
    st.session_state.setdefault("intro_done", False)
    st.session_state.setdefault("splash_done", False)
    st.session_state.setdefault("startup_seen", False)
    st.session_state.setdefault("splash_started_at", monotonic())
    st.session_state.setdefault("camera_running", False)
    st.session_state.setdefault("camera", None)
    st.session_state.setdefault("computer_control", False)
    st.session_state.setdefault("runtime_control_blocked", False)
    st.session_state.setdefault("app_mode", "PowerPoint")
    st.session_state.setdefault("gesture_history", [])
    st.session_state.setdefault("last_stable_gesture", None)
    st.session_state.setdefault("action_status", "Waiting for gesture")
    st.session_state.setdefault("last_action_time", 0.0)
    st.session_state.setdefault("last_action_gesture", None)
    st.session_state.setdefault("last_action_until", 0.0)
    st.session_state.setdefault("pinch_active", False)
    st.session_state.setdefault("last_processed_gesture", None)
    st.session_state.setdefault("powerpoint_launched", False)
    st.session_state.setdefault("gesture_event_debug", "")


def camera_status() -> str:
    return "Connected" if st.session_state.get("camera_running", False) else "Not Connected"


def model_status() -> str:
    return "Trained" if MODEL_PATH.exists() else "Not Trained"


def render_status(label: str, value: str, color: str) -> None:
    st.markdown(
        f"<div class='status-pill'><span style='color:{color};'>●</span> {label}: {value}</div>",
        unsafe_allow_html=True,
    )


def compute_hand_center_xy(hand_landmarks: Any) -> tuple[float, float] | None:
    if hand_landmarks is None:
        return None
    anchor_indices = [0, 5, 9, 13, 17]
    xs = []
    ys = []
    for index in anchor_indices:
        if index < len(hand_landmarks.landmark):
            xs.append(hand_landmarks.landmark[index].x)
            ys.append(hand_landmarks.landmark[index].y)
    if not xs or not ys:
        return None
    return float(np.mean(xs)), float(np.mean(ys))


def stop_camera() -> None:
    camera = st.session_state.pop("camera", None)
    if camera is not None:
        camera.release()
    st.session_state["camera_running"] = False


def enter_gestureai() -> None:
    st.session_state["intro_done"] = True
    st.session_state["splash_done"] = True
    st.session_state["startup_seen"] = True


def render_intro_video() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"], [data-testid="stApp"] {
            background: #050b16;
        }
        header, footer, #MainMenu {
            visibility: hidden;
        }
        .block-container {
            padding-top: 5vh;
        }
        div[data-testid="stVideo"] {
            max-width: 1280px;
            margin: 0 auto;
        }
        .intro-fallback {
            min-height: min(62vw, 720px);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            color: #eaf6ff;
            text-align: center;
            background: radial-gradient(ellipse at 50% 54%, #102136 0%, #050b16 68%);
        }
        .intro-fallback h1 {
            font-size: clamp(2.8rem, 7vw, 6rem);
            font-weight: 300;
            margin: 0;
        }
        .intro-fallback p {
            color: #a9c5da;
            font-size: 1.15rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    video_path = Path(__file__).resolve().parent / "assets" / "gestureai_intro.mp4"
    video_column = st.columns([0.15, 5, 0.15])[1]
    with video_column:
        if video_path.is_file():
            st.video(str(video_path), autoplay=True, muted=True)
        else:
            st.markdown(
                '<div class="intro-fallback"><h1>GestureAI</h1>'
                '<p>Move naturally, control digitally.</p></div>',
                unsafe_allow_html=True,
            )
    button_column = st.columns([1, 1, 1])[1]
    with button_column:
        st.button("Enter GestureAI", key="enter_gestureai_button", on_click=enter_gestureai)


def normalize_gesture(prediction: Any) -> str | None:
    token = re.sub(r"[\s_-]+", "", str(prediction).strip().upper())
    labels = {
        "OPENPALM": "OPEN_PALM",
        "FIST": "FIST",
        "THUMBSUP": "THUMBS_UP",
        "THUMBSDOWN": "THUMBS_DOWN",
        "PINCH": "PINCH",
    }
    return labels.get(token)


def get_mode_actions(mode: str) -> dict[str, str]:
    mode_map = {
        "PowerPoint": {
            "OPEN_PALM": "space",
            "THUMBS_UP": "next_slide",
            "THUMBS_DOWN": "previous_slide",
            "PINCH": "left_click",
            "FIST": "disable_control",
        },
        "Chrome": {
            "OPEN_PALM": "space",
            "THUMBS_UP": "right_arrow",
            "THUMBS_DOWN": "left_arrow",
            "PINCH": "left_click",
            "FIST": "disable_control",
        },
        "PDF": {
            "OPEN_PALM": "space",
            "THUMBS_UP": "page_down",
            "THUMBS_DOWN": "page_up",
            "PINCH": "left_click",
            "FIST": "disable_control",
        },
    }
    return mode_map.get(mode, mode_map["PowerPoint"])


def execute_stable_gesture(gesture_label: str, mode: str) -> None:
    canonical_gesture = normalize_gesture(gesture_label)
    if canonical_gesture is None:
        return

    now = monotonic()
    st.session_state["gesture_event_debug"] = ""
    if st.session_state.get("runtime_control_blocked", False):
        st.session_state["action_status"] = "Control Disabled — Fist detected"
        return

    mode_map = get_mode_actions(mode)
    action_name = mode_map.get(canonical_gesture)
    if not action_name or action_name == "noop":
        st.session_state["action_status"] = "Ready"
        return

    if canonical_gesture == "FIST":
        st.session_state["runtime_control_blocked"] = True
        st.session_state["action_status"] = "Control Disabled — Fist detected"
        st.session_state["last_action_time"] = now
        st.session_state["last_action_gesture"] = "fist"
        st.session_state["last_action_until"] = now + 1.0
        return

    if not st.session_state.get("computer_control", False):
        st.session_state["action_status"] = (
            f"Gesture recognized, action not triggered — Computer Control OFF; "
            f"mapped action: {action_name.upper()}"
        )
        st.session_state["gesture_event_debug"] = ""
        return

    st.session_state["gesture_event_debug"] = ""

    if canonical_gesture == "PINCH":
        if st.session_state.get("pinch_active", False):
            return
        st.session_state["pinch_active"] = True
        if now - st.session_state.get("last_action_time", 0.0) < 0.5:
            return
        result = get_action_manager().execute(action_name)
        st.session_state["action_status"] = result["message"]
        st.session_state["last_action_time"] = now
        st.session_state["last_action_gesture"] = "PINCH"
        st.session_state["last_action_until"] = now + 1.0
        return

    result = get_action_manager().execute(action_name)
    st.session_state["action_status"] = (
        f"✓ {result['message']}" if result["status"] == "success" else
        f"Gesture recognized, action not triggered — {result['message']}"
    )
    st.session_state["last_action_time"] = now
    st.session_state["last_action_gesture"] = canonical_gesture
    st.session_state["last_action_until"] = now + 1.0
    if action_name in {"next_slide", "previous_slide"}:
        event_lines = [
            "GESTURE EVENT",
            f"Gesture: {canonical_gesture}",
            f"Action: {action_name.upper()}",
            f"Controller: {result.get('controller', action_name + '()')}",
            f"PowerPoint: {result.get('powerpoint', 'UNKNOWN')}",
            f"Focus: {result.get('focus', 'UNKNOWN')}",
            f"Key: {result.get('key', 'UNKNOWN')}",
            f"Result: {result.get('result', 'NOT SENT')}",
        ]
        st.session_state["gesture_event_debug"] = "\n".join(event_lines)
        print("\n".join(event_lines))


def get_stable_prediction(history: List[str]) -> str | None:
    if len(history) < 5:
        return None
    counts = Counter(history)
    top_label, top_count = counts.most_common(1)[0]
    if top_count / len(history) >= 0.6:
        return top_label
    return None


def is_new_stable_gesture(current: str | None, previous: str | None) -> bool:
    return current is not None and current != previous


@st.fragment(run_every="150ms")
def render_live_frame(frame_placeholder: Any, gesture_display: Any, confidence_display: Any, context_display: Any, action_status_display: Any, debug_display: Any) -> None:
    if st.session_state.get("dynamic_collection_active"):
        return

    camera = st.session_state.get("camera")
    if camera is None or not st.session_state.get("camera_running"):
        return

    ret, frame = camera.read()
    if not ret:
        st.error("Camera could not provide a frame.")
        stop_camera()
        return

    detector = get_hand_detector()
    hand_landmarks, annotated, status = detector.detect(frame)
    if hand_landmarks is None:
        st.session_state["gesture_history"] = []
        st.session_state["last_stable_gesture"] = None
        st.session_state["last_processed_gesture"] = None
        st.session_state["gesture_event_debug"] = ""
        gesture_display.markdown("<h3>No Hand</h3>", unsafe_allow_html=True)
        confidence_display.markdown("<h3>--</h3>", unsafe_allow_html=True)
        context_display.markdown("<p>Not detected</p>", unsafe_allow_html=True)
        action_status_display.markdown("<p>Waiting for gesture</p>", unsafe_allow_html=True)
        debug_display.code(
            "Raw prediction: --\nNormalized gesture: --\nConfidence: --\n"
            "Stable gesture: --\nMapped action: --"
        )
        st.session_state["pinch_active"] = False
        return

    try:
        features = extract_feature_vector(hand_landmarks, frame.shape)
        prediction = get_predictor().predict(features)
    except Exception:
        gesture_display.markdown("<h3>Prediction unavailable</h3>", unsafe_allow_html=True)
        confidence_display.markdown("<h3>--</h3>", unsafe_allow_html=True)
        context_display.markdown("<p>Model not ready</p>", unsafe_allow_html=True)
        action_status_display.markdown("<p>Waiting for gesture</p>", unsafe_allow_html=True)
        return

    raw_prediction = prediction["label"]
    gesture_name = normalize_gesture(raw_prediction)
    confidence = float(prediction["confidence"])
    history = st.session_state.get("gesture_history", [])
    if gesture_name is None:
        history = []
    else:
        history.append(gesture_name)
    if len(history) > 8:
        history = history[-8:]
    st.session_state["gesture_history"] = history

    current_stable = get_stable_prediction(history)
    st.session_state["last_stable_gesture"] = current_stable

    if current_stable is not None:
        gesture_display.markdown(f"<h3>{current_stable.replace('_', ' ').title()}</h3>", unsafe_allow_html=True)
    else:
        display_gesture = gesture_name.replace("_", " ").title() if gesture_name else "Unrecognized"
        gesture_display.markdown(f"<h3>{display_gesture}</h3>", unsafe_allow_html=True)
    confidence_display.markdown(f"<h3>{confidence * 100:.2f}%</h3>", unsafe_allow_html=True)
    context_display.markdown(f"<p>{'Gesture Stable' if current_stable is not None else 'Waiting for stability'}</p>", unsafe_allow_html=True)

    mode_map = get_mode_actions(st.session_state.get("app_mode", "PowerPoint"))
    trace_gesture = current_stable or gesture_name
    trace_action = mode_map.get(trace_gesture, "--") if trace_gesture else "--"
    debug_display.code(
        f"Raw prediction: {raw_prediction!r}\n"
        f"Normalized gesture: {gesture_name or '--'}\n"
        f"Confidence: {confidence:.4f}\n"
        f"Stable gesture: {current_stable or '--'}\n"
        f"Mapped action: {trace_action.upper()}"
    )

    if is_new_stable_gesture(current_stable, st.session_state.get("last_processed_gesture")):
        execute_stable_gesture(current_stable, st.session_state.get("app_mode", "PowerPoint"))
        st.session_state["last_processed_gesture"] = current_stable
    elif current_stable is None:
        st.session_state["last_processed_gesture"] = None
        st.session_state["pinch_active"] = False

    if current_stable in {"OPEN_PALM", "FIST", "THUMBS_UP", "THUMBS_DOWN", "PINCH"}:
        action_status_display.markdown(f"<p>{st.session_state.get('action_status', 'Waiting for gesture')}</p>", unsafe_allow_html=True)
    else:
        action_status_display.markdown("<p>Waiting for gesture</p>", unsafe_allow_html=True)

    if st.session_state.get("runtime_control_blocked", False):
        st.session_state["pinch_active"] = False

    if st.session_state.get("gesture_event_debug"):
        debug_display.code(
            f"Raw prediction: {raw_prediction!r}\n"
            f"Normalized gesture: {gesture_name or '--'}\n"
            f"Confidence: {confidence:.4f}\n"
            f"Stable gesture: {current_stable or '--'}\n"
            f"Mapped action: {trace_action.upper()}\n\n"
            f"{st.session_state['gesture_event_debug']}"
        )

    cv2.putText(annotated, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if current_stable is not None:
        cv2.putText(annotated, current_stable.replace('_', ' ').upper(), (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 212, 255), 2)
    frame_placeholder.image(annotated, channels="BGR")


def dataset_status(dataset: GestureDataset) -> str:
    errors = dataset.validate()
    if not dataset.dataset_path.exists() or dataset.load().empty:
        return "Empty"
    return "Available" if not errors else "Needs More Data"


def main() -> None:
    ensure_session_state()

    if not st.session_state.get("intro_done", False):
        render_intro_video()
        return

    st.markdown(
        """
        <style>
        :root {
            --bg: #0B1020;
            --card: #111827;
            --surface: #172033;
            --primary: #00D4FF;
            --secondary: #7C3AED;
            --success: #22C55E;
            --warning: #F59E0B;
            --error: #EF4444;
            --text: #F8FAFC;
            --muted: #94A3B8;
        }
        html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
            background-color: var(--bg);
            color: var(--text);
            font-family: 'Inter', Arial, sans-serif;
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 1.2rem;
        }
        .status-pill {
            background: var(--card);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 0.85rem;
            color: var(--text);
            padding: 0.75rem 0.9rem;
            margin-bottom: 0.5rem;
        }
        .stButton > button {
            background: var(--surface);
            color: var(--text);
            border: 1px solid rgba(0,212,255,0.45);
            border-radius: 0.5rem;
            font-weight: 600;
        }
        .card {
            background: rgba(17, 24, 39, 0.9);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 1rem;
            padding: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("GestureAI")
    st.caption("Move naturally, control digitally.")

    status_col1, status_col2, status_col3 = st.columns(3)
    with status_col1:
        render_status("Camera", "Ready" if st.session_state.get("camera_running") else "Not Connected", "#22C55E" if st.session_state.get("camera_running") else "#94A3B8")
    with status_col2:
        render_status("Model", model_status(), "#00D4FF" if MODEL_PATH.exists() else "#94A3B8")
    with status_col3:
        if st.session_state.get("runtime_control_blocked"):
            control_value = "BLOCKED"
            control_color = "#F59E0B"
        else:
            control_value = "ON" if st.session_state.get("computer_control") else "OFF"
            control_color = "#22C55E" if st.session_state.get("computer_control") else "#F59E0B"
        render_status("Control", control_value, control_color)

    st.subheader("Application Mode")
    mode = st.selectbox("Select mode", ["PowerPoint"], index=0, key="app_mode")

    if mode == "PowerPoint":
        st.subheader("POWERPOINT CONTROL")
        powerpoint_status = "Launched" if st.session_state["powerpoint_launched"] else "Not Running"
        st.markdown(f"**PowerPoint Status:** {powerpoint_status}")
        if st.button("Launch PowerPoint", key="launch_powerpoint_button"):
            launch_status, launch_message = launch_powerpoint()
            st.session_state["powerpoint_launched"] = launch_status == "launched"
            st.session_state["powerpoint_launch_message"] = launch_message
            st.rerun()
        if st.session_state.get("powerpoint_launch_message"):
            if st.session_state["powerpoint_launched"]:
                st.success(st.session_state["powerpoint_launch_message"])
            else:
                st.warning(st.session_state["powerpoint_launch_message"])
        test_next_col, test_previous_col = st.columns(2)
        with test_next_col:
            if st.button("TEST NEXT SLIDE", key="test_next_slide_button"):
                execute_stable_gesture("THUMBS_UP", mode)
        with test_previous_col:
            if st.button("TEST PREVIOUS SLIDE", key="test_previous_slide_button"):
                execute_stable_gesture("THUMBS_DOWN", mode)
        if st.session_state.get("action_status", "").startswith("Gesture recognized, action not triggered"):
            st.warning(st.session_state["action_status"])
        elif st.session_state.get("gesture_event_debug"):
            st.code(st.session_state["gesture_event_debug"])
        st.markdown(
            "**Instructions**\n\n"
            "1. Open your presentation\n"
            "2. Start Slide Show\n"
            "3. Start Camera\n"
            "4. Enable Computer Control"
        )

    control_enabled = st.toggle("Computer Control", key="computer_control")

    if not control_enabled:
        st.session_state["runtime_control_blocked"] = False
        st.info("Control is OFF. Gesture recognition is still active, but no computer actions will execute.")
    else:
        st.session_state["runtime_control_blocked"] = st.session_state.get("runtime_control_blocked", False)

    left_col, right_col = st.columns([1.7, 1])

    with left_col:
        st.subheader("Live Camera")
        frame_placeholder = st.empty()
        if st.session_state.get("camera_running"):
            if st.button("Stop Camera", key="stop_camera_button"):
                stop_camera()
                st.rerun()
        else:
            if st.button("Start Camera", key="start_camera_button"):
                camera = cv2.VideoCapture(0)
                if not camera.isOpened():
                    st.error("Camera could not be accessed. Check permissions or another app using it.")
                else:
                    st.session_state["camera"] = camera
                    st.session_state["camera_running"] = True
                    st.rerun()

    with right_col:
        st.subheader("Recognition")
        st.markdown("### Current Gesture")
        gesture_display = st.empty()
        st.markdown("### Confidence")
        confidence_display = st.empty()
        st.markdown("### Status")
        context_display = st.empty()
        st.markdown("### Action")
        action_status_display = st.empty()
        debug_display = st.empty()

    st.subheader("PowerPoint Mapping")
    mapping_cols = st.columns(5)
    gestures = [
        ("Open Palm", "Space / Play-Pause"),
        ("Fist", "Disable Control"),
        ("Thumbs Up", "Next Slide"),
        ("Thumbs Down", "Previous Slide"),
        ("Pinch", "Mouse Click"),
    ]
    for idx, (label, action) in enumerate(gestures):
        with mapping_cols[idx]:
            st.markdown(
                f"<div class='card'><strong>{label}</strong><br><span style='color:#94A3B8;'>{action}</span></div>",
                unsafe_allow_html=True,
            )

    if st.session_state.get("camera_running"):
        render_live_frame(frame_placeholder, gesture_display, confidence_display, context_display, action_status_display, debug_display)
    else:
        frame_placeholder.info("Camera is not connected. Start the camera to begin recognition.")
        gesture_display.markdown("<h3>Waiting...</h3>", unsafe_allow_html=True)
        confidence_display.markdown("<h3>--</h3>", unsafe_allow_html=True)
        context_display.markdown("<p>Waiting for gesture</p>", unsafe_allow_html=True)
        action_status_display.markdown("<p>Waiting for gesture</p>", unsafe_allow_html=True)
        debug_display.code("Raw prediction: --\nNormalized gesture: --\nConfidence: --\nStable gesture: --\nMapped action: --")

    st.subheader("System Status")
    dataset = load_or_create_dataset()
    system_cols = st.columns(3)
    with system_cols[0]:
        st.metric("Camera", camera_status())
    with system_cols[1]:
        st.metric("Model", model_status())
    with system_cols[2]:
        st.metric("Dataset", dataset_status(dataset))

    if st.button("Disable Control", key="disable_control_button"):
        st.session_state["runtime_control_blocked"] = True
        st.session_state["action_status"] = "Control blocked — disabled"
        st.rerun()

    if st.button("Train Model", key="train_model_button"):
        try:
            trainer = get_trainer()
            dataset_manager = load_or_create_dataset()
            validation_errors = dataset_manager.validate()
            if validation_errors:
                st.warning("Training cannot start yet. Please collect enough samples.")
            else:
                evaluation = trainer.train(dataset_manager.load())
                st.success("Training complete.")
                st.json(evaluation)
        except Exception as exc:
            st.error(f"Training failed: {exc}")


if __name__ == "__main__":
    main()
