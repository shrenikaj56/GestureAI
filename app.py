from __future__ import annotations

import json
import time
from datetime import datetime
from time import monotonic
from typing import Any, List

import cv2
import numpy as np
import streamlit as st

from src.ml.dataset import GestureDataset
from src.ml.dynamic_dataset import DynamicSequenceDataset
from src.ml.dynamic_features import LANDMARK_COUNT, extract_dynamic_features
from src.ml.dynamic_predictor import DynamicGesturePredictor
from src.ml.dynamic_trainer import DynamicGestureTrainer
from src.ml.evaluator import GestureEvaluator
from src.ml.predictor import GesturePredictor
from src.ml.trainer import GestureTrainer
from src.utils.config import (
    DATASET_PATH,
    DEFAULT_GESTURES,
    DYNAMIC_CLASSES,
    DYNAMIC_CONFIDENCE_THRESHOLD,
    DYNAMIC_CONFIRMATION_COUNT,
    DYNAMIC_DATA_DIR,
    DYNAMIC_EVALUATION_PATH,
    DYNAMIC_MODEL_PATH,
    DYNAMIC_SEQUENCE_LENGTH,
    EVALUATION_PATH,
    MODEL_PATH,
    MODEL_VERSION,
    DYNAMIC_COLLECTION_COUNT_OPTIONS,
    DYNAMIC_COLLECTION_DURATION_SECONDS,
    DYNAMIC_COLLECTION_PAUSE_SECONDS,
    DYNAMIC_MIN_VALID_FRAMES,
)
from src.utils.temporal_swipe_detector import TemporalSwipeDetector
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
def get_dynamic_predictor() -> DynamicGesturePredictor:
    return DynamicGesturePredictor()


@st.cache_resource
def get_temporal_swipe_detector() -> TemporalSwipeDetector:
    return TemporalSwipeDetector()


def load_or_create_dataset() -> GestureDataset:
    return GestureDataset(DATASET_PATH)


def parse_dataset_for_ui() -> List[str]:
    dataset = load_or_create_dataset().load()
    if dataset.empty:
        return []
    return sorted(dataset["gesture"].dropna().unique().tolist())


def camera_status() -> str:
    return "Connected" if st.session_state.get("camera_running", False) else "Not Connected"


def model_status() -> str:
    return "Trained" if MODEL_PATH.exists() else "Not Trained"


def dataset_status(dataset: GestureDataset) -> str:
    errors = dataset.validate()
    if not dataset.dataset_path.exists() or dataset.load().empty:
        return "Empty"
    return "Available" if not errors else "Needs More Data"


def render_status(label: str, value: str, color: str) -> None:
    st.markdown(
        f"<div class='status-pill'><span style='color:{color};'>●</span> {label}: {value}</div>",
        unsafe_allow_html=True,
    )


def compute_hand_center_xy(hand_landmarks: Any) -> tuple[float, float] | None:
    if hand_landmarks is None:
        return None
    anchor_indices = [0, 5, 8, 9, 12, 13, 17, 20]
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
    st.session_state.pop("dynamic_sequence_buffer", None)


def reset_dynamic_collection() -> None:
    camera = st.session_state.pop("camera", None)
    if camera is not None:
        camera.release()
    st.session_state["camera_running"] = False
    for key in (
        "dynamic_collection_active",
        "dynamic_collection_state",
        "dynamic_collection_frames",
        "dynamic_collection_recording_started",
        "dynamic_collection_state_started",
        "dynamic_collection_total_frames",
    ):
        st.session_state.pop(key, None)


def start_dynamic_collection(label: str, target_count: int) -> None:
    """Initialize a fixed-duration collection session."""
    camera = st.session_state.get("camera")
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            camera.release()
            raise RuntimeError("Camera could not be accessed for dynamic collection.")
        st.session_state["camera"] = camera
    st.session_state.update(
        {
            "camera_running": True,
            "dynamic_collection_active": True,
            "dynamic_collection_label": label,
            "dynamic_collection_target": int(target_count),
            "dynamic_collection_current": 0,
            "dynamic_collection_state": "countdown",
            "dynamic_collection_state_started": monotonic(),
            "dynamic_collection_recording_started": None,
            "dynamic_collection_frames": [],
            "dynamic_collection_total_frames": 0,
        }
    )


@st.fragment(run_every="100ms")
def render_dynamic_collection(preview_display: Any, progress_display: Any, status_display: Any) -> None:
    """Capture one frame per tick; sequence boundaries use wall-clock time."""
    if not st.session_state.get("dynamic_collection_active"):
        return
    camera = st.session_state.get("camera")
    if camera is None or not camera.isOpened():
        status_display.error("Camera could not be accessed. Collection stopped safely.")
        reset_dynamic_collection()
        return

    now = monotonic()
    state = st.session_state["dynamic_collection_state"]
    label = st.session_state["dynamic_collection_label"]
    target = st.session_state["dynamic_collection_target"]
    current = st.session_state["dynamic_collection_current"]
    elapsed = now - st.session_state["dynamic_collection_state_started"]
    progress_display.progress(current / target)

    if state == "pause":
        status_display.info(f"Preparing Sequence {current + 1}/{target}...")
        if elapsed >= DYNAMIC_COLLECTION_PAUSE_SECONDS:
            st.session_state["dynamic_collection_state"] = "countdown"
            st.session_state["dynamic_collection_state_started"] = now
        return

    if state == "countdown":
        countdown = max(1, 3 - int(elapsed))
        status_display.info(f"Get Ready... {countdown}")
        if elapsed >= 3.0:
            st.session_state["dynamic_collection_state"] = "recording"
            st.session_state["dynamic_collection_state_started"] = now
            st.session_state["dynamic_collection_recording_started"] = now
            st.session_state["dynamic_collection_frames"] = []
            st.session_state["dynamic_collection_total_frames"] = 0
        return

    ret, frame = camera.read()
    if not ret:
        status_display.error("Camera stopped providing frames. Collection stopped safely.")
        reset_dynamic_collection()
        return
    landmarks, annotated, _ = get_hand_detector().detect(frame)
    preview_display.image(annotated, channels="BGR")
    recording_elapsed = now - st.session_state["dynamic_collection_recording_started"]
    frames = st.session_state["dynamic_collection_frames"]
    total_frames = st.session_state.get("dynamic_collection_total_frames", 0) + 1
    st.session_state["dynamic_collection_total_frames"] = total_frames
    if landmarks is not None:
        frames.append(np.asarray([[point.x, point.y, point.z] for point in landmarks.landmark], dtype=np.float32))

    status_display.info(
        f"RECORDING - Perform {label.replace('_', ' ').title()} | "
        f"Time: {min(recording_elapsed, DYNAMIC_COLLECTION_DURATION_SECONDS):.1f} / "
        f"{DYNAMIC_COLLECTION_DURATION_SECONDS:.1f}s | Valid frames: {len(frames)} | Total frames: {total_frames} | "
        f"Hand: {'Detected' if landmarks is not None else 'Not detected'}"
    )
    if recording_elapsed < DYNAMIC_COLLECTION_DURATION_SECONDS:
        return

    if len(frames) < DYNAMIC_MIN_VALID_FRAMES:
        status_display.warning(
            f"Insufficient hand frames — captured {len(frames)} valid frames out of {total_frames}. Retrying."
        )
        st.session_state["dynamic_collection_state"] = "pause"
        st.session_state["dynamic_collection_state_started"] = now
        st.session_state["dynamic_collection_frames"] = []
        st.session_state["dynamic_collection_total_frames"] = 0
        return

    DynamicSequenceDataset().add_sequence(label, np.asarray(frames, dtype=np.float32))
    current += 1
    st.session_state["dynamic_collection_current"] = current
    st.session_state["dynamic_collection_frames"] = []
    st.session_state["dynamic_collection_total_frames"] = 0
    status_display.success(f"Sequence {current}/{target} captured - {len(frames)} valid frames saved")
    if current >= target:
        status_display.success(f"{label.replace('_', ' ').title()} collection complete.")
        reset_dynamic_collection()
    else:
        st.session_state["dynamic_collection_state"] = "pause"
        st.session_state["dynamic_collection_state_started"] = now


@st.fragment(run_every="200ms")
def render_live_frame(
    frame_placeholder: Any,
    gesture_display: Any,
    confidence_display: Any,
    context_display: Any,
    intent_display: Any,
    action_display: Any,
    dynamic_display: Any,
    dynamic_confidence_display: Any,
    debug_display: Any,
) -> None:
    """Read and render one frame while the user has explicitly enabled the camera."""
    if st.session_state.get("dynamic_collection_active"):
        return
    camera = st.session_state.get("camera")
    if camera is None or not st.session_state.get("camera_running"):
        return

    ret, frame = camera.read()
    if not ret:
        st.error("Camera could not provide a frame. Check the connection and permissions.")
        stop_camera()
        return

    detector = get_hand_detector()
    hand_landmarks, annotated, status = detector.detect(frame)
    temporal_detector = get_temporal_swipe_detector()
    current_time = monotonic()
    hand_center = compute_hand_center_xy(hand_landmarks)
    if hand_center is not None:
        hand_center_x, hand_center_y = hand_center
    else:
        hand_center_x, hand_center_y = None, None
    temporal_result = temporal_detector.update(
        hand_center_x,
        current_time,
        hand_present=hand_landmarks is not None,
        y_value=hand_center_y,
    )
    dynamic_label = temporal_result.get("gesture")
    current_dynamic_x = temporal_result.get("current_x")
    current_dynamic_start_x = temporal_result.get("start_x")
    current_dynamic_displacement = temporal_result.get("displacement")
    current_dynamic_direction = temporal_result.get("direction")
    current_dynamic_state = temporal_result.get("state")
    current_dynamic_elapsed = temporal_result.get("elapsed")
    current_dynamic_cooldown = temporal_result.get("cooldown")
    detection_reason = temporal_result.get("reason", "Waiting for more horizontal movement")

    display_until = st.session_state.get("last_dynamic_until", 0.0)
    if dynamic_label is not None:
        st.session_state["last_dynamic_gesture"] = dynamic_label
        st.session_state["last_dynamic_until"] = current_time + 1.2
        st.session_state["last_dynamic_confidence"] = 1.0
    if current_time < display_until and st.session_state.get("last_dynamic_gesture") is not None:
        displayed_dynamic_label = st.session_state.get("last_dynamic_gesture")
        displayed_dynamic_confidence = st.session_state.get("last_dynamic_confidence", 1.0)
    else:
        displayed_dynamic_label = None
        displayed_dynamic_confidence = None
        st.session_state.pop("last_dynamic_gesture", None)
        st.session_state.pop("last_dynamic_confidence", None)

    if displayed_dynamic_label is not None:
        dynamic_display.markdown(f"<h3>{displayed_dynamic_label.replace('_', ' ').upper()}</h3>", unsafe_allow_html=True)
        dynamic_confidence_display.markdown(f"<p>Confidence: {displayed_dynamic_confidence * 100:.0f}%</p>", unsafe_allow_html=True)
    else:
        dynamic_display.markdown("<h3>—</h3>", unsafe_allow_html=True)
        dynamic_confidence_display.markdown("<p>Confidence: --</p>", unsafe_allow_html=True)

    debug_display.write(
        {
            "hand": "Detected" if hand_landmarks is not None else "Not detected",
            "trajectory_samples": len(temporal_detector.trajectory),
            "start_x": round(float(current_dynamic_start_x), 3) if current_dynamic_start_x is not None else None,
            "current_x": round(float(current_dynamic_x), 3) if current_dynamic_x is not None else None,
            "displacement_x": round(float(current_dynamic_displacement), 3) if current_dynamic_displacement is not None else 0.0,
            "direction": current_dynamic_direction or "idle",
            "state": current_dynamic_state or "IDLE",
            "elapsed_time": round(float(current_dynamic_elapsed), 2) if current_dynamic_elapsed is not None else 0.0,
            "dynamic_gesture": displayed_dynamic_label.replace('_', ' ').upper() if displayed_dynamic_label else "None",
            "cooldown": f"{round(float(current_dynamic_cooldown), 2)} s" if current_dynamic_cooldown else "Ready",
            "reason": detection_reason,
        }
    )

    if hand_landmarks is not None and MODEL_PATH.exists():
        try:
            features = extract_feature_vector(hand_landmarks, frame.shape)
            prediction = get_predictor().predict(features)
            gesture_name = str(prediction["label"])
            confidence = float(prediction["confidence"])
            gesture_display.markdown(f"<h3>{gesture_name.replace('_', ' ').upper()}</h3>", unsafe_allow_html=True)
            confidence_display.markdown(f"<h3>{confidence * 100:.2f}%</h3>", unsafe_allow_html=True)
            context_display.markdown("<p>Unknown</p>", unsafe_allow_html=True)
            if displayed_dynamic_label == "swipe_right":
                intent_display.markdown("<p>Swipe Right</p>", unsafe_allow_html=True)
                action_display.markdown("<p>Detection only</p>", unsafe_allow_html=True)
            elif displayed_dynamic_label == "swipe_left":
                intent_display.markdown("<p>Swipe Left</p>", unsafe_allow_html=True)
                action_display.markdown("<p>Detection only</p>", unsafe_allow_html=True)
            else:
                intent_display.markdown("<p>Waiting for gesture</p>", unsafe_allow_html=True)
                action_display.markdown("<p>Detection only</p>", unsafe_allow_html=True)
        except Exception as exc:
            gesture_display.markdown("<h3>Prediction unavailable</h3>", unsafe_allow_html=True)
            confidence_display.markdown("<h3>--</h3>", unsafe_allow_html=True)
            context_display.markdown("<p>Unknown</p>", unsafe_allow_html=True)
            intent_display.markdown("<p>Waiting for gesture</p>", unsafe_allow_html=True)
            action_display.markdown("<p>Detection only</p>", unsafe_allow_html=True)
            st.warning(f"Prediction failed: {exc}")
    elif hand_landmarks is not None:
        gesture_display.markdown("<h3>MODEL NOT TRAINED</h3>", unsafe_allow_html=True)
        confidence_display.markdown("<h3>--</h3>", unsafe_allow_html=True)
        context_display.markdown("<p>Not detected</p>", unsafe_allow_html=True)
        intent_display.markdown("<p>Train the model first</p>", unsafe_allow_html=True)
        action_display.markdown("<p>Detection only</p>", unsafe_allow_html=True)
    else:
        gesture_display.markdown("<h3>No Hand</h3>", unsafe_allow_html=True)
        confidence_display.markdown("<h3>--</h3>", unsafe_allow_html=True)
        context_display.markdown("<p>Not detected</p>", unsafe_allow_html=True)
        intent_display.markdown("<p>Waiting for gesture</p>", unsafe_allow_html=True)
        action_display.markdown("<p>Detection only</p>", unsafe_allow_html=True)

    cv2.putText(annotated, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if dynamic_label is not None:
        cv2.putText(annotated, dynamic_label.replace('_', ' ').upper(), (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 212, 255), 2)
    frame_placeholder.image(annotated, channels="BGR")


def safe_metric(value: float) -> float:
    return float(value) if value is not None else 0.0


def main() -> None:
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
            padding-top: 1.5rem;
            padding-bottom: 1.5rem;
        }
        div[data-testid="stMetric"] {
            background: var(--card);
            border: 1px solid rgba(148, 163, 184, 0.15);
            border-radius: 0.8rem;
            padding: 1rem;
        }
        .stTabs [role="tablist"] button {
            background: var(--card);
            color: var(--text);
        }
        .stTabs [role="tablist"] [aria-selected="true"] {
            border-bottom: 2px solid var(--primary);
        }
        .stButton > button {
            background: var(--surface);
            color: var(--text);
            border: 1px solid var(--primary);
            border-radius: 0.5rem;
            font-weight: 600;
        }
        .stProgress > div > div {
            background: linear-gradient(90deg, var(--primary), var(--secondary));
        }
        .status-pill {
            background: var(--card);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 0.5rem;
            color: var(--text);
            padding: 0.65rem 0.8rem;
            margin-bottom: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("GestureAI")
    st.caption("AI-Powered Personalized Gesture Control")

    dataset = load_or_create_dataset()
    status_col1, status_col2, status_col3 = st.columns(3)
    with status_col1:
        render_status("Camera", camera_status(), "#22C55E" if st.session_state.get("camera_running") else "#94A3B8")
    with status_col2:
        render_status("Model", model_status(), "#00D4FF" if MODEL_PATH.exists() else "#94A3B8")
    with status_col3:
        render_status("Recognition", "Ready" if MODEL_PATH.exists() else "Waiting", "#22C55E" if MODEL_PATH.exists() else "#F59E0B")

    tabs = st.tabs(["Live Control", "Teach Gesture", "Experimental / Dataset Lab", "Model Performance", "Gesture History", "Settings"])

    with tabs[0]:
        left, right = st.columns([1.6, 1])
        with left:
            st.subheader("Live Camera Feed")
            frame_placeholder = st.empty()
            if st.session_state.get("camera_running"):
                if st.button("Stop Camera", key="stop_live_camera"):
                    stop_camera()
                    st.rerun()
            else:
                if st.button("Start Camera", key="start_live_camera"):
                    camera = cv2.VideoCapture(0)
                    if not camera.isOpened():
                        camera.release()
                        st.error("Camera could not be accessed. Check camera permissions or whether another application is using it.")
                    else:
                        st.session_state["camera"] = camera
                        st.session_state["camera_running"] = True
                        st.rerun()

        with right:
            st.subheader("Prediction Panel")
            st.markdown("### Detected Gesture")
            gesture_display = st.empty()
            st.markdown("### Confidence")
            confidence_display = st.empty()
            st.markdown("### Context")
            context_display = st.empty()
            st.markdown("### Intent")
            intent_display = st.empty()
            st.markdown("### Action")
            action_display = st.empty()
            st.markdown("### Dynamic Gesture")
            dynamic_display = st.empty()
            dynamic_confidence_display = st.empty()
            with st.expander("Temporal debug", expanded=False):
                debug_display = st.empty()

        metric_cols = st.columns(4)
        evaluation = GestureEvaluator().summary()
        accuracy = evaluation.get("accuracy") if evaluation.get("status") == "ready" else None
        metric_cols[0].metric("Model Accuracy", f"{accuracy * 100:.1f}%" if accuracy is not None else "--")
        metric_cols[1].metric("Gestures Available", str(len(parse_dataset_for_ui())))
        metric_cols[2].metric("Custom Gestures", str(max(0, len(parse_dataset_for_ui()) - len(DEFAULT_GESTURES))))
        metric_cols[3].metric("Current FPS", "--")
        if st.session_state.get("camera_running"):
            render_live_frame(
                frame_placeholder,
                gesture_display,
                confidence_display,
                context_display,
                intent_display,
                action_display,
                dynamic_display,
                dynamic_confidence_display,
                debug_display,
            )
        else:
            frame_placeholder.info("Camera is not connected. Select Start Camera to begin.")
            gesture_display.markdown("<h3>Waiting...</h3>", unsafe_allow_html=True)
            confidence_display.markdown("<h3>--</h3>", unsafe_allow_html=True)
            context_display.markdown("<p>Not detected</p>", unsafe_allow_html=True)
            intent_display.markdown("<p>Waiting</p>", unsafe_allow_html=True)
            action_display.markdown("<p>Waiting</p>", unsafe_allow_html=True)
            dynamic_display.markdown("<h3>NONE</h3>", unsafe_allow_html=True)
            dynamic_confidence_display.markdown("<p>Confidence: --</p>", unsafe_allow_html=True)
            debug_display.write(
                {
                    "hand": "Not detected",
                    "temporal_buffer_frames": 0,
                    "dynamic_gesture": "None",
                    "movement": "-",
                    "horizontal_displacement": 0.0,
                    "vertical_displacement": 0.0,
                    "cooldown": "Ready",
                }
            )

    with tabs[1]:
        st.subheader("Teach Your Own Gesture")
        st.info("How it works: perform the gesture in front of your camera. GestureAI captures hand landmarks and converts them into ML training features automatically. You do not need to create CSV files manually.")
        gesture_name = st.text_input("Gesture name", placeholder="Example: Open Calculator")
        sample_count = st.slider("Number of samples", min_value=10, max_value=100, value=100)
        collect_button = st.button("Start Collection")
        if collect_button:
            if not gesture_name.strip():
                st.error("Please enter a valid gesture name.")
            else:
                st.session_state["gesture_label"] = gesture_name.strip()
                st.session_state["samples_target"] = sample_count
                st.session_state["samples_collected"] = 0
                st.session_state["collector_ready"] = True
                st.success(f"Collection started for '{gesture_name}'.")

        if st.session_state.get("collector_ready"):
            detector = get_hand_detector()
            camera = cv2.VideoCapture(0)
            progress_bar = st.progress(0)
            sample_counter = st.empty()
            try:
                collected = []
                while st.session_state["samples_collected"] < st.session_state["samples_target"]:
                    ret, frame = camera.read()
                    if not ret:
                        st.warning("Unable to read camera while collecting samples.")
                        break
                    hand_landmarks, annotated, _ = detector.detect(frame)
                    if hand_landmarks is not None:
                        features = extract_feature_vector(hand_landmarks, frame.shape)
                        collected.append(features)
                        st.session_state["samples_collected"] = len(collected)
                        sample_counter.markdown(f"Samples collected: {len(collected)} / {st.session_state['samples_target']}")
                        progress_bar.progress(len(collected) / st.session_state["samples_target"])
                        cv2.imshow("Collection Preview", annotated)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            break
                    time.sleep(0.05)

                if collected:
                    dataset = GestureDataset(DATASET_PATH)
                    for feature_vector in collected:
                        dataset.append_sample(st.session_state["gesture_label"], feature_vector)
                    st.success(f"Collected {len(collected)} samples for '{st.session_state['gesture_label']}'.")
                    st.session_state["collector_ready"] = False
                    st.session_state["samples_collected"] = 0
            except Exception as exc:
                st.error(f"Collection failed: {exc}")
            finally:
                camera.release()
                cv2.destroyAllWindows()

        train_button = st.button("Train Model")
        if train_button:
            try:
                dataset_manager = load_or_create_dataset()
                validation_errors = dataset_manager.validate()
                if validation_errors:
                    st.warning("Training cannot start yet.")
                    for error in validation_errors:
                        st.write(f"- {error}")
                else:
                    evaluation = get_trainer().train(dataset_manager.load())
                    st.success("Training complete. Metrics below are measured on the held-out test split.")
                    st.json(evaluation)
            except Exception as exc:
                st.error(f"Training failed: {exc}")

    with tabs[2]:
        st.subheader("Experimental / Dataset Lab")
        st.info("This lab is optional and not required for live control. Real-time swipe recognition uses the deterministic temporal detector immediately after the camera starts.")
        dynamic_label = st.selectbox("Select dynamic gesture", DYNAMIC_CLASSES, format_func=lambda value: value.replace("_", " ").title())
        st.caption(
            f"Use an open palm. For Swipe Left, start on the RIGHT side and move toward the LEFT. "
            f"For Swipe Right, start on the LEFT side and move toward the RIGHT. "
            f"Each sequence records for {DYNAMIC_COLLECTION_DURATION_SECONDS:.1f} seconds automatically."
        )
        sequence_count = st.selectbox("Number of sequences", DYNAMIC_COLLECTION_COUNT_OPTIONS, index=1)
        dynamic_counts = DynamicSequenceDataset().counts()
        st.write({"Swipe Left": dynamic_counts["swipe_left"], "Swipe Right": dynamic_counts["swipe_right"], "Total": sum(dynamic_counts.values())})
        collect_dynamic_button = st.button("Start Dynamic Collection")
        if collect_dynamic_button:
            try:
                start_dynamic_collection(dynamic_label, int(sequence_count))
                st.rerun()
            except Exception as exc:
                st.error(f"Dynamic collection failed: {exc}")

        if st.session_state.get("dynamic_collection_active"):
            collection_preview = st.empty()
            collection_progress = st.progress(
                st.session_state.get("dynamic_collection_current", 0)
                / st.session_state.get("dynamic_collection_target", 1)
            )
            collection_status = st.empty()
            render_dynamic_collection(collection_preview, collection_progress, collection_status)

        train_dynamic_button = st.button("Train Dynamic Model")
        if train_dynamic_button:
            try:
                dynamic_dataset = DynamicSequenceDataset()
                errors = dynamic_dataset.validate()
                if errors:
                    st.warning("Dynamic training cannot start yet.")
                    for error in errors:
                        st.write(f"- {error}")
                else:
                    sequences, labels = dynamic_dataset.load_sequences()
                    evaluation = DynamicGestureTrainer().train(sequences, labels)
                    st.success("Dynamic model training complete. Metrics are from the held-out test split.")
                    st.json(evaluation)
                    get_dynamic_predictor.clear()
            except Exception as exc:
                st.error(f"Dynamic training failed: {exc}")

    with tabs[3]:
        st.subheader("Model Performance")
        evaluator = GestureEvaluator()
        metrics = evaluator.summary()
        if metrics.get("status") == "ready":
            st.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
            st.metric("Precision", f"{metrics['precision'] * 100:.2f}%")
            st.metric("Recall", f"{metrics['recall'] * 100:.2f}%")
            st.metric("F1 Score", f"{metrics['f1_score'] * 100:.2f}%")
            st.write("Confusion Matrix")
            st.dataframe(np.asarray(metrics.get("confusion_matrix", [])))
        else:
            st.warning(metrics.get("message", "No evaluation available yet."))

        if DYNAMIC_EVALUATION_PATH.exists():
            st.subheader("Dynamic Model Performance")
            with open(DYNAMIC_EVALUATION_PATH, "r", encoding="utf-8") as file:
                st.json(json.load(file))
        else:
            st.info("No dynamic model evaluation available yet.")

    with tabs[4]:
        st.subheader("Gesture History")
        history = st.session_state.get("history", [])
        if history:
            st.dataframe(history)
            if st.button("Clear History"):
                st.session_state["history"] = []
                st.success("History cleared.")
        else:
            st.info("No recognition events yet.")

    with tabs[5]:
        st.subheader("System Status")
        current_dataset = load_or_create_dataset()
        status_items = [
            ("Camera Status", camera_status(), "Camera is ready only after Start Camera is selected."),
            ("Model Status", model_status(), "Collect gesture data and train the model to enable recognition."),
            ("Dataset Status", dataset_status(current_dataset), "Samples are collected through the webcam Teach Gesture workflow."),
            ("Available Gestures", str(len(DEFAULT_GESTURES)), "Static gesture classes configured for the initial model."),
            ("Custom Gestures", str(max(0, len(parse_dataset_for_ui()) - len(DEFAULT_GESTURES))), "Additional labels collected beyond the initial static classes."),
            ("Model Version", MODEL_VERSION, "Current static Random Forest model format."),
            ("Last Training Status", "Successful" if EVALUATION_PATH.exists() else "Not trained", "Only saved evaluation results are shown."),
        ]
        for index in range(0, len(status_items), 2):
            columns = st.columns(2)
            for column, (label, value, detail) in zip(columns, status_items[index:index + 2]):
                with column:
                    st.markdown(
                        f"<div class='status-pill'><strong>{label}</strong><br><span style='font-size:1.2rem;'>{value}</span><br><small style='color:#94A3B8;'>{detail}</small></div>",
                        unsafe_allow_html=True,
                    )


if __name__ == "__main__":
    main()
