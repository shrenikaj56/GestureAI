from __future__ import annotations

from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from time import monotonic
from typing import Any

import cv2
import streamlit as st

from src.ml.predictor import GesturePredictor
from src.services.ai_service import AIService, AIServiceError
from src.services.knowledge_service import KnowledgeService
from src.vision.feature_extractor import extract_feature_vector
from src.vision.hand_detector import HandDetector

APP_DIR = Path(__file__).resolve().parent
LOGO_PATH = APP_DIR / "assets" / "gestureai_logo.png"
CONFIDENCE_THRESHOLD = 0.60
STABILIZATION_FRAMES = 3
STABILIZATION_VOTES = 2
ACTION_COOLDOWN_SECONDS = 0.35

GESTURE_NAMES = {
    "OPEN_PALM": "Open Palm",
    "FIST": "Fist",
    "THUMBS_UP": "Thumbs Up",
    "THUMBS_DOWN": "Thumbs Down",
    "PINCH": "Pinch",
}
GESTURE_ACTIONS = {
    "OPEN_PALM": "EXPLAIN",
    "THUMBS_UP": "DEEP DIVE",
    "THUMBS_DOWN": "SIMPLIFY",
    "PINCH": "EXAMPLE",
    "FIST": "RESET",
}
GESTURE_ICONS = {
    "OPEN_PALM": "✋",
    "THUMBS_UP": "👍",
    "THUMBS_DOWN": "👎",
    "PINCH": "🤏",
    "FIST": "✊",
}

st.set_page_config(page_title="GestureAI", page_icon=str(LOGO_PATH), layout="wide")


@st.cache_resource
def get_predictor() -> GesturePredictor:
    predictor = GesturePredictor()
    predictor.load()
    return predictor


@st.cache_resource
def get_hand_detector() -> HandDetector:
    return HandDetector()


@st.cache_resource
def get_ai_service() -> AIService:
    return AIService()

@st.cache_resource
def get_knowledge_service() -> KnowledgeService:
    return KnowledgeService()

@st.cache_resource
def get_ai_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=2, thread_name_prefix="gestureai-ai")


def ensure_session_state() -> None:
    defaults = {
        "camera_enabled": False,
        "camera_running": False,
        "camera_capture": None,
        "camera_initial_frame": None,
        "camera_error": "",
        "gesture_enabled": True,
        "gesture_history": [],
        "last_stable_gesture": None,
        "active_gesture": None,
        "previous_gesture": None,
        "quiz_questions": [],
        "quiz_submitted": False,
        "gesture_confidence": 0.0,
        "gesture_status": "Show your hand to GestureAI",
        "gesture_count": 0,
        "current_topic": "",
        "current_response": "",
        "last_action": "No action yet",
        "last_gesture_event": "",
        "ai_busy": False,
        "ai_future": None,
        "pending_ai_action": "",
        "pending_ai_topic": "",
        "ai_error": "",
        "ai_connection_state": "unknown",
        "ai_interactions": 0,
        "last_action_at": 0.0,
        "topic_widget_version": 0,
        "topic_widget_key": "topic_input_0",
        "model_load_error": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def normalize_gesture(prediction: Any) -> str | None:
    token = "".join(character for character in str(prediction).strip().upper() if character.isalnum())
    aliases = {
        "OPENPALM": "OPEN_PALM",
        "FIST": "FIST",
        "THUMBSUP": "THUMBS_UP",
        "THUMBSDOWN": "THUMBS_DOWN",
        "PINCH": "PINCH",
    }
    return aliases.get(token)


def is_new_stable_gesture(current: str | None, previous: str | None) -> bool:
    return current is not None and current != previous


def get_stable_prediction(history: list[tuple[str | None, float]]) -> str | None:
    eligible = [
        gesture
        for gesture, confidence in history[-STABILIZATION_FRAMES:]
        if gesture is not None and confidence >= CONFIDENCE_THRESHOLD
    ]
    if not eligible:
        return None
    gesture, votes = Counter(eligible).most_common(1)[0]
    return gesture if votes >= STABILIZATION_VOTES else None


def is_new_stable_gesture(current: str | None, previous: str | None) -> bool:
    return current is not None and current != previous


def open_camera_capture() -> tuple[Any | None, str]:
    existing = st.session_state.get("camera_capture")
    if existing is not None:
        try:
            if st.session_state.get("camera_running") and existing.isOpened():
                return existing, "Camera is already running."
        except Exception:
            pass
        stop_camera()

    capture = None
    try:
        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            capture.release()
            return None, "Camera unavailable. Please check camera permissions."
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        ok, frame = capture.read()
        if not ok or frame is None or frame.size == 0:
            capture.release()
            return None, "Camera unavailable. Please check camera permissions."
    except Exception:
        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass
        return None, "Camera unavailable. Please check camera permissions."

    st.session_state["camera_initial_frame"] = frame
    return capture, "Camera ready."


def stop_camera() -> None:
    capture = st.session_state.get("camera_capture")
    if capture is not None:
        try:
            capture.release()
        except Exception:
            pass
    st.session_state["camera_capture"] = None
    st.session_state["camera_initial_frame"] = None
    st.session_state["camera_running"] = False
    st.session_state["gesture_history"] = []
    st.session_state["last_stable_gesture"] = None
    st.session_state["active_gesture"] = None
    st.session_state["gesture_confidence"] = 0.0
    st.session_state["gesture_status"] = "Camera is off"


def on_camera_toggle() -> None:
    if st.session_state.get("camera_enabled", False):
        capture, message = open_camera_capture()
        if capture is None:
            st.session_state["camera_error"] = message
            st.session_state["camera_enabled"] = False
            st.session_state["camera_running"] = False
            return
        st.session_state["camera_capture"] = capture
        st.session_state["camera_running"] = True
        st.session_state["camera_error"] = ""
        st.session_state["gesture_status"] = "Camera active"
    else:
        stop_camera()
        st.session_state["camera_error"] = ""

def generate_knowledge_response(action: str, topic: str) -> str:
    """Generate a topic-specific response for the selected study action."""

    topic = topic.strip()

    if not topic:
        return "Please enter a topic or question first."

    # Use the existing AIService demo knowledge when available.
    # It already contains detailed action-specific responses for
    # important Computer Engineering topics.
    service = get_ai_service()

    topic_lower = topic.casefold()

    rich_topics = (
        "binary search",
        "dijkstra",
        "normalization",
        "normalisation",
        "machine learning",
        "operating system",
        "operating systems",
    )

    if any(keyword in topic_lower for keyword in rich_topics):
        try:
            response = service.generate(
                action=action,
                topic=topic,
                current_response="",
            )

            if response:
                return response

        except Exception:
            pass

    # ---------------------------------------------------------
    # Fallback to the local knowledge base for other topics.
    # ---------------------------------------------------------

    knowledge = get_knowledge_service()
    result = knowledge.search(topic)

    if not result["found"]:
        return (
            "I couldn't find a reliable answer for that topic in my "
            "local knowledge base yet.\n\n"
            "Try asking about Binary Search, Dijkstra, SQL, DBMS, "
            "Normalization, Linked Lists, Stack, Queue, Machine Learning, "
            "Operating Systems, Computer Networks, AI, or Data Science."
        )

    answer = result["answer"]
    matched_topic = result["topic"]

    if action == "EXPLAIN":
        return (
            f"### Explanation\n\n"
            f"{answer}\n\n"
            f"**Topic:** {matched_topic}"
        )

    if action == "SIMPLIFY":
        return (
            f"### Simplified Explanation\n\n"
            f"{answer}\n\n"
            f"**In simple words:**\n"
            f"{matched_topic} can be understood by focusing first "
            f"on its main purpose and then understanding how its "
            f"individual steps work together."
        )

    if action == "EXAMPLE":
        return (
            f"### Example\n\n"
            f"**Concept:** {matched_topic}\n\n"
            f"{answer}\n\n"
            f"**Practical example:**\n"
            f"Consider a small real-world situation where you need "
            f"to apply {matched_topic.lower()}. Break the problem "
            f"into the same steps used by the concept and observe "
            f"how each step changes the result."
        )

    if action == "DEEP DIVE":
        return (
            f"### Deep Dive\n\n"
            f"**Core Concept**\n\n"
            f"{answer}\n\n"
            f"**Important points to study**\n\n"
            f"- Understand the purpose of {matched_topic}.\n"
            f"- Learn its main steps or components.\n"
            f"- Understand its efficiency and limitations.\n"
            f"- Know where it is used in real applications.\n"
            f"- Be able to explain it in your own words during a viva or interview.\n\n"
            f"**Topic:** {matched_topic}"
        )

    return (
        f"### Explanation\n\n"
        f"{answer}\n\n"
        f"**Topic:** {matched_topic}"
    )
def queue_ai_action(action: str, topic: str) -> bool:
    """Generate a local knowledge-base response immediately."""

    topic = topic.strip()

    if not topic:
        st.session_state["ai_error"] = "Enter a topic or question first."
        return False

    if st.session_state.get("ai_busy", False):
        st.session_state["ai_error"] = "AI is thinking... please wait."
        return False

    if topic != st.session_state.get("current_topic"):
        st.session_state["current_response"] = ""
        st.session_state["current_topic"] = topic

    if action == "EXPLAIN":
        st.session_state["current_response"] = ""

    st.session_state["ai_busy"] = True
    st.session_state["pending_ai_action"] = action
    st.session_state["pending_ai_topic"] = topic
    st.session_state["last_action"] = action.title()
    st.session_state["last_gesture_event"] = f"{action} requested"
    st.session_state["ai_error"] = ""

    try:
        response = generate_knowledge_response(action, topic)

        # Store the answer directly in Streamlit session state.
        st.session_state["current_response"] = response
        st.session_state["ai_connection_state"] = "demo"
        st.session_state["ai_interactions"] += 1
        st.session_state["last_gesture_event"] = f"{action} completed"

    except Exception as exc:
        st.session_state["ai_error"] = (
            f"Knowledge service error: {exc}"
        )
        st.session_state["last_gesture_event"] = f"{action} failed"

    finally:
        st.session_state["ai_busy"] = False
        st.session_state["pending_ai_action"] = ""
        st.session_state["pending_ai_topic"] = ""

    st.session_state["last_action_at"] = monotonic()

    return True


def complete_ai_request() -> None:
    future: Future[str] | None = st.session_state.get("ai_future")
    if not st.session_state.get("ai_busy") or future is None or not future.done():
        return

    action = st.session_state.get("pending_ai_action", "")
    try:
        response = future.result()
    except AIServiceError:
        st.session_state["ai_error"] = "AI service is temporarily unavailable."
        st.session_state["last_gesture_event"] = f"{action} failed"
        st.session_state["ai_connection_state"] = "unavailable"
    except Exception:
        st.session_state["ai_error"] = "AI service is temporarily unavailable."
        st.session_state["last_gesture_event"] = f"{action} failed"
        st.session_state["ai_connection_state"] = "unavailable"
    else:
        st.session_state["current_response"] = response
        st.session_state["ai_error"] = ""
        st.session_state["ai_interactions"] += 1
        st.session_state["last_gesture_event"] = f"{action} completed"
        service = get_ai_service()
        st.session_state["ai_connection_state"] = "connected" if service.api_key else "demo"
    finally:
        st.session_state["ai_busy"] = False
        st.session_state["ai_future"] = None
        st.session_state["pending_ai_action"] = ""
        st.session_state["pending_ai_topic"] = ""


def reset_interaction() -> None:
    future: Future[str] | None = st.session_state.get("ai_future")
    if future is not None:
        future.cancel()
    st.session_state["ai_future"] = None
    st.session_state["ai_busy"] = False
    st.session_state["pending_ai_action"] = ""
    st.session_state["pending_ai_topic"] = ""
    st.session_state["current_topic"] = ""
    st.session_state["current_response"] = ""
    st.session_state["ai_error"] = ""
    st.session_state["last_action"] = "Session reset"
    st.session_state["last_gesture_event"] = "Session reset."
    st.session_state["topic_widget_version"] += 1
    st.session_state["topic_widget_key"] = f"topic_input_{st.session_state['topic_widget_version']}"


def queue_gesture_action(gesture: str) -> bool:
    if gesture == "FIST":
        reset_interaction()
        return True

    action = GESTURE_ACTIONS[gesture]
    topic = st.session_state.get("current_topic", "")
    if not topic:
        topic = st.session_state.get(st.session_state.get("topic_widget_key", ""), "")
    if not topic.strip():
        st.session_state["ai_error"] = "Enter a topic or question first."
        st.session_state["last_action"] = action.title()
        st.session_state["last_gesture_event"] = f"{gesture} needs a topic"
        return False

    return queue_ai_action(action, topic)

def _handle_camera_gesture(gesture: str) -> None:
    """Handle stable gestures and gesture combinations."""

    previous_gesture = st.session_state.get("previous_gesture")

    # ============================================================
    # GESTURE COMBINATION:
    # PINCH → THUMBS UP = START QUIZ
    # ============================================================
    if previous_gesture == "PINCH" and gesture == "THUMBS_UP":

        st.session_state["quiz_questions"] = generate_quiz(
            st.session_state.get("current_topic", "")
        )

        if st.session_state["quiz_questions"]:
            st.session_state["quiz_submitted"] = False
            st.session_state["last_action"] = "Quiz Mode"
            st.session_state["last_gesture_event"] = (
                "PINCH + THUMBS UP · Quiz started"
            )
            st.session_state["gesture_status"] = (
                "🧠 Quiz Mode activated!"
            )
        else:
            st.session_state["last_gesture_event"] = (
                "PINCH + THUMBS UP · No quiz available"
            )
            st.session_state["gesture_status"] = (
                "Enter a supported topic first."
            )

        st.session_state["previous_gesture"] = gesture
        st.session_state["last_stable_gesture"] = gesture
        st.session_state["active_gesture"] = gesture
        st.session_state["gesture_count"] += 1

        return

    # ============================================================
    # FIST = RESET
    # ============================================================
    if gesture == "FIST":

        reset_interaction()

        st.session_state["quiz_questions"] = []
        st.session_state["quiz_submitted"] = False
        st.session_state["previous_gesture"] = None

        st.session_state["last_stable_gesture"] = gesture
        st.session_state["active_gesture"] = gesture
        st.session_state["gesture_count"] += 1

        st.session_state["last_gesture_event"] = (
            "FIST · Session reset."
        )
        st.session_state["gesture_status"] = "Session reset."

        return

    # ============================================================
    # IGNORE SAME HELD GESTURE
    # ============================================================
    if gesture == st.session_state.get("last_stable_gesture"):
        return

    # ============================================================
    # NORMAL GESTURE HANDLING
    # ============================================================
    st.session_state["previous_gesture"] = gesture
    st.session_state["last_stable_gesture"] = gesture
    st.session_state["active_gesture"] = gesture
    st.session_state["gesture_count"] += 1

    action = GESTURE_ACTIONS[gesture]

    st.session_state["last_gesture_event"] = (
        f"{gesture} · {action}"
    )

    if st.session_state.get("ai_busy", False):
        st.session_state["gesture_status"] = "AI is thinking..."
        return

    started = queue_gesture_action(gesture)

    if started:
        st.session_state["gesture_status"] = (
            f"{action} activated"
        )
    else:
        st.session_state["gesture_status"] = st.session_state.get(
            "ai_error",
            "Enter a topic or question first."
        )
@st.fragment(run_every="120ms")
def render_live_camera(
    image_placeholder: Any,
    gesture_placeholder: Any,
    confidence_placeholder: Any,
    progress_placeholder: Any,
    status_placeholder: Any,
) -> None:
    if not st.session_state.get("camera_enabled") or not st.session_state.get("camera_running"):
        if st.session_state.get("camera_error"):
            image_placeholder.error("Camera unavailable. Please check camera permissions.")
        else:
            image_placeholder.info("Show your hand to GestureAI when the camera is on.")
        gesture_placeholder.markdown("**Detected Gesture**\n\n—")
        confidence_placeholder.caption("Confidence: —")
        progress_placeholder.markdown(
            '<div class="confidence-track"><div class="confidence-fill" style="width:0%"></div></div>',
            unsafe_allow_html=True,
        )
        status_placeholder.caption("Camera off")
        return

    camera = st.session_state.get("camera_capture")
    if camera is None:
        st.session_state["camera_running"] = False
        image_placeholder.error("Camera unavailable. Please check camera permissions.")
        return

    frame = st.session_state.pop("camera_initial_frame", None)
    if frame is None:
        try:
            ok, frame = camera.read()
        except Exception:
            ok, frame = False, None
        if not ok or frame is None or frame.size == 0:
            stop_camera()
            st.session_state["camera_error"] = "Camera unavailable. Please check camera permissions."
            image_placeholder.error(st.session_state["camera_error"])
            gesture_placeholder.markdown("**Detected Gesture**\n\n—")
            confidence_placeholder.caption("Confidence: —")
            status_placeholder.caption("Camera unavailable")
            return

    try:
        hand_landmarks, annotated, detection_status = get_hand_detector().detect(frame)
    except Exception:
        image_placeholder.image(frame, channels="BGR", width=620)
        st.session_state["gesture_status"] = "Gesture detector unavailable"
        status_placeholder.caption(st.session_state["gesture_status"])
        return

    current_confidence = 0.0
    current_gesture = None
    if hand_landmarks is not None and not st.session_state.get("model_load_error"):
        try:
            prediction = get_predictor().predict(extract_feature_vector(hand_landmarks, frame.shape))
            current_gesture = normalize_gesture(prediction["label"])
            current_confidence = float(prediction["confidence"])
        except Exception:
            st.session_state["model_load_error"] = "Gesture model could not be loaded."

    if not st.session_state.get("gesture_enabled", True):
        st.session_state["gesture_history"] = []
        st.session_state["last_stable_gesture"] = None
        st.session_state["active_gesture"] = None
        stable = None
        st.session_state["gesture_status"] = "Gesture recognition paused"
    elif hand_landmarks is None:
        st.session_state["gesture_history"] = []
        st.session_state["last_stable_gesture"] = None
        st.session_state["active_gesture"] = None
        stable = None
        st.session_state["gesture_status"] = "Show your hand to GestureAI"
    else:
        history: list[tuple[str | None, float]] = st.session_state.get("gesture_history", [])
        history.append((current_gesture, current_confidence))
        st.session_state["gesture_history"] = history[-STABILIZATION_FRAMES:]
        stable = get_stable_prediction(st.session_state["gesture_history"])
        if stable is not None:
            st.session_state["active_gesture"] = stable
            st.session_state["gesture_confidence"] = current_confidence
            st.session_state["gesture_status"] = "Gesture detected"
            if is_new_stable_gesture(stable, st.session_state.get("last_stable_gesture")):
                _handle_camera_gesture(stable)
        elif st.session_state.get("last_stable_gesture") is not None:
            st.session_state["active_gesture"] = st.session_state["last_stable_gesture"]
            st.session_state["gesture_confidence"] = current_confidence
            st.session_state["gesture_status"] = "Gesture held"
        else:
            st.session_state["active_gesture"] = current_gesture
            st.session_state["gesture_confidence"] = current_confidence
            st.session_state["gesture_status"] = "Waiting for stability" if current_gesture else detection_status

    image_placeholder.image(annotated, channels="BGR", width=620)
    active = st.session_state.get("active_gesture")
    gesture_label = GESTURE_NAMES.get(active, active.replace("_", " ") if active else "—")
    confidence = float(st.session_state.get("gesture_confidence", 0.0))
    gesture_placeholder.markdown(f"**Detected Gesture**\n\n### {gesture_label}")
    confidence_placeholder.caption(f"Confidence: {confidence:.0%}" if active else "Confidence: —")
    progress_placeholder.markdown(
        f'<div class="confidence-track"><div class="confidence-fill" style="width:{confidence * 100:.1f}%"></div></div>',
        unsafe_allow_html=True,
    )
    status_placeholder.caption(st.session_state.get("gesture_status", "READY"))


@st.fragment(run_every="250ms")
def render_ai_response(
    topic_placeholder: Any,
    action_placeholder: Any,
    status_placeholder: Any,
    response_placeholder: Any,
) -> None:
    complete_ai_request()
    topic = st.session_state.get("current_topic", "")
    response = st.session_state.get("current_response", "")
    if topic:
        topic_placeholder.markdown(f"**CURRENT TOPIC**\n\n{topic}")
    else:
        topic_placeholder.markdown("**CURRENT TOPIC**\n\nEnter a topic or question to begin.")

    action = st.session_state.get("last_action", "No action yet")
    event = st.session_state.get("last_gesture_event", "")
    action_placeholder.markdown(
        f'<span class="action-badge">LAST GESTURE ACTION · {action.upper()}</span>'
        + (f"<br><span class=\"muted-copy\">{event}</span>" if event else ""),
        unsafe_allow_html=True,
    )
    if st.session_state.get("ai_busy"):
        status_placeholder.info("AI is thinking...")
    elif st.session_state.get("ai_error"):
        status_placeholder.warning(st.session_state["ai_error"])
    else:
        status_placeholder.empty()

    if response:
        with response_placeholder.container(border=True):
            st.markdown("**AI EXPLANATION**")
            st.markdown(response)
    else:
        response_placeholder.info("Your explanation will appear here.")

    render_quiz_mode()  


def _render_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #0B1020;
            --surface: #111827;
            --card: #151D2E;
            --active: #1B263B;
            --purple: #7C5CFC;
            --cyan: #00D4FF;
            --green: #22C55E;
            --amber: #F59E0B;
            --red: #EF4444;
            --text: #F8FAFC;
            --secondary: #94A3B8;
            --muted: #64748B;
            --border: #263247;
        }
        html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
            background: var(--bg);
            color: var(--text);
            font-family: Inter, Arial, sans-serif;
        }
        [data-testid="stSidebar"] { background: #111827; border-right: 1px solid var(--border); }
        .block-container { max-width: 1500px; padding: 1.25rem 1.5rem 1.75rem; }
        h1, h2, h3 { color: var(--text); }
        h1 { font-size: 30px !important; font-weight: 700 !important; }
        h2 { font-size: 20px !important; font-weight: 600 !important; }
        h3 { font-size: 17px !important; font-weight: 600 !important; }
        p, label, li { color: var(--secondary); }
        [data-testid="stMetricValue"] { color: var(--text); }
        [data-testid="stImage"] img { border-radius: 8px; }
        .brand-tagline { color: var(--secondary); font-size: 14px; }
        .mode-badge {
            display: inline-flex; align-items: center; gap: 8px; padding: 8px 12px;
            border: 1px solid var(--border); border-radius: 999px; background: var(--card);
            color: var(--text); font-size: 13px; font-weight: 600;
        }
        .mode-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
        .confidence-track { height: 7px; width: 100%; overflow: hidden; border-radius: 9px; background: var(--border); }
        .confidence-fill { height: 100%; background: var(--purple); border-radius: 9px; }
        .gesture-tile {
            min-height: 92px; padding: 12px 8px; background: var(--card);
            border: 1px solid var(--border); border-radius: 10px; text-align: center;
        }
        .gesture-tile.active { background: var(--active); border: 1px solid var(--purple); }
        .gesture-icon { color: var(--text); font-size: 23px; line-height: 1.2; }
        .gesture-name { color: var(--text); font-size: 13px; font-weight: 600; margin-top: 5px; }
        .gesture-action { color: var(--secondary); font-size: 11px; margin-top: 2px; }
        .action-badge { display: inline-block; padding: 5px 9px; color: var(--text); background: var(--active); border: 1px solid var(--border); border-radius: 999px; font-size: 11px; font-weight: 600; }
        .muted-copy { color: var(--secondary); font-size: 13px; }
        .stButton > button, [data-testid="stFormSubmitButton"] button {
            min-height: 44px; color: white; background: var(--purple); border: 1px solid var(--purple);
            border-radius: 9px; font-weight: 600;
        }
        .stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover { background: var(--active); border-color: var(--purple); color: white; }
        [data-testid="stTextInput"] input { background: var(--surface); color: var(--text); border-color: var(--border); }
        [data-testid="stProgress"] > div > div { background: var(--purple); }
        [data-testid="stAlert"] { background: var(--surface); border-color: var(--border); }
        @media (max-width: 760px) {
            .block-container { padding: 1rem 0.8rem; }
            .gesture-tile { min-height: 82px; padding: 9px 3px; }
            .gesture-name { font-size: 11px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _ai_badge(service: AIService) -> tuple[str, str]:
    if not service.api_key:
        return "DEMO MODE", "#F59E0B"
    state = st.session_state.get("ai_connection_state", "unknown")
    if state == "connected":
        return "AI ACTIVE", "#22C55E"
    if state == "unavailable":
        return "AI UNAVAILABLE", "#F59E0B"
    return "AI API CONFIGURED", "#F59E0B"

def generate_quiz(topic: str) -> list[dict]:
    """Generate a small local quiz for the selected topic."""

    topic = topic.strip()

    if not topic:
        return []

    knowledge = get_knowledge_service()
    result = knowledge.search(topic)

    if not result["found"]:
        return []

    matched_topic = result["topic"]
    answer = result["answer"]

    quiz_bank = {
        "binary search": [
            {
                "question": "What is required before applying Binary Search?",
                "options": [
                    "The data must be sorted",
                    "The data must be random",
                    "The data must be duplicated",
                    "The data must be stored in a stack",
                ],
                "answer": "The data must be sorted",
            },
            {
                "question": "What is the typical time complexity of Binary Search?",
                "options": ["O(n)", "O(log n)", "O(n²)", "O(1)"],
                "answer": "O(log n)",
            },
            {
                "question": "Which element is checked first in Binary Search?",
                "options": [
                    "First element",
                    "Last element",
                    "Middle element",
                    "Random element",
                ],
                "answer": "Middle element",
            },
            {
                "question": "Binary Search repeatedly eliminates approximately what fraction of the search space?",
                "options": [
                    "One quarter",
                    "One half",
                    "All of it",
                    "None",
                ],
                "answer": "One half",
            },
            {
                "question": "Binary Search works directly on which type of data?",
                "options": [
                    "Sorted sequence",
                    "Only linked lists",
                    "Only graphs",
                    "Only trees",
                ],
                "answer": "Sorted sequence",
            },
        ],
    }

    topic_key = matched_topic.casefold()

    if "binary search" in topic_key:
        return quiz_bank["binary search"]

    # Safe fallback for topics without dedicated questions yet.
    return [
        {
            "question": f"Which statement best describes {matched_topic}?",
            "options": [
                answer,
                "It is unrelated to computer science.",
                "It can never be used in practical applications.",
                "It is only used for hardware design.",
            ],
            "answer": answer,
        }
    ]
def render_quiz_mode() -> None:
    questions = st.session_state.get("quiz_questions", [])

    if not questions:
        return

    st.markdown("## 🧠 Quiz Mode")
    st.caption("Test your understanding of the current topic.")

    with st.container(border=True):
        answers = []

        for i, question in enumerate(questions, start=1):
            st.markdown(f"**Q{i}. {question['question']}**")

            selected = st.radio(
                "Choose an answer:",
                question["options"],
                key=f"quiz_answer_{i}",
                index=None,
            )

            answers.append(selected)

        if st.button("Submit Quiz", type="primary"):
            score = 0

            for selected, question in zip(answers, questions):
                if selected == question["answer"]:
                    score += 1

            st.session_state["quiz_submitted"] = True
            st.session_state["quiz_score"] = score

        if st.session_state.get("quiz_submitted", False):
            score = st.session_state.get("quiz_score", 0)
            total = len(questions)

            st.success(
                f"Quiz completed! Score: **{score}/{total}**"
            )

def main() -> None:
    ensure_session_state()
    _render_styles()

    service = get_ai_service()
    predictor = None
    if not st.session_state.get("model_load_error"):
        try:
            predictor = get_predictor()
        except Exception:
            st.session_state["model_load_error"] = "Gesture model could not be loaded."

    with st.sidebar:
        st.markdown("## GestureAI")
        st.caption("Hands-free AI learning")
        st.markdown("**CONTROLS**")
        st.toggle("Camera", key="camera_enabled", on_change=on_camera_toggle)
        st.toggle("Gesture Recognition", key="gesture_enabled")
        badge_text, badge_color = _ai_badge(service)
        st.markdown("**AI Mode**")
        st.markdown(
            f'<span class="mode-badge"><span class="mode-dot" style="background:{badge_color}"></span>{badge_text}</span>',
            unsafe_allow_html=True,
        )
        st.markdown("**ABOUT**")
        st.caption(
            "GestureAI is a hands-free AI learning interface that combines computer vision, "
            "machine learning, and AI-assisted learning."
        )
        st.divider()
        st.markdown("**SYSTEM STATUS**")
        st.caption(f"Camera · {'Ready' if st.session_state.get('camera_running') else 'Off'}")
        model_ready = predictor is not None
        st.caption(f"Model · {'Loaded' if model_ready else 'Unavailable'}")
        st.caption(f"Gesture Engine · {'Active' if st.session_state.get('gesture_enabled') else 'Paused'}")
        if not service.api_key:
            st.caption("AI · Demo Mode")
        elif st.session_state.get("ai_connection_state") == "connected":
            st.caption("AI · Connected")
        else:
            st.caption("AI · Not verified")

    badge_text, badge_color = _ai_badge(service)
    header_left, header_right = st.columns([0.78, 0.22], vertical_alignment="center")
    with header_left:
        logo_col, title_col = st.columns([0.12, 0.88], vertical_alignment="center")
        with logo_col:
            if LOGO_PATH.exists():
                st.image(str(LOGO_PATH), width=54)
        with title_col:
            st.markdown("# GestureAI")
            st.markdown('<div class="brand-tagline">Move naturally. Learn intelligently.</div>', unsafe_allow_html=True)
    with header_right:
        st.markdown(
            f'<div style="text-align:right"><span class="mode-badge"><span class="mode-dot" style="background:{badge_color}"></span>{badge_text}</span></div>',
            unsafe_allow_html=True,
        )
    st.divider()

    left_column, right_column = st.columns([0.9, 1.1], gap="large")
    with left_column:
        st.markdown("## LIVE GESTURE INPUT")
        camera_label = "Camera Active" if st.session_state.get("camera_running") else "Camera Off"
        st.caption(camera_label)
        image_placeholder = st.empty()
        gesture_placeholder = st.empty()
        confidence_placeholder = st.empty()
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        render_live_camera(
            image_placeholder,
            gesture_placeholder,
            confidence_placeholder,
            progress_placeholder,
            status_placeholder,
        )
        if st.session_state.get("model_load_error"):
            st.error("Gesture model could not be loaded.")

    with right_column:
        st.markdown("## AI STUDY ASSISTANT")
        st.caption("Ask anything. Learn naturally.")
        with st.form("ask_ai_form", clear_on_submit=False):
            st.text_input("Ask a question or enter a topic...", key=st.session_state["topic_widget_key"])
            ask = st.form_submit_button("ASK AI", use_container_width=True)
        if ask:
            typed_topic = st.session_state.get(st.session_state["topic_widget_key"], "")
            started = queue_ai_action("EXPLAIN", typed_topic)
            if started:
                st.session_state["gesture_status"] = "AI is thinking..."

        topic_placeholder = st.empty()
        action_placeholder = st.empty()
        ai_status_placeholder = st.empty()
        response_placeholder = st.empty()
        render_ai_response(
            topic_placeholder,
            action_placeholder,
            ai_status_placeholder,
            response_placeholder,
        )
# ============================================================
# AI QUIZ MODE
# ============================================================

st.markdown("## 🧠 AI QUIZ MODE")
st.caption("Test your understanding of the current topic.")

current_topic = st.session_state.get("current_topic", "").strip()

if "quiz_questions" not in st.session_state:
    st.session_state["quiz_questions"] = []

if "quiz_score" not in st.session_state:
    st.session_state["quiz_score"] = 0

if "quiz_submitted" not in st.session_state:
    st.session_state["quiz_submitted"] = False

quiz_col1, quiz_col2 = st.columns([3, 1])

with quiz_col1:
    if current_topic:
        st.info(f"Topic: **{current_topic}**")
    else:
        st.info("Enter a topic above first.")

with quiz_col2:
    generate_button = st.button(
        "🧠 Generate Quiz",
        use_container_width=True,
    )

if generate_button:
    if not current_topic:
        st.warning("Please enter a topic first.")
    else:
        questions = generate_quiz(current_topic)

        if questions:
            st.session_state["quiz_questions"] = questions
            st.session_state["quiz_score"] = 0
            st.session_state["quiz_submitted"] = False
            st.rerun()
        else:
            st.warning(
                "No quiz is available for this topic yet."
            )

questions = st.session_state.get("quiz_questions", [])

if questions:
    st.markdown("### Test Yourself")

    with st.form("quiz_form"):

        selected_answers = {}

        for index, question in enumerate(questions):

            st.markdown(
                f"**Q{index + 1}. {question['question']}**"
            )

            selected_answers[index] = st.radio(
                "Choose an answer:",
                question["options"],
                key=f"quiz_answer_{index}",
                label_visibility="collapsed",
            )

            st.divider()

        submit_quiz = st.form_submit_button(
            "✅ Submit Quiz",
            use_container_width=True,
        )

    if submit_quiz:

        score = 0

        for index, question in enumerate(questions):

            if selected_answers[index] == question["answer"]:
                score += 1

        st.session_state["quiz_score"] = score
        st.session_state["quiz_submitted"] = True

    if st.session_state.get("quiz_submitted"):

        score = st.session_state["quiz_score"]
        total = len(questions)

        st.success(
            f"🎯 Your Score: **{score}/{total}**"
        )

        percentage = int((score / total) * 100)

        if percentage == 100:
            st.balloons()
            st.success("Excellent! You mastered this topic.")

        elif percentage >= 60:
            st.info(
                "Good job! Review the topic once more to strengthen your understanding."
            )

        else:
            st.warning(
                "Keep practicing. Try using Simplify or Example before attempting the quiz again."
            )

    st.divider()
    st.markdown("### GESTURE COMMANDS")
    command_columns = st.columns(5, gap="small")
    active_gesture = st.session_state.get("active_gesture")
    command_rows = [
        ("OPEN_PALM", "EXPLAIN"),
        ("THUMBS_UP", "DEEP DIVE"),
        ("THUMBS_DOWN", "SIMPLIFY"),
        ("PINCH", "EXAMPLE"),
        ("FIST", "RESET"),
    ]
    for column, (gesture, action) in zip(command_columns, command_rows):
        tile_class = "gesture-tile active" if active_gesture == gesture else "gesture-tile"
        column.markdown(
            f'<div class="{tile_class}"><div class="gesture-icon">{GESTURE_ICONS[gesture]}</div>'
            f'<div class="gesture-name">{GESTURE_NAMES[gesture].upper()}</div>'
            f'<div class="gesture-action">{action}</div></div>',
            unsafe_allow_html=True,
        )

    st.caption(
        f"Gestures used: {st.session_state['gesture_count']} · "
        f"AI interactions: {st.session_state['ai_interactions']} · "
        f"Last action: {st.session_state['last_action']}"
    )


if __name__ == "__main__":
    main()
