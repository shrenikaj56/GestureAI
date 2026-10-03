import pytest

import app
from src.services.ai_service import AIService


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("open_palm", "OPEN_PALM"),
        ("OPENPALM", "OPEN_PALM"),
        ("thumbs_up", "THUMBS_UP"),
        ("THUMBSUP", "THUMBS_UP"),
        ("thumbs down", "THUMBS_DOWN"),
        ("THUMBSDOWN", "THUMBS_DOWN"),
        ("pinch", "PINCH"),
        ("fist", "FIST"),
        ("unknown", None),
    ],
)
def test_normalize_existing_model_labels(raw, expected):
    assert app.normalize_gesture(raw) == expected


def test_stable_prediction_uses_two_of_last_three_and_confidence():
    assert app.get_stable_prediction(
        [("OPEN_PALM", 0.9), ("THUMBS_UP", 0.9), ("OPEN_PALM", 0.61)]
    ) == "OPEN_PALM"
    assert app.get_stable_prediction(
        [("OPEN_PALM", 0.9), ("THUMBS_UP", 0.9), ("THUMBS_DOWN", 0.9)]
    ) is None
    assert app.get_stable_prediction(
        [("OPEN_PALM", 0.9), ("OPEN_PALM", 0.4), ("THUMBS_UP", 0.9)]
    ) is None


def test_stable_gesture_is_one_shot_until_state_changes():
    assert app.is_new_stable_gesture("OPEN_PALM", None)
    assert not app.is_new_stable_gesture("OPEN_PALM", "OPEN_PALM")
    assert app.is_new_stable_gesture("THUMBS_UP", "OPEN_PALM")
    assert not app.is_new_stable_gesture(None, "PINCH")


@pytest.mark.parametrize(
    "topic",
    [
        "Binary Search",
        "Dijkstra's Algorithm",
        "DBMS Normalization",
        "Machine Learning",
        "Operating System",
    ],
)
@pytest.mark.parametrize("action", ["EXPLAIN", "DEEP DIVE", "SIMPLIFY", "EXAMPLE"])
def test_demo_mode_has_all_study_actions_without_api_key(monkeypatch, topic, action):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    service = AIService()

    response = service.generate(action, topic, "Existing explanation about this topic.")

    assert service.mode == "Demo Mode"
    assert response
    assert topic.split()[0].casefold() in response.casefold() or action == "EXAMPLE"


def test_unknown_demo_topic_stays_relevant_to_requested_topic(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    service = AIService()

    response = service.generate("DEEP DIVE", "Graph coloring", "Graph coloring assigns colors to vertices.")

    assert "Graph coloring" in response
    assert "Graph coloring assigns colors to vertices." in response


def test_busy_ai_request_blocks_duplicate_action(monkeypatch):
    state = {"ai_busy": True, "ai_error": "", "last_action_at": 0.0}
    monkeypatch.setattr(app.st, "session_state", state)

    assert not app.queue_ai_action("EXPLAIN", "Binary Search")
    assert state["ai_error"] == "AI is thinking... please wait."
