from src.utils.temporal_swipe_detector import TemporalSwipeDetector


def detect_any(xs):
    detector = TemporalSwipeDetector()
    now = 0.0
    for x in xs:
        result = detector.update(x, now, hand_present=True)
        if result["gesture"] is not None:
            return result
        now += 0.25
    return detector.update(xs[-1], now, hand_present=True)


def test_swipe_right_detected():
    result = detect_any([0.20, 0.30, 0.40, 0.52, 0.66, 0.72])
    assert result["gesture"] == "swipe_right"


def test_swipe_left_detected():
    result = detect_any([0.80, 0.68, 0.58, 0.47, 0.34, 0.27])
    assert result["gesture"] == "swipe_left"


def test_stationary_motion_does_not_trigger_swipe():
    result = detect_any([0.50, 0.51, 0.50, 0.49, 0.50])
    assert result["gesture"] is None


def test_vertical_motion_does_not_trigger_swipe():
    result = detect_any([0.50, 0.52, 0.51, 0.53, 0.50])
    assert result["gesture"] is None


def test_random_motion_does_not_trigger_swipe():
    result = detect_any([0.18, 0.42, 0.20, 0.51, 0.26, 0.48, 0.21])
    assert result["gesture"] is None


def test_cooldown_blocks_repeated_detection():
    detector = TemporalSwipeDetector()
    now = 0.0
    detected = False
    for x in [0.20, 0.30, 0.42, 0.58, 0.70]:
        result = detector.update(x, now, hand_present=True)
        detected = detected or result["gesture"] is not None
        now += 0.25
    assert detected is True
    result = detector.update(0.74, now, hand_present=True)
    assert result["gesture"] is None
    assert result["state"] == "cooldown"
