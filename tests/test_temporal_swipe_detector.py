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
    result = detect_any([0.80, 0.72, 0.64, 0.52, 0.36, 0.28])
    assert result["gesture"] == "swipe_right"


def test_swipe_left_detected():
    result = detect_any([0.20, 0.32, 0.42, 0.55, 0.66, 0.78])
    assert result["gesture"] == "swipe_left"


def test_mirrored_camera_direction_is_converted_to_screen_space():
    detector = TemporalSwipeDetector()
    now = 0.0
    detected = None
    for x in [0.80, 0.72, 0.63, 0.55, 0.39, 0.28]:
        result = detector.update(x, now, hand_present=True, y_value=0.45)
        if result["gesture"] is not None:
            detected = result
            break
        now += 0.25
    assert detected is not None
    assert detected["gesture"] == "swipe_right"
    assert detected["screen_delta_x"] > 0
    assert detected["direction"] == "right"


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
