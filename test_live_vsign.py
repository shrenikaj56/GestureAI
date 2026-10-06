import cv2
import joblib
import json

from src.vision.hand_detector import HandDetector
from src.vision.feature_extractor import extract_feature_vector

model = joblib.load("models/gesture_model.joblib")

with open("models/gesture_labels.json", "r", encoding="utf-8") as f:
    labels = json.load(f)

detector = HandDetector()
camera = cv2.VideoCapture(0)

print("Show V-SIGN ✌️ to the camera.")
print("Press Q to quit.")

while True:
    success, frame = camera.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    landmarks, annotated, status = detector.detect(frame)

    if landmarks is not None:
        features = extract_feature_vector(
            landmarks,
            frame.shape
        )

        prediction = model.predict([features])[0]
        probabilities = model.predict_proba([features])[0]

        gesture = labels[str(prediction)]
        confidence = max(probabilities) * 100

        cv2.putText(
            annotated,
            f"{gesture} - {confidence:.1f}%",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    else:
        cv2.putText(
            annotated,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )

    cv2.imshow("GestureAI - Live Model Test", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
detector.release()
cv2.destroyAllWindows()