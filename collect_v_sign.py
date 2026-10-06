import cv2
import numpy as np

from src.ml.dataset import GestureDataset
from src.vision.feature_extractor import extract_feature_vector
from src.vision.hand_detector import HandDetector
from src.utils.config import DATASET_PATH


TARGET_SAMPLES = 100


def main():
    detector = HandDetector()
    dataset = GestureDataset(DATASET_PATH)

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("❌ Could not open webcam.")
        detector.release()
        return

    collected = 0

    print("\n========================================")
    print("       GestureAI - V-SIGN Collector")
    print("========================================")
    print("Show a clear V-SIGN ✌️ to the camera.")
    print("Move your hand slightly between samples.")
    print("Press Q to stop.")
    print(f"Target samples: {TARGET_SAMPLES}")
    print("========================================\n")

    while collected < TARGET_SAMPLES:
        success, frame = camera.read()

        if not success:
            print("Could not read webcam frame.")
            break

        frame = cv2.flip(frame, 1)

        landmarks, annotated, status = detector.detect(frame)

        if landmarks is not None:
            try:
                features = extract_feature_vector(
                    landmarks,
                    annotated.shape,
                )

                dataset.append_sample(
                    "V_SIGN",
                    features,
                )

                collected += 1

                cv2.putText(
                    annotated,
                    f"V-SIGN samples: {collected}/{TARGET_SAMPLES}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

            except Exception as exc:
                cv2.putText(
                    annotated,
                    f"Error: {exc}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

        else:
            cv2.putText(
                annotated,
                status,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        cv2.imshow("GestureAI - V-SIGN Collection", annotated)

        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            break

    camera.release()
    detector.release()
    cv2.destroyAllWindows()

    print("\n========================================")
    print(f"Collection finished: {collected} V_SIGN samples")
    print("========================================")


if __name__ == "__main__":
    main()