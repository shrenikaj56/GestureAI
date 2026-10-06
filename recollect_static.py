import cv2
import pandas as pd
from src.ml.dataset import GestureDataset
from src.vision.feature_extractor import extract_feature_vector
from src.vision.hand_detector import HandDetector
from src.utils.config import DATASET_PATH

GESTURES = [
    "OPEN_PALM",
    "FIST",
    "THUMBS_UP",
    "THUMBS_DOWN",
]

SAMPLES_PER_GESTURE = 100


def collect_gesture(detector, gesture_name):
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Could not open webcam.")
        return []

    samples = []

    print("\n================================")
    print(f"Collecting: {gesture_name}")
    print("================================")
    print("Show the gesture clearly.")
    print("Move your hand slightly between samples.")
    print("Press Q to stop.\n")

    while len(samples) < SAMPLES_PER_GESTURE:

        success, frame = camera.read()

        if not success:
            break

        # IMPORTANT: same preprocessing as live recognition
        frame = cv2.flip(frame, 1)

        landmarks, annotated, status = detector.detect(frame)

        if landmarks is not None:
            features = extract_feature_vector(
                landmarks,
                frame.shape
            )

            samples.append(features)

            cv2.putText(
                annotated,
                f"{gesture_name}: {len(samples)}/{SAMPLES_PER_GESTURE}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
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

        cv2.imshow("Gesture Collection", annotated)

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

    print(f"{gesture_name}: {len(samples)} samples collected.")

    return samples


def main():

    detector = HandDetector()

    all_samples = []

    for gesture in GESTURES:

        samples = collect_gesture(
            detector,
            gesture
        )

        for features in samples:
            row = {"gesture": gesture}

            for i, value in enumerate(features):
                row[f"feature_{i}"] = float(value)

            all_samples.append(row)

    detector.release()

    if not all_samples:
        print("No samples collected.")
        return

    # Load existing dataset
    dataset = GestureDataset(DATASET_PATH)
    existing = dataset.load()

    # Keep V_SIGN and remove old versions of the 4 gestures
    existing = existing[
        ~existing["gesture"].isin(GESTURES)
    ]

    new_data = pd.DataFrame(all_samples)

    final_data = pd.concat(
        [existing, new_data],
        ignore_index=True
    )

    final_data.to_csv(
        DATASET_PATH,
        index=False
    )

    print("\n================================")
    print("Collection completed.")
    print("================================")
    print(final_data["gesture"].value_counts())


if __name__ == "__main__":
    main()