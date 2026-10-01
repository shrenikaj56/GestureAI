# GestureAI

GestureAI is an AI-powered personalized gesture control system that uses an explicitly started webcam, MediaPipe hand landmarks, and a scikit-learn machine learning pipeline to recognize custom static gestures in real time. The project is designed as a college AIML MVP with a clean modular structure and a professional Streamlit dashboard.

## Problem statement

Traditional gesture interfaces often depend on hard-coded rules or limited predefined motions. This makes them fragile, difficult to personalize, and poor for custom use cases such as controlling applications to match individual habits and workflows.

## Proposed solution

GestureAI combines computer vision, feature engineering, and supervised machine learning to detect and classify hand gestures from webcam input. The system supports an initial gesture set and allows users to teach custom gestures by collecting landmark samples and retraining the model.

## Features

- Explicit Start Camera / Stop Camera lifecycle
- Real-time webcam hand tracking after camera start
- Mediapipe landmark extraction and visualization
- Normalized landmark feature engineering
- Random Forest-based classifier pipeline
- Gesture prediction with confidence values
- Model evaluation with accuracy, precision, recall, F1-score, and confusion matrix
- Custom gesture training workflow
- Dashboard for live recognition and performance monitoring
- Honest empty-data and not-trained states; no synthetic training rows or fake metrics
- Modular structure for future extensions such as SVM/KNN and context-aware actions

## Architecture

- Computer vision: OpenCV + MediaPipe
- Feature extraction: normalized hand landmarks and distances
- Machine learning: scikit-learn Random Forest
- UI: Streamlit
- Control layer: PyAutoGUI is isolated behind a whitelist for later activation

## Tech stack

- Python
- OpenCV
- MediaPipe
- Streamlit
- scikit-learn
- NumPy
- Pandas
- Matplotlib
- PyAutoGUI
- Joblib

## ML workflow

1. Start the camera explicitly from Live Control or Teach Gesture.
2. Collect landmark samples for static gestures.
3. Normalize landmark coordinates relative to the wrist.
4. Extract engineered numeric features, including distances.
5. Validate labels, class counts, and dataset size.
6. Train/test split the dataset.
7. Train a Random Forest classifier.
8. Evaluate using standard metrics.
9. Save model, label mapping, feature metadata, and measured metrics.
10. Run real-time prediction on webcam frames.

## Static and dynamic gestures

The static classifier contains only `open_palm`, `fist`, `thumbs_up`, `thumbs_down`, and `pinch`. A static gesture is one hand configuration classified by the original Random Forest model. A dynamic gesture is a sequence of hand configurations over time classified by a separate dynamic Random Forest model. Swipe labels are never added to the static model.

## Dynamic sequence pipeline

The dynamic pipeline is independent from static recognition:

```text
MediaPipe landmarks
-> sequence buffer
-> resampling to 20 frames
-> wrist-relative shape normalization
-> scale-normalized wrist/hand-center trajectory
-> frame deltas and engineered features
-> dynamic Random Forest
-> confidence and movement rejection
-> swipe_left / swipe_right event
```

Dynamic sequences are stored separately under `data/dynamic/swipe_left` and `data/dynamic/swipe_right`. Each `.npy` file represents one real webcam movement sequence. The Train Dynamic Gesture tab collects sequences automatically, validates direction and movement, and trains `models/dynamic_gesture_model.joblib` without changing the static model.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running the application

```bash
streamlit run app.py
```

## Dataset collection instructions

1. Open the Teach Gesture page in the Streamlit app.
2. Enter a gesture name.
3. Choose the number of samples to collect.
4. Click Start Collection and allow camera access.
5. Perform the static gesture in front of the webcam. The app automatically saves normalized features and the label; no manual CSV creation is required.
6. Repeat for at least two classes, with enough samples per class.
7. Click Train Model and review the actual held-out evaluation metrics.

For dynamic gestures, open Train Dynamic Gesture, choose Swipe Left or Swipe Right, select a sequence count, and start collection. Each accepted sample contains multiple landmark frames from one complete movement. The application resamples every sequence to 20 frames and rejects missing-hand, low-motion, and wrong-direction attempts.

## Model training instructions

Static training is handled through the existing Teach Gesture workflow. Before static training, the system checks that the dataset exists, is readable, contains feature columns, has at least two classes, and has enough samples per class and overall. Dynamic training uses a separate stratified train/test split over real sequence files and saves independent model metadata and evaluation metrics. Neither path creates placeholder data or fabricated metrics.

## Screenshots

Placeholder for screenshots of the dashboard, live webcam view, and model performance page.

## Future scope

- SVM, KNN, and deep learning comparisons
- More dynamic sequence data and model comparisons
- Context-aware actions for PowerPoint, Chrome, and PDF workflows after temporal recognition is stable
- Voice + gesture multimodal interaction
- Multi-camera and deployment options
- Improved robustness for custom gesture learning

## Limitations

- Requires a real camera and good lighting conditions.
- Gesture accuracy depends on consistent hand positioning and sample quality.
- Swipe recognition, active-window context detection, and automatic computer actions are intentionally deferred until their temporal/context foundations are implemented.

## Project structure

```text
GestureAI/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── gestures/
├── models/
├── config/
│   └── actions.json
├── src/
│   ├── vision/
│   │   ├── hand_detector.py
│   │   ├── feature_extractor.py
│   │   └── temporal_gesture.py
│   ├── ml/
│   │   ├── dataset.py
│   │   ├── trainer.py
│   │   ├── predictor.py
│   │   └── evaluator.py
│   ├── control/
│   │   ├── action_manager.py
│   │   └── context_manager.py
│   ├── data/
│   │   └── collector.py
│   └── utils/
│       ├── config.py
│       └── logger.py
└── assets/
```
