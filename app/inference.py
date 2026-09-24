import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pickle

from pathlib import Path


# -------------------------
# PROJECT PATHS
# -------------------------

# app/ is inside the main project folder
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "bsl_lstm_best_model.h5"
ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"

SEQUENCE_LENGTH = 10


# -------------------------
# LOAD MODEL AND ENCODER
# -------------------------

model = tf.keras.models.load_model(MODEL_PATH)

with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)


# -------------------------
# MEDIAPIPE
# -------------------------

mp_hands = mp.solutions.hands


# -------------------------
# LANDMARK PREPROCESSING
# -------------------------

def normalize_landmarks(landmarks):
    """
    Centre each hand around its wrist landmark.
    """

    landmarks = np.array(
        landmarks
    ).reshape((2, 21, 3))

    normalised_landmarks = []

    for hand in landmarks:

        wrist = hand[0]

        hand = hand - wrist

        normalised_landmarks.append(hand)

    return (
        np.array(normalised_landmarks)
        .flatten()
        .tolist()
    )


def extract_landmarks_from_frame(results):
    """
    Convert MediaPipe hand landmarks into
    exactly 126 values:
    2 hands x 21 landmarks x 3 coordinates.
    """

    left_hand = [0.0] * 63
    right_hand = [0.0] * 63

    if not results.multi_hand_landmarks:
        return [0.0] * 126

    for hand_landmarks, handedness in zip(
        results.multi_hand_landmarks,
        results.multi_handedness
    ):

        coords = []

        for landmark in hand_landmarks.landmark:
            coords.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        hand_label = (
            handedness
            .classification[0]
            .label
        )

        if hand_label == "Left":
            left_hand = coords

        elif hand_label == "Right":
            right_hand = coords

    landmarks = left_hand + right_hand

    if len(landmarks) != 126:
        return [0.0] * 126

    return normalize_landmarks(landmarks)


# -------------------------
# VIDEO PREDICTION
# -------------------------

def predict_bsl(video_path):
    """
    Read a video, create a 10-frame landmark
    sequence and return the predicted BSL letter.
    """

    cap = cv2.VideoCapture(str(video_path))

    sequence = []

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            # Same mirror transformation used during training
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            results = hands.process(rgb)

            # Only use frames where a hand was detected
            if results.multi_hand_landmarks:

                landmarks = extract_landmarks_from_frame(
                    results
                )

                sequence.append(landmarks)

            # Stop once 10 valid frames are collected
            if len(sequence) == SEQUENCE_LENGTH:
                break

    cap.release()

    # Not enough usable frames
    if len(sequence) < SEQUENCE_LENGTH:
        return None, None

    input_sequence = np.expand_dims(
        np.array(sequence),
        axis=0
    )

    prediction = model.predict(
        input_sequence,
        verbose=0
    )[0]

    predicted_index = np.argmax(prediction)

    predicted_label = label_encoder.inverse_transform(
        [predicted_index]
    )[0]

    confidence = float(
        prediction[predicted_index]
    )

    return predicted_label, confidence