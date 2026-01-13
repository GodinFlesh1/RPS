import cv2
import mediapipe as mp
import numpy as np
import joblib


model = joblib.load("initial_model/mediapipe_rps/mediapipe_rps.pkl")
scaler = joblib.load("initial_model/mediapipe_rps/mediapipe_scaler.pkl")

LABELS = {0: "Rock", 1: "Paper", 2: "Scissor", 3: "None"}

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

def extract_landmarks(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if not result.multi_hand_landmarks:
        return None, frame

    hand_landmarks = result.multi_hand_landmarks[0]
    features = []

    for lm in hand_landmarks.landmark:
        features.extend([lm.x, lm.y, lm.z])

    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    return features, frame


def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        features, annotated_frame = extract_landmarks(frame)

        text = "No Hand Detected"

        if features is not None:
            X = np.array(features).reshape(1, -1)
            X = scaler.transform(X)
            
            pred = model.predict(X)[0]
            text = LABELS[pred]

        cv2.putText(annotated_frame, text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("RPS Model Test", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
