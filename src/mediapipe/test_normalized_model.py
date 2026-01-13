import cv2
import mediapipe as mp
import numpy as np
import joblib
import time

# Load model & scaler
# model = joblib.load("models/svm_rps/rps.pkl")
model = joblib.load("models/gradient_boost_rps/rps.pkl")
# model = joblib.load("models/random_forest_rps/rps.pkl")

# scaler = joblib.load("models/svm_rps/scaler.pkl")
scaler = joblib.load("models/gradient_boost_rps/scaler.pkl")
# scaler = joblib.load("models/random_forest_rps/scaler.pkl")


LABELS = {0: "Rock", 1: "Paper", 2: "Scissor", 3: "None"}


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils


ROI_TOP_LEFT = (150, 100)
ROI_BOTTOM_RIGHT = (450, 400)


def normalize_landmarks(features):
    landmarks = np.array(features).reshape(21, 3)

    wrist = landmarks[0]
    centered = landmarks - wrist

    scale = np.linalg.norm(centered[9])
    if scale < 1e-6:
        scale = 1.0

    normalized = centered / scale
    return normalized.flatten()


def predict_from_features(features):
    normalized = normalize_landmarks(features)
    X = scaler.transform([normalized])
    
    probs = model.predict_proba(X)[0]

    pred_class = model.classes_[np.argmax(probs)]
    confidence = probs.max()

    # Confidence threshold
    THRESHOLD = 0.8

    if confidence < THRESHOLD:
        return 3
    else:
        return int(pred_class)




def extract_landmarks(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if not result.multi_hand_landmarks:
        return None

    lm = result.multi_hand_landmarks[0]

    h, w, _ = frame.shape
    f = []

    for p in lm.landmark:
        f.extend([p.x, p.y, p.z])

    mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

    return f


def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        # Draw ROI
        cv2.rectangle(frame, ROI_TOP_LEFT, ROI_BOTTOM_RIGHT, (0, 255, 0), 2)

        # Extract landmarks
        features = extract_landmarks(frame)

        if features is not None:
            pred = predict_from_features(features)
            label = LABELS[pred]
        else:
            label = "None"

        # Display prediction
        cv2.putText(frame, f"Prediction: {label}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Model Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
