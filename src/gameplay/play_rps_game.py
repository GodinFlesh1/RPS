import cv2
import mediapipe as mp
import numpy as np
import time
import joblib
from ai_opponent import AIOpponent


# Using the best model that came out from testing
model = joblib.load("models/gradient_boost_rps/rps.pkl")
scaler = joblib.load("models/gradient_boost_rps/scaler.pkl")


#Label Mapping
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

# AI Opponent
ai = AIOpponent()

player_score = 0
computer_score = 0



# Hand or gesture detection function
def detect_gesture(frame):
    # ROI Box
    cv2.rectangle(frame, ROI_TOP_LEFT, ROI_BOTTOM_RIGHT, (0, 255, 0), 2)

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if not result.multi_hand_landmarks:
        return None

    lm = result.multi_hand_landmarks[0]

    h, w, _ = frame.shape
    wx = int(lm.landmark[0].x * w)
    wy = int(lm.landmark[0].y * h)

    x1, y1 = ROI_TOP_LEFT
    x2, y2 = ROI_BOTTOM_RIGHT

    inside_roi = (x1 < wx < x2 and y1 < wy < y2)
    if not inside_roi:
        return None

    features = []
    for p in lm.landmark:
        features.extend([p.x, p.y, p.z])

    mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

    return features


def predict_from_features(features):
    wrist = features[0:3]
    landmarks = np.array(features).reshape(21, 3)
    centered = landmarks - np.array(wrist)

    scale = np.linalg.norm(centered[9])
    if scale < 1e-6:
        scale = 1.0

    normalized = centered / scale
    flat = normalized.flatten().reshape(1, -1)

    X = scaler.transform(flat)
    return model.predict(X)[0]

    
def get_winner(player, computer):
    if player == computer:
        return "Draw"

    # Rock = 0, Paper = 1, Scissor = 2
    if (player == 0 and computer == 2) or (player == 1 and computer == 0) or (player == 2 and computer == 1):
        return "Player"

    return "Computer"




# Game loop to play game
# Press q to quit
def main():
    global player_score, computer_score

    # Video capture from inbuilt webcam
    # cap = cv2.VideoCapture(0)

    # Video capture from external webcam (if needed) 
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 

    while True:

        # AI chooses move initially
        computer_move = ai.choose_move()
        comp_label = LABELS[computer_move]

        countdown_start = time.time()
        countdown_time = 6
        last_detected_gesture = -1

        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            # Live gesture detection during countdown
            features = detect_gesture(frame)
            if features is not None:
                last_detected_gesture = predict_from_features(features)
                gesture_text = LABELS[last_detected_gesture]
            else:
                gesture_text = "None"

            cv2.putText(frame, f"Gesture: {gesture_text}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            # Countdown display
            elapsed = int(time.time() - countdown_start)
            remaining = countdown_time - elapsed

            if remaining <= 0:
                break

            cv2.putText(frame, f"{remaining}", (250, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (0,255,255), 8)
            cv2.putText(frame, "Place your hand inside the green box", (40, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            cv2.imshow("Rock Paper Scissors", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return

        # The last detected gesture is the player's move
        player_move = last_detected_gesture
        player_label = LABELS[player_move]

        # SKIP INVALID ROUND
        if player_move == -1:
            print("Skipping round — No gesture detected.")
            time.sleep(1)
            continue

       
        ai.update_history(player_move)
        result = get_winner(player_move, computer_move)

        if result == "Player":
            player_score += 1
        elif result == "Computer":
            computer_score += 1


        show_start = time.time()
        while time.time() - show_start < 2:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            cv2.putText(frame, f"Player: {player_label}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
            cv2.putText(frame, f"Computer: {comp_label}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3)
            cv2.putText(frame, f"Result: {result}", (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3)
            cv2.putText(frame, f"Score You: {player_score} | AI: {computer_score}",
                        (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)

            cv2.imshow("Rock Paper Scissors", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
