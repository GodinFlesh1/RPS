import cv2
import mediapipe as mp
import numpy as np
import joblib
from ai_opponent import AIOpponent


model = joblib.load("models/gradient_boost_rps/rps.pkl")
scaler = joblib.load("models/gradient_boost_rps/scaler.pkl")

LABELS = {0: "Rock", 1: "Paper", 2: "Scissor", 3: "None"}


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

ai = AIOpponent()

player_score = 0
computer_score = 0
round_result = ""

def extract_landmarks(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if not results.multi_hand_landmarks:
        return None, frame

    hand_points = results.multi_hand_landmarks[0]
    features = []

    for lm in hand_points.landmark:
        features.extend([lm.x, lm.y, lm.z])

    mp_draw.draw_landmarks(frame, hand_points, mp_hands.HAND_CONNECTIONS)
    return features, frame


def get_winner(player, computer):
    if player == computer:
        return "Draw"

    # Rock = 0, Paper = 1, Scissor = 2
    if (player == 0 and computer == 2) or (player == 1 and computer == 0) or (player == 2 and computer == 1):
        return "Player"

    return "Computer"


def main():
    global player_score, computer_score, round_result

    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        features, annotated_frame = extract_landmarks(frame)

        player_move = None
        display_player = "Detecting..."

        if features is not None:
            X = np.array(features).reshape(1, -1)
            X_scaled = scaler.transform(X)
            pred = model.predict(X_scaled)[0]
            player_move = pred
            display_player = LABELS[pred]

        # AI move
        ai.update_history(player_move)
        computer_move = ai.choose_move()
        display_computer = LABELS[computer_move]

        # Determine winner
        if player_move is not None:
            result = get_winner(player_move, computer_move)
            round_result = result

            if result == "Player":
                player_score += 1
            elif result == "Computer":
                computer_score += 1

        # Display overlay
        cv2.putText(annotated_frame, f"Player: {display_player}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        
        cv2.putText(annotated_frame, f"Computer: {display_computer}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
        
        cv2.putText(annotated_frame, f"Result: {round_result}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 2)
        
        cv2.putText(annotated_frame, f"Score  You: {player_score}   AI: {computer_score}", (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

        cv2.imshow("Rock Paper Scissors Game", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
