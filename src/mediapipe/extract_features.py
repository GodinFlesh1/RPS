import os
import cv2
import mediapipe as mp
import pandas as pd

INPUT_DIR = "dataset_new"
OUTPUT_CSV = "feature_extraction\mediapipe_features.csv"

LABEL_MAP = {
    "rock": 0,
    "paper": 1,
    "scissor": 2,
    "none": 3
}

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

def extract_landmarks(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if not result.multi_hand_landmarks:
        return None

    lm_points = result.multi_hand_landmarks[0].landmark
    features = []

    for lm in lm_points:
        features.extend([lm.x, lm.y, lm.z])

    return features


def main():
    rows = []
    os.makedirs("data_csv", exist_ok=True)

    for label_name, label_id in LABEL_MAP.items():
        folder = os.path.join(INPUT_DIR, label_name)
        print(f"Processing: {folder}")

        for filename in os.listdir(folder):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)

            if img is None:
                print("Cannot read:", img_path)
                continue

            features = extract_landmarks(img)

            if features is None:
                print("No hand detected in:", img_path)
                continue

            rows.append(features + [label_id])

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False, header=False)

    print(f"\nSaved {len(df)} samples to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
