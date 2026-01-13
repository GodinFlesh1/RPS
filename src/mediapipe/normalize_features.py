import pandas as pd
import numpy as np

INPUT_CSV = "feature_extraction\mediapipe_features.csv"
OUTPUT_CSV = "feature_extraction\mediapipe_features_normalized.csv"

NUM_LM = 21
VALUES_PER_LM = 3


def normalize_landmarks(row):
    landmarks = row[:-1].reshape(NUM_LM, VALUES_PER_LM)
    label = row[-1]

    wrist = landmarks[0]
    centered = landmarks - wrist

    # Use distance between wrist and middle finger
    scale = np.linalg.norm(centered[9])
    if scale < 1e-6:
        scale = 1.0

    normalized = centered / scale

    normalized_flat = normalized.flatten()

    return np.append(normalized_flat, label)


def main():
    df = pd.read_csv(INPUT_CSV, header=None)
    data = df.values

    fixed_rows = []

    for row in data:
        fixed = normalize_landmarks(row)
        fixed_rows.append(fixed)

    fixed_df = pd.DataFrame(fixed_rows)
    fixed_df.to_csv(OUTPUT_CSV, header=False, index=False)

    print(f"Normalized dataset saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
