import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import os


INPUT_CSV = "feature_extraction\mediapipe_features_normalized.csv"
MODEL_DIR = "initial_model/mediapipe_normalized_rps"

os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(INPUT_CSV, header=None)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(
    n_estimators=500,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_scaled, y_train)

pred = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

joblib.dump(model, f"{MODEL_DIR}/mediapipe_rps.pkl")
joblib.dump(scaler, f"{MODEL_DIR}/mediapipe_rps_scaler.pkl")

print("Model saved.")
