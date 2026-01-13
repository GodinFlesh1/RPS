import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

CSV_PATH = "feature_extraction\mediapipe_features.csv"
MODEL_DIR = "initial_model/mediapipe_rps"
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    df = pd.read_csv(CSV_PATH, header=None)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=4))

    joblib.dump(clf, f"{MODEL_DIR}/mediapipe_rps.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/mediapipe_scaler.pkl")

    print("\nModel saved successfully!")

if __name__ == "__main__":
    main()
