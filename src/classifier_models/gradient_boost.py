import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# INPUT_CSV = "feature_extraction\mediapipe_features.csv"
INPUT_CSV = "feature_extraction\mediapipe_features_normalized.csv" 
MODEL_DIR = "models/gradient_boost_rps"
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(INPUT_CSV, header=None)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.9,
    random_state=42
)


model.fit(X_train_scaled, y_train)


pred = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred, digits=4))

joblib.dump(model, f"{MODEL_DIR}/rps.pkl")
joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")

print("\nGradient Boosting model saved successfully!")
