import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib, os

INPUT_CSV = "feature_extraction\mediapipe_features_normalized.csv" 
MODEL_DIR = "models/knn_rps"
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(INPUT_CSV, header=None)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = KNeighborsClassifier(
    n_neighbors=5,
    weights="distance"
)

model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

joblib.dump(model, f"{MODEL_DIR}/rps.pkl")
joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
