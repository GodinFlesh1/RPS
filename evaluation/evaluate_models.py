import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import joblib

import applicationconfig as config


FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def evaluate_cnn_model():
    print("\n Evaluating CNN Model")

    model = load_model(
        f"{config.MODEL_DIR}/model1_cnn/best_model.keras"
    )

    from src.model1.data_preprocessing import prepare_dataset
    (_, _), (_, _), (X_test, y_test) = prepare_dataset()

    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(model.predict(X_test), axis=1)

    print("\nCNN Classification Report:")
    print(classification_report(y_true, y_pred, target_names=config.CLASSES))

    acc = accuracy_score(y_true, y_pred)
    print("CNN Accuracy:", acc)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=config.CLASSES)
    disp.plot(cmap="Blues")
    plt.title("CNN Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "cnn_confusion.png"))
    plt.close()

    return acc



def evaluate_mediapipe_model():
    print("\nEvaluating MediaPipe Model ")

    model = joblib.load("models/gradient_boost_rps/rps.pkl")
    scaler = joblib.load("models/gradient_boost_rps/scaler.pkl")

    df = pd.read_csv(
        os.path.join("feature_extraction", "mediapipe_features_normalized.csv"),
        header=None
    )

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X = scaler.transform(X)

    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    y_pred = model.predict(X_test)

    print("\nMediaPipe Classification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=config.CLASSES
    ))

    acc = accuracy_score(y_test, y_pred)
    print("MediaPipe Accuracy:", acc)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=config.CLASSES)
    disp.plot(cmap="Greens")
    plt.title("MediaPipe Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "mediapipe_confusion.png"))
    plt.close()

    return acc



def plot_accuracy_comparison(cnn_acc, mp_acc):
    models = ["CNN Model", "MediaPipe Model"]
    accuracies = [cnn_acc, mp_acc]

    plt.bar(models, accuracies)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "accuracy_comparison.png"))
    plt.close()



if __name__ == "__main__":
    cnn_acc = evaluate_cnn_model()
    mp_acc = evaluate_mediapipe_model()
    plot_accuracy_comparison(cnn_acc, mp_acc)
