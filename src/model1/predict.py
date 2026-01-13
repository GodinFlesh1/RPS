import cv2
import os
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import applicationconfig as config
from tensorflow.keras.models import load_model
from src.model1.data_preprocessing import preprocess_image

MODEL_DIR = "models\model1_cnn"

def predict_gesture(model, roi):

    img = preprocess_image(roi, config.IMG_SIZE)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img, verbose=0)
    class_idx = np.argmax(preds)
    confidence = float(preds[0][class_idx])

    return config.CLASSES[class_idx], confidence



def real_time_prediction(model_path, camera_index=0):
    model = load_model(model_path)
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return

    print("Press 'q' to quit")

    ROI_SIZE = 250 

    cx, cy = 325, 265  

    x1 = cx - ROI_SIZE // 2
    y1 = cy - ROI_SIZE // 2
    x2 = cx + ROI_SIZE // 2
    y2 = cy + ROI_SIZE // 2

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        x1c = max(0, min(x1, w - 1))
        x2c = max(0, min(x2, w))
        y1c = max(0, min(y1, h - 1))
        y2c = max(0, min(y2, h))

        cv2.rectangle(frame, (x1c, y1c), (x2c, y2c), (0, 255, 0), 2)

        roi = frame[y1c:y2c, x1c:x2c]

        if roi.size == 0:
            label, conf = "none", 0.0
        else:
            rh, rw = roi.shape[:2]
            side = min(rh, rw)
            roi = roi[:side, :side]

            label, conf = predict_gesture(model, roi)

        text = f"{label} ({conf:.2f})"
        cv2.putText(frame, text, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

        cv2.imshow("Rock Paper Scissors", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    model_path = os.path.join(MODEL_DIR, "best_model.keras")
    print("Loading model from:", model_path)
    real_time_prediction(model_path, camera_index=0)
