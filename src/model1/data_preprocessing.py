import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import applicationconfig as config

def preprocess_image(img, img_size):
    """
    Canonical preprocessing for both training and inference.
    """
    h, w = img.shape[:2]
    side = min(h, w)
    start_x = (w - side) // 2
    start_y = (h - side) // 2
    img = img[start_y:start_y+side, start_x:start_x+side]
    img = cv2.resize(img, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def load_images_from_folder(folder, img_size):
    images = []
    labels = []
    DEBUG_SAVE = True
    DEBUG_DIR = "debug_preprocessed"

    print("folder", folder)

    for idx, class_name in enumerate(config.CLASSES):
        class_path = os.path.join(folder, class_name)

        if not os.path.exists(class_path):
            print(f"Warning: {class_path} does not exist")
            continue

        print(f"Loading {class_name} images...")

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)

            img = cv2.imread(img_path)
            if img is None:
                continue

            img = preprocess_image(img, img_size)
            # cv2.imshow("TRAIN PREPROCESS OUTPUT", cv2.resize(img, (256, 256)))
            # cv2.waitKey(0)   # press any key to go to next image
            images.append(img)
            labels.append(idx)

            if DEBUG_SAVE:
                os.makedirs(os.path.join(DEBUG_DIR, class_name), exist_ok=True)
                debug_count = 0

            if DEBUG_SAVE and debug_count < 10:
                save_path = os.path.join(DEBUG_DIR, class_name, img_name)
                cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                debug_count += 1


    return np.array(images), np.array(labels)


def preprocess_data(images, labels):
    """Normalize and prepare data"""
    # Normalize pixel values to [0, 1]
    images = images.astype('float32') / 255.0
    
    # One-hot encode labels
    labels = to_categorical(labels, num_classes=config.NUM_CLASSES)
    
    return images, labels

def prepare_dataset():
    """Complete data preparation pipeline"""
    print("Loading dataset...")
    images, labels = load_images_from_folder(config.DATA_DIR, config.IMG_SIZE)
    print("Loading data set")
    
    print(f"Total images loaded: {len(images)}")
    print(f"Image shape: {images[0].shape}")
    
    # Preprocess
    # images, labels = preprocess_data(images, labels)
    # Convert to numpy
    images = images.astype("float32") / 255.0
    labels = np.array(labels) 

    # -------- First split (Train / Temp) --------
    X_train, X_temp, y_train_idx, y_temp_idx = train_test_split(
        images,
        labels,
        test_size=(1 - config.TRAIN_SPLIT),
        random_state=42,
        stratify=labels
    )

    # -------- Second split (Val / Test) --------
    val_frac_of_temp = config.VAL_SPLIT / (config.VAL_SPLIT + config.TEST_SPLIT)

    X_val, X_test, y_val_idx, y_test_idx = train_test_split(
        X_temp,
        y_temp_idx,
        test_size=(1 - val_frac_of_temp),
        random_state=42,
        stratify=y_temp_idx
    )

    # -------- One-hot encode AFTER splitting --------
    y_train = to_categorical(y_train_idx, num_classes=config.NUM_CLASSES)
    y_val   = to_categorical(y_val_idx,   num_classes=config.NUM_CLASSES)
    y_test  = to_categorical(y_test_idx,  num_classes=config.NUM_CLASSES)

    print("\nDataset split:")
    print(f"Training:   {len(X_train)}")
    print(f"Validation: {len(X_val)}")
    print(f"Testing:    {len(X_test)}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

