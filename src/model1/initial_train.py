import os
import numpy as np
import sys
import random
import tensorflow as tf
from collections import Counter
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import applicationconfig as config
from src.model1.data_preprocessing import prepare_dataset
from src.model1.model import create_simple_cnn

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)


def plot_training_history(history, save_path='training_history.png'):
    """Plot training and validation accuracy/loss"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training history plot saved to {save_path}")




def train_model():
    """Main training function"""
    # Prepare data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_dataset()
    
    print("\nCreating model...")
    model = create_simple_cnn()
    
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(config.MODEL_DIR, 'best_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

    
    # train_aug = ImageDataGenerator(
    #     rotation_range=12,
    #     width_shift_range=0.08,
    #     height_shift_range=0.08,
    #     zoom_range=0.08,
    #     brightness_range=(0.85,1.2),
    #     horizontal_flip=True,
    #     fill_mode='nearest'
    # )


    train_aug = ImageDataGenerator(
    rotation_range=8,
    width_shift_range=0.04,
    height_shift_range=0.04,
    zoom_range=0.03,
    brightness_range=(0.9, 1.1),
    horizontal_flip=True,
    fill_mode='nearest'
    )
    val_aug = ImageDataGenerator()

    y_train_labels = np.argmax(y_train, axis=1)
    counts = Counter(y_train_labels)
    print("Train class counts before upsampling:", counts)

    target = max(
        counts[cls]
        for cls in counts
        if config.CLASSES[cls] != "none"
    )

    Xs, Ys = [], []
    for cls in range(config.NUM_CLASSES):

        cls_name = config.CLASSES[cls]
        idxs = np.where(y_train_labels == cls)[0]
        if len(idxs) == 0:
            continue

        Xs.append(X_train[idxs])
        Ys.append(y_train[idxs])
        need = target - len(idxs)
        if need > 0:
            rep_idx = np.random.choice(idxs, size=need, replace=True)
            Xs.append(X_train[rep_idx])
            Ys.append(y_train[rep_idx])

    # Concatenate and shuffle
    X_train_bal = np.concatenate(Xs, axis=0)
    y_train_bal = np.concatenate(Ys, axis=0)
    perm = np.random.permutation(len(X_train_bal))
    X_train_bal = X_train_bal[perm]
    y_train_bal = y_train_bal[perm]

    train_gen = train_aug.flow(X_train_bal, y_train_bal, batch_size=config.BATCH_SIZE, shuffle=True, seed=SEED)
    val_gen = val_aug.flow(X_val, y_val, batch_size=config.BATCH_SIZE, shuffle=False)

    
    history = model.fit(
        train_gen,
        epochs=config.EPOCHS,
        validation_data=val_gen,
        callbacks=[checkpoint, early_stop, reduce_lr],
        steps_per_epoch=max(1, len(X_train_bal) // config.BATCH_SIZE),
        verbose=1
    )


    print("\nStarting training...")
    plot_training_history(history)
    
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    final_model_path = os.path.join(config.MODEL_DIR, 'best_model.h5')
    model.save(final_model_path)
    print(f"\nFinal model saved to {final_model_path}")
    
    return model, history

if __name__ == "__main__":
    train_model()