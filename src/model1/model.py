import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import applicationconfig as config


def create_simple_cnn():
    """
    Simple CNN designed to LEARN features first.
    Minimal regularization, strong gradients.
    """

    model = Sequential([
        # Block 1
        Conv2D(
            32, (3, 3),
            activation='relu',
            input_shape=(config.IMG_SIZE, config.IMG_SIZE, config.IMG_CHANNELS)
        ),
        MaxPooling2D(pool_size=(2, 2)),

        # Block 2
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        # Block 3
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        # Classification head
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.25),        # reduced dropout
        Dense(config.NUM_CLASSES, activation='softmax')
    ])

    return model



# def create_deeper_cnn():
#     """
#     Deeper CNN with CAREFUL regularization.
#     Use this only after simple model learns.
#     """

#     model = Sequential([
#         # Block 1
#         Conv2D(
#             32, (3, 3),
#             activation='relu',
#             padding='same',
#             input_shape=(config.IMG_SIZE, config.IMG_SIZE, config.IMG_CHANNELS)
#         ),
#         Conv2D(32, (3, 3), activation='relu', padding='same'),
#         MaxPooling2D(pool_size=(2, 2)),
#         Dropout(0.2),

#         # Block 2
#         Conv2D(64, (3, 3), activation='relu', padding='same'),
#         Conv2D(64, (3, 3), activation='relu', padding='same'),
#         MaxPooling2D(pool_size=(2, 2)),
#         Dropout(0.25),

#         # Block 3
#         Conv2D(128, (3, 3), activation='relu', padding='same'),
#         Conv2D(128, (3, 3), activation='relu', padding='same'),
#         MaxPooling2D(pool_size=(2, 2)),
#         Dropout(0.3),

#         # Head
#         Flatten(),
#         Dense(256, activation='relu'),
#         Dropout(0.4),
#         Dense(128, activation='relu'),
#         Dropout(0.4),
#         Dense(config.NUM_CLASSES, activation='softmax')
#     ])

#     return model
