import os

# Paths
BASE_DIR = os.path.dirname((os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')
MODEL_DIR = os.path.join(BASE_DIR, 'models')


# Model parameters
IMG_SIZE = 128
IMG_CHANNELS = 3 
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001

# 4 classes
CLASSES = ['rock', 'paper', 'scissor', 'none']
NUM_CLASSES = len(CLASSES)

# Data split
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15