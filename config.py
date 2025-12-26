"""
Visual Model Configuration
"""
from pathlib import Path

# Paths
COMBINATIONS_DIR = Path("C:/Users/user/Desktop/STATIONARY/Combinations")
OUTPUT_DIR = Path("c:/Users/user/Desktop/STATIONARY/tsfresh ensemble/visual_model/trained_models")

# Training hyperparameters
NUM_EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.001
IMAGE_SIZE = 224
TRAIN_SPLIT = 0.8

# Data loading
MAX_IMAGES_PER_COMBO = None  # None = use all images, or set to number (e.g., 100)

# Model architecture
BACKBONE = 'resnet50'  # Options: 'resnet50', 'resnet34', 'efficientnet_b0'
PRETRAINED = True

# Device
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataloader
NUM_WORKERS = 4  # For parallel data loading (set to 0 if having issues on Windows)
