# ml/config.py

import torch
import os

# -- Project Root --
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# -- Project Paths --
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'models')

# -- Model & Tokenizer --
MODEL_NAME = 'distilbert-base-uncased'

# -- Training Hyperparameters --
# These are optimized for a 2GB GPU as per the project prompt.
# If you have more VRAM, you can increase BATCH_SIZE and reduce GRADIENT_ACCUMULATION_STEPS.
# The effective batch size is BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 512  # Max sequence length for the model
BATCH_SIZE = 4    # Batch size per device
GRADIENT_ACCUMULATION_STEPS = 4 # Number of steps to accumulate gradients for
EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500 # Number of warmup steps for the learning rate scheduler

# -- Logging --
LOG_STEP = 100 # Log training loss every N steps
