"""
config.py — Central Configuration for Dogri Machine Translation Pipeline
=========================================================================
All hyperparameters, file paths, and special tokens are defined here.
This ensures reproducibility and easy tuning for experiments.
"""

import os
import torch

# ──────────────────────────────────────────────
# 1. PROJECT PATHS
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PARALLEL_DIR = os.path.join(DATA_DIR, "parallel")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOGS_DIR = os.path.join(RESULTS_DIR, "training_logs")
EVAL_DIR = os.path.join(RESULTS_DIR, "evaluation")
ERROR_DIR = os.path.join(RESULTS_DIR, "error_analysis")

# Original dataset path
DATASET_PATH = os.path.join(BASE_DIR, "doogri-sentences.txt")

# Create directories if they don't exist
for d in [RAW_DIR, PARALLEL_DIR, PROCESSED_DIR, MODELS_DIR,
          LOGS_DIR, EVAL_DIR, ERROR_DIR]:
    os.makedirs(d, exist_ok=True)

# ──────────────────────────────────────────────
# 2. SPECIAL TOKENS
# ──────────────────────────────────────────────
PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

# ──────────────────────────────────────────────
# 3. DATA CONFIGURATION
# ──────────────────────────────────────────────
TRAIN_ID_RANGE = (1, 1000)       # IDs 1–1000: training set
TEST_ID_RANGE  = (1001, 1200)    # IDs 1001–1200: held-out test set
TRAIN_SPLIT_RATIO = 0.9          # Preferred train share when dataset grows
MIN_TRAIN_SAMPLES = 1000         # Keep at least this many training examples if available
MIN_TEST_SAMPLES = 120           # Keep a held-out set for evaluation
MIN_WORD_FREQ  = 1               # Minimum word frequency for vocabulary
MAX_SEQ_LENGTH = 50              # Maximum sequence length (tokens)

# ──────────────────────────────────────────────
# 4. MODEL HYPERPARAMETERS
# ──────────────────────────────────────────────
EMBEDDING_DIM  = 256    # Word embedding dimensionality
HIDDEN_DIM     = 512    # GRU hidden state dimensionality
NUM_LAYERS     = 2      # Number of GRU layers (encoder & decoder)
DROPOUT        = 0.3    # Dropout probability
BIDIRECTIONAL  = True   # Bidirectional encoder

# ──────────────────────────────────────────────
# 5. TRAINING HYPERPARAMETERS
# ──────────────────────────────────────────────
BATCH_SIZE            = 32
NUM_EPOCHS            = 140
LEARNING_RATE         = 0.0008
WEIGHT_DECAY          = 1e-5
GRADIENT_CLIP         = 1.0
TEACHER_FORCING_START = 1.0
TEACHER_FORCING_END   = 0.3
TEACHER_FORCING_DECAY = 0.03   # Decrease per epoch
EARLY_STOP_PATIENCE   = 10
LR_SCHEDULER_FACTOR   = 0.5
LR_SCHEDULER_PATIENCE = 5
LABEL_SMOOTHING       = 0.1


# ──────────────────────────────────────────────
# 7. BEAM SEARCH (Inference)
# ──────────────────────────────────────────────
BEAM_WIDTH = 5

# ──────────────────────────────────────────────
# 8. DEVICE CONFIGURATION
# ──────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────
# 9. RANDOM SEED (Reproducibility)
# ──────────────────────────────────────────────
SEED = 42
