# ============================================================
# config.py — Central configuration for SRDC v2
# Change settings here, they apply everywhere
# ============================================================

# Model
MODEL_NAME    = "microsoft/deberta-v3-base"  # HuggingFace model
NUM_CLASSES   = 12                            # 11 families + goodware
MAX_LENGTH    = 512                           # DeBERTa max tokens

# Training
BATCH_SIZE    = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS    = 10
WARMUP_RATIO  = 0.1
WEIGHT_DECAY  = 0.01

# Data
DATA_PATH     = "data/raw/RansomwareData.csv"
TRAIN_SPLIT   = 0.8
RANDOM_SEED   = 42

# Paths
MODEL_SAVE    = "outputs/model"
LOG_DIR       = "outputs/logs"

# Family mapping
FAMILY_MAP = {
    0: "Goodware",    1: "Citroni",
    2: "CryptLocker", 3: "CryptoWall",
    4: "KOLLAH",      5: "Kovter",
    6: "Locker",      7: "MATSNU",
    8: "PGPCODER",    9: "Reveton",
    10: "TeslaCrypt", 11: "TrojanRansom"
}