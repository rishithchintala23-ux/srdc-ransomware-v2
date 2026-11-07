# ============================================================
# config.py - Central configuration for SRDC v2
# Change settings here, they apply everywhere.
# ============================================================

# ---- Model ----
MODEL_NAME  = "microsoft/deberta-v3-base"
NUM_CLASSES = 12          # 11 ransomware families + goodware
MAX_LENGTH  = 512         # DeBERTa max tokens (split across 7 feature groups)

# ---- Training ----
# Effective batch size = BATCH_SIZE * GRAD_ACCUM = 16.
# BATCH_SIZE 8 keeps a 512-token DeBERTa-base inside a 16 GB T4.
BATCH_SIZE    = 8
GRAD_ACCUM    = 2
LEARNING_RATE = 2e-5
NUM_EPOCHS    = 8
WARMUP_RATIO  = 0.1
WEIGHT_DECAY  = 0.01
RANDOM_SEED   = 42

# Class weight strength for the family task: weight = (1/count) ** CLASS_WEIGHT_POWER
# 0 = no weighting, 1 = full inverse frequency (too aggressive for 4-sample classes)
CLASS_WEIGHT_POWER = 0.5

# ---- Evaluation ----
CV_FOLDS = 4              # forced by PGPCODER having only 4 samples

# Zero-day split (matches the paper's Table 7)
SEEN_FAMILIES   = [1, 2, 3, 4, 5, 6, 7]   # Citroni, CryptLocker, CryptoWall, KOLLAH, Kovter, Locker, MATSNU
UNSEEN_FAMILIES = [8, 9, 10, 11]          # PGPCODER, Reveton, TeslaCrypt, TrojanRansom

# ---- Paths ----
RAW_CSV       = "data/raw/RansomwareData.csv"
RAW_VARNAMES  = "data/raw/VariableNames.txt"
PROCESSED_CSV = "data/processed/processed.csv"
OUTPUT_DIR    = "outputs"
MODEL_SAVE    = "outputs/model"

# ---- Family mapping ----
FAMILY_MAP = {
    0: "Goodware",    1: "Citroni",
    2: "CryptLocker", 3: "CryptoWall",
    4: "KOLLAH",      5: "Kovter",
    6: "Locker",      7: "MATSNU",
    8: "PGPCODER",    9: "Reveton",
    10: "TeslaCrypt", 11: "TrojanRansom",
}

# ---- Paper targets (AAAI-25 SRDC) ----
PAPER_TASK1_BALANCED_ACC = 0.5483
PAPER_TASK2_ACCURACY     = 0.96
PAPER_TASK2_F1           = 0.96