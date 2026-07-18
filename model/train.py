# ============================================================
# train.py — Fine-tune DeBERTa-v3 on ransomware dataset
#
# WHAT FINE-TUNING MEANS:
#   DeBERTa-v3 was pre-trained on 160GB of text.
#   It already understands English deeply.
#   Fine-tuning = update its weights on OUR specific task
#   (ransomware classification) with a small learning rate.
#   Result: DeBERTa learns ransomware patterns fast.
#
# WHY SMALL LEARNING RATE (2e-5):
#   Large LR = destroy pre-trained knowledge (catastrophic forgetting)
#   Small LR = gently update toward ransomware understanding
#
# RUN ON GOOGLE COLAB (T4 GPU):
#   Runtime → Change runtime → T4 GPU
#   Then run: python model/train.py
# ============================================================

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import pandas as pd

from model.config import *
from data.dataset import RansomwareDataset

# ── Metrics ──────────────────────────────────────────────────
def compute_metrics(eval_pred):
    """
    Called after each epoch to evaluate the model.

    WHY BALANCED ACCURACY:
    Dataset is imbalanced (942 goodware, 4 PGPCODER).
    Regular accuracy would give 99% by predicting everything
    as goodware. Balanced accuracy treats all classes equally.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    bal_acc = balanced_accuracy_score(labels, preds)
    f1      = f1_score(labels, preds, average='weighted',
                       zero_division=0)
    return {
        'balanced_accuracy': bal_acc,
        'f1_weighted':       f1,
    }

def train(task: str = 'family'):
    """
    Fine-tune DeBERTa-v3 for ransomware classification.

    Args:
        task: 'family' (12-class) or 'binary' (2-class)
    """
    print("="*60)
    print(f"SRDC v2 — Fine-tuning DeBERTa-v3")
    print(f"Task: {task}")
    print(f"Model: {MODEL_NAME}")
    print("="*60)

    # ── Step 1: Load tokenizer ────────────────────────────────
    # WHY TOKENIZER:
    # Converts text → token IDs DeBERTa understands
    # Example: "crypt encrypt" → [2, 15234, 9821, 3]
    print("\n[1/5] Loading DeBERTa tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ── Step 2: Load dataset ──────────────────────────────────
    print("[2/5] Loading dataset...")
    dataset = RansomwareDataset(
        csv_path  = "data/processed/processed.csv",
        tokenizer = tokenizer,
        task      = task,
    )

    # Stratified train/val split
    # WHAT STRATIFIED MEANS:
    # Keeps same class proportions in train and val sets.
    # Without stratification, PGPCODER (4 samples) might
    # all end up in train set with none in val.
    labels  = [dataset.df.iloc[i]['family'] for i in range(len(dataset))]
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(
        indices,
        test_size    = 1 - TRAIN_SPLIT,
        stratify     = labels,
        random_state = RANDOM_SEED,
    )
    train_ds = Subset(dataset, train_idx)
    val_ds   = Subset(dataset, val_idx)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Step 3: Load model ────────────────────────────────────
    # WHY AutoModelForSequenceClassification:
    # Takes DeBERTa-v3 and adds a classification head on top.
    # The head is: Linear(768 → num_classes)
    # DeBERTa [CLS] token → classification head → class scores
    num_classes = NUM_CLASSES if task == 'family' else 2
    print(f"[3/5] Loading DeBERTa-v3 ({num_classes} classes)...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels = num_classes,
        ignore_mismatched_sizes = True,
    )

    # ── Step 4: Training arguments ────────────────────────────
    # WHY THESE SETTINGS:
    # evaluation_strategy='epoch': evaluate after every epoch
    # load_best_model_at_end: keep the epoch with best val accuracy
    # fp16=True: use 16-bit floats → 2x faster on GPU, same accuracy
    # warmup_ratio=0.1: slowly increase LR for first 10% of training
    #   → prevents early instability
    output_dir = f"{MODEL_SAVE}/{task}"
    args = TrainingArguments(
        output_dir              = output_dir,
        num_train_epochs        = NUM_EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        learning_rate           = LEARNING_RATE,
        weight_decay            = WEIGHT_DECAY,
        warmup_ratio            = WARMUP_RATIO,
        evaluation_strategy     = "epoch",
        save_strategy           = "epoch",
        load_best_model_at_end  = True,
        metric_for_best_model   = "balanced_accuracy",
        greater_is_better       = True,
        fp16                    = torch.cuda.is_available(),
        logging_dir             = LOG_DIR,
        logging_steps           = 50,
        report_to               = "none",
    )

    # ── Step 5: Train ─────────────────────────────────────────
    trainer = Trainer(
        model           = model,
        args            = args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        compute_metrics = compute_metrics,
        callbacks       = [
            # Stop training if no improvement for 3 epochs
            EarlyStoppingCallback(early_stopping_patience=3)
        ],
    )

    print(f"[4/5] Training...")
    print(f"GPU available: {torch.cuda.is_available()}")
    trainer.train()

    # ── Step 6: Save ──────────────────────────────────────────
    print(f"[5/5] Saving model to {output_dir}/final...")
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")

    # Final evaluation
    results = trainer.evaluate()
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Balanced Accuracy: {results['eval_balanced_accuracy']*100:.2f}%")
    print(f"F1 Weighted:       {results['eval_f1_weighted']*100:.2f}%")
    print(f"Paper target:      54.83%")
    if results['eval_balanced_accuracy'] > 0.5483:
        print("STATUS: BEATS PAPER!")
    return results

if __name__ == "__main__":
    # Train family classifier (Task 1)
    train(task='family')
    # Train binary detector (Task 2)
    train(task='binary')