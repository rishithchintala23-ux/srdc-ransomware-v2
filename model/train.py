# ============================================================
# train.py - Fine-tune DeBERTa-v3 for SRDC v2   (RUN ON COLAB, T4 GPU)
#
# TASK 1  family   12-class family classification, 4-fold stratified CV
# TASK 2  binary   zero-day detection: train on SEEN families + goodware,
#                  test on UNSEEN families + goodware (paper's Table 7 split)
#
# Usage (from project root):
#   python model/train.py --task binary                 # Task 2, saves model
#   python model/train.py --task family                 # Task 1, all 4 folds
#   python model/train.py --task family --folds 0,1     # only some folds
#   python model/train.py --task family --summarize     # pool saved folds
#   python model/train.py --task family --full          # train on ALL data, save
#
# No best-epoch selection on the test data: we train a fixed number of
# epochs and report the LAST epoch, so results are not inflated.
# ============================================================

import argparse
import glob
import json
import math
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.config import *            # noqa: F401,F403
from data.dataset import RansomwareDataset


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_class_weights(labels, num_classes, power=CLASS_WEIGHT_POWER):
    """weight_c = (1/count_c)**power, normalised to mean 1.
    Goodware(942) vs PGPCODER(4) is a >200x imbalance; sqrt-inverse
    weighting (power 0.5) lifts rare classes without exploding the loss."""
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0
    w = (1.0 / counts) ** power
    return w / w.mean()


def build_model(num_classes):
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_classes)
    return model.float()   # checkpoint is fp16; AMP needs fp32 master weights


@torch.no_grad()
def predict_proba(model, dataset, device, batch_size=32):
    """Softmax class probabilities, shape [N, num_classes]."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    use_amp = device.type == 'cuda'
    probs = []
    for batch in loader:
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=use_amp):
            logits = model(input_ids=ids, attention_mask=mask).logits
        probs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(probs, axis=0)


def predict(model, dataset, device, batch_size=32):
    return predict_proba(model, dataset, device, batch_size).argmax(axis=1)


def train_model(model, train_ds, class_weights, device, epochs):
    """Fixed-epoch fine-tuning with class-weighted CE, AMP, warmup + linear decay."""
    model.to(device)
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    steps_per_epoch = math.ceil(len(loader) / GRAD_ACCUM)
    total_steps = steps_per_epoch * epochs

    no_decay = ('bias', 'LayerNorm.weight')
    groups = [
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': WEIGHT_DECAY},
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(WARMUP_RATIO * total_steps), total_steps)

    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float, device=device))

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        running, seen = 0.0, 0
        for step, batch in enumerate(loader):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            y = batch['label'].to(device)

            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=use_amp):
                logits = model(input_ids=ids, attention_mask=mask).logits
            loss = loss_fn(logits.float(), y) / GRAD_ACCUM
            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            running += loss.item() * GRAD_ACCUM * len(y)
            seen += len(y)
        print(f"  epoch {epoch + 1}/{epochs}  train loss {running / seen:.4f}")
    return model


def free_gpu(*objs):
    for o in objs:
        del o
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def per_class_recall(y_true, y_pred):
    out = {}
    for c, name in FAMILY_MAP.items():
        mask = y_true == c
        if mask.sum() > 0:
            out[name] = round(float((y_pred[mask] == c).mean()), 4)
    return out


# ------------------------------------------------------------
# TASK 1: family classification, stratified k-fold CV
# ------------------------------------------------------------
def summarize_cv(n_samples):
    """Pool whatever fold files exist into one out-of-fold report."""
    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, 'cv', 'fold*.json')))
    if not files:
        print("No fold results found in outputs/cv/.")
        return None
    y_true, y_pred, fold_scores = [], [], {}
    for f in files:
        d = json.load(open(f))
        y_true += d['y_true']
        y_pred += d['y_pred']
        fold_scores[d['fold']] = d['balanced_accuracy']
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    pooled = balanced_accuracy_score(y_true, y_pred)

    print("\n" + "=" * 60)
    print(f"TASK 1 RESULTS  ({len(files)}/{CV_FOLDS} folds, {len(y_true)}/{n_samples} samples)")
    print("=" * 60)
    for k in sorted(fold_scores):
        print(f"  fold {k}: balanced acc {fold_scores[k] * 100:.2f}%")
    print(f"Pooled balanced accuracy: {pooled * 100:.2f}%")
    print(f"Paper target:             {PAPER_TASK1_BALANCED_ACC * 100:.2f}%")
    print("Per-class recall:")
    for name, r in per_class_recall(y_true, y_pred).items():
        print(f"  {name:14s} {r:.3f}")
    if len(files) == CV_FOLDS and pooled > PAPER_TASK1_BALANCED_ACC:
        print("STATUS: BEATS PAPER")
    result = {'pooled_balanced_accuracy': pooled, 'folds': fold_scores,
              'per_class_recall': per_class_recall(y_true, y_pred)}
    json.dump(result, open(os.path.join(OUTPUT_DIR, 'results_family.json'), 'w'), indent=2)
    return result


def run_family_cv(dataset, device, epochs, folds):
    labels = np.array(dataset.labels)
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    os.makedirs(os.path.join(OUTPUT_DIR, 'cv'), exist_ok=True)

    for fold, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        if fold not in folds:
            continue
        print(f"\n--- Fold {fold}: train {len(tr_idx)} | test {len(te_idx)} ---")
        set_seed(RANDOM_SEED + fold)
        weights = make_class_weights(labels[tr_idx], NUM_CLASSES)
        model = build_model(NUM_CLASSES)
        model = train_model(model, Subset(dataset, tr_idx), weights, device, epochs)
        probs = predict_proba(model, Subset(dataset, te_idx), device)
        preds = probs.argmax(axis=1)
        bal = balanced_accuracy_score(labels[te_idx], preds)
        print(f"  fold {fold} balanced accuracy: {bal * 100:.2f}%")
        json.dump({'fold': fold, 'balanced_accuracy': bal,
                   'y_true': labels[te_idx].tolist(), 'y_pred': preds.tolist(),
                   'te_idx': te_idx.tolist(),
                   'y_prob': np.round(probs, 5).tolist()},
                  open(os.path.join(OUTPUT_DIR, 'cv', f'fold{fold}.json'), 'w'))
        free_gpu(model)
    return summarize_cv(len(labels))


def run_family_full(dataset, tokenizer, device, epochs):
    """Train on ALL samples and save - this is the model you deploy."""
    labels = np.array(dataset.labels)
    set_seed(RANDOM_SEED)
    model = build_model(NUM_CLASSES)
    model = train_model(model, dataset, make_class_weights(labels, NUM_CLASSES),
                        device, epochs)
    out = os.path.join(MODEL_SAVE, 'family', 'final')
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"Saved full family model -> {out}")


# ------------------------------------------------------------
# TASK 2: zero-day detection (seen/unseen family holdout)
# ------------------------------------------------------------
def zero_day_split(df):
    fam = df['family'].values
    rng = np.random.RandomState(RANDOM_SEED)
    unseen_idx = np.where(np.isin(fam, UNSEEN_FAMILIES))[0]
    seen_idx = np.where(np.isin(fam, SEEN_FAMILIES))[0]
    good_idx = np.where(fam == 0)[0]
    rng.shuffle(good_idx)
    n = len(unseen_idx)                        # equal goodware in the test set
    test_idx = np.concatenate([unseen_idx, good_idx[:n]])
    train_idx = np.concatenate([seen_idx, good_idx[n:]])
    return train_idx, test_idx


def run_binary(dataset, tokenizer, device, epochs):
    labels = np.array(dataset.labels)           # 0 goodware / 1 ransomware
    fam = dataset.df['family'].values
    train_idx, test_idx = zero_day_split(dataset.df)
    print(f"Train: {len(train_idx)} ({int(labels[train_idx].sum())} ransomware, "
          f"{int((labels[train_idx] == 0).sum())} goodware)")
    print(f"Test:  {len(test_idx)} ({int(labels[test_idx].sum())} UNSEEN ransomware, "
          f"{int((labels[test_idx] == 0).sum())} goodware)")

    set_seed(RANDOM_SEED)
    model = build_model(2)
    model = train_model(model, Subset(dataset, train_idx),
                        make_class_weights(labels[train_idx], 2), device, epochs)
    probs = predict_proba(model, Subset(dataset, test_idx), device)
    preds = probs.argmax(axis=1)
    y = labels[test_idx]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, 'deberta_test_probs.npy'), probs[:, 1])
    np.save(os.path.join(OUTPUT_DIR, 'deberta_test_idx.npy'), test_idx)

    acc = accuracy_score(y, preds)
    rec = recall_score(y, preds, zero_division=0)
    prec = precision_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)

    print("\n" + "=" * 60)
    print("TASK 2 RESULTS (zero-day detection, unseen families)")
    print("=" * 60)
    print(f"Accuracy:  {acc * 100:.2f}%   (paper {PAPER_TASK2_ACCURACY * 100:.0f}%)")
    print(f"Recall:    {rec * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"F1:        {f1 * 100:.2f}%   (paper {PAPER_TASK2_F1 * 100:.0f}%)")
    per_fam = {}
    for c in UNSEEN_FAMILIES:
        m = fam[test_idx] == c
        per_fam[FAMILY_MAP[c]] = round(float((preds[m] == 1).mean()), 4)
    print("Detection rate per unseen family:", per_fam)
    if acc > PAPER_TASK2_ACCURACY:
        print("STATUS: BEATS PAPER")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    json.dump({'accuracy': acc, 'recall': rec, 'precision': prec, 'f1': f1,
               'per_unseen_family_detection': per_fam},
              open(os.path.join(OUTPUT_DIR, 'results_binary.json'), 'w'), indent=2)

    out = os.path.join(MODEL_SAVE, 'binary', 'final')
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"Saved binary model -> {out}")


# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=['family', 'binary'], required=True)
    ap.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    ap.add_argument('--folds', default=','.join(str(i) for i in range(CV_FOLDS)),
                    help="comma-separated fold ids for the family task, e.g. 0,1")
    ap.add_argument('--full', action='store_true',
                    help="family task: train on all data and save the model")
    ap.add_argument('--summarize', action='store_true',
                    help="family task: pool saved fold results and exit")
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  | model: {MODEL_NAME}  | epochs: {args.epochs}")
    if device.type == 'cpu':
        print("WARNING: no GPU found - run this on Colab (Runtime > T4 GPU).")

    if args.task == 'family' and args.summarize:
        n = len(pd.read_csv(PROCESSED_CSV, usecols=['label']))
        summarize_cv(n)
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = RansomwareDataset(PROCESSED_CSV, tokenizer, task=args.task)

    if args.task == 'binary':
        run_binary(dataset, tokenizer, device, args.epochs)
    elif args.full:
        run_family_full(dataset, tokenizer, device, args.epochs)
    else:
        folds = [int(f) for f in args.folds.split(',')]
        run_family_cv(dataset, device, args.epochs, folds)


if __name__ == '__main__':
    main()