# ============================================================
# sparse_baseline.py - Sparse-feature ensemble for zero-day detection
#
# Why this exists: a logistic regression on the raw 30,967 binary features
# already beats DeBERTa-on-truncated-text (it sees EVERY feature; DeBERTa
# only sees ~460 tokens). Industry systems combine both views, so this is
# the second half of the final ensemble:
#     sparse model (LR + RandomForest)  +  semantic model (DeBERTa)
#
# Runs on CPU in a couple of minutes. Run from project root:
#   python model/sparse_baseline.py                 # everything
#   python model/sparse_baseline.py --skip_seeds    # faster
#   python model/sparse_baseline.py --skip_lofo
#
# EVALUATION IS LEAKAGE-FREE:
#   * the LR regularisation strength C is chosen ONLY on the training set,
#     with leave-one-SEEN-family-out validation (a pseudo zero-day test)
#   * the test set (unseen families) is touched once, at the end
# ============================================================

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.config import (FAMILY_MAP, OUTPUT_DIR, PAPER_TASK2_ACCURACY,
                          RANDOM_SEED, RAW_CSV, SEEN_FAMILIES,
                          UNSEEN_FAMILIES)

C_GRID = (0.1, 0.3, 1.0, 3.0)
N_TREES = 300


# ------------------------------------------------------------
def load_raw():
    try:
        mat = pd.read_csv(RAW_CSV, header=None, encoding='latin-1',
                          dtype=np.int32).values
    except ValueError:
        mat = pd.read_csv(RAW_CSV, header=None, encoding='latin-1',
                          skiprows=1, dtype=np.int32).values
    y = mat[:, 1]
    fam = mat[:, 2]
    X = sparse.csr_matrix(mat[:, 3:].astype(np.float32))
    return X, y, fam


def zero_day_split(fam, seed):
    """Same logic as train.py (seed 42 gives the identical split)."""
    unseen = np.where(np.isin(fam, UNSEEN_FAMILIES))[0]
    seen = np.where(np.isin(fam, SEEN_FAMILIES))[0]
    good = np.where(fam == 0)[0]
    np.random.RandomState(seed).shuffle(good)
    n = len(unseen)
    return np.concatenate([seen, good[n:]]), np.concatenate([unseen, good[:n]])


def wilson_ci(correct, n, z=1.96):
    p = correct / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return 100 * (centre - half), 100 * (centre + half)


def new_lr(C):
    return LogisticRegression(C=C, max_iter=3000, class_weight='balanced')


def new_rf():
    return RandomForestClassifier(N_TREES, n_jobs=-1, class_weight='balanced',
                                  random_state=0)


def ensemble_proba(lr, rf, X):
    return (lr.predict_proba(X)[:, 1] + rf.predict_proba(X)[:, 1]) / 2


# ------------------------------------------------------------
def select_C(X, y, fam, train_idx):
    """Leave-one-SEEN-family-out CV on the training set only."""
    rng = np.random.RandomState(0)
    tr_fam = fam[train_idx]
    scores = {C: [] for C in C_GRID}
    for held in SEEN_FAMILIES:
        pos_val = train_idx[tr_fam == held]
        rest = train_idx[tr_fam != held]
        good_pool = rest[fam[rest] == 0]
        val_good = rng.choice(good_pool, size=len(pos_val), replace=False)
        fit_idx = np.setdiff1d(rest, val_good)
        val_idx = np.concatenate([pos_val, val_good])
        for C in C_GRID:
            m = new_lr(C).fit(X[fit_idx], y[fit_idx])
            scores[C].append(accuracy_score(y[val_idx], m.predict(X[val_idx])))
    mean = {C: float(np.mean(v)) for C, v in scores.items()}
    best = max(mean, key=mean.get)
    print("C selection (leave-one-seen-family-out, training data only):")
    for C in C_GRID:
        print(f"  C={C:<4} mean pseudo-zero-day acc {mean[C] * 100:.2f}%"
              + ("   <-- chosen" if C == best else ""))
    return best


def metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }


def line(name, m):
    return (f"{name:12s} acc {m['accuracy'] * 100:5.2f}  rec {m['recall'] * 100:5.2f}  "
            f"prec {m['precision'] * 100:5.2f}  f1 {m['f1'] * 100:5.2f}")


# ------------------------------------------------------------
def main_eval(X, y, fam, C):
    tr, te = zero_day_split(fam, RANDOM_SEED)
    print(f"\nMAIN ZERO-DAY EVAL  (train {len(tr)}, test {len(te)})")
    lr = new_lr(C).fit(X[tr], y[tr])
    rf = new_rf().fit(X[tr], y[tr])
    p_lr = lr.predict_proba(X[te])[:, 1]
    p_rf = rf.predict_proba(X[te])[:, 1]
    p_ens = (p_lr + p_rf) / 2
    out = {}
    for name, p in [('LogReg', p_lr), ('RandomForest', p_rf), ('LR+RF', p_ens)]:
        pred = (p >= 0.5).astype(int)
        m = metrics(y[te], pred)
        out[name] = m
        print(line(name, m))
    pred = (p_ens >= 0.5).astype(int)
    correct = int((pred == y[te]).sum())
    lo, hi = wilson_ci(correct, len(te))
    print(f"LR+RF accuracy {correct}/{len(te)} = {correct / len(te) * 100:.2f}%  "
          f"(95% CI {lo:.1f}-{hi:.1f}%)   paper: {PAPER_TASK2_ACCURACY * 100:.0f}%")
    per_fam = {FAMILY_MAP[c]: round(float(pred[fam[te] == c].mean()), 3)
               for c in UNSEEN_FAMILIES}
    print("Detection rate per unseen family:", per_fam)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, 'sparse_test_idx.npy'), te)
    np.save(os.path.join(OUTPUT_DIR, 'sparse_test_probs.npy'), p_ens)
    return out, per_fam


def multi_seed(X, y, fam, C, seeds=range(10)):
    print("\nSTABILITY: 10 different goodware draws for the test set")
    accs = []
    for s in seeds:
        tr, te = zero_day_split(fam, s)
        lr = new_lr(C).fit(X[tr], y[tr])
        rf = new_rf().fit(X[tr], y[tr])
        pred = (ensemble_proba(lr, rf, X[te]) >= 0.5).astype(int)
        accs.append(accuracy_score(y[te], pred) * 100)
    print("  accuracy per draw:", [round(a, 1) for a in accs])
    print(f"  mean {np.mean(accs):.2f}%  std {np.std(accs):.2f}%  "
          f"min {np.min(accs):.1f}%  max {np.max(accs):.1f}%")
    return accs


def leave_one_family_out(X, y, fam, C):
    """Strictest zero-day test: hold out ONE family completely, train on the other 10."""
    print("\nLEAVE-ONE-FAMILY-OUT ZERO-DAY (each family is 'never seen')")
    rng = np.random.RandomState(RANDOM_SEED)
    rows = {}
    for f in range(1, 12):
        pos = np.where(fam == f)[0]
        good = np.where(fam == 0)[0].copy()
        rng.shuffle(good)
        test_good, train_good = good[:len(pos)], good[len(pos):]
        train_pos = np.where((fam != 0) & (fam != f))[0]
        tr = np.concatenate([train_pos, train_good])
        te = np.concatenate([pos, test_good])
        lr = new_lr(C).fit(X[tr], y[tr])
        rf = new_rf().fit(X[tr], y[tr])
        pred = (ensemble_proba(lr, rf, X[te]) >= 0.5).astype(int)
        det = float(pred[:len(pos)].mean())          # ransomware detected
        spec = float(1 - pred[len(pos):].mean())     # goodware correctly passed
        rows[FAMILY_MAP[f]] = {'n': int(len(pos)), 'detection': det, 'goodware_ok': spec}
        print(f"  {FAMILY_MAP[f]:13s} n={len(pos):3d}  detected {det * 100:5.1f}%   goodware passed {spec * 100:5.1f}%")
    mean_det = np.mean([r['detection'] for r in rows.values()])
    print(f"  mean detection across families: {mean_det * 100:.1f}%")
    return rows


# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip_seeds', action='store_true')
    ap.add_argument('--skip_lofo', action='store_true')
    args = ap.parse_args()

    X, y, fam = load_raw()
    print(f"Loaded {X.shape[0]} samples x {X.shape[1]} binary features")
    tr, _ = zero_day_split(fam, RANDOM_SEED)
    C = select_C(X, y, fam, tr)

    results = {'chosen_C': C}
    results['main'], results['per_unseen_family'] = main_eval(X, y, fam, C)
    if not args.skip_seeds:
        results['seed_accuracies'] = multi_seed(X, y, fam, C)
    if not args.skip_lofo:
        results['leave_one_family_out'] = leave_one_family_out(X, y, fam, C)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, 'results_sparse.json'), 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved {OUTPUT_DIR}/results_sparse.json")


if __name__ == '__main__':
    main()