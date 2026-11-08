# ============================================================
# ensemble.py - Combine sparse (LR+RF) and DeBERTa zero-day predictions
#
# Needs two files pairs in outputs/ :
#   sparse_test_probs.npy + sparse_test_idx.npy     (from sparse_baseline.py)
#   deberta_test_probs.npy + deberta_test_idx.npy   (from train.py --task binary, on Colab)
#
# Run from project root:  python model/ensemble.py
#
# The PRIMARY result is the plain 50/50 average - no weight is tuned on the
# test set. The weight sweep is printed only to show how sensitive it is.
# ============================================================

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.config import (FAMILY_MAP, OUTPUT_DIR, PAPER_TASK2_ACCURACY,
                          RAW_CSV, UNSEEN_FAMILIES)


def load_pair(prefix):
    probs = np.load(os.path.join(OUTPUT_DIR, f'{prefix}_test_probs.npy'))
    idx = np.load(os.path.join(OUTPUT_DIR, f'{prefix}_test_idx.npy'))
    return probs, idx


def report(name, y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    print(f"{name:22s} acc {accuracy_score(y, pred) * 100:5.2f}  "
          f"rec {recall_score(y, pred) * 100:5.2f}  "
          f"prec {precision_score(y, pred, zero_division=0) * 100:5.2f}  "
          f"f1 {f1_score(y, pred) * 100:5.2f}")
    return pred


def main():
    p_sp, idx_sp = load_pair('sparse')
    p_db, idx_db = load_pair('deberta')
    assert len(idx_sp) == len(idx_db) and (idx_sp == idx_db).all(), (
        "Test sets differ between the two models - both must use the same "
        "zero-day split (seed 42). Re-run whichever is out of date.")

    meta = pd.read_csv(RAW_CSV, header=None, encoding='latin-1',
                       usecols=[1, 2], dtype=np.int32).values
    y_all, fam_all = meta[:, 0], meta[:, 1]
    y, fam = y_all[idx_sp], fam_all[idx_sp]
    print(f"Test set: {len(y)} samples ({int(y.sum())} unseen ransomware, "
          f"{int((y == 0).sum())} goodware)\n")

    pred_sp = report('Sparse (LR+RF)', y, p_sp)
    pred_db = report('DeBERTa', y, p_db)
    p_ens = (p_sp + p_db) / 2
    pred_en = report('ENSEMBLE (50/50)', y, p_ens)
    print(f"{'paper':22s} acc {PAPER_TASK2_ACCURACY * 100:5.2f}")

    # --- do the two models make DIFFERENT mistakes? (that's what ensembling needs)
    wrong_sp, wrong_db = pred_sp != y, pred_db != y
    print("\nError overlap")
    print(f"  both wrong:         {int((wrong_sp & wrong_db).sum())}")
    print(f"  only sparse wrong:  {int((wrong_sp & ~wrong_db).sum())}")
    print(f"  only DeBERTa wrong: {int((~wrong_sp & wrong_db).sum())}")
    print(f"  both right:         {int((~wrong_sp & ~wrong_db).sum())}")

    print("\nDetection rate per unseen family")
    for name, pred in [('Sparse', pred_sp), ('DeBERTa', pred_db), ('Ensemble', pred_en)]:
        d = {FAMILY_MAP[c]: round(float(pred[fam == c].mean()), 3)
             for c in UNSEEN_FAMILIES}
        print(f"  {name:9s} {d}")

    print("\nWeight sweep (INFORMATION ONLY - do not pick the best one and report it)")
    for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
        pw = (1 - w) * p_sp + w * p_db
        print(f"  DeBERTa weight {w:.2f}: acc {accuracy_score(y, (pw >= 0.5).astype(int)) * 100:5.2f}")


if __name__ == '__main__':
    main()