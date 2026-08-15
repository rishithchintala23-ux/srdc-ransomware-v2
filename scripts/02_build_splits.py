"""
SRDC v2 — Step 2: Build evaluation splits.

Task 1: Stratified 4-fold CV (k=4, forced by PGPCODER's 4 samples).
Task 2: Seen/unseen family holdout matching the paper's Table 7.
"""
import json
import random
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold

IN_PATH = "../data/processed/samples.jsonl"
OUT_TASK1 = "../data/processed/task1_splits.json"
OUT_TASK2 = "../data/processed/task2_splits.json"

TASK2_SEEN = {"Critroni", "CryptLocker", "CryptoWall", "KOLLAH", "Kovter", "Locker", "MATSNU"}
TASK2_UNSEEN = {"PGPCODER", "Reveton", "TeslaCrypt", "Trojan-Ransom"}


def main():
    samples = []
    with open(IN_PATH) as f:
        for line in f:
            samples.append(json.loads(line))

    ids = [s["id"] for s in samples]
    families = [s["family"] for s in samples]

    class_counts = defaultdict(int)
    for f in families:
        class_counts[f] += 1
    k = min(class_counts.values())
    print(f"Task 1: using k={k} folds (smallest class has {k} samples)")

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold_of_id = {}
    for fold_idx, (_, test_idx) in enumerate(skf.split(ids, families)):
        for i in test_idx:
            fold_of_id[ids[i]] = fold_idx

    with open(OUT_TASK1, "w") as f:
        json.dump({"k": k, "fold_of_id": fold_of_id}, f, indent=2)
    print(f"Wrote Task 1 splits -> {OUT_TASK1}")

    test_ids = [s["id"] for s in samples if s["family_name"] in TASK2_UNSEEN]
    n_unseen_ransom = len(test_ids)

    goodware_ids = [s["id"] for s in samples if s["family_name"] == "Goodware"]
    random.Random(42).shuffle(goodware_ids)
    goodware_test = goodware_ids[:n_unseen_ransom]
    goodware_train = goodware_ids[n_unseen_ransom:]

    train_ids = [s["id"] for s in samples if s["family_name"] in TASK2_SEEN] + goodware_train
    test_ids = test_ids + goodware_test

    with open(OUT_TASK2, "w") as f:
        json.dump({"seen_families": sorted(TASK2_SEEN), "unseen_families": sorted(TASK2_UNSEEN),
                   "train_ids": train_ids, "test_ids": test_ids}, f, indent=2)
    print(f"Wrote Task 2 splits -> {OUT_TASK2}")
    print(f"  train: {len(train_ids)} ({len(train_ids)-len(goodware_train)} ransomware + {len(goodware_train)} goodware)")
    print(f"  test:  {len(test_ids)} ({n_unseen_ransom} ransomware + {len(goodware_test)} goodware)")


if __name__ == "__main__":
    main()