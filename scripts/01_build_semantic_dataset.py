"""
SRDC v2 — Step 1: Raw feature extraction + semantic reconstruction.

Reads the raw Sgandurra/EldeRan CSV (30,970 columns: ID, Label, Family, then
30,967 binary features across API/DROP/REG/FILES/FILES_EXT/DIR/STR groups)
and reconstructs each sample into 7 natural-language "semantic texts"
(one per feature group).
"""
import re
import csv
import json
import os
from collections import defaultdict, Counter

RAW_DIR = "../raw_dataset_src"
OUT_DIR = "../data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

FAMILY_NAMES = {
    0: "Goodware", 1: "Critroni", 2: "CryptLocker", 3: "CryptoWall",
    4: "KOLLAH", 5: "Kovter", 6: "Locker", 7: "MATSNU", 8: "PGPCODER",
    9: "Reveton", 10: "TeslaCrypt", 11: "Trojan-Ransom",
}

MAX_FEATURES_PER_GROUP = 60
MAX_STR_LEN = 80


def camel_to_phrase(name: str) -> str:
    name = re.sub(r'(?<=[a-z])(?:A|W)$', '', name)
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', name)
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', s)
    return s.lower().strip()


def load_variable_names():
    cols = {}
    with open(os.path.join(RAW_DIR, "VariableNames.txt"), encoding="latin1") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if not line:
                continue
            idx, rest = line.split(";", 1)
            idx = int(idx)
            if idx <= 3:
                cols[idx] = ("META", rest)
                continue
            if ":" in rest:
                group, val = rest.split(":", 1)
            else:
                group, val = "OTHER", rest
            if group == "STR":
                val = re.sub(r'^\d+;', '', val)
            cols[idx] = (group, val)
    return cols


def semantic_for_group(group, raw_values):
    raw_values = raw_values[:MAX_FEATURES_PER_GROUP]
    if not raw_values:
        return ""
    if group == "API":
        phrases = sorted(set(camel_to_phrase(v) for v in raw_values))
        return ". ".join(f"called windows api to {p}" for p in phrases) + "."
    if group == "REG":
        parts = []
        for v in raw_values:
            if ":" in v:
                verb, path = v.split(":", 1)
            else:
                verb, path = "accessed", v
            verb_map = {"OPENED": "opened", "DELETED": "deleted",
                        "SETVALUE": "set a value in", "CREATED": "created",
                        "READ": "read"}
            verb_word = verb_map.get(verb, verb.lower())
            parts.append(f"{verb_word} registry key {path.strip(chr(92))}")
        return ". ".join(parts) + "."
    if group in ("FILES",):
        parts = []
        for v in raw_values:
            if ":" in v:
                verb, path = v.split(":", 1)
            else:
                verb, path = "accessed", v
            verb_map = {"OPENED": "opened", "DELETED": "deleted",
                        "READ": "read", "WRITTEN": "wrote to", "CREATED": "created"}
            verb_word = verb_map.get(verb, verb.lower())
            parts.append(f"{verb_word} the file {path.strip(chr(92))}")
        return ". ".join(parts) + "."
    if group == "DIR":
        parts = []
        for v in raw_values:
            if ":" in v:
                verb, path = v.split(":", 1)
            else:
                verb, path = "accessed", v
            verb_map = {"CREATED": "created directory", "ENUMERATED": "enumerated directory",
                        "DELETED": "deleted directory"}
            verb_word = verb_map.get(verb, f"{verb.lower()} directory")
            parts.append(f"{verb_word} {path.strip(chr(92))}")
        return ". ".join(parts) + "."
    if group == "FILES_EXT":
        exts = sorted(set(v.strip(".") for v in raw_values if v.strip(".")))
        return "accessed files with extensions: " + ", ".join(exts) + "."
    if group == "DROP":
        exts = sorted(set(v.strip(".") for v in raw_values if v.strip(".")))
        return "dropped new files with extensions: " + ", ".join(exts) + "."
    if group == "STR":
        vals = [v for v in raw_values if v and len(v) <= MAX_STR_LEN]
        vals = list(dict.fromkeys(vals))[:MAX_FEATURES_PER_GROUP]
        if not vals:
            return ""
        return "contained embedded strings: " + " | ".join(vals) + "."
    return ". ".join(raw_values) + "."


def main():
    col_meta = load_variable_names()
    group_of_col = {i: col_meta[i][0] for i in col_meta if i > 3}
    raw_name_of_col = {i: col_meta[i][1] for i in col_meta if i > 3}

    csv_path = os.path.join(RAW_DIR, "unzipped", "RansomwareData.csv")
    out_path = os.path.join(OUT_DIR, "samples.jsonl")

    n_written = 0
    family_counter = Counter()
    with open(csv_path, encoding="latin1", newline="") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        reader = csv.reader(fin)
        for row in reader:
            sample_id = row[0]
            label = int(row[1])
            family = int(row[2])
            active_by_group = defaultdict(list)
            for i in range(4, len(row) + 1):
                if i not in group_of_col:
                    continue
                if row[i - 1] == "1":
                    active_by_group[group_of_col[i]].append(raw_name_of_col[i])

            texts = {}
            for group in ["API", "DROP", "REG", "FILES", "FILES_EXT", "DIR", "STR"]:
                texts[group] = semantic_for_group(group, active_by_group.get(group, []))

            rec = {
                "id": sample_id,
                "label": label,
                "family": family,
                "family_name": FAMILY_NAMES.get(family, "UNKNOWN"),
                "texts": texts,
                "active_feature_counts": {g: len(active_by_group.get(g, [])) for g in
                                           ["API", "DROP", "REG", "FILES", "FILES_EXT", "DIR", "STR"]},
            }
            fout.write(json.dumps(rec) + "\n")
            n_written += 1
            family_counter[FAMILY_NAMES.get(family, "UNKNOWN")] += 1

    print(f"Wrote {n_written} samples to {out_path}")
    print("Family distribution:")
    for fam, cnt in sorted(family_counter.items(), key=lambda x: -x[1]):
        print(f"  {fam:15s} {cnt}")


if __name__ == "__main__":
    main()