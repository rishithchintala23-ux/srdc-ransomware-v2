# ============================================================
# process.py - Convert Sgandurra binary features -> text
#
# The raw CSV has no header. Column meanings come from VariableNames.txt.
# Row layout: [ID, Label, Family, 30,967 binary feature flags]
# Each active (=1) feature is turned into a short phrase and grouped
# into 7 categories: api, files, ext, reg, dir, drop, str.
#
# Run from the project root:  python data/process.py
# ============================================================

import os
import re

import numpy as np
import pandas as pd

RAW_CSV      = "data/raw/RansomwareData.csv"
RAW_VARNAMES = "data/raw/VariableNames.txt"
OUT_PATH     = "data/processed/processed.csv"

FAMILY_MAP = {
    0: 'Goodware', 1: 'Citroni', 2: 'CryptLocker', 3: 'CryptoWall',
    4: 'KOLLAH', 5: 'Kovter', 6: 'Locker', 7: 'MATSNU',
    8: 'PGPCODER', 9: 'Reveton', 10: 'TeslaCrypt', 11: 'TrojanRansom',
}

CATS = ['api', 'files', 'ext', 'reg', 'dir', 'drop', 'str']
MAX_STR_CHARS = 80   # embedded strings longer than this are binary junk


# ---------- name -> phrase helpers ----------

def camel_to_words(name):
    name = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', name)
    name = re.sub(r'([a-z\d])([A-Z])',      r'\1 \2', name)
    return name.lower().strip()


def strip_suffix(name):
    """Drop the Win32 ANSI/Wide suffix (CreateFileA / CreateFileW -> CreateFile)."""
    if len(name) > 2 and name[-1] in ('W', 'A') and name[-2].islower():
        return name[:-1]
    return name


def clean_path(path):
    """Make registry/file paths readable and token-cheap:
    random GUIDs -> 'guid', backslashes -> spaces, collapse whitespace."""
    path = re.sub(r'\{[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}\}', 'guid', path)
    path = path.replace('\\', ' ')
    return re.sub(r'\s+', ' ', path).strip()


def get_category(col_name):
    u = col_name.upper()
    # FILES_EXT must be checked before FILES (shared prefix)
    if u.startswith('FILES_EXT:'):                       return 'ext'
    if u.startswith('API:'):                             return 'api'
    if u.startswith('FILES:') or u.startswith('FILE:'): return 'files'
    if u.startswith('EXT:'):                             return 'ext'
    if u.startswith('REG:'):                             return 'reg'
    if u.startswith('DIR:'):                             return 'dir'
    if u.startswith('DROP:'):                            return 'drop'
    return 'str'


def col_to_phrase(col_name):
    col = col_name.strip()
    u = col.upper()

    if u.startswith('API:'):
        return camel_to_words(strip_suffix(col.split(':', 1)[1]))

    if u.startswith('REG:'):
        parts = col.split(':', 2)
        verb = parts[1].lower() if len(parts) > 1 else 'accessed'
        key  = clean_path(parts[2]) if len(parts) > 2 else ''
        return f"{verb} registry {key}".strip()

    if u.startswith('FILES_EXT:'):
        parts = col.split(':', 2)
        verb = parts[1].lower() if len(parts) > 2 else 'accessed'
        ext  = (parts[2] if len(parts) > 2 else parts[-1]).strip().lstrip('.')
        return f"{verb} {ext} file"

    if u.startswith('FILES:') or u.startswith('FILE:'):
        parts = col.split(':', 2)
        verb = parts[1].lower() if len(parts) > 1 else 'accessed'
        path = clean_path(parts[2]) if len(parts) > 2 else ''
        return f"{verb} file {path}".strip()

    if u.startswith('DROP:'):
        return f"dropped file {col.split(':', 1)[1]}".strip()

    if u.startswith('DIR:'):
        parts = col.split(':', 2)
        verb = parts[1].lower() if len(parts) > 1 else 'accessed'
        path = clean_path(parts[2]) if len(parts) > 2 else ''
        return f"{verb} directory {path}".strip()

    if u.startswith('EXT:'):
        return f"accessed {col.split(':', 1)[1].lstrip('.')} files"

    if u.startswith('STR:'):
        raw = col.split(':', 1)[1]
        # STR values are stored as "<internal_index>;<string>" - drop the index
        val = re.sub(r'^\d+;', '', raw)
        return val.strip()

    return camel_to_words(col)


# ---------- loading ----------

def load_variable_names(path):
    names = []
    with open(path, encoding='latin-1') as f:
        for line in f:
            line = line.rstrip('\n\r')
            if not line:
                continue
            parts = line.split(';', 1)
            names.append(parts[1] if len(parts) > 1 else parts[0])
    return names


def load_matrix(path):
    """Read the headerless CSV as an int matrix (tolerates a stray header row)."""
    try:
        return pd.read_csv(path, header=None, encoding='latin-1',
                           dtype=np.int32).values
    except ValueError:
        print("Non-numeric first row detected - skipping it as a header.")
        return pd.read_csv(path, header=None, encoding='latin-1', skiprows=1,
                           dtype=np.int32).values


def process_dataset(input_csv=RAW_CSV, varnames_txt=RAW_VARNAMES,
                    output_path=OUT_PATH):
    print("Loading variable names...")
    col_names = load_variable_names(varnames_txt)
    print(f"Total columns in VariableNames.txt: {len(col_names)}")

    print("Loading dataset matrix...")
    mat = load_matrix(input_csv)
    print(f"Matrix shape: {mat.shape}")
    assert mat.shape[1] == len(col_names), (
        f"CSV has {mat.shape[1]} columns but VariableNames.txt has "
        f"{len(col_names)} entries")

    sample_ids = mat[:, 0]
    labels     = mat[:, 1]
    families   = mat[:, 2]
    feats      = mat[:, 3:]
    feat_names = col_names[3:]
    print(f"Feature columns: {feats.shape[1]}")

    print("Pre-computing feature phrases...")
    categories = [get_category(c) for c in feat_names]
    phrases    = [col_to_phrase(c) for c in feat_names]

    print("Building text features...")
    records = []
    for i in range(mat.shape[0]):
        groups = {c: [] for c in CATS}
        for j in np.flatnonzero(feats[i] == 1):
            phrase = phrases[j]
            if categories[j] == 'str' and (not phrase or len(phrase) > MAX_STR_CHARS):
                continue
            groups[categories[j]].append(phrase)

        rec = {
            'sample_id':   sample_ids[i],
            'label':       int(labels[i]),
            'family':      int(families[i]),
            'family_name': FAMILY_MAP.get(int(families[i]), 'Unknown'),
        }
        for c in CATS:
            uniq = list(dict.fromkeys(groups[c]))      # de-duplicate, keep order
            rec[f'{c}Features'] = ' , '.join(uniq) if uniq else 'no activity'
        records.append(rec)

    out_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_csv(output_path, index=False)

    print(f"\nSaved: {output_path}")
    print(f"Shape: {out_df.shape}")
    print("\nFamily distribution:")
    print(out_df['family_name'].value_counts())
    print("\nSample apiFeatures (row 1):")
    print(out_df['apiFeatures'].iloc[1][:300])
    print("\nSample extFeatures (row 1):")
    print(out_df['extFeatures'].iloc[1][:200])
    return out_df


if __name__ == "__main__":
    process_dataset()