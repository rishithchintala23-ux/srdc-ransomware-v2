# ============================================================
# process.py — Convert Sgandurra dataset → 7 text columns
# Handles the exact format of VariableNames.txt
# ============================================================

import pandas as pd
import re
import os
from tqdm import tqdm

# ── Load column names from VariableNames.txt ─────────────────
def load_column_names(var_path: str) -> list:
    """
    VariableNames.txt format:
      1;ID
      2;Label (1 Ransomware / 0 Goodware)
      3;Ransomware Family
      4;API:GetSystemDirectoryA
      30966;STR:16263;MP_PERSISTEDSTORE_NOTIFY_DISABLE
              ↑ STR cols have extra internal index — strip it!
    """
    cols = []
    with open(var_path, encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(';', 1)          # split on FIRST ; only
            name  = parts[1] if len(parts) > 1 else parts[0]

            # STR columns: "STR:16263;MP_PERSIST..." → "STR:MP_PERSIST..."
            if name.upper().startswith('STR:'):
                after = name[4:]                # remove "STR:"
                # if next part looks like a number;name, strip the number
                sub = after.split(';', 1)
                if len(sub) == 2 and sub[0].isdigit():
                    name = 'STR:' + sub[1]

            cols.append(name)
    return cols

# ── Text helpers ──────────────────────────────────────────────
def camel_to_words(name: str) -> str:
    name = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', name)
    name = re.sub(r'([a-z\d])([A-Z])',      r'\1 \2', name)
    return name.lower().strip()

def strip_suffix(name: str) -> str:
    if len(name) > 2 and name[-1] in ('W','A') and name[-2].islower():
        return name[:-1]
    return name

def col_to_phrase(col: str) -> str:
    u = col.upper()
    if u.startswith('API:'):
        return camel_to_words(strip_suffix(col[4:]))
    if u.startswith('REG:'):
        parts = col.split(':', 2)
        verb  = parts[1].lower() if len(parts) > 1 else 'accessed'
        key   = parts[2].replace('\\', ' ') if len(parts) > 2 else ''
        return f"{verb} registry {key}".strip()
    if u.startswith('FILES:') or u.startswith('FILE:'):
        parts = col.split(':', 2)
        verb  = parts[1].lower() if len(parts) > 1 else 'accessed'
        path  = parts[2] if len(parts) > 2 else ''
        return f"{verb} file {path}".strip()
    if u.startswith('DROP:'):
        return f"dropped file {col[5:]}".strip()
    if u.startswith('DIR:'):
        parts = col.split(':', 2)
        verb  = parts[1].lower() if len(parts) > 1 else 'accessed'
        path  = parts[2] if len(parts) > 2 else ''
        return f"{verb} directory {path}".strip()
    if u.startswith('EXT:'):
        return f"opened extension {col[4:].lstrip('.')}"
    if u.startswith('STR:'):
        return col[4:].lower().replace('_', ' ').replace(';', ' ')
    return camel_to_words(col)

def get_category(col: str) -> str:
    u = col.upper()
    if u.startswith('API:'):   return 'api'
    if u.startswith('FILES:') or u.startswith('FILE:'): return 'files'
    if u.startswith('EXT:'):   return 'ext'
    if u.startswith('REG:'):   return 'reg'
    if u.startswith('DIR:'):   return 'dir'
    if u.startswith('DROP:'):  return 'drop'
    return 'str'

# ── Main ──────────────────────────────────────────────────────
def process_dataset(csv_path: str, var_path: str, out_path: str):
    print("Loading column names...")
    all_cols = load_column_names(var_path)
    print(f"Total columns: {len(all_cols)}")
    print(f"First 5: {all_cols[:5]}")
    print(f"Last  5: {all_cols[-5:]}")

    print("\nLoading dataset...")
    df = pd.read_csv(csv_path, header=None, names=all_cols)
    print(f"Shape: {df.shape}")

    # First 3 cols are metadata
    label_cols   = ['ID',
                    'Label (1 Ransomware / 0 Goodware)',
                    'Ransomware Family']
    feature_cols = [c for c in all_cols if c not in label_cols]
    print(f"Feature cols: {len(feature_cols)}")

    # Pre-compute phrase for each feature column
    print("\nPre-computing phrases...")
    col_info = {
        c: {'cat': get_category(c), 'phrase': col_to_phrase(c)}
        for c in tqdm(feature_cols)
    }

    # Verify categories
    from collections import Counter
    cats_count = Counter(v['cat'] for v in col_info.values())
    print("Category counts:", dict(cats_count))

    # Build 7 text strings per sample
    print("\nBuilding text features...")
    CATS = ['api', 'files', 'ext', 'reg', 'dir', 'drop', 'str']
    records = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        active = [c for c in feature_cols if row[c] == 1]
        groups = {c: [] for c in CATS}
        for c in active:
            info = col_info[c]
            groups[info['cat']].append(info['phrase'])

        record = {
            'label':       int(row['Label (1 Ransomware / 0 Goodware)']),
            'family':      int(row['Ransomware Family']),
        }
        for cat in CATS:
            record[f'{cat}Features'] = (
                ' , '.join(groups[cat]) if groups[cat] else 'no activity'
            )
        records.append(record)

    out_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")
    print(f"Shape: {out_df.shape}")
    print(f"\nLabel distribution:")
    print(out_df['label'].value_counts())
    print(f"\nFamily distribution:")
    print(out_df['family'].value_counts().sort_index())
    print(f"\nSample apiFeatures row 0:")
    print(out_df['apiFeatures'].iloc[0][:300])
    return out_df

if __name__ == "__main__":
    process_dataset(
        csv_path = "data/raw/RansomwareData.csv",
        var_path = "data/raw/VariableNames.txt",
        out_path = "data/processed/processed.csv",
    )