# ============================================================
# process.py — Convert Sgandurra binary features → text
# Dataset has no headers — we load them from VariableNames.txt
# ============================================================

import pandas as pd
import re
import os
from tqdm import tqdm

def camel_to_words(name):
    name = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', name)
    name = re.sub(r'([a-z\d])([A-Z])',      r'\1 \2', name)
    return name.lower().strip()

def strip_suffix(name):
    if len(name) > 2 and name[-1] in ('W','A') and name[-2].islower():
        return name[:-1]
    return name

def col_to_phrase(col_name):
    col = col_name.strip()
    u   = col.upper()
    if u.startswith('API:'):
        return camel_to_words(strip_suffix(col.split(':',1)[1]))
    if u.startswith('REG:'):
        parts = col.split(':',2)
        verb  = parts[1].lower() if len(parts)>1 else 'accessed'
        key   = parts[2].replace('\\',' ') if len(parts)>2 else ''
        return f"{verb} registry {key}".strip()
    if u.startswith('FILES:') or u.startswith('FILE:'):
        parts = col.split(':',2)
        verb  = parts[1].lower() if len(parts)>1 else 'accessed'
        path  = parts[2].replace('\\',' ') if len(parts)>2 else ''
        return f"{verb} file {path}".strip()
    if u.startswith('DROP:'):
        return f"dropped file {col.split(':',1)[1]}".strip()
    if u.startswith('DIR:'):
        parts = col.split(':',2)
        verb  = parts[1].lower() if len(parts)>1 else 'accessed'
        path  = parts[2].replace('\\',' ') if len(parts)>2 else ''
        return f"{verb} directory {path}".strip()
    if u.startswith('EXT:'):
        return f"opened extension {col.split(':',1)[1].lstrip('.')}"
    if u.startswith('STR:'):
        raw = col.split(':',1)[1]
        val = raw.split(';',1)[-1] if ';' in raw else raw
        return val.lower().strip()
    return camel_to_words(col)

def get_category(col_name):
    u = col_name.upper()
    if u.startswith('API:'):                             return 'api'
    if u.startswith('FILES:') or u.startswith('FILE:'): return 'files'
    if u.startswith('EXT:'):                             return 'ext'
    if u.startswith('REG:'):                             return 'reg'
    if u.startswith('DIR:'):                             return 'dir'
    if u.startswith('DROP:'):                            return 'drop'
    return 'str'

def load_variable_names(path):
    names = []
    with open(path, encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(';', 1)
            name  = parts[1] if len(parts) > 1 else parts[0]
            names.append(name)
    return names

def process_dataset(input_csv, varnames_txt, output_path):
    print("Loading variable names...")
    col_names = load_variable_names(varnames_txt)
    print(f"Total columns: {len(col_names)}")
    print(f"First 3: {col_names[:3]}")

    print("\nLoading dataset...")
    df = pd.read_csv(input_csv, header=None, names=col_names)
    print(f"Shape: {df.shape}")

    # Rename first 3 columns
    df.columns.values[0] = 'sample_id'
    df.columns.values[1] = 'label'
    df.columns.values[2] = 'family'

    # Check if row 0 is a header row (non-numeric)
    try:
        float(df.iloc[0, 3])
    except:
        print("Dropping header row...")
        df = df.iloc[1:].reset_index(drop=True)

    feature_cols = list(df.columns[3:])
    print(f"Feature columns: {len(feature_cols)}")

    family_map = {
        0:'Goodware', 1:'Citroni', 2:'CryptLocker',
        3:'CryptoWall', 4:'KOLLAH', 5:'Kovter',
        6:'Locker', 7:'MATSNU', 8:'PGPCODER',
        9:'Reveton', 10:'TeslaCrypt', 11:'TrojanRansom'
    }
    df['family']      = pd.to_numeric(df['family'], errors='coerce').fillna(0).astype(int)
    df['label']       = pd.to_numeric(df['label'],  errors='coerce').fillna(0).astype(int)
    df['family_name'] = df['family'].map(family_map).fillna('Unknown')

    print("\nPre-computing feature phrases...")
    col_info = {}
    for col in tqdm(feature_cols):
        col_info[col] = {
            'category': get_category(col),
            'phrase':   col_to_phrase(col)
        }

    cats = ['api','files','ext','reg','dir','drop','str']
    print("\nBuilding text features...")
    records = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        groups = {c: [] for c in cats}
        for col in feature_cols:
            try:
                if int(row[col]) == 1:
                    info = col_info[col]
                    groups[info['category']].append(info['phrase'])
            except:
                pass

        record = {
            'sample_id':   row['sample_id'],
            'label':       row['label'],
            'family':      row['family'],
            'family_name': row['family_name'],
        }
        for cat in cats:
            record[f'{cat}Features'] = (
                ' , '.join(groups[cat]) if groups[cat] else 'no activity'
            )
        records.append(record)

    out_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_csv(output_path, index=False)

    print(f"\nSaved: {output_path}")
    print(f"Shape: {out_df.shape}")
    print(f"\nFamily distribution:")
    print(out_df['family_name'].value_counts())
    print(f"\nSample apiFeatures row 1:")
    print(out_df['apiFeatures'].iloc[1][:300])
    return out_df

if __name__ == "__main__":
    process_dataset(
        input_csv    = "data/raw/RansomwareData.csv",
        varnames_txt = "data/raw/VariableNames.txt",
        output_path  = "data/processed/processed.csv"
    )
