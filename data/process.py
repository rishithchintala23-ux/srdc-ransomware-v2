# ============================================================
# process.py — Convert Sgandurra binary features → text
# WHAT HAPPENS HERE:
#   Input:  RansomwareData.csv (1524 rows × 30967 columns)
#   Output: processed.csv with 7 text columns per sample
#
# WHY: DeBERTa reads English text, not binary 0/1 columns
# HOW: We convert each active feature to an English phrase
#      then group into 7 categories
# ============================================================

import pandas as pd
import re
import os
from tqdm import tqdm

# ── Text conversion helpers ───────────────────────────────────
def camel_to_words(name: str) -> str:
    """
    CamelCase → lowercase words
    WHY: DeBERTa understands 'crypt encrypt' not 'CryptEncrypt'
    
    Example: CreateRemoteThread → 'create remote thread'
    """
    name = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', name)
    name = re.sub(r'([a-z\d])([A-Z])',      r'\1 \2', name)
    return name.lower().strip()

def strip_suffix(name: str) -> str:
    """
    Remove W/A suffix from Windows API names
    WHY: CryptEncryptW and CryptEncryptA are the same function
    
    Example: CryptEncryptW → CryptEncrypt
    """
    if len(name) > 2 and name[-1] in ('W','A') and name[-2].islower():
        return name[:-1]
    return name

def col_to_phrase(col_name: str) -> str:
    """
    Convert raw column name → English phrase
    
    Examples:
    'API:CryptEncryptA'           → 'crypt encrypt'
    'REG:OPENED:HKLM\\Run'        → 'opened registry HKLM Run'
    'DROP:C:\\READ_ME.txt'         → 'dropped file READ ME txt'
    """
    col = col_name.strip()
    u   = col.upper()

    if u.startswith('API:'):
        api = col.split(':',1)[1]
        return camel_to_words(strip_suffix(api))

    if u.startswith('REG:'):
        parts = col.split(':',2)
        verb  = parts[1].lower() if len(parts)>1 else 'accessed'
        key   = parts[2].replace('\\',' ').replace('_',' ') if len(parts)>2 else ''
        return f"{verb} registry {key}".strip()

    if u.startswith('FILES:') or u.startswith('FILE:'):
        parts = col.split(':',2)
        verb  = parts[1].lower() if len(parts)>1 else 'accessed'
        path  = parts[2].replace('\\',' ') if len(parts)>2 else ''
        return f"{verb} file {path}".strip()

    if u.startswith('DROP:'):
        path = col.split(':',1)[1].replace('\\',' ')
        return f"dropped file {path}".strip()

    if u.startswith('DIR:'):
        parts = col.split(':',2)
        verb  = parts[1].lower() if len(parts)>1 else 'accessed'
        path  = parts[2].replace('\\',' ') if len(parts)>2 else ''
        return f"{verb} directory {path}".strip()

    if u.startswith('EXT:'):
        return f"opened extension {col.split(':',1)[1].lstrip('.')}"

    if u.startswith('STR:'):
        return col.split(':',1)[1].lower().strip()

    return camel_to_words(col)

def get_category(col_name: str) -> str:
    """Which of 7 categories does this feature belong to?"""
    u = col_name.upper()
    if u.startswith('API:'):                        return 'api'
    if u.startswith('FILES:') or u.startswith('FILE:'): return 'files'
    if u.startswith('EXT:'):                        return 'ext'
    if u.startswith('REG:'):                        return 'reg'
    if u.startswith('DIR:'):                        return 'dir'
    if u.startswith('DROP:'):                       return 'drop'
    return 'str'

# ── Main processing function ──────────────────────────────────
def process_dataset(input_path: str, output_path: str):
    """
    WHAT THIS DOES:
    1. Load the raw Sgandurra CSV (1524 rows × 30967 columns)
    2. For each row, find all columns with value=1 (active features)
    3. Convert each active feature name to English phrase
    4. Group phrases into 7 categories
    5. Save as new CSV with 7 text columns

    WHY 7 CATEGORIES:
    Different feature types need separate encoding.
    Feeding all 30967 features as one text = too long for DeBERTa.
    Split into 7 = each fits within 512 token limit.
    We encode each separately then combine (like SRDC paper).
    """
    print(f"Loading dataset from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

    # Identify label columns vs feature columns
    label_cols   = ['label', 'family', 'family_name', 'sample_id']
    feature_cols = [c for c in df.columns if c not in label_cols]
    print(f"Feature columns: {len(feature_cols)}")

    # Pre-compute category and phrase for each feature column
    print("Pre-computing feature phrases...")
    col_info = {}
    for col in tqdm(feature_cols):
        col_info[col] = {
            'category': get_category(col),
            'phrase':   col_to_phrase(col)
        }

    # Build 7 text strings per row
    print("Building text features for each sample...")
    cats = ['api', 'files', 'ext', 'reg', 'dir', 'drop', 'str']
    records = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Find all active features (value = 1)
        active = [col for col in feature_cols if row[col] == 1]

        # Group by category
        groups = {c: [] for c in cats}
        for col in active:
            info = col_info[col]
            groups[info['category']].append(info['phrase'])

        # Join phrases per category
        record = {}
        for lc in label_cols:
            if lc in df.columns:
                record[lc] = row[lc]
        for cat in cats:
            record[f'{cat}Features'] = (
                ' , '.join(groups[cat]) if groups[cat] else 'no activity'
            )
        records.append(record)

    # Save
    out_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_csv(output_path, index=False)

    print(f"\nSaved to {output_path}")
    print(f"Shape: {out_df.shape}")
    print("\nSample row (apiFeatures):")
    print(out_df['apiFeatures'].iloc[0][:200])
    return out_df

if __name__ == "__main__":
    process_dataset(
        input_path  = "data/raw/RansomwareData.csv",
        output_path = "data/processed/processed.csv"
    )