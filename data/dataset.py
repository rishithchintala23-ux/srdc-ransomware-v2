# ============================================================
# dataset.py - PyTorch Dataset for DeBERTa fine-tuning
#
# Each sample has 7 text groups (api, files, ext, reg, dir, drop, str).
# Samples can be thousands of words long, but DeBERTa takes 512 tokens.
# Plain concatenation + truncation would cut off the LAST groups
# (drop, str) for most samples - and those carry strong ransomware signal
# (dropped ransom notes, strings like "encrypt").
#
# Fix: every group gets its own token budget, so all 7 groups are
# always represented in the final 512-token input.
# ============================================================

import os
import sys

import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.config import MAX_LENGTH

# (csv column, tag, max body tokens)
# Budgets sum to 460; with tags + [CLS]/[SEP] this stays inside 512.
GROUPS = [
    ('apiFeatures',   'api',   110),
    ('filesFeatures', 'files',  70),
    ('extFeatures',   'ext',    20),
    ('regFeatures',   'reg',    90),
    ('dirFeatures',   'dir',    50),
    ('dropFeatures',  'drop',   30),
    ('strFeatures',   'str',    90),
]


class RansomwareDataset(Dataset):
    """
    Returns per item:
      input_ids       [MAX_LENGTH]
      attention_mask  [MAX_LENGTH]
      label           family id (0-11) if task='family', else 0/1 if task='binary'
      binary_label    0 = goodware, 1 = ransomware
    """

    def __init__(self, csv_path, tokenizer, task='family'):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.task = task
        self.labels = (self.df['label'] if task == 'binary'
                       else self.df['family']).astype(int).tolist()

        print(f"Tokenizing {len(self.df)} samples (per-group budgets)...")
        self.encoded = [self._encode_row(row) for _, row in self.df.iterrows()]

    def _encode_row(self, row):
        tok = self.tokenizer
        ids = [tok.cls_token_id]
        for col, tag, budget in GROUPS:
            text = str(row.get(col, '')).strip()
            if not text or text == 'nan':
                text = 'no activity'
            tag_ids = tok(tag + ':', add_special_tokens=False)['input_ids']
            body = tok(text, add_special_tokens=False, truncation=True,
                       max_length=budget)['input_ids']
            ids += tag_ids + body
        ids = ids[:MAX_LENGTH - 1] + [tok.sep_token_id]
        mask = [1] * len(ids)
        pad = MAX_LENGTH - len(ids)
        ids += [tok.pad_token_id] * pad
        mask += [0] * pad
        return ids, mask

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        ids, mask = self.encoded[idx]
        row = self.df.iloc[idx]
        return {
            'input_ids':      torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            'label':          torch.tensor(self.labels[idx], dtype=torch.long),
            'binary_label':   torch.tensor(int(row['label']), dtype=torch.long),
        }