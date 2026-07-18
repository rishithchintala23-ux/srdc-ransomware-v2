# ============================================================
# dataset.py — PyTorch Dataset for DeBERTa fine-tuning
#
# WHAT HAPPENS HERE:
#   Input:  processed.csv (7 text columns per sample)
#   Output: PyTorch tensors DeBERTa can train on
#
# WHY THIS DESIGN:
#   DeBERTa takes text input → we combine all 7 categories
#   into one string with special separators.
#   This lets DeBERTa attend across ALL feature types at once.
# ============================================================

import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer
from model.config import MODEL_NAME, MAX_LENGTH, FAMILY_MAP

FEATURE_COLS = [
    'apiFeatures', 'filesFeatures', 'filesExtFeatures',
    'regFeatures', 'dirFeatures', 'dropFeatures', 'strFeatures'
]

# Labels for readable display
CAT_LABELS = {
    'apiFeatures':      '[API]',
    'filesFeatures':    '[FILES]',
    'filesExtFeatures': '[EXT]',
    'regFeatures':      '[REG]',
    'dirFeatures':      '[DIR]',
    'dropFeatures':     '[DROP]',
    'strFeatures':      '[STR]',
}

def build_input_text(row: pd.Series) -> str:
    """
    Combine 7 feature categories into one DeBERTa input string.

    WHY: DeBERTa's bidirectional attention means it can see
    relationships ACROSS all feature types simultaneously.
    Example: [API] crypt encrypt [FILES] opened doc.docx
             [DROP] dropped READ_ME.txt
    DeBERTa sees that crypt_encrypt + dropped READ_ME.txt
    TOGETHER = strong ransomware signal.

    Format:
    [API] api text here [FILES] files text [REG] reg text ...
    """
    parts = []
    for col in FEATURE_COLS:
        text = str(row.get(col, 'no activity')).strip()
        if not text or text == 'nan':
            text = 'no activity'
        label = CAT_LABELS.get(col, col)
        parts.append(f"{label} {text}")
    return ' '.join(parts)


class RansomwareDataset(Dataset):
    """
    PyTorch Dataset for ransomware detection.

    HOW IT WORKS:
    1. Load processed CSV
    2. For each sample, combine 7 text columns → one string
    3. Tokenize with DeBERTa tokenizer
    4. Return: input_ids, attention_mask, label

    WHAT __getitem__ RETURNS:
    {
        'input_ids':      tensor of token IDs [MAX_LENGTH]
        'attention_mask': tensor of 1s/0s     [MAX_LENGTH]
        'label':          integer (family class 0-11)
        'binary_label':   integer (0=goodware, 1=ransomware)
    }
    """

    def __init__(self, csv_path: str, tokenizer, task: str = 'family'):
        """
        Args:
            csv_path:  path to processed.csv
            tokenizer: DeBERTa tokenizer
            task:      'family' (12-class) or 'binary' (2-class)
        """
        self.df        = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.task      = task

        # Build input texts once (saves time during training)
        print(f"Building input texts for {len(self.df)} samples...")
        self.texts = [
            build_input_text(row)
            for _, row in self.df.iterrows()
        ]
        print(f"Average text length: "
              f"{sum(len(t) for t in self.texts)//len(self.texts)} chars")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text  = self.texts[idx]
        row   = self.df.iloc[idx]

        # Tokenize
        # truncation=True: cut if > MAX_LENGTH (DeBERTa limit)
        # padding='max_length': pad shorter sequences to MAX_LENGTH
        encoding = self.tokenizer(
            text,
            max_length    = MAX_LENGTH,
            truncation    = True,
            padding       = 'max_length',
            return_tensors = 'pt'
        )

        label = (
            int(row['label'])   # 0 or 1
            if self.task == 'binary'
            else int(row['family'])  # 0-11
        )

        return {
            'input_ids':      encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label':          torch.tensor(label, dtype=torch.long),
            'binary_label':   torch.tensor(int(row['label']),
                                           dtype=torch.long),
        }