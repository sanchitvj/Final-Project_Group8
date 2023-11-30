from abc import ABC

import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from tqdm.auto import tqdm  ########
import tokenizers
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig
print(f"tokenizers.__version__: {tokenizers.__version__}")
print(f"transformers.__version__: {transformers.__version__}")


# tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
# encoded_input = tokenizer("Tokenization is essential.")
# print(encoded_input)


class FeedbackDataset(Dataset):

    def __init__(self, cfg, df, tokenizer, test=False):
        self.df = df.reset_index(drop=True)
        self.classes = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
        self.max_len = cfg.dataset.max_len
        self.tokenizer = tokenizer
        self.test = test

    def __getitem__(self, index):
        text = self.df["full_text"][index]
        tokenization = self.tokenizer(text,
                                      add_special_tokens=True,
                                      max_length=self.max_len,
                                      truncation=True,
                                      return_offsets_mapping=False)

        inputs = {
            "input_ids": torch.tensor(tokenization['input_ids'], dtype=torch.long),
            "token_type_ids": torch.tensor(tokenization['token_type_ids'], dtype=torch.long),
            "attention_mask": torch.tensor(tokenization['attention_mask'], dtype=torch.long)
        }

        if self.test:
            return inputs

        label = self.df.loc[index, self.classes].to_list()
        targets = {
            "labels": torch.tensor(label, dtype=torch.float32),
        }

        return inputs, targets

    def __len__(self) -> int:
        return len(self.df)


def collate_fn(batch):
    # Separate input and target in the batch
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch if len(item) > 1]

    # Batch input
    batched_input = {
        'input_ids': torch.stack([x['input_ids'] for x in inputs]),
        'token_type_ids': torch.stack([x['token_type_ids'] for x in inputs]),
        'attention_mask': torch.stack([x['attention_mask'] for x in inputs])
    }

    # Check if targets are available
    if len(targets) > 0:
        batched_targets = {
            'labels': torch.stack([x['labels'] for x in targets])
        }
        return batched_input, batched_targets

    return batched_input


