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

    def __init__(self):
        pass

    def __getitem__(self, index) -> T_co:
        self.items[index]
