import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
# import pytorch_lightning as pl
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pandas as pd
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
import gc

df = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


tokenizer = Tokenizer(lower=False,oov_token='<OOV>')
tokenizer.fit_on_texts(df['full_text'])

class EssayDataset:
    def __init__(self, df, max_len, tokenizer, test=False):
        self.test = test
        self.max_length = max_len
        self.texts = list(df['full_text'].values)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        text = self.tokenizer.texts_to_sequences([text])[0]
        text = pad_sequences([text], maxlen=self.max_length, padding='pre', truncating='post')[0]
        text = torch.tensor(text, dtype=torch.long)
        if not self.test:
            label_cols = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
            labels = df.loc[idx, label_cols].values / 5.
            label = torch.tensor(labels, dtype=torch.float32)
            return text, label
        return text

# sample_ds = EssayDataset(df,512,tokenizer)
# sample_ds[0]


#%%
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, seq_len, n_layers, output_dim, lr):
        super(RNNModel, self).__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.lr = lr

        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.lstm = nn.LSTM(input_size=self.embed_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.n_layers,
                            batch_first=True,
                            dropout=0.3)
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)

        self.loss_fn = nn.MSELoss()
        self.optimizer = AdamW(self.parameters(), lr=self.lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=20, eta_min=1e-6)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.linear(x[:, -1, :])
        return x

    def train_step(self, loader):
        self.train()
        epoch_loss = 0.0
        for inputs, targets in loader:
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(loader)

    def val_step(self, loader):
        self.eval()
        epoch_loss = 0.0
        with torch.no_grad():
            for data in loader:
                if len(data) == 2:  # Check if both inputs and targets are available
                    inputs, targets = data
                else:  # For the test set where only inputs are present
                    inputs = data
                    targets = None

                outputs = self(inputs)

                # Calculate loss if targets are available
                if targets is not None:
                    loss = self.loss_fn(outputs, targets)
                    epoch_loss += loss.item()

        # Return average loss if targets were available, else return None
        if targets is not None:
            return epoch_loss / len(loader)
        else:
            return None

score_cols = ['cohesion','syntax','vocabulary','phraseology','grammar','conventions']
df['total_score'] = df.loc[:,score_cols].sum(axis=1)

config = {
    'vocab': len(tokenizer.word_index),
    'embed_dim': 13,
    'hidden_dim': 8,
    'seq_len': 512,
    'n_layers': 4,
    'output_dim': len(score_cols),
    'lr': 3e-4,
    'epochs' : 25,
    'batch_size' : 16,
    'seed' : 1357
}
#print(config)

def prepare_datasets(df, test_size=0.2):
    train_df, val_df = train_test_split(df,
                                        test_size=test_size,
                                        shuffle=True,
                                        random_state=config['seed']
                                        )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_ds = EssayDataset(train_df, config['seq_len'], tokenizer)
    val_ds = EssayDataset(val_df, config['seq_len'], tokenizer)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=config['batch_size'])

    return train_loader, val_loader

train_loader, val_loader = prepare_datasets(df)
len(train_loader), len(val_loader)

test_ds = EssayDataset(test,config['seq_len'],tokenizer,test=True)
test_loader = DataLoader(test_ds, batch_size=1, num_workers=1, shuffle=False)

_x,_y = next(iter(train_loader))
_x.shape, _y.shape

model = RNNModel(config['vocab'], config['embed_dim'], config['hidden_dim'], config['seq_len'],
                 config['n_layers'], config['output_dim'], config['lr'])

# model.eval()
# test_loss = model.val_step(test_loader)
# print(f"Test Loss: {test_loss:.4f}")
#
# predictions = []
# model.eval()
# with torch.no_grad():
#     for inputs in test_loader:
#         outputs = model(inputs)
#         predictions.append(outputs.squeeze().tolist())

predictions =[]
model.eval()
test_loss = model.val_step(test_loader)

if test_loss is not None:
    print(f"Test Loss: {test_loss:.4f}")
else:
    print("No targets available for evaluation in the test set.")
