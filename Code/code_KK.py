# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize, sent_tokenize

#%%
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#train.head()
train.info()
train.describe().T

# %% EDA
train['total_score'] = train.sum(axis=1, numeric_only=True)
plt.figure(figsize=(18, 5))
sns.histplot(x = train['total_score'], data = train)
plt.show()

score_cols = ['cohesion','syntax','vocabulary','phraseology','grammar','conventions']
for i,col in enumerate(score_cols):
    fig = sns.countplot(data = train, x=col)
    plt.title(col + ' Scores Distribution')
    plt.show()
    
# %% Text EDA
text_1 = train['full_text'].values[0]
text_2 = train['full_text'].values[2]
text_1
text_2

train['text_len'] = train['full_text'].astype(str).apply(len)

train["text_word_count"] = train["full_text"].apply(lambda x: len(x.replace('\n', ' ').split()))

def get_sent_count(text):
    tokens = sent_tokenize(text, language='english')
    return len(tokens)

train['sent_count'] = train['full_text'].apply(get_sent_count)

cols = ['text_len', 'text_word_count', 'sent_count']
for i,col in enumerate(cols):
    fig = sns.histplot(data = train, x=col)
    plt.title(col + '  Distribution')
    plt.show()
    
print('The maximum number of sentences: {}'.format(train['sent_count'].max()))
print('The minimum number of sentences: {}'.format(train['sent_count'].min()))
print('The average number of sentences: {}'.format(train['sent_count'].mean()))

print('The maximum text length: {}'.format(train['text_len'].max()))
print('The minimum text length: {}'.format(train['text_len'].min()))
print('The average text length: {}'.format(train['text_len'].mean()))

print('The maximum number of words: {}'.format(train['text_word_count'].max()))
print('The minimum number of words: {}'.format(train['text_word_count'].min()))
print('The average number of words: {}'.format(train['text_word_count'].mean()))


# %%
# a look at the essay's scores with maximum words:
train[train.text_word_count == train.text_word_count.max()]
# get essay =='3814F9116CD1'
train._get_value(725, 'full_text')

# a look at the essay's score with minimum words:
train[train.text_word_count == train.text_word_count.min()]
#get essay == 'F69C85F4C3CA':
train._get_value(3679, 'full_text')

# single sentence essays:
train[train.sent_count == train.sent_count.min()]
#get essay == '39:
train._get_value(39, 'full_text')
#get essay == '3607:
train._get_value(3607, 'full_text')

# %% Text length vs. scoring
score_text_len = train.groupby('total_score')['text_len'].mean().sort_values()

score_text_len.plot(kind = 'barh', figsize = (12,8))
plt.xlabel('Average Text length')
plt.title(' Relationship between length of texts and scoring')

# Number of words vs. scoring
score_word_count = train.groupby('total_score')['text_word_count'].mean().sort_values()
score_word_count.plot(kind = 'barh', figsize = (12,8))
plt.xlabel('Average Word_count')
plt.title(' Relationship between number of words and scoring')

# Number of sentences vs. scoring
score_sent_count = train.groupby('total_score')['sent_count'].mean().sort_values()
score_sent_count.plot(kind = 'barh', figsize = (12,8))
plt.xlabel('Average Sent_count')
plt.title(' Relationship between number of sentences and scoring')


# plt.figure(figsize=(15,15))
# colormap = sns.color_palette('Blues')
# sns.heatmap(train.corr(), annot=True, cmap=colormap)

#%%
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

tokenize = Tokenizer(lower=False,oov_token='<OOV>')
tokenize.fit_on_texts(train['full_text'])

class EssayDataset:
    def __init__(self, train, max_len, tokenizer, test=False):
        self.test = test
        self.max_length = max_len
        self.classes = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
        self.texts = list(train['full_text'].values)
        if self.test is False:
            self.labels = train.loc[:, self.classes].values
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        text = self.tokenizer.texts_to_sequences([text])[0]
        text = pad_sequences([text], maxlen=self.max_length, padding='pre', truncating='post')[0]
        text = torch.tensor(text, dtype=torch.long)
        if self.test is False:
            label = self.labels[idx, :] / 5.
            label = torch.tensor(label, dtype=torch.float32)
            return text, label
        return text

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
        x = self.linear(x[:, -1, :])  # Consider the last output of LSTM for each sequence
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
            for inputs, targets in loader:
                outputs = self(inputs)
                loss = self.loss_fn(outputs, targets)
                epoch_loss += loss.item()
        return epoch_loss / len(loader)

score_cols = ['cohesion','syntax','vocabulary','phraseology','grammar','conventions']
train['total_score'] = train.loc[:,score_cols].sum(axis=1)

config = {
    'vocab': len(tokenize.word_index),
    'embed_dim': 13,
    'hidden_dim': 8,
    'seq_len': 512,
    'n_layers': 4,
    'output_dim': len(score_cols),
    'lr': 3e-4,
}
print(config)

def prepare_datasets(df, test_size=0.2):
    train_df, val_df = train_test_split(df,
                                        test_size=test_size,
                                        shuffle=True,
                                        random_state=config['seed']
                                        )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_ds = EssayDataset(train_df, config['seq_len'], tokenize)
    val_ds = EssayDataset(val_df, config['seq_len'], tokenize)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=config['batch_size'])

    return train_loader, val_loader

train_loader, val_loader = prepare_datasets(train)
len(train_loader), len(val_loader)

test_ds = EssayDataset(test,config['seq_len'],tokenize,test=True)
test_loader = DataLoader(test_ds, batch_size=1, num_workers=1, shuffle=False)

_x,_y = next(iter(train_loader))
_x.shape, _y.shape

model = RNNModel(config['vocab'], config['embed_dim'], config['hidden_dim'], config['seq_len'],
                 config['n_layers'], config['output_dim'], config['lr'])

trainer = pl.Trainer(accelerator='gpu',
                     callbacks=[
                         EarlyStopping(monitor="val_loss",
                                       mode="min",
                                       patience=5,
                                      )
                     ],
                     max_epochs = config['epochs']
                    )

lr_finder = trainer.tuner.lr_find(model, train_loader, val_loader)

# Results can be found in
lr_finder.results

# Plot with
fig = lr_finder.plot(suggest=True)
fig.show()

# Pick point based on plot, or get suggestion
new_lr = lr_finder.suggestion()
print(new_lr)

model.hparams.lr = new_lr
model.hparams

trainer.fit(model, train_loader, val_loader)
metrics = trainer.logged_metrics

logs = {
    'train_loss': metrics['train_loss_epoch'].item(),
    'val_loss': metrics['val_loss'].item()
}
logs

trainer.test(model,test_loader)

p = model.get_predictions()
p.shape
