import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pandas as pd
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
import gc
from sklearn.metrics import mean_squared_error



df = pd.read_csv('train_df.csv')
test = pd.read_csv('test_df.csv')

tkz = Tokenizer(lower=False,oov_token='<OOV>')
tkz.fit_on_texts(df['full_text'])


class EssayDataset:
    def __init__(self, df, max_len, tokenizer, test=False):
        self.test = test
        self.max_length = max_len
        self.classes = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
        self.texts = list(df['full_text'].values)
        if self.test is False:
            self.labels = df.loc[:, self.classes].values
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


# sample_ds = EssayDataset(df, 512, tkz)
# sample_ds[0]

class RNNModel(pl.LightningModule):
    def __init__(self, config):
        super(RNNModel, self).__init__()
        self.save_hyperparameters()
        self.config = config
        self.vocab_size = self.config['vocab']
        self.embed_dim = self.config['embed_dim']
        self.hidden_dim = self.config['hidden_dim']
        self.seq_len = self.config['seq_len']
        self.n_layers = self.config['n_layers']
        self.output_dim = self.config['output_dim']

        self.lr = config['lr']

        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)

        self.lstm = nn.LSTM(input_size=self.embed_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.n_layers,
                            batch_first=True,
                            dropout=0.3
                            )

        self.linear = nn.Linear(self.hidden_dim, self.output_dim)

        self.test_preds = []

    def forward(self, x):
        x = self.embedding(x)
        x, (h, c) = self.lstm(x)
        x = self.linear(h[-1])
        return x

    def loss_fn(self, outputs, targets):
        # MCRMSE: https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/348985
        colwise_mse = torch.mean(torch.square(targets - outputs), dim=0)
        loss = torch.mean(torch.sqrt(colwise_mse), dim=0)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        #         scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3,gamma=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)

        loss = self.loss_fn(outputs, y)

        self.log('train_loss', loss.item(), on_epoch=True)

        return loss

    def calculate_mcrmse(self, outputs, y):
            mse_per_col = torch.mean((outputs - y) ** 2, dim=0)
            mcrmse = torch.sqrt(mse_per_col).mean()  # Mean across columns
            return mcrmse

    def validation_step(self, batch, batch_idx):
            x, y = batch

            outputs = self(x)
            loss = self.loss_fn(outputs, y)

            mcrmse_batch = self.calculate_mcrmse(outputs, y)
            self.log('val_loss', loss.item(), on_epoch=True)
            self.log('val_mcrmse_batch', mcrmse_batch.item())  # Logging per batch MCRMSE

            return {'val_loss': loss, 'val_mcrmse_batch': mcrmse_batch}

    def validation_epoch_end(self, outputs):
        avg_mcrmse = torch.stack([x['val_mcrmse_batch'] for x in outputs]).mean()
        self.log('val_mcrmse_avg', avg_mcrmse.item(), on_epoch=True)

        if self.trainer.current_epoch == self.trainer.max_epochs - 1:
            filename = f'rnn_model_epoch_{self.trainer.current_epoch}_val_mcrmse_{avg_mcrmse:.4f}.pth'
            torch.save({'model': self.state_dict()}, filename)
    def get_predictions(self):
        return torch.cat(self.test_preds).numpy()


score_cols = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
df['total_score'] = df.loc[:, score_cols].sum(axis=1)
config = {
    'vocab': len(tkz.word_index),
    'embed_dim': 13,
    'hidden_dim': 8,
    'seq_len': 512,
    'n_layers': 4,
    'output_dim': len(score_cols),
    'lr': 3e-4,
    'epochs': 25,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'batch_size': 16,
    'seed': 1357,
    'model_name': 'lstm-embeddings'
}

def prepare_datasets(df, test_size=0.2):
    train_df, val_df = train_test_split(df,
                                        test_size=test_size,
                                        shuffle=True,
                                        random_state=config['seed']
                                        )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_ds = EssayDataset(train_df, config['seq_len'], tkz)
    val_ds = EssayDataset(val_df, config['seq_len'], tkz)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=config['batch_size'])

    return train_loader, val_loader


train_loader, val_loader = prepare_datasets(df)
len(train_loader), len(val_loader)

model = RNNModel(config)


trainer = pl.Trainer(accelerator='gpu',
                     devices = 1,
                     callbacks=[
                         EarlyStopping(monitor="val_loss",
                                       mode="min",
                                       patience=5,
                                      )
                     ],
                     max_epochs = config['epochs']
                    )