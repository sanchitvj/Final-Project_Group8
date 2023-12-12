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


#%% Logistic Regression
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
import re
from wordcloud import WordCloud
from wordcloud import STOPWORDS

import nltk
nltk.download('wordnet')

from textblob import TextBlob
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


df = pd.read_csv("train_df.csv")


# Basic text cleaning function
def remove_noise(text):
    # Make lowercase
    text = text.apply(lambda x: " ".join(x.lower() for x in x.split()))

    # Remove whitespaces
    text = text.apply(lambda x: " ".join(x.strip() for x in x.split()))

    # # Remove special characters
    # text = text.apply(lambda x: "".join([" " if ord(i) < 32 or ord(i) > 126 else i for i in x]))
    #
    # # Remove punctuation
    # text = text.str.replace('[^\w\s]', '')

    # Remove numbers
    text = text.str.replace('\d+', '')

    # # Remove Stopwords
    # text = text.apply(lambda x: ' '.join([word for word in x.split() if word not in (STOPWORDS)]))

    # Convert to string
    text = text.astype(str)

    return text

df['filtered_text'] = remove_noise(df['full_text'])
df['text_len'] = df['full_text'].apply(lambda x: len(x))
df['words_num'] = df['full_text'].apply(lambda x: len(x.split()))


# def sentiment_analyser(text):
#     return text.apply(lambda Text: pd.Series(TextBlob(Text).sentiment.polarity))
#
# # Applying function to reviews
# df['polarity'] = sentiment_analyser(df['filtered_text'])
#
# # Length of full_text and words num
# fig, ax = plt.subplots(2, 2, figsize=(15, 8))
# sns.boxplot(df['text_len'], palette='PRGn', ax = ax[0, 0])
# sns.histplot(df['text_len'], ax = ax[1, 0])
# sns.boxplot(df['words_num'], palette='PRGn', ax = ax[0, 1])
# sns.histplot(df['words_num'], ax = ax[1, 1])
#
#
# text = " ".join(i for i in df['filtered_text'])
# stopwords = set(STOPWORDS)
# wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
# plt.figure( figsize=(15,10))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.show()


#%%
def modelRes(mod, y_name, x_train, y_train, x_test, y_test):
    mod.fit(x_train, y_train)
    print(y_name)
    acc = cross_val_score(mod, x_train, y_train, scoring = "accuracy", cv = 5)
    predictions = cross_val_predict(mod, x_test, y_test, cv = 5)
    print("Accuracy:", round(acc.mean(),3))
    print("Classification Report \n",classification_report(predictions, y_test))

max_words = round(df['filtered_text'].apply(lambda x: len(x.split())).max())

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['filtered_text'])
word_index = tokenizer.word_index

train_seq = tokenizer.texts_to_sequences(df['filtered_text'])
pad_train = pad_sequences(train_seq, maxlen=max_words, truncating='post')


word_idx_count = len(word_index)
print(word_idx_count)

tokenizer.word_counts

scList = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']

X = pad_train

for score in scList:
    y = df[score].replace([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], [0, 1, 2, 3, 4, 5, 6, 7, 8])
    # Create a train-test split of these variables
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)
    model = LogisticRegression()
    modelRes(model, score, X_train, y_train, X_test, y_test)

#%%

test = pd.read_csv("test_df.csv")
test['filtered_text'] = remove_noise(test['full_text'])

test_seq = tokenizer.texts_to_sequences(test['filtered_text'])
pad_test = pad_sequences(test_seq, maxlen=max_words, truncating='post')

submission = pd.DataFrame()


submission['text_id'] = test['text_id'].copy()

for score in scList:
    y = df[score].replace([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], [0, 1, 2, 3, 4, 5, 6, 7, 8])
    model = LogisticRegression()
    model.fit(pad_train, y)
    print(score, model.score(pad_train, y))
    submission[score] = model.predict(pad_test).tolist()


#%% LSTM (not complete)
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pandas as pd
from nltk.tokenize import word_tokenize
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
import gc
from sklearn.metrics import mean_squared_error


df = pd.read_csv('train_df.csv')
test = pd.read_csv('test_df.csv')



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

        text = torch.tensor(self.tokenizer.texts_to_sequences([text])[0])

        text = nn.functional.pad(text, (0, self.max_length - text.shape[0]))

        if self.test is False:
            label = self.labels[idx, :] / 5.
            label = torch.tensor(label, dtype=torch.float32)
        return text, label


#%%
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

#%% Bert
import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.linear_model import LogisticRegression
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam
from sklearn.metrics import mean_squared_error

df = pd.read_csv('train_df.csv')
test = pd.read_csv('test_df.csv')

def process_text(df):
    df = df.copy()
    df['full_text'] = df['full_text'].apply(lambda x: x.replace('\n', ' '))
    df['full_text'] = df['full_text'].apply(lambda x: x.strip())
    df['full_text'] = df['full_text'].apply(lambda x: x.lower())
    return df

df = process_text(df)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):
        self.labels = df[["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]].reset_index()
        self.texts = df[["full_text"]].reset_index()

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels.loc[idx].values[1:]).astype(float)

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return tokenizer(self.texts.loc[idx].values[1],
                         padding='max_length', max_length=512, truncation=True,
                         return_tensors="pt")

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y


class FeedbackModel(nn.Module):
    def __init__(self, dropout=0.5):
        super(FeedbackModel, self).__init__()

        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 256)
        self.relu = nn.ReLU()
        self.out = nn.Linear(256, 6)

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        x = self.dropout(pooled_output)
        x = self.linear(x)
        x = self.relu(x)
        final_layer = self.out(x)
        return final_layer



def mcrmse_labelwise_score(true_values, predicted_values):
    individual_scores = []

    num_targets = true_values.shape[1]

    for target_index in range(num_targets):
        true_target = true_values[:, target_index]

        predicted_target = predicted_values[:, target_index]

        mse_score = mean_squared_error(true_target, predicted_target, squared=False)

        individual_scores.append(mse_score)

    average_mcrmse = np.mean(individual_scores)

    return average_mcrmse, individual_scores


def train(model, train_data, val_data, epochs, device, criterion):
    train_set = Dataset(train_data)
    val_set = Dataset(val_data)

    train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_set, batch_size=64, shuffle=False, pin_memory=True)

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=1e-5)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=500,
                                                         eta_min=1e-6)

    for epoch in range(epochs):

        print(f"Epoch {epoch + 1}/{epochs}")

        total_train_loss = 0
        total_train_samples = 0

        for train_input, train_labels in train_dataloader:
            train_labels = train_labels.to(device).float()
            batch_size = train_labels.size(0)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)
            batch_loss = criterion(output, train_labels)

            total_train_loss += batch_loss.item() * batch_size
            total_train_samples += batch_size

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
            scheduler.step()

        total_loss_val = 0
        total_val_samples = 0

        with torch.no_grad():
            avg_mcrmse, _ = mcrmse_labelwise_score(true_labels, predicted)

            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                batch_size = val_label.size(0)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)
                batch_loss = criterion(output, val_label)

                total_loss_val += batch_loss.item() * batch_size
                total_val_samples += batch_size


        avg_train_loss = total_train_loss / total_train_samples
        avg_val_loss = total_loss_val / total_val_samples

        print(f'Epoch: {epoch + 1} | Train Loss: {avg_train_loss:.3f} | Val Loss: {avg_val_loss:.3f}')

    return avg_val_loss

# def evaluate(model, test_data):
#     test = Dataset(test_data)
#
#     test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)
#     criterion = nn.MSELoss()
#     use_cuda = torch.cuda.is_available()
#     device = torch.device("cuda" if use_cuda else "cpu")
#
#     if use_cuda:
#         model = model.cuda()
#
#     total_loss_test = 0
#     with torch.no_grad():
#
#         for test_input, test_labels in enumerate(test_dataloader):
#             test_labels = test_labels.to(device)
#             mask = test_input['attention_mask'].to(device)
#             input_id = test_input['input_ids'].squeeze(1).to(device)
#
#             output = model(input_id, mask)
#
#             loss = criterion(output, test_labels)
#             total_loss_test += loss
#
#     print(f'Test Loss: {total_loss_test / len(test_data): .3f}')


np.random.seed(42)
df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),
                                     [int(.9 * len(df)), int(.95 * len(df))])

print(len(df_train), len(df_val), len(df_test))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPOCHS = 15
model = FeedbackModel()
model.to(device)

criterion = nn.MSELoss()

avg_val_loss = train(model, df_train, df_val, EPOCHS, device, criterion)

# evaluate(model, df_test)

torch.save({'model': model.state_dict()}, f'bert-base-uncased_{avg_val_loss}.pth')


# model = FeedbackModel()
# model.load_state_dict(torch.load('feedback_model.pt'))
# model.to(device)
# model.eval()