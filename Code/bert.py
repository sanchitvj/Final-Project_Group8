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


class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 6)

    def forward(self, text):
        # text should be tokenized
        embedded = self.embedding(text)
        output, hidden = self.rnn(embedded)

        # use final hidden state as feature for classifier
        predictions = self.fc(hidden.squeeze(0))
        return predictions

class LogisticRegressionModel:

    def __init__(self):
        self.model = LogisticRegression()

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)


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