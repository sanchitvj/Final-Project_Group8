import pickle
import torch
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



df = pd.read_csv("train_df.csv")

df["text"] = df["full_text"].apply(lambda x: x.lower())

X = df["text"]
y = df[["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]]


vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)


X_train, X_val, y_train, y_val = train_test_split(X_vec, y, test_size=0.2)

model = LogisticRegression()

epochs = 100
patience = 5
best_loss = float("inf")

for i in range(epochs):

    model.fit(X_train, y_train)

    val_predictions = model.predict(X_val)
    val_loss = np.mean((val_predictions - y_val) ** 2)

    mse = mean_squared_error(y_val, val_predictions, squared=False)
    mcrmse = np.mean(np.sqrt(mse))

    print(f"Epoch: {i + 1}, Val MCRMSE: {mcrmse:.3f}")

    if val_loss < best_loss:
        best_loss = val_loss
        best_epoch = epoch + 1

        # Create state dict
        state_dict = {"coef_": model.coef_,
                      "intercept_": model.intercept_}

        # Save to pth
        path = f"best_logreg_{best_epoch}_{val_loss:.3f}.pth"
        torch.save(state_dict, path)

    print(f"Saved best model for epoch {best_epoch} with loss {best_loss:.3f}")