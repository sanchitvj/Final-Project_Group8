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
