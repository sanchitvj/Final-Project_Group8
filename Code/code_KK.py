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
