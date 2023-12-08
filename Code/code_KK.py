# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize, sent_tokenize

#%%
train = pd.read_csv('train_df.csv')
test = pd.read_csv('test_df.csv')

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


