import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style(style='dark')
import operator
import string, re
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
eng_stopwords = set(stopwords.words('english'))

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# %%
print(train.head().to_string())
print(test.head().to_string())

print("train shape: ", train.shape)
print("test shape: ", test.shape)

print(train.columns)
print(train.nunique())
print(train.info())

# %%
train_lists = list(train['full_text'])
length_train_list = [len(train_list) for train_list in train_lists]

plt.figure(figsize=(15, 10))

plt.hist(length_train_list, bins=30, alpha=0.5, color='blue', label='word')  # bins => Number of Data in Xlim
plt.yscale('log')
plt.title("Log-Histplot of Text length", fontsize=20)
plt.xlabel("length of Text", fontsize=16)
plt.ylabel("number of Text", fontsize=16)
plt.show()

# %%
# Make DataFrame for Statistic Info
measure_name = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
cohesion_list = list(train['cohesion'])
syntax_list = list(train['syntax'])
vocabulary_list = list(train['vocabulary'])
phraseology_list = list(train['phraseology'])
grammar_list = list(train['grammar'])
conventions_list = list(train['conventions'])

measure_df = pd.DataFrame({'Max': [np.max(cohesion_list), np.max(syntax_list), np.max(vocabulary_list),
                                   np.max(phraseology_list), np.max(grammar_list), np.max(conventions_list)],
                           'Min': [np.min(cohesion_list), np.min(syntax_list), np.min(vocabulary_list),
                                   np.min(phraseology_list), np.min(grammar_list), np.min(conventions_list)],
                           'Mean': [np.mean(cohesion_list), np.mean(syntax_list), np.mean(vocabulary_list),
                                    np.mean(phraseology_list), np.mean(grammar_list), np.mean(conventions_list)],
                           'Std': [np.std(cohesion_list), np.std(syntax_list), np.std(vocabulary_list),
                                   np.std(phraseology_list), np.std(grammar_list), np.std(conventions_list)],
                           'Median': [np.median(cohesion_list), np.median(syntax_list), np.median(vocabulary_list),
                                      np.median(phraseology_list), np.median(grammar_list),
                                      np.median(conventions_list)]})
measure_df.index = [name for name in measure_name]
print(measure_df)

# %%
sns.set_style(style='dark')
plt.figure(figsize=(20, 15))

# cohesion_list
plt.subplot(2, 3, 1)
plt.hist(cohesion_list, bins=9, alpha=0.5, color='blue', label='word')
plt.title("Cohesion", fontsize=15)
plt.xlabel("Score of Cohesion", fontsize=10)
plt.ylabel("number of Cohesion", fontsize=10)

# syntax_list
plt.subplot(2, 3, 2)
plt.hist(syntax_list, bins=9, alpha=0.5, color='red', label='word')
plt.title("Syntax", fontsize=15)
plt.xlabel("Score of Syntax", fontsize=10)
plt.ylabel("number of Syntax", fontsize=10)

# vocabulary_list
plt.subplot(2, 3, 3)
plt.hist(vocabulary_list, bins=9, alpha=0.5, color='green', label='word')
plt.title("vocabulary", fontsize=15)
plt.xlabel("Score of vocabulary", fontsize=10)
plt.ylabel("number of vocabulary", fontsize=10)

# phraseology_list
plt.subplot(2, 3, 4)
plt.hist(phraseology_list, bins=9, alpha=0.5, color='black', label='word')
plt.title("phraseology", fontsize=15)
plt.xlabel("Score of phraseology", fontsize=10)
plt.ylabel("number of phraseology", fontsize=10)

# grammar_list
plt.subplot(2, 3, 5)
plt.hist(grammar_list, bins=9, alpha=0.5, color='yellow', label='word')
plt.title("grammar", fontsize=15)
plt.xlabel("Score of grammar", fontsize=10)
plt.ylabel("number of grammar", fontsize=10)

# conventions_list
plt.subplot(2, 3, 6)
plt.hist(conventions_list, bins=9, alpha=0.5, color='violet', label='word')
plt.title("conventions", fontsize=15)
plt.xlabel("Score of conventions", fontsize=10)
plt.ylabel("number of conventions", fontsize=10)

plt.show()

# %%
sns.heatmap(train.drop(['text_id', 'full_text'], axis=1).corr(), annot=True)
plt.tight_layout()
plt.show()

# %%
# Let's now examine the total punctuation obtained by each essay.
# In other words, we're going to sum all the scores given to obtain total_score.
train['total_score'] = train['cohesion'] + train['syntax'] + train['vocabulary'] + \
                       train['phraseology'] + train['grammar'] + train['conventions']

avg_char = round(train['total_score'].mean())
plt.figure(figsize=(22, 5))
sns.distplot(train['total_score'])
plt.axvline(x=avg_char, color='red')
plt.title('Character Count')
plt.show()

#%%
# stop words count visualization
corpus = ''.join(train.full_text).split()
dic = defaultdict(int)

for w in corpus:
    if w in eng_stopwords:
        dic[w] += 1

dic_sorted = sorted(dic.items(), key=operator.itemgetter(1), reverse=True)
x, y = zip(*dic_sorted[:10])
plt.bar(x, y)
plt.title("Stopwords count")
plt.show()

#%%
# most frequent n-grams
def preprocess_text(text):
    text = text.lower()
    # Eliminate punctuation
    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
    # Remove characters that are not ASCII
    text = re.sub("([^\x00-\x7F])+", " ", text)
    return text

def get_top_ngrams(start_size=2, end_size=2):

    vectorizer = CountVectorizer(stop_words='english', ngram_range=(start_size, end_size))
    ngram_matrix = vectorizer.fit_transform(train_copy["full_text"])

    # Sum frequencies of ngrams
    freqs = ngram_matrix.toarray().sum(axis=0)

    # Convert frequencies to DataFrame
    freq_df = pd.DataFrame(sorted([(freqs[i], word) for word, i in vectorizer.vocabulary_.items()], reverse=True))
    freq_df.columns = ["frequency", "ngram"]

    return freq_df


train_copy = train.copy()
train_copy['full_text'] = train_copy['full_text'].apply(preprocess_text)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(22, 6))
for index in range(2, 4):
    frequent_ngrams = get_top_ngrams(index, index)
    sns.barplot(x='frequency', y='ngram', data=frequent_ngrams[:10], ax=ax[index - 2])
    del frequent_ngrams

fig.tight_layout(h_pad=1.0, w_pad=0.5)
plt.suptitle('Top N-Grams in the Dataset', y=1.02)
plt.show()

#%%
# POS tagging
def extract_specific_tokens(pos_tags, desired_tag='ADJ'):
    """Extracts tokens from pos_tags that are labeled as desired_tag."""

    filtered_tokens = [token for token, pos in pos_tags if (pos == desired_tag) and (token not in eng_stopwords)]
    return filtered_tokens


def display_top_tokens(pos_tags, tag='ADJ', row_num=-1, col_num=-1):
    """Identify and display the most frequent tokens of a specific tag.
       Visual representation using a bar chart."""

    tokens = extract_specific_tokens(pos_tags, tag)

    # Count the most frequent tokens
    top_tokens = Counter(tokens).most_common(10)

    # Prepare data for visualization
    tokens_list, counts = zip(*top_tokens)

    # Generate bar plot for the top tokens
    if col_num == -1:
        sns.barplot(x=counts, y=tokens_list)
    else:
        sns.barplot(x=counts, y=tokens_list, ax=axes[row_num][col_num])
    plt.show()


# Remove stopwords from the corpus
filtered_corpus = [word for word in corpus if word not in eng_stopwords]
# Tag the corpus with POS tags
pos_tags = nltk.pos_tag(filtered_corpus, tagset="universal")

# Display the most common adjectives
display_top_tokens(pos_tags, 'ADJ')

#%%


