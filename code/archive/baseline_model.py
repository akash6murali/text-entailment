import nltk
nltk.download('punkt')
import pandas as pd
from nltk.tokenize import word_tokenize
# from gensim.models import Word2Vec
import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import PunktSentenceTokenizer, word_tokenize

import json

import logging
import sys


## Importing dataset.

# Drive path

# from google.colab import drive
# drive.mount('/content/drive')

# train_file_path = '/content/drive/MyDrive/NLP/Project/Input Dataset/snli_1.0_train.jsonl'

# Local Path
train_file_path = r'D:\Dhananjay\Github\text_entailment\text-entailment\data\input_data\snli_1.0_train.jsonl'


data = []
with open(train_file_path, 'r') as f:
  for line in f:
    data.append(json.loads(line))

train_df = pd.DataFrame(data)



nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')

# Initialize the WordNet lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Initialize the Punkt sentence tokenizer
punkt_tokenizer = PunktSentenceTokenizer()


def preprocess_sentences(text):

    # Tokenize the text
    sentences = punkt_tokenizer.tokenize(text)

    processed_sentences = []
    for sentence in sentences:
        tokens = word_tokenize(sentence)
        cleaned_words = []

        for token in tokens:
            if token.lower() not in stop_words and token.isalpha():
                cleaned_words.append(lemmatizer.lemmatize(token.lower()))

        processed_sentences.append(cleaned_words)
    return processed_sentences


# Apply preprocessing to the 'sentence1' and 'sentence2' columns
train_df['sentence1_processed'] = train_df['sentence1'].apply(preprocess_sentences)
train_df['sentence2_processed'] = train_df['sentence2'].apply(preprocess_sentences)


import pandas as pd
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


train_df.rename(columns={'sentence1_processed': 'Fact', 'sentence2_processed': 'hypothesis'}, inplace=True)
train_df.rename(columns={'gold_label': 'true label'}, inplace=True)


# Tokenize the text
# train_df['tokenized_fact'] = train_df['Fact'].apply(word_tokenize)
# train_df['tokenized_hypothesis'] = train_df['hypothesis'].apply(word_tokenize)

# Combine all tokens for training Word2Vec (CBOW)
all_tokens = train_df['Fact'].tolist() + train_df['hypothesis'].tolist()


# prompt: I have a column that has a list in a list in each row, I want each row to be having only a single list and have all the data

train_df['Fact_1'] = train_df['Fact'].apply(lambda x: [item for sublist in x for item in sublist])
train_df['hypothesis_1'] = train_df['hypothesis'].apply(lambda x: [item for sublist in x for item in sublist])


# Tokenize the text
train_df['tokenized_fact'] = train_df['sentence1'].apply(word_tokenize)
train_df['tokenized_hypothesis'] = train_df['sentence2'].apply(word_tokenize)

all_tokens = train_df['Fact_1'].tolist() + train_df['hypothesis_1'].tolist()



print(train_df['tokenized_fact'][0])
print(train_df['Fact'][0])
print(train_df['sentence1'][0])
print(all_tokens)


model = Word2Vec(sentences=all_tokens, vector_size=100, window=5, min_count=1, sg=0)


# Function to average word vectors for a sentence
def sentence_vector(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

# Create sentence vectors for Fact and Hypothesis
train_df['fact_vector'] = train_df['Fact_1'].apply(lambda tokens: sentence_vector(tokens, model))
train_df['hypothesis_vector'] = train_df['hypothesis_1'].apply(lambda tokens: sentence_vector(tokens, model))


# Create feature vectors by concatenating Fact and Hypothesis vectors
X = np.array(train_df['fact_vector'].tolist()) - np.array(train_df['hypothesis_vector'].tolist()) # Using difference as a feature
# Alternatively, you could concatenate:
# X = np.concatenate((np.array(df['fact_vector'].tolist()), np.array(df['hypothesis_vector'].tolist())), axis=1)


y = train_df['true label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classifier (Logistic Regression in this example)
# You can explore other classifiers like SVM, Random Forest, etc.
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = classifier.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))