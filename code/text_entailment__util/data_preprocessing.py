import json
import numpy as np
import pandas as pd

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import PunktSentenceTokenizer, word_tokenize

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')


# Initialize the WordNet lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Initialize the Punkt sentence tokenizer
punkt_tokenizer = PunktSentenceTokenizer()

from text_entailment__util.project_log import logger_msg


def preprocessing(
      file_path = None
      ):

    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    df = pd.DataFrame(data)

    # Apply preprocessing to the 'sentence1' and 'sentence2' columns
    df['sentence1_processed'] = df['sentence1'].apply(preprocess_sentences)
    df['sentence2_processed'] = df['sentence2'].apply(preprocess_sentences)

    df.rename(columns={'sentence1_processed': 'Fact', 'sentence2_processed': 'hypothesis'}, inplace=True)
    df.rename(columns={'gold_label': 'true label'}, inplace=True)

    return df

def preprocess_sentences(text):
    """
        Tokenize the text
    """
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