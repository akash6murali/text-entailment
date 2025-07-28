import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from nltk.tokenize import PunktSentenceTokenizer, word_tokenize


# Initialize the WordNet lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Initialize the Punkt sentence tokenizer
punkt_tokenizer = PunktSentenceTokenizer()

from gensim.models import Word2Vec

from sklearn.model_selection import train_test_split

from text_entailment__util.project_log import logger_msg

# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report





def sentence_vector(tokens, model):
    """
    Function to average word vectors for a sentence  
    """
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)


def data_prep(
        df,
        test_size = 0.2,
        random_state = 42
):

        # Combine all tokens for training Word2Vec (CBOW)
    all_tokens = df['Fact'].tolist() + df['hypothesis'].tolist()

    df['Fact_1'] = df['Fact'].apply(lambda x: [item for sublist in x for item in sublist])
    df['hypothesis_1'] = df['hypothesis'].apply(lambda x: [item for sublist in x for item in sublist])


    # Tokenize the text
    df['tokenized_fact'] = df['sentence1'].apply(word_tokenize)
    df['tokenized_hypothesis'] = df['sentence2'].apply(word_tokenize)

    all_tokens = df['Fact_1'].tolist() + df['hypothesis_1'].tolist()

    print(df['tokenized_fact'][0])
    print(df['Fact'][0])
    print(df['sentence1'][0])
    print(all_tokens)
    
    model = Word2Vec(sentences=all_tokens, vector_size=100, window=5, min_count=1, sg=0)

    # Create sentence vectors for Fact and Hypothesis
    df['fact_vector'] = df['Fact_1'].apply(lambda tokens: sentence_vector(tokens, model))
    df['hypothesis_vector'] = df['hypothesis_1'].apply(lambda tokens: sentence_vector(tokens, model))

    # Create feature vectors by concatenating Fact and Hypothesis vectors
    X = np.array(df['fact_vector'].tolist()) - np.array(df['hypothesis_vector'].tolist()) # Using difference as a feature
    # Alternatively, you could concatenate:
    # X = np.concatenate((np.array(df['fact_vector'].tolist()), np.array(df['hypothesis_vector'].tolist())), axis=1)


    y = df['true label']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test