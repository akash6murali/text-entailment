from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import sys

from text_entailment__util.project_log import logger_msg

def data_prep__attention_models(
    df
):
    """
    """
    try:
        
        df['Fact_1'] = df['Fact'].apply(lambda x: [item for sublist in x for item in sublist])
        df['hypothesis_1'] = df['hypothesis'].apply(lambda x: [item for sublist in x for item in sublist])
        
        all_processed_tokens = df['Fact_1'].tolist() + df['hypothesis_1'].tolist()

        # Build vocabulary and get word index
        word_index = {}
        for sentence in all_processed_tokens:
            for word in sentence:
                if word not in word_index:
                    word_index[word] = len(word_index) + 1  # Start index from 1

        vocab_size = len(word_index) + 1

        # Convert tokenized sentences to sequences of integers
        df['Fact_sequence'] = df['Fact_1'].apply(lambda tokens: [word_index.get(word, 0) for word in tokens])
        df['hypothesis_sequence'] = df['hypothesis_1'].apply(lambda tokens: [word_index.get(word, 0) for word in tokens])

        # Pad sequences to a fixed length
        max_sequence_length = 100  # You can adjust this
        X_fact = pad_sequences(df['Fact_sequence'], maxlen=max_sequence_length)
        X_hypothesis = pad_sequences(df['hypothesis_sequence'], maxlen=max_sequence_length)

        # Combine Fact and Hypothesis sequences
        X_combined = np.concatenate((X_fact, X_hypothesis), axis=1)

        # Encode the true labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(df['true label'])

        # Split data into training and testing sets
        X_train_combined, X_test_combined, y_train_encoded, y_test_encoded = train_test_split(X_combined, y_encoded, test_size=0.2, random_state=42)

    except Exception as e:

        logger_msg("Data Prep Att | Error in Data Prep.")
        logger_msg(e, level='error')
        sys.exit()

    return X_train_combined, X_test_combined, y_train_encoded, y_test_encoded , vocab_size, max_sequence_length, label_encoder