import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder


from text_entailment__util.project_log import logger_msg

# Assuming 'Fact_1' and 'hypothesis_1' contain the preprocessed token lists
# Combine the tokenized sentences for vocabulary building
all_processed_tokens = train_df['Fact_1'].tolist() + train_df['hypothesis_1'].tolist()

# Build vocabulary and get word index
word_index = {}
for sentence in all_processed_tokens:
    for word in sentence:
        if word not in word_index:
            word_index[word] = len(word_index) + 1  # Start index from 1

vocab_size = len(word_index) + 1

# Convert tokenized sentences to sequences of integers
train_df['Fact_sequence'] = train_df['Fact_1'].apply(lambda tokens: [word_index.get(word, 0) for word in tokens])
train_df['hypothesis_sequence'] = train_df['hypothesis_1'].apply(lambda tokens: [word_index.get(word, 0) for word in tokens])

# Pad sequences to a fixed length
max_sequence_length = 100  # You can adjust this
X_fact = pad_sequences(train_df['Fact_sequence'], maxlen=max_sequence_length)
X_hypothesis = pad_sequences(train_df['hypothesis_sequence'], maxlen=max_sequence_length)

# Combine Fact and Hypothesis sequences
X_combined = np.concatenate((X_fact, X_hypothesis), axis=1)

# Encode the true labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(train_df['true label'])

# Split data into training and testing sets
X_train_combined, X_test_combined, y_train_encoded, y_test_encoded = train_test_split(X_combined, y_encoded, test_size=0.2, random_state=42)

# Define the LSTM model
lstm_model = Sequential([
    Embedding(vocab_size, 100, input_length=max_sequence_length * 2),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(len(label_encoder.classes_), activation='softmax')
])

lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("LSTM Model Summary:")
lstm_model.summary()

# Train the LSTM model
lstm_model.fit(X_train_combined, y_train_encoded, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate the LSTM model
loss_lstm, accuracy_lstm = lstm_model.evaluate(X_test_combined, y_test_encoded)
print(f"LSTM Model Accuracy: {accuracy_lstm}")

# Define the Bidirectional LSTM model
blstm_model = Sequential([
    Embedding(vocab_size, 100, input_length=max_sequence_length * 2),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.2),
    Bidirectional(LSTM(64)),
    Dropout(0.2),
    Dense(len(label_encoder.classes_), activation='softmax')
])

blstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("\nBidirectional LSTM Model Summary:")
blstm_model.summary()

# Train the Bidirectional LSTM model
blstm_model.fit(X_train_combined, y_train_encoded, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate the Bidirectional LSTM model
loss_blstm, accuracy_blstm = blstm_model.evaluate(X_test_combined, y_test_encoded)
print(f"Bidirectional LSTM Model Accuracy: {accuracy_blstm}")