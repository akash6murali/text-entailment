import os
from text_entailment__util.project_log import initiate_log, logger_msg
from text_entailment__util.data_preprocessing import preprocessing
from text_entailment__util.data_modeling import data_prep
from text_entailment__util.baseline_models import logistic_regression, svm, random_forest
from text_entailment__util.data_modeling_att import data_prep__attention_models
from text_entailment__util.attention_models import lstm_model, blstm_model


if __name__ == "__main__":

    logger = initiate_log()

    logger_msg(f"Main script is running.")

    # Get current working directory
    current_path = os.getcwd()
    # Go one step back
    parent_dir = os.path.dirname(current_path)

    file_path = os.path.join(parent_dir, 'data','input_data','snli_1.0_train.jsonl')
    # file_path = r'D:\Dhananjay\Github\text_entailment\text-entailment\data\input_data\snli_1.0_train.jsonl'

    preprocessed_df = preprocessing(file_path)

    X_train, X_test, y_train, y_test = data_prep(preprocessed_df)

    logger_msg("Running baseline models.")
    logistic_regression(X_train, X_test, y_train, y_test)

    random_forest(X_train, X_test, y_train, y_test)

    # svm(X_train, X_test, y_train, y_test)


    X_train, X_test, y_train, y_test, vocab_size, max_sequence_length, label_encoder = data_prep__attention_models(preprocessed_df)

    logger_msg("Running attention models.")
    lstm_model(X_train_combined,X_test_combined,y_train_encoded,y_test_encoded,vocab_size,max_sequence_length,label_encoder)

    blstm_model(X_train,X_test,y_train,y_test,vocab_size,max_sequence_length,label_encoder)
