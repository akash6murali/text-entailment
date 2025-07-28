import os
from text_entailment__util.project_log import initiate_log, logger_msg
from text_entailment__util.data_preprocessing import preprocessing
from text_entailment__util.data_modeling import data_prep




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
