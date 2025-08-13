import logging
import sys
from datetime import date, datetime
import inspect
import os


rundate = datetime.now().strftime("%Y%m%d%H%M%S")
LOG_FILE_NAME = f'text_entailment__{rundate}.log'

# Get current working directory
current_path = os.getcwd()
# Go one step back
parent_dir = os.path.dirname(current_path)
# Path to 'log' folder in parent directory
LOG_DIR = os.path.join(parent_dir, "log")
# Full path to log file
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE_NAME)




def initiate_log():
    """
    Initiates log file and a logg.
    """
    try:
        if sys.platform.lower() in ['windows','win32','win64']:

            if not os.path.exists(LOG_DIR):
                os.makedirs(LOG_DIR)

            rundate = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            logger = logging.getLogger("project_logger")
            logger.setLevel(logging.INFO)
            
            
            # Write log message
            log_message = 'Initated Log file successfully.'

            if not logger.handlers:
                
                # File Handler
                file_handler = logging.FileHandler(LOG_FILE_PATH)
                formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)

                # Stream (console) handler
                stream_handler = logging.StreamHandler()
                stream_formatter = logging.Formatter('%(levelname)s: %(message)s')
                stream_handler.setFormatter(stream_formatter)
                logger.addHandler(stream_handler)
            
            logger.info(log_message)


    except Exception as e:

        print('%s error initializing the log')

    return logger


def logger_msg(
        log_message = None,
        level = "info"
):

    """
    Logs a message using the configured logger.
    level: "info", "warning", or "error"
    """
    logger = logging.getLogger("project_logger")
    
    if level == "info":
        logger.info(log_message)
    elif level == "warning":
        logger.warning(log_message)
    elif level == "error":
        logger.error(log_message)
    else:
        logger.info(log_message)

    print(log_message)
    
    return None