import logging
import os

def initialize_logging(output_directory: str = None):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    log_file = "file.log"
    log_file_path = os.path.join(output_directory, log_file)

    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(filename=log_file_path, mode="a", encoding="utf8"),
    ]

    logging.basicConfig(
        format='%(asctime)s [%(name)s] %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=logging.INFO,
        handlers=handlers
    )

    # Use a logger with the same name as the module
    logger = logging.getLogger(__name__)

    # Log a message to indicate the start of the process
    logger.info("Process Starting........")

    return logger