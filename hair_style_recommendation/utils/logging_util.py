import logging
import os 

def initialize_logging(output_directory: str = None):
    # let's initialise the Streamhandler so that we can show the information
    # in the output 
    
    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)
    logFile = "file.log"

    handlers = [logging.StreamHandler(), logging.FileHandler(filename=os.path.join(output_directory, logFile), mode="a", encoding="utf8" )]
    logging.basicConfig( format='%(asctime)s %(message)s', \
                         datefmt='%m/%d/%Y %I:%M:%S %p', \
                         level=logging.INFO, \
                         handlers=handlers
                         )

    logging.info("Process Starting........")
