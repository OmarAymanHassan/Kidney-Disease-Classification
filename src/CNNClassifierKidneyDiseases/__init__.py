# Logging Every step we are doing 


import os 
import sys
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

log_dir = "logs"
log_filepath= os.path.join(log_dir,"running_logs.log")
os.makedirs(log_dir,exist_ok=True)


""" 
Each %(...)s is a placeholder used by the Python logging module’s formatting system. The s at the end means:

format this value as a string

So, for example:

%(asctime)s: insert the timestamp as a string

%(levelname)s: insert the log level (e.g., INFO, ERROR) as a string

%(module)s: insert the module name as a string

%(message)s: insert the actual log message as a string

The % syntax is similar to the older Python string formatting style using %s, %d, etc.

"""


logging.basicConfig(
level = logging.INFO,
format=logging_str,

handlers= [logging.FileHandler(log_filepath),
            logging.StreamHandler(sys.stdout)]
)


logger = logging.getLogger("CNN-Classifier-Logger")
# name of the logger object


