"""This file define the configuration variables."""
# Logging
LOGLEVEL = 20 # logging.INFO
LOG_FORMAT = "%(name)-20s %(levelname)-8s %(message)s"
FILE_LOGLEVEL = 10 # logging.DEBUG
FILE_LOG_FORMAT = "%(asctime)s %(name)-20s %(levelname)-8s %(message)s"

# Tensorflow dataset
SHUFFLE_BUFFER_SIZE = 1000
PREFETCH_SIZE = 1
