import logging
import logging.config
import os
import sys

# Set ROOT logger
logger = logging.getLogger()
formatter = logging.Formatter('%(asctime)s - %(filename)-10s - %(funcName)-8s - %(levelname)-6s - %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.setLevel(logging.WARNING)


def init(logger_name: str, level=logging.INFO) -> logging.Logger:
    """Initiate logger with input name."""
    logger_name = os.path.basename(logger_name)
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    return logger
