# Author: Yonglin Wang
# Date: 2021/5/30 4:00 PM
# Returns a customized Python Logger object for displaying and saving messages


import logging
import os


def init_logger(dunder_name: str, output_dir: str, output_file_name: str) -> logging.Logger:
    """Initialize a logger that takes all levels of messages"""
    log_format = (
        '%(asctime)s - '
        '%(name)s - '
        '%(funcName)s - '
        '%(levelname)s - '
        '%(message)s'
    )

    logger = logging.getLogger(dunder_name)

    logger.setLevel(logging.DEBUG)

    # Output full log
    fh = logging.FileHandler(os.path.join(output_dir, output_file_name + ".log"))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(log_format)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # log to stdout
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(log_format)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

if __name__ == "__main__":
    logger = init_logger(__name__, "data/", "test_log")
    logger.info("Hello!!!")
    logger.debug("Here is a debug message, not sure if it'll make it to the log file.")