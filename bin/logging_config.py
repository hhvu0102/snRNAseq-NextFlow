import logging
import os
from datetime import datetime

import argparse


parser = argparse.ArgumentParser("Plot QC metrics per sample")
parser.add_argument("--sample", help="Donor ID.", type=str)


def setup_logging(log_file="logs", level=logging.DEBUG):
    """
    Configure logging to write to both a file and the console.

    Parameters
    ----------
    log_dir : str
        Directory in which to store log files.
    level : int
        Minimum logging level for the file handler.

    Returns
    -------
    logging.Logger
        Configured root logger.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_file

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Prevent duplicate handlers if called multiple times
    if root_logger.handlers:
        root_logger.handlers.clear()

    # File handler — captures everything
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)

    # Console handler — only INFO and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "\n" + "=" * 50 + "\n%(levelname)s: %(message)s\n" + "=" * 50 + "\n"
    )
    console_handler.setFormatter(console_formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    root_logger.info(f"Logging initialized. Log file: {log_file}")

    return root_logger
