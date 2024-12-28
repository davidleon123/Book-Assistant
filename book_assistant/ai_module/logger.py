from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING
#from .config import BASE_DIR

if TYPE_CHECKING:
    from logging import Logger
# Ensure the log directory exists

base_dir = os.getenv('BASE_DIR')

if base_dir is None:
    raise ValueError("BASE_DIR environment variable is not set")

BASE_DIR = Path(base_dir)
log_directory = BASE_DIR / 'log'
log_directory.mkdir(parents=True, exist_ok=True)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# File handler for logging questions
file_handler = logging.FileHandler(log_directory / 'questions.log')
formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def log_question(question: str, logger:Logger =logger) -> None:
    logger.info(f"Question: {question}")