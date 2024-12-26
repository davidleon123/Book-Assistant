from __future__ import annotations
import logging
import os
from typing import TYPE_CHECKING
from .config import PROJECT_ROOT

if TYPE_CHECKING:
    from logging import Logger
# Ensure the log directory exists
log_directory = os.path.join(PROJECT_ROOT, 'log')
os.makedirs(log_directory, exist_ok=True)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# File handler for logging questions
file_handler = logging.FileHandler(os.path.join(log_directory, 'questions.log'))
formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def log_question(question: str, logger:Logger =logger) -> None:
    logger.info(f"Question: {question}")