import logging

import os
from datetime import datetime

# Create a timestamped log directory and file
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Configure more readable, structured logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format=(
        "\n-----------------------------\n"
        "Time       : %(asctime)s\n"
        "Level      : %(levelname)s\n"
        "Module     : %(name)s\n"
        "Line       : %(lineno)d\n"
        "Message    : %(message)s\n"
        "-----------------------------\n"
    ),
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Example usage


