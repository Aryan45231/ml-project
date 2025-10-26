import os 
import sys
import pandas as pd
import  numpy as np
import dill

from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging


def save_object(file_path: str, obj: object):
    """
    Saves a Python object to a file using pickle.

    Parameters:
    file_path (str): The path where the object should be saved.
    obj (object): The Python object to be saved.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Object saved successfully at {file_path}")
    except Exception as e:
        logging.error("Error saving object: {0}".format(e))
        raise CustomException(e, sys)