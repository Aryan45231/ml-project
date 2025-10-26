import os 
import sys
import pandas as pd
import  numpy as np
import dill

from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import r2_score


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
    

def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Evaluates multiple regression models and returns their R2 scores.

    Parameters:
    X_train (array-like): Training features.
    y_train (array-like): Training target.
    X_test (array-like): Testing features.
    y_test (array-like): Testing target.
    models (dict): A dictionary where keys are model names and values are model instances.

    Returns:
    dict: A dictionary with model names as keys and their R2 scores as values.
    """
    try:
        model_report = {}

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_test  = model.predict(X_test)

            train_r2_score = r2_score(y_train, y_pred_train)    

            test_r2_score = r2_score(y_test, y_pred_test)
            model_report[model_name] = test_r2_score
            logging.info(f"{model_name} R2 Score: {test_r2_score}")

        return model_report
    except Exception as e:
        logging.error("Error evaluating models: {0}".format(e))
        raise CustomException(e, sys)