import os 
import sys
import pandas as pd
import  numpy as np
import dill

from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


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
    

def evaluate_models(X_train, y_train, X_test, y_test, models , param_grids) -> dict:
   
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
            param_grid = param_grids.get(model_name, {})
            gs = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=0)
            gs.fit(X_train, y_train)
            model.set_params(**gs.best_params_)
            logging.info(f"Best parameters for {model_name}: {gs.best_params_}")

            model.fit(X_train, y_train)
        
            y_pred_test  = model.predict(X_test)

            test_r2_score = r2_score(y_test, y_pred_test)
            model_report[model_name] = test_r2_score
            logging.info(f"{model_name} R2 Score: {test_r2_score}")

        return model_report
    except Exception as e:
        logging.error("Error evaluating models: {0}".format(e))
        raise CustomException(e, sys)
    
def load_object(file_path: str) -> object:
    """
    Loads a Python object from a file using pickle.

    Parameters:
    file_path (str): The path to the file from which the object should be loaded.

    Returns:
    object: The loaded Python object.
    """
    try:
        with open(file_path, 'rb') as file_obj:
            obj = dill.load(file_obj)
        logging.info(f"Object loaded successfully from {file_path}")
        return obj
    except Exception as e:
        logging.error("Error loading object: {0}".format(e))
        raise CustomException(e, sys)