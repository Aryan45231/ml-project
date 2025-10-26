#  Libraries 
import os 
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass


#  local Imports 
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object , evaluate_models


# Models
from sklearn.linear_model import LinearRegression , Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor , AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

# Imports for hyperparameter tuning
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Metric Imports to test model performance
from sklearn.metrics import r2_score
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Model Trainer initiated")
            X_train, y_train = train_array[:,:-1], train_array[:,-1]
            X_test, y_test = test_array[:,:-1], test_array[:,-1]

            models = {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "SVR": SVR(),
                "KNeighbors": KNeighborsRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False),
                "XGBRegressor": XGBRegressor()
            }

            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter': ['best', 'random'],
                    'max_depth': [3, 5, 10, 15, 20, None]
                },
                "Random Forest": {
                    'n_estimators': [50, 100, 200],
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_depth': [3, 5, 10, 15, 20, None]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.01, 0.1, 0.2, 0.3],
                    'n_estimators': [50, 100, 200],
                    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0]
                },
                "SVR": {
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto']
                },
                "KNeighbors": {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                },
                "CatBoost": {
                    'depth': [4, 6, 8],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'iterations': [100, 200, 300]
                },
                "XGBRegressor": {
                    'learning_rate': [0.01, 0.1, 0.2],
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7]
                }
            }

            logging.info("Models defined for training")
            model_report = evaluate_models(X_train, y_train, X_test, y_test, models , params)


            logging.info("Model evaluation completed finding best model")
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]
            best_r2_score = model_report[best_model_name]

            if(best_r2_score < 0.6):
                logging.info("No model found with R2 score greater than 0.6")
                raise CustomException("No best model found with R2 score greater than 0.6", sys)


            logging.info(f"Best Model: {best_model_name} with R2 Score: {best_r2_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_r2_score
        except Exception as e:
            logging.error("Error in Model Trainer {0}".format(e))
            raise CustomException(e, sys)