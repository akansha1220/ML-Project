import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from src.logger import logging
from src.exception import CustomException
import os

import warnings

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from src.utlis import save_object
from src.utlis import evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config - ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,preprocessor_path):
        try:
            logging.info("Split training and test input data")
            X_train,Y_train,X_test,Y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                train_array[:,:-1],
                train_array[:,-1],
            )

            models = {
                    "Linear Regression": LinearRegression(),
                    "K-Neighbours Regression" : KNeighborsRegressor(),
                    "Decision Tree": DecisionTreeRegressor(),
                    "Random Forest Regressor" : RandomForestRegressor(),
                    "XGBRegressor" : XGBRegressor(),
                    "CatBoosting Regressor": CatBoostRegressor(),
                    "AdaBoost Regressor" : AdaBoostRegressor()
            }


            params = {
                "Decision Tree":{
                    'criterion': ['squared_error','absolute_error','friedman_errors','poisson'],
                    #'splitter' :['best','random'],
                    # 'max_features' : ['sqrt','log2]
                },
                "Random Forest Regressor":{
                    'criterion': ['squared_error','absolute_error','friedman_errors','poisson'],
                     'n_estimators' : [8,16,32,64,128,256]
                    # 'max_features' : ['sqrt','log2]
                },
                "Linear Regression": { },

                "K-Neighbours Regression":{
                    'n_neighbor': [5,7,9,11],
                    #'weights' :['uniform','distance'],
                    # 'algorithm' : ['ball_tree','kd_tree','brute']
                },
                "XGBRegressor":{
                    'n_estimators' : [8,16,32,64,128,256],
                    'learning_rate' : [.1,.01,.05,.001] 
                },

                "CatBoosting Regression":{
                    'iterations' : [30,50,100],
                    'learning_rate' : [.1,.01,.05,.001],
                    'depth' : [6,8,10]  
                },
                "AdaBoost Regressor":{
                    'n_estimators' : [8,16,32,64,128,256],
                    'learning_rate' : [.1,.01,.05,.001] 
                },


            }
            model_report:dict = evaluate_model(X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test,models=models,params=params)

            #to get the best model score

            best_model_score = max(sorted(model_report.values()))

            #to get the model name from dict

            best_model_name = list(model_report.keys())[ list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )
        
            predicted = best_model.predict(X_test)

            r2_square = r2_score(Y_test,predicted)
            return r2_square
        
        except Exception as E:
            raise CustomException(E,sys)