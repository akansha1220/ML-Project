import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from logger import logging
from exception import CustomException


import warnings

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from utlis import save_object
from utlis import evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,Y_train,X_test,Y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]

            )

           
            models = {
                    
                    "Decision Tree": DecisionTreeRegressor(),
                    "Random Forest" : RandomForestRegressor(),
                    "XGBRegressor" : XGBRegressor(),
                    "Gradient Boosting":GradientBoostingRegressor(),
                    "CatBoosting Regressor": CatBoostRegressor(),
                    "AdaBoost Regressor" : AdaBoostRegressor(),
                    "Linear Regression": LinearRegression()
            }


            params = {

                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                }
            logging.info("Done with models and their respective params")
            model_report:dict = evaluate_model(X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test,models=models,params=params)

            logging.info("Done with evaluate model")

            #to get the best model score

            best_model_score = max(sorted(model_report.values()))

            #to get the model name from dict

            best_model_name = list(model_report.keys())[ list(model_report.values()).index(best_model_score)]

            #print(model_report[best_model_name][1])

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )
        
            
            predicted =  best_model.predict(X_test)

            r2_square = r2_score(Y_test,predicted)
            return r2_square
        
        except Exception as E:
            raise CustomException(E,sys)