import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import dill
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error
import joblib



def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            joblib.dump(obj,file_obj)

    except Exception as E:
        raise CustomException(E,sys)

def evaluate_model(X_train,Y_train,X_test,Y_test,models,params):
    try:
        report = {}

        

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]
              #train model

            #make predictions
            
            logging.info("Started with grid search CV")
            gs = GridSearchCV(estimator=model, param_grid=para, cv=2)
        
            gs.fit(X_train,Y_train)

            logging.info("trained the open search cv")

            best_model = gs.best_estimator_
            y_pred = best_model.predict(X_test)
          

    #Evaluate tran and test dataset

            mse = mean_squared_error(Y_test, y_pred)
            rmse = np.sqrt(mse)  

    
            report[list(models.keys())[i]] = rmse
        return report
    
    except Exception as E:
        raise CustomException(E,sys)
    
def load_object(file_path):
        try:
            with open(file_path,"rb") as file_obj:
                 logging.info("loading is successfully in utlis file")
                 return joblib.load(file_obj)
        except Exception as E:
            raise CustomException(E,sys)