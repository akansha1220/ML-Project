import os
import sys

import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import dill
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import r2_score



def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as E:
        raise CustomException(E,sys)

def evaluate_model(X_train,Y_train,X_test,Y_test,models,param):
    try:
        report = {}

        r2_list = []

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]
              #train model

            #make predictions
            gs = GridSearchCV(model = model,param_grid= para,cv=2)
            gs.fit(X_train,Y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,Y_train)

            Y_train_predict = model.predict(X_train)
            Y_test_predict = model.predict(X_test)

    #Evaluate tran and test dataset

            model_train_r2 = r2_score(Y_train, Y_train_predict)
            model_test_r2 = r2_score(Y_test, Y_test_predict)

    
            report[list(models.keys())[i]] = model_test_r2
        return report
    
    except Exception as E:
        raise CustomException(E,sys)
    
def load_object(file_path):
        try:
            with open(file_path,"rb") as file_obj:
                 return dill.load(file_obj)
        except Exception as E:
            raise CustomException(E,sys)