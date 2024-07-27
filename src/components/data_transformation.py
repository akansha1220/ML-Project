import sys
from dataclasses import dataclass

import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from utlis import save_object

from exception import CustomException
from logger import logging
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','processor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
          this function is responsible for data transformation
        '''
        try:
            numerical_columns = ["reading_score","writing_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder(handle_unknown='ignore')),
                ("Scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info("Numerical columns standard scaling completed")

            logging.info("Categorical columns standard scaling completed")

            preprocessor = ColumnTransformer(
                [
                ('num_pipeline',num_pipeline,numerical_columns),
                ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor

        except Exception as E:
            raise CustomException(E,sys)


    def initiate_date_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read the train and test path")
            
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

           
            input_feature_train_df = train_df.drop(["math_score"],axis=1)
            target_feature_train_df = train_df["math_score"]

            input_feature_test_df = test_df.drop(["math_score"],axis=1)
            target_feature_test_df = test_df["math_score"]

            logging.info("applying preprocessing object on the training dataframe and testing dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)            

            train_arr = np.c_[
                input_feature_train_arr,np.array(target_feature_train_df).reshape(-1, 1)
            ]

            test_arr = np.c_[
                input_feature_test_arr,np.array(target_feature_test_df).reshape(-1, 1)
            ]

          
            logging.info("Saved preprocessing object.")
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
        
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as E:
            raise CustomException(E,sys)

                




