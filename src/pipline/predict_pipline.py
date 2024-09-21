import sys
import pandas as pd
from src.exception import CustomException
from src.utlis import load_object
from src.logger import logging
from src.exception import CustomException

class PredictPipeline:
    def __init__(self) -> None:
        pass

#like model prediction
    def predict(self,features):
        try:
            logging.info("loading the Model pickel file")
            model=load_object(file_path='artifacts\model.pkl')
            preprocessor=load_object(file_path='artifacts\processor.pkl')
            logging.info("Load the model file successfully")
            data_scaled=preprocessor.transform(features)
            logging.info("transformation of input done sucessful")
            preds=model.predict(data_scaled)
            return preds
        

        except Exception as E:
            logging.critical(f"Prediction failed at {E}")
            raise CustomException(E,sys)


#this will map the all the input given in html to the backend code

class CustomData:
    def __init__(self,gender: str, race_ethnicity:str,
                 parental_level_of_education:str,
                 lunch: str,
                 test_preparation_course:str,
                 reading_score: int,
                 writing_score: int) :
        

        self.gender = gender
        self.race_ethnicity =race_ethnicity
        self.parental_level_of_education =  parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
        
    def get_data_as_dataframe(self):            #convert the data into the dataframe
        try:
            custom_data_input_dict = {
               "gender":[self.gender], 
               "race_ethnicity":[self.race_ethnicity],
               "parental_level_of_education":[self.parental_level_of_education],
               "lunch":[self.lunch],
               "test_preparation_course":[self.test_preparation_course],
               "reading_score":[self.reading_score],
               "writing_score":[self.writing_score]
            }
            logging.info("Changing the input data into datframe is done successfully")
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            logging.critical("Error occur while predicting the data")
            raise CustomException(e,sys)