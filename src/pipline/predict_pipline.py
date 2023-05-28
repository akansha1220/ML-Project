import sys
import pandas as pd
from src.exception import CustomException
from src.utlis import load_object

class PredictPipelin:
    def __init__(self) -> None:
        pass

#like model prediction
    def predict(self,features):
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl' #handle the categorical feature and do feature scaling
            model = load_object(file_path=model_path) # load the file as it is defined in utlis (opening the file at gven path and reading it and loading it)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)   #scaling the data
            preds = model.predict(data_scaled) #here predict the scaled data 
            return preds
        
        except Exception as E:
            raise CustomException(E,sys)


#this will map the all the input given in html to the backend code

class CustomData:
    def __init__(self,gender: str, race_ethnicity:str,
                 parental_level_of_education,
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

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e,sys)