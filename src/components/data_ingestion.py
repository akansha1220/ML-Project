import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from exception import CustomException
from logger import logging
import pandas as pd
import sys
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from components.model_trainer import ModelTrainerConfig
from components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path : str=os.path.join("artifacts",'train.csv')

    test_data_path : str=os.path.join("artifacts",'test.csv')
    raw_data_path : str=os.path.join("artifacts",'data.csv')


class DataIndegestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv("Notebook\data\stud.csv")
            logging.info("Exported /read the dataset as Dataframe")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("train tesy split initiated")
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index = False,header = True)
            test_set.to_csv(self.ingestion_config.test_data_path,index = False,header = True)
            logging.info("ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            
            )
        except Exception as e:
            pass
            #raise CustomException(e,sys)


if __name__=="__main__":
    obj = DataIndegestion()
    train_data,test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_date_transformation(train_data,test_data)
    
    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))