import os
import sys

from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass

from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

script_name = os.path.basename(__file__) # to get the name of the current script file

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info(f"{script_name} Entered the data ingestion method or component")
        try:
            
            df = pd.read_csv(os.path.join('C:\\Users\\kesha\\OneDrive\\Desktop\\EndToEndMLProjectBYKN\\notebook\\data\\stud.csv'))
            logging.info(f"{script_name} Read the dataset as dataframe")
            if df is None:
                raise CustomException(f"{script_name} Failed to enter the data into dataframe df. do your due diligance", sys)
            
            '''
            👆 this area is actually the area of creativity. Here you can read the data from 
            different sources and perform different operations before it could be used for training. 
            You can read the data from a database, you can read the data from a csv file, you can read 
            the data from an API, you can read the data from a cloud storage like AWS S3 bucket or GCP bucket. 
            You can perform different operations on the data like handling missing values, handling categorical 
            variables, handling outliers, handling imbalanced data, next. The possibilities are endless. 
            You can do whatever you want to do with the data before it could be used for training.
            and for this purpose, you can create a separate method for each operation and call those methods here 
            in the initiate_data_ingestion method. This will make your code more modular and easier to maintain. 
            You can also create a separate class for each operation and call those classes here in the initiate_data_ingestion method. 
            This will make your code more object-oriented and easier to maintain.
            '''
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info(f"{script_name} Raw data is saved at {self.ingestion_config.raw_data_path}")

            logging.info(f"{script_name} Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            logging.info(f"{script_name} Train data is saved at {self.ingestion_config.train_data_path}")

            test_set.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info(f"{script_name} Test data is saved at {self.ingestion_config.test_data_path}")

            logging.info(f"{script_name} Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                # self.ingestion_config.raw_data_path
            )

        except Exception as e:
            logging.error(f"{script_name} Error occurred in data ingestion: {e}")
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    print( model_trainer.initiate_model_trainer(
        train_arr, 
        test_arr, 
        # preprocessor_path=data_transformation.data_transformation_config.preprocessor_obj_file_path
        ))
    # print(f"R2 square value: {r2_square}")


