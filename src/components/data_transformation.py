import sys
from dataclasses import dataclass
import os

from src.exception import CustomException
from src.logger import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = ['gender', 
                                   'race_ethnicity', 
                                   'parental_level_of_education', 
                                   'lunch', 
                                   'test_preparation_course'
                                   ]
            logging.info(f"Numerical columns being used: {numerical_columns}")
            logging.info(f"Categorical columns being used: {categorical_columns}")
            
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            logging.info("Numerical pipeline created successfully with simple imputer and standard scaler.")
            
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            logging.info("Categorical pipeline created successfully with simple imputer, one-hot encoder, and standard scaler.")
            
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )
            logging.info("Column transformer created successfully with numerical and categorical pipelines.")

            return preprocessor
        except Exception as e:
            logging.error(f"Error in creating data transformer object: {e}")
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and test data read successfully from the provided paths.")

            logging.info("Obtaining the preprocessor object.")
            preprocessor_obj = self.get_data_transformer_object()

            target_column_name = 'math_score'
            numerical_columns = ['writing_score', 'reading_score']
            
            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]
            
            logging.info("Separated input features and target feature for both train and test datasets.")
            logging.info("Applying the preprocessor object on the training and testing data.")

            input_feature_train_df = preprocessor_obj.fit_transform(input_feature_train_df)
            logging.info("Preprocessor object fitted successfully on the training data.")

            input_feature_test_df = preprocessor_obj.transform(input_feature_test_df)
            logging.info("Preprocessor object applied successfully on the testing data.")

            train_arr = np.c_[
                input_feature_train_df, np.array(target_feature_train_df)
                ]
            test_arr = np.c_[
                input_feature_test_df, np.array(target_feature_test_df)
                ]
            logging.info("Combined input features and target feature into numpy arrays for both train and test datasets.")
            logging.info("Data transformation completed successfully.")
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            logging.error(f"Error in data transformation: {e}")
            raise CustomException(e, sys)
        

