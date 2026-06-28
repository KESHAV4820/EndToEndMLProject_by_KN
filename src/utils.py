import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    r2_score, accuracy_score, 
    classification_report, confusion_matrix, 
    f1_score, precision_score, recall_score
    )
import dill as pickle
import yaml

script_name = os.path.basename(__file__) # to get the name of the current script file

import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # points to src/
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(X_train,y_train,X_test,y_test,models,param):
    try:
        logging.info(f"{script_name}: Evaluating models using GridSearchCV and r2 score.")
        report={}
        for i in range(len(list(models))):
            model=list(models.values())[i] #getting the model from the dictionary
            para=param[list(models.keys())[i]] #getting the parameters for that model from the param dictionary

            gs=GridSearchCV(model,para,cv=3) #creating the GridSearchCV object with the model and its parameters
            gs.fit(X_train,y_train) #fitting the GridSearchCV object on the training data

            # model.fit(X_train,y_train) #training that model # legacy code, replaced with GridSearchCV

            model.set_params(**gs.best_params_) #setting the best parameters for that model
            logging.info(f"{script_name}: Best parameters for {list(models.keys())[i]}: {gs.best_params_}")
            logging.info(f"{script_name}: Training {list(models.keys())[i]} with best parameters.")
            model.fit(X_train,y_train) #training that model with the best parameters

            y_train_pred=model.predict(X_train) #predicting the training set results
            y_test_pred=model.predict(X_test) #predicting the test set results

            train_model_score: float=r2_score(y_train,y_train_pred) #calculating the r2 score for the model on training data
            test_model_score: float=r2_score(y_test,y_test_pred) #calculating the r2 score for the model on test data

            report[list(models.keys())[i]]=test_model_score #storing the r2 score in the report dictionary with the name of the model as the key and the r2 score as the value

        logging.info(f"{script_name}: Model evaluation completed.")
        return report

    except Exception as e:
        raise CustomException(e,sys)
def read_yaml(file_path: str) -> dict:
    try:
        logging.info(f"{script_name}: Trying to read YAML file from path: {file_path}")
        with open(file_path, 'r') as f:
            content = yaml.safe_load(f)
        if content is None:
            raise CustomException(f"{script_name}: YAML file is empty or null: {file_path}", sys)
        logging.info(f"{script_name}: YAML file read successfully: {file_path}")
        return content
    except CustomException:
        raise                        # let your own exceptions pass through clean
    except Exception as e:
        raise CustomException(e, sys)