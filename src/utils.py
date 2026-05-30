import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import dill as pickle

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(X_train,y_train,X_test,y_test,models):
    try:
        report={}
        for i in range(len(list(models))):
            model=list(models.values())[i] #getting the model from the dictionary

            model.fit(X_train,y_train) #training that model

            y_train_pred=model.predict(X_train) #predicting the training set results
            y_test_pred=model.predict(X_test) #predicting the test set results

            train_model_score=r2_score(y_train,y_train_pred) #calculating the r2 score for the model on training data
            test_model_score=r2_score(y_test,y_test_pred) #calculating the r2 score for the model on test data

            report[list(models.keys())[i]]=test_model_score #storing the r2 score in the report dictionary with the name of the model as the key and the r2 score as the value

        return report

    except Exception as e:
        raise CustomException(e,sys)