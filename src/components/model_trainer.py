import os
import sys

from src.exception import CustomException
from src.logger import logging
from src.utils import (
    save_object,
    evaluate_models,
    read_yaml
)
# from src.components.data_transformation import DataTransformation


from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    )
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from sklearn.metrics import r2_score

from src.constants import ROOT_DIR

script_name = os.path.basename(__file__) # to get the name of the current script file

from dataclasses import dataclass
@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')
    # model_params_file_path: str = os.path.join(ROOT_DIR, 'config', 'model_params.yaml')
    model_params_file_path: str = os.path.join('config', 'model_params.yaml')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.config: str = read_yaml(self.model_trainer_config.model_params_file_path)  # Load the YAML configuration


    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info(f"{script_name} Splitting training and testing input data")
            X_train, y_train = train_array[:,:-1], train_array[:,-1]
            X_test, y_test = test_array[:,:-1], test_array[:,-1]

            models = {
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "XGB Regressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False)
            }

            # load hyperparameter grid from YAML config # lagacy code
            # config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'model_params.yaml')
            # config_path = os.path.abspath(config_path)
            # params = read_yaml(config_path)


            # config_path = os.path.join(ROOT_DIR, 'config', 'model_params.yaml') #legacy code
            # params = read_yaml(self.model_trainer_config.model_params_file_path) #legacy code
            params = self.config['model_params'] # Load hyperparameter grid from YAML config
            threshold = self.config['training']['model_score_threshold'] # Load threshold from YAML config

            logging.info(f"{script_name} Training and evaluating models")
            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                models=models,
                param=params
            )
            logging.info(f"{script_name} Model evaluation completed.")
            
            # to get the best model score from the dictionary
            logging.info(f"{script_name} finding the best model score from the evaluation report.")
            best_model_score = max(sorted(model_report.values())) 
            logging.info(f"{script_name} Best model score found: {best_model_score}")

            # to get the best model name from the dictionary
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            logging.info(f"{script_name} Best model name found: {best_model_name}")

            best_model = models[best_model_name]

            if best_model_score < threshold:
                raise CustomException(f"{script_name} No best model found with score greater than {threshold}", sys)
            logging.info(f"{script_name} Best model score: {best_model_score:.4f} and Best model name: {best_model_name}")
            logging.info(f"{script_name} Best found model on both training and testing dataset is {best_model_name} with r2 score: {best_model_score:.4f}")
            
            logging.info(f"{script_name} Saving the best model to the file: {self.model_trainer_config.trained_model_file_path}")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info(f"{script_name} Best model saved successfully at {self.model_trainer_config.trained_model_file_path}")

            prediction = best_model.predict(X_test)
            logging.info(f"{script_name} Prediction on test data completed using the best model: {best_model_name}.")
            r2_square = r2_score(y_test, prediction)
            logging.info(f"{script_name} R2 square value: {r2_square:.4f}")
            return r2_square

            
        except Exception as e:
            logging.error(f"Error in model training: {e}")
            raise CustomException(e, sys)
