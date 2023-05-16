import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_model
from sklearn.tree import DecisionTreeClassifier
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_model

from dataclasses import dataclass
import sys
import os


@dataclass
class ModelTrainerConfig:

    trained_model_file_path=os.path.join('artifacts','model.pkl')
class ModelTrainer:

    def __init__(self):

        self.model_trainer_config=ModelTrainerConfig()
    def initiate_model_trainer(self,train_array,test_array):
        logging.info("splitting dependent and independent features from train and test data")
        try:
                
                X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
                )



                model_report:dict = evaluate_model(X_train,y_train,X_test,y_test)
                print(model_report)
                print('\n====================================================================================\n')
                logging.info(f'Model Report : {model_report}')


                best_model=list(model_report.values())[0]
                print(best_model)

                save_object(
                    file_path=self.model_trainer_config.trained_model_file_path,
                    obj=best_model
                )
          

        
        except Exception as e:
             logging.info("Exception occured at model trainer")
             raise CustomException(e,sys)