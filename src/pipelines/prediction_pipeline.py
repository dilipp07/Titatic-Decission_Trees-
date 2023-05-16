import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd
import numpy as np

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')
            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)
            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            logging.info("done prediction")
            return pred
            logging.info("returned pred value")

            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)




class CustomData:

    def __init__(self,
    
    Pclass:float,
    Sex:str,
    
    SibSp:float,
    
    Parch:float,
    Embarked:str):
    
        
        
        
        self.Pclass=Pclass,
        self.Sex=Sex,
        self.SibSp=SibSp,
        
        self.Parch=Parch,
        self.Embarked=Embarked
    
        

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                
                
                'Pclass':[self.Pclass][0],
                'Sex':[self.Sex][0],
                'SibSp':[self.SibSp][0],
                
                'Parch':[self.Parch][0],
                'Embarked':[self.Embarked][0],
         
            
                }

            df = pd.DataFrame(custom_data_input_dict)
           
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)