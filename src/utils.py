
import os
import sys
import pickle
import numpy as np 
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(X_train,y_train,X_test,y_test):
    try:
        model=DecisionTreeClassifier()
        param_grid = {
            'max_depth': [3, 4, 5, 6],
            'min_samples_split': [2, 3, 4, 5],
            'min_samples_leaf': [1, 2, 3]
        }
        grid_search=GridSearchCV(estimator=model,param_grid=param_grid,cv=10)
        grid_search.fit(X_train,y_train)

        

        model.set_params(**grid_search.best_params_)
        model.fit(X_train,y_train)

        best_params=grid_search.best_params_
        
        # Predict Testing data
        y_test_pred =model.predict(X_test)

        tn, fp, fn, tp = confusion_matrix(y_test,y_test_pred).ravel()
        accuracy = (tp+tn)/(tp+tn+fp+fn)
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        specificity = tn/(fp+tn)
        F1_score = 2*recall*precision/(recall+precision)
        auc = roc_auc_score(y_test,y_test_pred)
        report = {  "model":model,
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "Specificity": specificity,
                    "F1 Score": F1_score,
                    'auc score':auc,
                    'best_para':best_params}
        

        return report

    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys)

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)
        
