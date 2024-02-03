import os
import sys
from dataclasses import dataclass
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
# Modelling
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
#import warnings
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder,OrdinalEncoder
import os
from box.exceptions import BoxValueError
import yaml
import json
import joblib
from DiamondPricePredictor.logger import logging
from DiamondPricePredictor.exception import CustomException
from DiamondPricePredictor.utils.common import save_obj
from ensure import ensure_annotations
from box import ConfigBox
from typing import Any,List
import base64
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


@ensure_annotations
def PrepareData(path_of_files: Path,Target_Column:str,Drop_Columns:List):
    try:
        print(path_of_files)
        df=pd.read_csv(path_of_files)
        X = df.drop(columns=[Target_Column],axis=1)
       
        X.drop(columns=Drop_Columns,inplace=True)
        
        y=df[[Target_Column]]
        num_features = X.select_dtypes(exclude="object").columns
        cat_features = X.select_dtypes(include="object").columns
        return X,y
    except BoxValueError:
        raise ValueError("file is empty")
    except Exception as e:
        raise e

@ensure_annotations
def SplitDataSet(feature_data:pd.DataFrame,Target_data:pd.DataFrame,split_ratio:float,random_state:int):
    
    
    try:
        split_ratio=float(split_ratio)
        x_train,x_test,y_train,y_test = train_test_split(feature_data,Target_data,test_size=split_ratio,random_state=random_state)
        logging.info(f"The shape of X_train {x_train.shape}")
        logging.info(f"The head of X_test {x_train.head}")
        logging.info(f"The shape of X_test {x_test.shape}")
        logging.info(f"The head of X_train {x_test.head}")
        logging.info(f"The shape of y_train {y_train.shape}")
        logging.info(f"The head of y_train {y_train.head}")
        logging.info(f"The shape of y_train {y_test.shape}")
        logging.info(f"The head of y_test {y_test.head}")
        return x_train,y_train,x_test,y_test
    except BoxValueError:
        raise ValueError("file is empty")
    except Exception as e:
        raise e
    

def evaluate_model(x_train,y_train,x_test,y_test,models,param):
    try:
        report={}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param_grid=param[list(models.keys())[i]]
            print(model)
            print(param_grid)
            model_name=model
            logging.info(f"the Current model select {param_grid} for the model {model}")
            model = eval(model)
            gs=GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
            
            gs.fit(x_train,y_train)
            
            model.set_params(**gs.best_params_)
            logging.info(f"The Best Working Model {model}")
            model.fit(x_train,y_train)
            joblib.dump(model, 'artifacts/prepare_base_model/'+model_name[:-2]+'.joblib')
            y_train_pred=model.predict(x_train)
            y_test_pred=model.predict(x_test)

            train_model_score=r2_score(y_train,y_train_pred)
            test_model_score=r2_score(y_test,y_test_pred)

        
            report[list(models.keys())[i]] = test_model_score
        return report

    except Exception as e:
        raise e

@ensure_annotations   
def DataTransformation(x_train:pd.DataFrame,y_train:pd.DataFrame,x_test:pd.DataFrame,y_test:pd.DataFrame,repro_path):
    
    
    try:
        num_features = x_train.select_dtypes(exclude="object").columns
        cat_features = x_train.select_dtypes(include="object").columns
        
        logging.info(f"{cat_features} These Are Categorical features ")
        logging.info(f"{num_features} These Are Numerical features")

        num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())

                ]

            )
            
            # Categorigal Pipeline
        cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder()),
                ('scaler',StandardScaler())
                ]

            )
            
        preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,num_features),
            ('cat_pipeline',cat_pipeline,cat_features)
            ])
        
        x_train_t=preprocessor.fit_transform(x_train)
       
        x_test_t=preprocessor.transform(x_test)
    
        preproc_path = Path(repro_path).joinpath('preprocessor.joblib')

        save_obj(
                file_path=preproc_path,
                obj=preprocessor
            )
       
        #def transform_lebal_data(data):
        #    log_price_data = np.log1p(data)
        #    normalized_log_price_data = (log_price_data - log_price_data.min()) / (log_price_data.max() - log_price_data.min())
        #    return normalized_log_price_data
        

        #y_train_t = transform_lebal_data(y_train)
        #y_test_t = transform_lebal_data(y_test)
        
        y_train_t = y_train
        y_test_t = y_test

        

        return  x_train_t,y_train_t,x_test_t,y_test_t
    except BoxValueError:
        raise ValueError("file is empty")
    except Exception as e:
        raise CustomException(e,sys)
    



def best_score(path_of_files,best_model,x_test_t,y_test_t):
    try:

        path_of_model = os.path.join(path_of_files, best_model[:-2] + '.joblib')
        model = joblib.load(path_of_model)

   
        predictions = model.predict(x_test_t)
        r2_score_best_model=r2_score(y_test_t,predictions)
        return r2_score_best_model 
    

    except BoxValueError:
        raise ValueError("file is empty")
    except Exception as e:
        raise e
    

    