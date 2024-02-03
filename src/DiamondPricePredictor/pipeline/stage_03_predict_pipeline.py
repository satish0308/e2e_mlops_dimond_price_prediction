import sys
import os
import pandas as pd
from DiamondPricePredictor.exception import CustomException
from DiamondPricePredictor.logger import logging
import joblib
import json
from pathlib import Path


class predictpipeline:
    def __init__(self):
        pass
    def predict(self,feaures):
        try: 
            
            path_of_files = 'artifacts/prepare_base_model/best_model.json'

            # Read the content of the JSON file
            with open(path_of_files, 'r') as file:
                model_dict = json.load(file)

            model_name = model_dict.get("best_model", {})


            base_path=Path('artifacts\prepare_base_model')

            path_of_model = os.path.join(base_path, model_name[:-2] + '.joblib')
                                
                        

            model = joblib.load(path_of_model)
        
            path_of_preprocessor=Path('artifacts\prepare_base_model\preprocessor.joblib')
            preprocessor=joblib.load(path_of_preprocessor)

            data_scaled=preprocessor.transform(feaures)
            preds=model.predict(data_scaled) 
            return preds
        except Exception as e:
            raise CustomException(e,sys)
class CustomData:
    def __init__(self,carat:float,cut:str,color:str,clarity:float,depth:float,table:float,x:float,y:float,z:float):
        self.carat=carat
        self.cut=cut
        self.color=color
        self.clarity=clarity
        self.depth=depth
        self.table=table
        self.x=x
        self.y=y
        self.z=z

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict={
                "carat":[self.carat],
                "cut":[self.cut],
                "color":[self.color],
                "clarity":[self.clarity],
                "depth":[self.depth],
                "table":[self.table],
                "x":[self.x],
                "y":[self.y],
                "z":[self.z]
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)

