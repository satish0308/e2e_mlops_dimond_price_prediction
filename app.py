from flask import Flask,request,render_template

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from src.DiamondPricePredictor.pipeline.stage_03_predict_pipeline import  CustomException,CustomData,predictpipeline


application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            carat=float(request.form.get('carat')),
            color=request.form.get('color'),
            cut=request.form.get('cut'),
            depth=float(request.form.get('depth')),
            table=float(request.form.get('table')),
            clarity=request.form.get('clarity'),
            x=float(request.form.get('x')),
            y=float(request.form.get('y')),
            z=float(request.form.get('z'))
        )  
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=predictpipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)