
from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipline.predict_pipline import CustomData,PredictPipeline

app=Flask(__name__)



## Route for a home page

#@app.route('/')
#def index():
#   return render_template('index.html') 
predict_pipeline=PredictPipeline()  #loading model

@app.route('/',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))

        )
        pred_df=data.get_data_as_dataframe()
        print(pred_df)
        print("__Mid Prediction__")
        results=predict_pipeline.predict(pred_df)
        print("__after Prediction__")
        return render_template('home.html',results=round(results[0],2))
    

if __name__=="__main__":
    app.run(host="0.0.0.0")        