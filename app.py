from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipline.predict_pipline import CustomData,PredictPipelin

app = Flask(__name__)

#route for a home page

@app.route('/predictdata',methods = ['GET','POST'])
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        #here reading all the gender by using request.form.post
        data=CustomData(
            gender= request.form.get('gender'),
            race_ethnicity= request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get("parental_level_of_education"),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score= request.form.get('reading_score'),
            writing_score=request.form.get('writing_score')
        )

        pred_df = data.get_data_as_dataframe()    #it is converting all the data into datframe
        print(pred_df)

        Predict_Pipelin= PredictPipelin()
        res = Predict_Pipelin.predict(pred_df)
        return render_template('home.html',results = res[0])   #returning the home page with the result output
    

if __name__=="__main__":
    app.run(host="0.0.0.0",debug = True)