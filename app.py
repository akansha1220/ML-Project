
from flask import Flask,request,render_template
import sys

from sklearn.preprocessing import StandardScaler
from src.pipline.predict_pipline import CustomData,PredictPipeline
from src.logger import logging
from src.exception import CustomException

# Register a new event source
# Now log an event
app=Flask(__name__)

predict_pipeline=PredictPipeline()  #loading model

@app.route('/',methods=['GET','POST'])
def predict_datapoint():
    try:
        if request.method=='GET':
            logging.info("Get method recieve")
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
            logging.info("User inputs recieved")
            logging.warning("User input recieved")

            pred_df=data.get_data_as_dataframe()

            logging.info("Mid prediction completed")
            results=predict_pipeline.predict(pred_df)

            logging.info("Get the result successfully")
            return render_template('home.html',results=round(results[0],2))
        
    except Exception as E:
        logging.critical(f"Error occur while geting the output at {E}")
        raise CustomException(E,sys)
    

if __name__=="__main__":
    app.run(host="0.0.0.0")  
          