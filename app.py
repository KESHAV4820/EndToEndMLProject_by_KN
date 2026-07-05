from flask import Flask,request,render_template
import pandas as pd
import numpy as np
import os

from src.exception import CustomException
from src.logger import logging

from sklearn.preprocessing import StandardScaler
from src.pipeline.predicting_pipeline import CustomData, PredictPipeline

script_name = os.path.basename(__file__) # to get the name of the current script file

application = Flask(__name__)
app = application

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        logging.info(dict(request.form)) # to log the form data received from the user in the POST request
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))


        )
       
        pred_df = data.get_data_as_dataframe()
        print(pred_df)
        logging.info(f"{script_name}:Dataframe for prediction: {pred_df}")

        logging.info(f"{script_name}: Prediction initiated.")
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        logging.info(f"{script_name}: Prediction completed.")
        logging.info(f"{script_name}: Prediction results: {results[0]}")

        return render_template('home.html', results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0",port=8087,debug=True)


