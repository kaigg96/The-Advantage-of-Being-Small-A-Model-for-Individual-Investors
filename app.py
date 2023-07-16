# 1. Imports
from flask import Flask, render_template, session, url_for, redirect
from flask_wtf import FlaskForm
from wtforms import SubmitField, StringField
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import datetime
import tables

# 2. create an instance of the Flask class
app = Flask(__name__)
app.config['SECRET_KEY'] = 'asecretkey'

# 3. define a prediction function
# 3a. load data
combined_df = pd.read_hdf('combined_df.h5', key='combined_df')

def return_prediction(deployment_test_dates, model):
    """
    Full deployment function.
    """
    
    X = combined_df.drop(columns=['Fwd rets'])

    return X[model.predict(X)==1]

# 4. load our moment predictor model
model = joblib.load('stock_predictor.joblib')

# 5. create a WTForm Class
class PredictForm(FlaskForm):
    print('predictform')
    text = StringField("Stock")
    submit = SubmitField("Predict")

# 6. set up our home page
@app.route("/", methods=["GET", "POST"])
def index():
    # Create instance of the form
    form = PredictForm()

    # Validate the form
    if form.validate_on_submit():
        session['Stock'] = form.text.data
        return redirect(url_for("prediction"))

    return render_template('home.html', form=form)

# 7. define a new "prediction" route that processes form input and returns a model prediction
@app.route('/prediction')
def prediction():
    content = {}
    content['text'] = str(session['Stock'])
    results = return_prediction(content['text'], model)
    return render_template('prediction.html', results=results)

# 8. allows us to run flask using $ python app.py
if __name__ == '__main__':
    app.run()
