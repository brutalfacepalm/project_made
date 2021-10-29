import json

from flask import Flask, render_template, redirect, url_for, request, session
from flask_wtf import FlaskForm
from requests.exceptions import ConnectionError
from wtforms import IntegerField, SelectField, StringField
from wtforms.validators import DataRequired, InputRequired, NumberRange

import pandas as pd
import numpy as np

from model import get_model
from preprocessing import Preprocessing
from feature_extractor import FeatureExtractor

MEAN_COUNT_CLAIMS = 1
MEAN_COUNT_CLAIMS_PROBA = 30.54
MEAN_AVG_CLAIM = 8641
MEAN_CLAIM_AMOUNT = 17283


model = get_model()

app = Flask(__name__)
app.config.update(
    CSRF_ENABLED=True,
    SECRET_KEY='you-will-never-guess',
)

class ClientDataForm(FlaskForm):
    name_dish = StringField('Название блюда', validators=[DataRequired()])
    product_description = StringField('Описание блюда', validators=[DataRequired()])


class Predictions:

    def __init__(self):
        self.results = []

    def predict(self, insurance_data):
        df_for_response = pd.DataFrame(data=[[insurance_data['name_dish'],
                                              insurance_data['product_description']]],
                                       columns=['name_dish', 'product_description'])

        df_for_response = Preprocessing.transform(df_for_response)
        df_for_response = FeatureExtractor.transform(df_for_response)

        predict = model(df_for_response)

        result = {'name_dish': insurance_data['name_dish'],
                  'product_description': insurance_data['product_description'],
                  'prediction': predict}
        
        if result not in self.results:
            self.results.append(result)

        return json.dumps(self.results)


predictions = Predictions()


@app.route("/")
def index():
    predictions.results = []
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict_form():
    form = ClientDataForm(request.form)
    data = {}
    while True:
        if request.method == 'POST' and form.validate_on_submit():
            data['name_dish'] = request.form.get('name_dish')
            data['product_description'] = request.form.get('product_description')
            try:
                response = json.loads(predictions.predict(data))
            except ConnectionError:
                response = json.dumps({"error": "ConnectionError"})
            return render_template('predicted.html', response=response, form=form)
        return render_template('form.html', form=form)

    

if __name__ == '__main__':
    app.run()
