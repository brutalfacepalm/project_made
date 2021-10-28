import json

from flask import Flask, render_template, redirect, url_for, request
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

def xgb_eval_dev_gamma(yhat, dtrain):
    y = dtrain.get_label()
    return 'dev_gamma', 2 * np.sum(-np.log(y / yhat) + (y - yhat) / yhat)


app = Flask(__name__)
app.config.update(
    CSRF_ENABLED=True,
    SECRET_KEY='you-will-never-guess',
)

class ClientDataForm(FlaskForm):
    name_dish = StringField('Название блюда', validators=[DataRequired()])
    product_description = StringField('Описание блюда', validators=[DataRequired()])


def predict(insurance_data):
    df_for_response = pd.DataFrame(data=[[insurance_data['Dish'],
                                          insurance_data['Description']]],
                                   columns=['Dish', 'Description'])

    df_for_response = Preprocessing.transform(df_for_response)
    df_for_response = FeatureExtractor.transform(df_for_response)

    prediction = model(df_for_response)

    result = {'prediction': prediction}

    return json.dumps(result)


@app.route("/")
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict_form():
    form = ClientDataForm(request.form)
    data = {}
    if request.method == 'POST' and form.validate_on_submit():
        data['Dish'] = request.form.get('dish')
        data['Description'] = request.form.get('description')
        try:
            response = json.loads(predict(data))
        except ConnectionError:
            response = json.dumps({"error": "ConnectionError"})
        return render_template('predicted.html', response=response)
    return render_template('form.html', form=form)


if __name__ == '__main__':
    app.run()
