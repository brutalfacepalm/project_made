import json
import sys
import logging
import click

from flask import Flask, render_template, redirect, url_for, request, session
from flask_wtf import FlaskForm
from requests.exceptions import ConnectionError
from wtforms import IntegerField, SelectField, StringField
from wtforms.validators import DataRequired, InputRequired, NumberRange

import pandas as pd
import numpy as np
from pymystem3 import Mystem

PATHFORMODEL = '../prodmodel'
sys.path.append(PATHFORMODEL)

from model import get_model
from preprocessing import get_cleaner
from featureextractor import get_feature_extractor
from decodepredict import get_decoder
from utils import get_logger


app = Flask(__name__)
app.config.update(
    CSRF_ENABLED=True,
    SECRET_KEY='you-will-never-guess',
)


class ClientDataForm(FlaskForm):
    name_dish = StringField('Название блюда', validators=[DataRequired()])
    product_description = StringField('Описание блюда', validators=[DataRequired()])
    price = IntegerField('Цена', validators=[DataRequired()])


class Predictions:

    def __init__(self, cleaner, feature_extractor, model, decode_predict):
        self.cleaner = cleaner
        self.feature_extractor = feature_extractor
        self.model = model
        self.decode_predict = decode_predict
        # self.results = []

    def predict(self, insurance_data):
        name_dish = insurance_data['name_dish'].strip()
        product_description = insurance_data['product_description'].strip()
        price = insurance_data['price'].strip()
        data = pd.DataFrame(data=[[name_dish, product_description]],
                           columns=['name_dish', 'product_description'])

        
        clean_data = self.cleaner.clean_data(data)
        logger.info('Data was cleaned')
        data_for_predict = self.feature_extractor.get_features(clean_data)
        logger.info('Work of extractor was complete')
        predict = self.model(*data_for_predict).detach().numpy()
        logger.info('Predict complete')
        predict_label = np.argmax(predict)

        predict_proba = predict[:, predict_label]


        logger.info('Start of process getting importance of words')
        words_in_data = list(set(name_dish.split(' ') + product_description.split(' ')))
        word_importance = {}
        for word in words_in_data:
            name_dish_ww = ' '.join([w for w in name_dish.split(' ') if w!=word])
            product_description_ww = ' '.join([w for w in product_description.split(' ') if w!=word])
            
            data_ww = pd.DataFrame(data=[[name_dish_ww, product_description_ww]],
                                            columns=['name_dish', 'product_description'])
            
            clean_data_ww = self.cleaner.clean_data(data_ww)
            data_for_predict_ww = self.feature_extractor.get_features(clean_data_ww)
            predict_ww = self.model(*data_for_predict_ww).detach().numpy()

            predict_label_ww = np.argmax(predict_ww)
            if predict_label_ww == predict_label:
                word_importance[word] = list(predict_ww[:, predict_label_ww] - predict_proba)[0]
            else:
                word_importance[word] = 'another_predict'

        logger.info('Finish of process getting importance of words')

        logger.info('Normalize importance')
        min_importance = np.inf
        max_importance = -np.inf        
        for word in word_importance:
            if word_importance[word] != 'another_predict':
                if word_importance[word] > max_importance:
                    max_importance = word_importance[word]
                if word_importance[word] < min_importance:
                    min_importance = word_importance[word]


        logger.info('Get class css for words')
        for word in word_importance:
            if word_importance[word] != 'another_predict':
                if min_importance != max_importance:
                    imp = int(((word_importance[word] - min_importance)/
                        (max_importance - min_importance) - 0.5)*6)*(-1)
                    if imp == -3:
                        word_importance[word] = {'class': 'high_negative', 'imp': imp}
                    elif imp == -2:
                        word_importance[word] = {'class': 'mid_negative', 'imp': imp}
                    elif imp == -1:
                        word_importance[word] = {'class': 'low_negative', 'imp': imp}
                    elif imp == 1:
                        word_importance[word] = {'class': 'low_positive', 'imp': imp}
                    elif imp == 2:
                        word_importance[word] = {'class': 'mid_positive', 'imp': imp}
                    elif imp == 3:
                        word_importance[word] = {'class': 'high_positive', 'imp': imp}
                    else:
                        word_importance[word] = {'class': 'normal', 'imp': imp}
                else:
                    word_importance[word] = {'class': 'normal', 'imp': 'NaN'}
            else:
                word_importance[word] = {'class': 'another_predict', 'imp': 'another_predict'}


        logger.info(word_importance)


        logger.info('Decode main prediction label')
        cat_prediction = self.decode_predict.decode_answer(predict_label)

        logger.info('Get result')
        result = {'name_dish': {'current_name': name_dish.split(' ')},
                  'product_description':  {'current_description': product_description.split(' ')},
                  'price': price,
                  'prediction': cat_prediction,
                  'word_importance': word_importance}
        
        # if result not in self.results:
        #     self.results.append(result)

        return json.dumps(result)


@app.route("/")
def index():
    logger.info('Someone connect to service')
    # predictions.results = []
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict_form():
    logger.info('Load page with form of prediction')
    form = ClientDataForm(request.form)
    data = {}
    # while True:
    if request.method == 'POST' and form.validate_on_submit():
        data['name_dish'] = request.form.get('name_dish')
        data['product_description'] = request.form.get('product_description')
        data['price'] = request.form.get('price')
        logger.info('Data for prediction received')
        try:
            logger.info('Try to predict')
            response = json.loads(predictions.predict(data))
        except ConnectionError:
            response = json.dumps({"error": "ConnectionError"})
            logger.error('Error connection to service')
        
        logger.info('Load predicted.heml')
        return render_template('predicted.html', response=response, form=form)
    logger.info('Load form.html') 
    return render_template('form.html', form=form)


@click.command()
@click.option('--host', '-h', default='0.0.0.0')
@click.option('--port', '-p', default='5000')
@click.option('--debug', default=False)
def run_app(host, port, debug):
    app.run(host, port, debug)


    
if __name__ == '__main__':
    logger = get_logger()

    cleaner = get_cleaner(PATHFORMODEL)
    logger.info('Create cleaner')
    feature_extractor = get_feature_extractor(PATHFORMODEL)
    logger.info('Create feature extractor')
    model = get_model(PATHFORMODEL)
    logger.info('Create model')
    decode_predict = get_decoder(PATHFORMODEL)
    logger.info('Create decoder of prediction')

    predictions = Predictions(cleaner, feature_extractor, model, decode_predict)

    run_app()
