import json
import sys
import re
import logging
import click

from flask import Flask, render_template, redirect, url_for, request, session
from flask_wtf import FlaskForm
from requests.exceptions import ConnectionError
from wtforms import widgets, IntegerField, SelectField, StringField, Field
from wtforms.validators import DataRequired, Regexp, Optional, ValidationError

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


class IntegerFieldCustom(Field):
    widget = widgets.TextInput()

    def __init__(self, label=None, validators=None, **kwargs):
        super(IntegerFieldCustom, self).__init__(label, validators, **kwargs)

    def _value(self):
        if self.raw_data:
            return self.raw_data[0]
        elif self.data is not None:
            return text_type(self.data)
        else:
            return ''

    def process_data(self, value):
        if value is not None and value is not unset_value:
            try:
                self.data = int(value)
            except (ValueError, TypeError):
                self.data = None
                raise ValueError(self.gettext('%s: поле может содержать только число.' % self.label.text))
        else:
            self.data = None

    def process_formdata(self, valuelist):
        if valuelist:
            try:
                self.data = int(valuelist[0])
            except ValueError:
                self.data = None
                raise ValueError(self.gettext('%s: поле может содержать только число.' % self.label.text))


class ClientDataForm(FlaskForm):

    def validate_name_dish(form, field):
        included_chars = re.findall(r'[а-яА-Я]+', field.data)
        logger.info('Custom validate string RUN')
        logger.info(field.data)
        logger.info(included_chars)

        if not included_chars:
            logger.info('validate name_dish fail')
            raise ValidationError('%s: поле должно содержать текст кириллицей.' % field.label.text)

    name_dish = StringField('Название блюда', 
        validators=[#Regexp(r'^\W+$', message='Поле должно содержать текст кириллицей.'),
                    validate_name_dish,
                    DataRequired(),
                    ])
    product_description = StringField('Описание блюда', validators=[Optional()])
    price = IntegerFieldCustom('Цена', validators=[Optional()])


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
        words_in_data = [x for x in list(set(data['name_dish'].values[0].split(' ') \
                         + data['product_description'].values[0].split(' '))) if x != '']
        word_importance = {}
        
        if len(words_in_data) > 1:
            for word in words_in_data:
                name_dish_ww = ' '.join([w for w in name_dish.split(' ') if w!=word])
                product_description_ww = ' '.join([w for w in product_description.split(' ') if w!=word])
                
                data_ww = pd.DataFrame(data=[[name_dish_ww, product_description_ww]],
                                                columns=['name_dish', 'product_description'])
                
                clean_data_ww = self.cleaner.clean_data(data_ww)

                clean_w = self.cleaner.clean_data(pd.DataFrame(data=[[word, '']],
                                                columns=['name_dish', 'product_description']))
                
                invalid_data = [x for x in list(set(clean_w['name_dish'].values[0].split(' ') \
                         + clean_w['product_description'].values[0].split(' '))) if x != '']
                empty_data = [x for x in list(set(clean_data_ww['name_dish'].values[0].split(' ') \
                         + clean_data_ww['product_description'].values[0].split(' '))) if x != '']
                
                if invalid_data and empty_data:
                    data_for_predict_ww = self.feature_extractor.get_features(clean_data_ww)
                    predict_ww = self.model(*data_for_predict_ww).detach().numpy()

                    predict_label_ww = np.argmax(predict_ww)
                    if predict_label_ww == predict_label:
                        word_importance[word] = list(predict_ww[:, predict_label_ww] - predict_proba)[0]
                    else:
                        word_importance[word] = 'another_predict'
                elif not invalid_data:
                    word_importance[word] = 'invalid_data'
                elif not empty_data:
                    word_importance[word] = 'normal'

            logger.info('Finish of process getting importance of words')

            logger.info('Normalize importance')
            min_importance = np.inf
            max_importance = -np.inf        
            for word in word_importance:
                if word_importance[word] not in ['another_predict', 'normal', 'invalid_data']:
                    if word_importance[word] > max_importance:
                        max_importance = word_importance[word]
                    if word_importance[word] < min_importance:
                        min_importance = word_importance[word]


            logger.info('Get class css for words')
            for word in word_importance:
                if word_importance[word] not in ['another_predict', 'normal', 'invalid_data']:
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
                elif word_importance[word] == 'another_predict':
                    word_importance[word] = {'class': 'another_predict', 'imp': 'another_predict'}
                elif word_importance[word] == 'normal':
                    word_importance[word] = {'class': 'normal', 'imp': 'NaN'}
                elif word_importance[word] == 'invalid_data':
                    word_importance[word] = {'class': 'invalid_data', 'imp': 'NaN'}
        else:
            word_importance[words_in_data[0]] = {'class': 'normal', 'imp': 'NaN'}


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
        except KeyError as ke:
            response = {'name_dish': data['name_dish'], 
                      'product_description': data['product_description'], 
                      'price': data['price'],
                      'error': str(ke)
                      }
            # response = json.dumps(result)
            return render_template('form.html', response=response, form=form)
        
        logger.info('Load predicted.html')
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
