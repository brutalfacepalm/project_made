import json
import sys
import re
import logging
import click
import torch.nn.functional

from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from requests.exceptions import ConnectionError
from wtforms import widgets, StringField, Field
from wtforms.validators import DataRequired, Optional, ValidationError
from werkzeug.datastructures import ImmutableMultiDict

import pandas as pd
import numpy as np
from torch import no_grad

PATHFORMODEL = '../prodmodel'
sys.path.append(PATHFORMODEL)

from modelbert import get_model
from preprocessing import get_cleaner
from berttokenizer import get_feature_extractor
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

    name_dish = StringField('Название блюда', validators=[validate_name_dish, DataRequired()])
    product_description = StringField('Описание блюда', validators=[validate_name_dish, DataRequired()])
    price = IntegerFieldCustom('Цена', validators=[Optional()])


class Predictions:

    def __init__(self, cleaner, feature_extractor, model, decode_predict):
        self.cleaner = cleaner
        self.feature_extractor = feature_extractor
        self.model = model
        self.model.eval()
        self.decode_predict = decode_predict
        self.results = []
        self.last_response = {}
        self.form = None

    def create_data(self, insurance_data):
        name_dish = insurance_data['name_dish'].strip()
        product_description = insurance_data['product_description'].strip()
        price = insurance_data['price'].strip() if insurance_data['price'] else 0
        data = pd.DataFrame(data=[[name_dish, product_description, price]],
                            columns=['name_dish', 'product_description', 'price'])
        return data, name_dish, product_description, price

    def preprocess(self, data):
        cleared_data = self.cleaner.clean_data(data)
        logger.info('Data was cleaned')
        return cleared_data

    def get_data_for_predict(self, clean_data):
        data_for_predict = self.feature_extractor.get_features(clean_data)
        logger.info('Work of extractor was complete')
        return data_for_predict

    def predict_from_model(self, input_ids, attention_masks):
        with no_grad():
            predict = self.model(input_ids, token_type_ids=None,
                                 attention_mask=attention_masks)
        logger.info('Predict complete')
        return predict

    def get_top_5_predictions(self, predict):
        top_5_labels = np.argsort(predict)[0][-5:][::-1]
        top_5_cat = [self.predict_as_word(label) for label in top_5_labels]
        top_5_proba = list(map(lambda x: round(x*100, 2), predict[0][top_5_labels]))
        top_5_prediction = dict(zip(top_5_cat, top_5_proba))
        return top_5_prediction

    def predict_as_word(self, label):
        return self.decode_predict.decode_answer(label)

    def get_word_importance(self, data, clean_data, predict_label, predict_proba):
        logger.info('Start of process getting importance of words')
        words_in_data = [x for x in list(set(data['name_dish'].values[0].split(' ') \
                         + data['product_description'].values[0].split(' '))) if x != '']
        word_importance = {}
        word_collect = {}
        data_ww = {}

        for word in words_in_data:
            word_c = self.cleaner.stemmer.lemmatize(word.strip())[0]
            data_ww[word_c] = ' '.join([w for w in clean_data.split() if w != word_c])
            if word_c in word_collect:
                word_collect[word_c].append(word)
            else:
                word_collect[word_c] = [word]

        if len(data_ww) > 1:
            for word_deleted in data_ww:
                data_temp = data_ww[word_deleted]
                input_ids_temp, attention_masks_temp = self.get_data_for_predict(data_temp)
                predict_temp = self.predict_from_model(input_ids_temp, attention_masks_temp)[0]
                predict_temp = torch.nn.functional.softmax(predict_temp).numpy()
                predict_label_temp = np.argmax(predict_temp)
                predict_proba_temp = predict_temp[:, predict_label_temp]
                if predict_label_temp == predict_label:
                    imp = int((predict_proba - predict_proba_temp) * 100)
                    if imp >= 15:
                        for w in list(word_collect[word_deleted]):
                            word_importance[w] = {'class': 'high_positive', 'imp': 3}
                    elif imp >= 10:
                        for w in list(word_collect[word_deleted]):
                            word_importance[w] = {'class': 'mid_positive', 'imp': 2}
                    elif imp >= 5:
                        for w in list(word_collect[word_deleted]):
                            word_importance[w] = {'class': 'low_positive', 'imp': 1}
                    elif imp > -5:
                        for w in list(word_collect[word_deleted]):
                            word_importance[w] = {'class': 'normal', 'imp': 0}
                    elif imp > -10:
                        for w in list(word_collect[word_deleted]):
                            word_importance[w] = {'class': 'low_negative', 'imp': -1}
                    elif imp > -15:
                        for w in list(word_collect[word_deleted]):
                            word_importance[w] = {'class': 'mid_negative', 'imp': -2}
                    elif imp <= -15:
                        for w in list(word_collect[word_deleted]):
                            word_importance[w] = {'class': 'high_negative', 'imp': -3}
                else:
                    for w in list(word_collect[word_deleted]):
                        word_importance[w] = {'class': 'another_predict', 'imp': 'another_predict'}
        else:
            for w in words_in_data:
                word_importance[w] = {'class': 'normal', 'imp': 'NaN'}

        return word_importance

    def predict(self, insurance_data):
        data, name_dish, product_description, price = self.create_data(insurance_data)
        clean_data = self.preprocess(data)
        input_ids, attention_masks = self.get_data_for_predict(clean_data)

        model_output = self.predict_from_model(input_ids, attention_masks)
        predict = model_output[0]

        predict = torch.nn.functional.softmax(predict).numpy()
        predict_label = np.argmax(predict)
        predict_proba = predict[:, predict_label]

        top_5_prediction = self.get_top_5_predictions(predict)

        logger.info('Decode main prediction label')
        cat_prediction = self.decode_predict.decode_answer(predict_label)

        word_importance = self.get_word_importance(data, clean_data, predict_label, predict_proba)

        logger.info(word_importance)

        logger.info('Get result')
        result = {'name_dish': {'current_name': name_dish.split(' ')},
                  'product_description':  {'current_description': product_description.split(' ')},
                  'price': price,
                  'prediction_primary': cat_prediction,
                  'predictions': top_5_prediction,
                  'word_importance': word_importance}
        
        if result not in self.results:
            self.results.append(result)

        return json.dumps(self.results)


@app.route("/")
def index():
    logger.info('Someone connect to service')
    # results = []
    predictions.form = ClientDataForm(ImmutableMultiDict([]))
    predictions.results = []
    predictions.last_response = {}
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict_form():
    logger.info('Load page with form of prediction')
    form = ClientDataForm(request.form) if request.form.get('csrf_token') else predictions.form
    predictions.form = form
    data = {}
    if request.method == 'POST' and request.form.get('csrf_token') and form.validate_on_submit():
        data['name_dish'] = request.form.get('name_dish')
        data['product_description'] = request.form.get('product_description')
        data['price'] = request.form.get('price')
        predictions.results = []
        predictions.last_response = {'name_dish': request.form.get('name_dish'),
                    'product_description': request.form.get('product_description'),
                    'price': request.form.get('price')}
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
            return render_template('form.html', response=response, form=form)
        
        logger.info('Load predicted.html')
        return render_template('predicted.html', response=response, form=form)
    logger.info('Load form.html')

    if not request.form:
        return render_template('form.html', response=predictions.last_response, form=form)
    else:
        if 'csrf_token' not in list(request.form.keys()):
            deleted_key = list(request.form.keys())[0]
            data_request = request.form.get(deleted_key).replace('\'', '"')
            data_request = json.loads(data_request)

            deleted_word = data_request['deleted']
            data_request[deleted_key] = ' '.join([word for word in data_request[deleted_key].split() if word != deleted_word])

            data['name_dish'] = data_request.get('name_dish')
            data['product_description'] = data_request.get('product_description')
            data['price'] = data_request.get('price')

            logger.info('Predict without one word')
            response = json.loads(predictions.predict(data))
            return render_template('predicted.html', response=response, form=form)
        else:
            return render_template('form.html', response='', form=form)


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
    feature_extractor = get_feature_extractor()
    logger.info('Create feature extractor')
    model = get_model(PATHFORMODEL)
    logger.info('Create model')
    decode_predict = get_decoder(PATHFORMODEL)
    logger.info('Create decoder of prediction')

    predictions = Predictions(cleaner, feature_extractor, model, decode_predict)

    run_app()
