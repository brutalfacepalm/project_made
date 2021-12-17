import json
import sys
import re
import logging
import click
from collections import OrderedDict

from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from requests.exceptions import ConnectionError
from wtforms import widgets, StringField, Field
from wtforms.validators import DataRequired, Optional, ValidationError
from werkzeug.datastructures import ImmutableMultiDict

import pandas as pd
import numpy as np
from pymystem3 import Mystem

PATHFORMODEL = '../prodmodel'
S3PATH = 'models/log_reg'
sys.path.append(PATHFORMODEL)

from utils import get_logger, load_files_from_s3

from inference import InferenceModel




app = Flask(__name__)
app.config.update(
    CSRF_ENABLED=True,
    SECRET_KEY='you-will-never-guess',
)

class LimitedSizeDict(OrderedDict):
    def __init__(self, *args, **kwds):
        self.maxlen = kwds.pop("maxlen", None)
        OrderedDict.__init__(self, *args, **kwds)
        self._check_size_limit()

    def __setitem__(self, key, value):
        OrderedDict.__setitem__(self, key, value)
        self._check_size_limit()

    def append(self, key, value):
        self.__setitem__(key, value)

    def extend(self, key, value):
        if key in list(self.keys()):
            self[key].append(value)

    def search(self, key):
        if key in list(self.keys()):
            return self[key]

    def _check_size_limit(self):
        if self.maxlen is not None:
            while len(self) > self.maxlen:
                self.popitem(last=False)

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

    def __init__(self, model, stemmer):
        self.stemmer = stemmer
        self.model = model
        self.results = LimitedSizeDict(maxlen=1000)
        self.last_response = {}
        self.form = None

    def create_data(self, insurance_data):
        name_dish = insurance_data['name_dish'].strip()
        product_description = insurance_data['product_description'].strip()
        price = insurance_data['price'].strip() if insurance_data['price'] else 0
        data = pd.DataFrame(data=[[name_dish, product_description, price]],
                            columns=['name_dish', 'product_description', 'price'])
        return data, name_dish, product_description, price


    def get_top_5_predictions(self, predict):
        top_5_labels = np.argsort(predict)[0][-5:][::-1]
        top_5_cat = [self.model.target_encoder.inverse_transform(label.reshape(1, -1))[0] for label in top_5_labels]
        top_5_proba = list(map(lambda x: round(x*100, 2), predict[0][top_5_labels]))
        top_5_prediction = dict(zip(top_5_cat, top_5_proba))

        return top_5_prediction


    def get_word_importance(self, data, clean_data, predict_label, predict_proba):
        logger.info('Start of process getting importance of words')
        predict_proba = predict_proba[:, predict_label].ravel()[0]
        words_in_data = [x for x in list(set(data['name_dish'].values[0].split(' ') \
                         + data['product_description'].values[0].split(' '))) if x != '']
        word_importance = {}
        word_collect = {}
        data_ww = {}
        for word in words_in_data:
            clean_data_ww = clean_data.copy()
            word_c = self.stemmer.lemmatize(word.strip())[0]
            clean_data_ww['name_dish'] = ' '.join([w for w in clean_data_ww['name_dish'].values[0].split() if w != word_c])
            clean_data_ww['product_description'] = ' '.join([w for w in clean_data_ww['product_description'].values[0].split() if w != word_c])
            clean_data_ww['price'] = data['price']
            data_ww[word_c] = clean_data_ww
            if word_c in word_collect:
                word_collect[word_c].append(word)
            else:
                word_collect[word_c] = [word]


        if len(data_ww) > 1:
            for word_deleted in data_ww:
                data_temp = data_ww[word_deleted]

                _, predict_label_temp, predict_temp = self.model.predict(data_temp)
                predict_proba_temp = predict_temp[:, predict_label_temp].ravel()[0]

                predict_label_temp = predict_label_temp.ravel()[0]

                if predict_label_temp == predict_label:
                    imp = int((predict_proba - predict_proba_temp) * 100)
                    if imp >= 9:
                        for w in list(word_collect[word_deleted]):
                            word_importance[w] = {'class': 'high_positive', 'imp': 3}
                    elif imp >= 6:
                        for w in list(word_collect[word_deleted]):
                            word_importance[w] = {'class': 'mid_positive', 'imp': 2}
                    elif imp >= 3:
                        for w in list(word_collect[word_deleted]):
                            word_importance[w] = {'class': 'low_positive', 'imp': 1}
                    elif imp > -3:
                        for w in list(word_collect[word_deleted]):
                            word_importance[w] = {'class': 'normal', 'imp': 0}
                    elif imp > -6:
                        for w in list(word_collect[word_deleted]):
                            word_importance[w] = {'class': 'low_negative', 'imp': -1}
                    elif imp > -9:
                        for w in list(word_collect[word_deleted]):
                            word_importance[w] = {'class': 'mid_negative', 'imp': -2}
                    elif imp <= -9:
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
        logger.info('Create data from form')
        data, name_dish, product_description, price = self.create_data(insurance_data)

        logger.info('Get clean data')
        clean_data = self.model.preprocess.transform(data)

        logger.info('Main predict')
        cat_prediction, predict_label, predict_proba = self.model.predict(data)

        logger.info('Get top-5 predict')
        top_5_prediction = self.get_top_5_predictions(predict_proba)

        logger.info('Decode main prediction label')

        word_importance = self.get_word_importance(data, clean_data, predict_label, predict_proba)

        logger.info(word_importance)

        logger.info('Get result')
        result = {'name_dish': {'current_name': name_dish.split(' ')},
                  'product_description':  {'current_description': product_description.split(' ')},
                  'price': price,
                  'prediction_primary': cat_prediction,
                  'predictions': top_5_prediction,
                  'word_importance': word_importance}
        
        # if result not in self.results:
        #     self.results.append(result)

        return json.dumps([result])


@app.route("/")
def index():
    logger.info('Someone connect to service')
    # results = []
    predictions.form = ClientDataForm(ImmutableMultiDict([]))
    # predictions.results = []
    predictions.last_response = {}
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict_form():
    logger.info('Load page with form of prediction')
    form = ClientDataForm(request.form) if request.form.get('csrf_token') else predictions.form
    predictions.form = form
    data = {}
    if request.method == 'POST' and 'first_predict' in list(request.form.keys()) and form.validate_on_submit():
        data['name_dish'] = request.form.get('name_dish')
        data['product_description'] = request.form.get('product_description')
        data['price'] = request.form.get('price')
        logger.info('Data for prediction received')
        try:
            logger.info('Try to predict')
            response = json.loads(predictions.predict(data))
            response[0]['csrf_token'] = request.form.get('csrf_token')
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

        predictions.results.append(request.form.get('csrf_token'), response)
        logger.info('Load predicted.html')
        return render_template('predicted.html', response=response, form=form)

    # elif not request.form:
    #     logger.info('Load form.html')
    #     return render_template('form.html', response=predictions.last_response, form=form)
    elif 'back' in list(request.form.keys()):
        logger.info('Load form.html')
        csrf = json.loads(request.form.get('back').replace('\'', '"'))['csrf_token']
        req = predictions.results.search(csrf)[0]
        response = {}
        response['name_dish'] = ' '.join(req['name_dish']['current_name'])
        response['product_description'] = ' '.join(req['product_description']['current_description'])
        response['price'] = req['price']
        return render_template('form.html', response=response, form=form)
    else:
        if 'name_dish' in list(request.form.keys()) or 'product_description' in list(request.form.keys()):
            deleted_key = list(request.form.keys())[0]
            data_request = request.form.get(deleted_key).replace('\'', '"')
            data_request = json.loads(data_request)
            csrf = data_request['csrf_token']
            deleted_word = data_request['deleted']
            data_request[deleted_key] = ' '.join([word for word in data_request[deleted_key].split() if word != deleted_word])

            data['name_dish'] = data_request.get('name_dish')
            data['product_description'] = data_request.get('product_description')
            data['price'] = data_request.get('price')

            logger.info('Predict without one word')
            response = json.loads(predictions.predict(data))
            predictions.results.extend(csrf, response[0])
            response = predictions.results.search(csrf)
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

    load_files_from_s3(S3PATH + '/model.pkl', PATHFORMODEL, '/dataformodel/model.pkl')
    model = InferenceModel.load(PATHFORMODEL + '/dataformodel/model.pkl')
    stemmer = Mystem()

    predictions = Predictions(model, stemmer)

    run_app()
