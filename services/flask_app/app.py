import json

from flask import Flask, render_template, redirect, url_for, request
from flask_wtf import FlaskForm
from requests.exceptions import ConnectionError
from wtforms import IntegerField, SelectField
from wtforms.validators import DataRequired, InputRequired, NumberRange

import pandas as pd
import numpy as np

from model import get_model

MEAN_COUNT_CLAIMS = 1
MEAN_COUNT_CLAIMS_PROBA = 30.54
MEAN_AVG_CLAIM = 8641
MEAN_CLAIM_AMOUNT = 17283


model_avgclaim, model_claimcounts = get_model()


def xgb_eval_dev_gamma(yhat, dtrain):
    y = dtrain.get_label()
    return 'dev_gamma', 2 * np.sum(-np.log(y / yhat) + (y - yhat) / yhat)


app = Flask(__name__)
app.config.update(
    CSRF_ENABLED=True,
    SECRET_KEY='you-will-never-guess',
)


class ClientDataForm(FlaskForm):
    exposure = IntegerField('Срок действия полиса(мес)', validators=[DataRequired()])
    age = IntegerField('Возраст водителя(лет)',
                       validators=[NumberRange(min=18, max=110, message="Возраст от 18 до 110"),
                                   InputRequired()])
    gender = SelectField('Пол', choices=[(1, 'М'), (0, 'Ж')])
    mari_stat = SelectField('Семейное положение', choices=[(0, 'Холост'), (1, 'Другое')])
    lic_age = IntegerField('Водительский стаж(лет)',
                           validators=[NumberRange(min=1, max=92, message="Водительский стаж от 1 до 92"),
                                       InputRequired()])
    bonus_malus = SelectField('Класс КБМ', choices=[(0, '0'), (1, '1'), (2, '2'), (3, '3'), (4, '4'), (5, '5'),
                                                    (6, '6'), (7, '7'), (8, '8'), (9, '9'), (10, '10'), (11, '11'),
                                                    (12, '12'), (13, '13')])


def predict(insurance_data):
    df_for_response = pd.DataFrame(data=[[insurance_data['Exposure'],
                                          insurance_data['Gender'],
                                          insurance_data['MariStat'],
                                          insurance_data['BonusMalus'],
                                          insurance_data['DrivAgeSq'],
                                          insurance_data['LicAgeYr']]],
                                   columns=['Exposure', 'Gender', 'MariStat', 'BonusMalus', 'DrivAgeSq', 'LicAgeYr'])

    prediction_avgclaim = model_avgclaim.predict(df_for_response)
    prediction_claimcounts = [[i + 1, int(v * 100)] for i, v in
                              enumerate(model_claimcounts.predict_proba(df_for_response)[:,1].tolist())]
    prediction_claimcounts = sorted(prediction_claimcounts, key=lambda x: x[1], reverse=True)[0]

    claimamount = prediction_avgclaim * prediction_claimcounts[0]

    value_avgclaim = prediction_avgclaim[0].round(2)
    value_claimcounts = int(prediction_claimcounts[0])
    value_claimcounts_probability = int(prediction_claimcounts[1])
    value_claimamount = claimamount[0].round(2)

    if value_claimcounts > MEAN_COUNT_CLAIMS:
        verdict_claimcounts = 'Высокая аварийность'
    elif value_claimcounts == MEAN_COUNT_CLAIMS and value_claimcounts_probability > MEAN_COUNT_CLAIMS_PROBA:
        verdict_claimcounts = 'Умеренная аварийность'
    else:
        verdict_claimcounts = 'Низкая аварийность'

    if value_avgclaim > MEAN_AVG_CLAIM * 1.2:
        verdict_avgclaim = 'Высокий средний убыток'
    elif value_avgclaim < MEAN_AVG_CLAIM * 1.2:
        verdict_avgclaim = 'Низкий средний убыток'
    else:
        verdict_avgclaim = 'Умеренный средний убыток'

    if value_claimamount > MEAN_CLAIM_AMOUNT * 1.2:
        verdict_claimamount = 'Высокий общий убыток'
    elif value_claimamount < MEAN_CLAIM_AMOUNT * 1.2:
        verdict_claimamount = 'Низкий общий убыток'
    else:
        verdict_claimamount = 'Умеренный общий убыток'

    result = {'value_avgclaim': str(value_avgclaim),
              'value_claimcounts': str(value_claimcounts),
              'value_claimcounts_probability': str(value_claimcounts_probability),
              'value_claimamount': str(value_claimamount),
              'verdict_claimcounts': verdict_claimcounts,
              'verdict_avgclaim': verdict_avgclaim,
              'verdict_claimamount': verdict_claimamount}

    return json.dumps(result)


@app.route("/")
def index():
    return render_template('index.html')


# @app.route('/predicted/')
# def predicted():
#
#     response = json.loads(request.args.get('response'))
#     return render_template('predicted.html', response=response)


@app.route('/predict', methods=['GET', 'POST'])
def predict_form():
    form = ClientDataForm(request.form)
    data = {}
    if request.method == 'POST' and form.validate_on_submit():
        data['DrivAgeSq'] = float(request.form.get('age')) ** 2
        data['LicAgeYr'] = float(request.form.get('lic_age'))
        data['Gender'] = float(request.form.get('gender'))
        data['MariStat'] = float(request.form.get('mari_stat'))
        data['Exposure'] = float(request.form.get('exposure')) / 12
        data['BonusMalus'] = float(request.form.get('bonus_malus'))
        try:
            response = json.loads(predict(data))
        except ConnectionError:
            response = json.dumps({"error": "ConnectionError"})
        # return redirect(url_for('predicted', response=response))
        return render_template('predicted.html', response=response)
    return render_template('form.html', form=form)


if __name__ == '__main__':
    app.run()
