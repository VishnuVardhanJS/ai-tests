import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import datetime

config = {
    'days_to_forecast': 14,
    'global_degree': 5,
    'new_case_degree': 3,
    'terapia_intensiva': 2,
    'time': int(str(datetime.datetime.now().time())[0:2]),
    'update_data': 18,
    'start_date': datetime.date(2020, 3, 16)
}

def train_model(x, y, degree):
    polynomial_features = PolynomialFeatures(degree=degree)
    x_poly = polynomial_features.fit_transform(x)

    model = LinearRegression()
    model.fit(x_poly, y)
    return model


def get_predictions(x, model, degree):
    polynomial_features = PolynomialFeatures(degree=degree)
    x_poly = polynomial_features.fit_transform(x)

    return model.predict(x_poly)

def call_model(model_name, model, x, y, days_to_predict, degree):
    y_pred = np.round(get_predictions(x, model, degree), 0).astype(np.int32)

    predictions = forecast(model_name, model, degree, beginning_day=len(x), limit=days_to_predict)
    print("")
    return predictions

def forecast(model_name, model, degree, beginning_day=0, limit=10):
    next_days_x = np.array(range(beginning_day, beginning_day + limit)).reshape(-1, 1)
    next_days_pred = np.round(get_predictions(next_days_x, model, degree), 0).astype(np.int32)

    print("The results for " + model_name + " in the following " + str(limit) + " days is:")
    print(next_days_pred[-1])

    return next_days_pred

def plot_prediction(y, predictions, title):
    total_days = [datetime.date(2020, 2, 24) + datetime.timedelta(days=int(i)) for i in range(int(y.shape[0]) + predictions.shape[0])]

    if config['time'] >= config['update_data']:
        today = str(datetime.date.today())
        last_day = str(datetime.date.today() + datetime.timedelta(days=config['days_to_forecast']))
    else:
        today = str(datetime.date.today() - datetime.timedelta(1))
        last_day = str(datetime.date.today() - datetime.timedelta(1) + datetime.timedelta(days=config['days_to_forecast']))

    final_dates = []
    for i in total_days:
        i = str(i)
        final_dates.append(i[5:])

    y = np.array(y)
    y = y.reshape((y.shape[0]), 1)
    predictions = np.array(predictions)
    predictions = predictions.reshape((predictions.shape[0]), 1)

    series = np.concatenate([y, predictions], axis=0)


def routine(series, title, degree):
    first_c = np.array(range(0, series.shape[0]))
    first_c = first_c.reshape((first_c.shape[0]), 1)
    series = series.reshape((series.shape[0], 1))
    series = np.concatenate([first_c, series], axis=1)

    x = series[:, 0].reshape(-1, 1)
    y = series[:, 1]

    model = train_model(x, y, degree)
    predictions = call_model(title, model, x, y, config["days_to_forecast"], degree)
    plot_prediction(y, predictions, title)

series = pd.read_csv('./covid_data.csv')
series_nuovi_positivi = np.array(series['total_positive'])
series_totale_casi = np.array(series['total_case'])

series_terapia_intensiva = series['intensity']
first = series_terapia_intensiva.head(1)
series_terapia_intensiva = np.array(series_terapia_intensiva.diff().fillna(first))
partition = series_totale_casi.shape[0]

routine(series_nuovi_positivi[0:partition], 'New Delhi new-daily cases prediction', config['new_case_degree'])
config['days_to_forecast']=30
routine(series_nuovi_positivi[0:partition], 'New Delhi new-daily cases prediction', config['new_case_degree'])
config['days_to_forecast']=60
routine(series_nuovi_positivi[0:partition], 'New Delhi new-daily cases prediction', config['new_case_degree'])