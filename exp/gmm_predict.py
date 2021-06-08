import sys
sys.path.insert(0, '..')

from utils import data
import os
import sklearn
import numpy as np
from sklearn.neighbors import (
    KNeighborsClassifier,
    DistanceMetric,
    
)
from sklearn.mixture import GaussianMixture

import json
import matplotlib.pyplot as plt


# ------------ HYPERPARAMETERS -------------
BASE_PATH = '..\\COVID-19\\csse_covid_19_data\\'
POLY_DEGREE = 4
MIN_CASES = 0
CLASSES = 15
CASE_THRESHOLD = 2000
#COUNTRIES = ["Canada", "China", "Botswana", "Mexico", "Denmark", "France", "Nigeria", "Japan"]
# ------------------------------------------

confirmed = os.path.join(
    BASE_PATH, 
    'csse_covid_19_time_series',
    'time_series_covid19_confirmed_global.csv')
confirmed = data.load_csv_data(confirmed)
features = []

for val in np.unique(confirmed["Country/Region"]):
    df = data.filter_by_attribute(
        confirmed, "Country/Region", val)
    cases, labels = data.get_cases_chronologically(df)
    cases = cases.sum(axis=0)

    if cases[-1] < CASE_THRESHOLD:
        continue


    cases = np.asarray(cases)
    train_cases = cases[cases > MIN_CASES]
    x = np.arange(cases.shape[0])
    x2 = np.arange(np.argmax(cases>MIN_CASES), cases.shape[0]+30)
    
    coeff = np.polyfit(x[cases > MIN_CASES], np.log(np.ma.array(train_cases.astype(float))), POLY_DEGREE)

    features.append(coeff)
    
gm = GaussianMixture(n_components = CLASSES)
gm.fit(features)

countries = {}

for val in np.unique(confirmed["Country/Region"]):
    df = data.filter_by_attribute(
        confirmed, "Country/Region", val)
    cases, labels = data.get_cases_chronologically(df)
    cases = cases.sum(axis=0)

    if cases[-1] < CASE_THRESHOLD:
        continue

    cases = np.asarray(cases)
    train_cases = cases[cases > MIN_CASES]
    x = np.arange(cases.shape[0])
    
    coeff = np.polyfit(x[cases > MIN_CASES], np.log(np.ma.array(train_cases.astype(float))), POLY_DEGREE)

    predicted = gm.predict(coeff.reshape(1, -1))
    predicted_class = int(predicted[0])
    
    if predicted_class in countries:
        countries[predicted_class].append(val)
    else:
        countries[predicted_class] = [val]    

print(countries)

with open('results/gmm_predict.json', 'w') as f:
    json.dump(countries, f, indent=4)
    






