import sys
sys.path.insert(0, '..')

from utils import data
import os
import sklearn
import numpy as np
from sklearn.neighbors import (
    KNeighborsClassifier,
    DistanceMetric
)
import json
import matplotlib.pyplot as plt


# ------------ HYPERPARAMETERS -------------
BASE_PATH = '..\\COVID-19\\csse_covid_19_data\\'
POLY_DEGREE = 5
MIN_CASES = 0
COUNTRIES = [
        "Andorra",
        "Argentina",
        "Barbados",
        "Bosnia and Herzegovina",
        "Burkina Faso",
        "Eritrea",
        "Estonia",
        "Finland",
        "India",
        "Indonesia",
        "Israel",
        "Kuwait",
        "Kyrgyzstan",
        "Latvia",
        "Luxembourg",
        "Madagascar",
        "Monaco",
        "Morocco",
        "Panama",
        "Philippines",
        "Poland",
        "Portugal",
        "Romania",
        "Rwanda",
        "Saint Lucia",
        "Spain",
        "Sri Lanka",
        "Sweden",
        "Uzbekistan"
    ]#['Ghana', 'Iceland', 'Ecuador'] #["Canada", "China", "Botswana", "Mexico", "Denmark", "France", "Nigeria", "Japan"]
# ------------------------------------------

confirmed = os.path.join(
    BASE_PATH, 
    'csse_covid_19_time_series',
    'time_series_covid19_confirmed_global.csv')
confirmed = data.load_csv_data(confirmed)
features = []
targets = []

cm = plt.get_cmap('jet')
NUM_COLORS = len(COUNTRIES)
LINE_STYLES = ['solid', 'dashed', 'dotted']
NUM_STYLES = len(LINE_STYLES)

colors = [cm(i) for i in np.linspace(0, 1, NUM_COLORS)]

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111)

i = 0
for val in COUNTRIES:
    df = data.filter_by_attribute(
        confirmed, "Country/Region", val)
    cases, labels = data.get_cases_chronologically(df)
    cases = cases.sum(axis=0)

    cases = np.asarray(cases)
    train_cases = cases[cases > MIN_CASES]
    x = np.arange(cases.shape[0])
    x2 = np.arange(np.argmax(cases>MIN_CASES), cases.shape[0]+30)
    
    coeff = np.polyfit(x[cases > MIN_CASES], np.log(np.ma.array(train_cases.astype(float))), POLY_DEGREE)
    
    lines_org = ax.plot(x, cases, label=labels[0,1])
    lines_org[0].set_linestyle('solid')
    lines_org[0].set_color(colors[i])
    i += 1
    '''
    y = np.polyval(coeff, x2)
    lines = ax.plot(x2, np.exp(y), label=labels[0,1])
    lines[0].set_linestyle('dashed')
    lines[0].set_color(colors[i])
    
    '''

   

ax.set_ylabel('# of confirmed cases')
ax.set_xlabel("Time (days since Jan 22, 2020)")
ax.legend()
ax.set_yscale('log')

plt.show()





