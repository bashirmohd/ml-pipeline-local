from datetime import date
import os
import time
from math import pi

from numpy.lib.utils import source

import fire
import joblib

import pandas as pd
from bokeh.plotting import figure, curdoc, show
from bokeh.models import DatetimeTickFormatter, ColumnDataSource, HoverTool
from sklearn.model_selection import train_test_split

my_model = joblib.load("xgb_model.joblib")


tp_data = pd.read_csv("data/throughput_metrics_modified.csv")
X = tp_data[["hour", "day", "month", "year"]]
y = tp_data["SiteF"]
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.15)

len_xtrain = X_train.shape[0]
len_xtest =  X_test.shape[0]

x_train_date = pd.to_datetime(tp_data.Time.iloc[0:len_xtrain])
x_test_date = pd.to_datetime(tp_data.Time.iloc[len_xtrain:])

predictions = my_model.predict(X_test)

train_source = ColumnDataSource(data=dict(
    date=x_train_date,
    metrics=y_train
))

test_source = ColumnDataSource(data=dict(
    date=x_test_date,
    metrics=y_test
))

prediction_source = ColumnDataSource(data=dict(
    date=x_test_date,
    metrics=predictions
))

TOOLTIPS = [
    ("date", "@date{%H:%M %d-%m-%Y}"),
    ("metrics", "@metrics{0,0.00}")
]

plot = figure(title="Throughput Metrics Prediction", x_axis_type="datetime", 
                plot_height=650, plot_width=1500, tooltips=TOOLTIPS)

plot.line("date", "metrics", color="red", line_width=1, legend_label="train data", source=train_source)
plot.line("date", "metrics", color="green", line_width=1, legend_label="true value", source=test_source)
plot.line("date", "metrics", color="blue", line_width=1, legend_label="predictions", source=prediction_source)

plot.xaxis.formatter = DatetimeTickFormatter(months=['%b %Y'])
plot.xaxis.major_label_orientation = pi/4

plot.hover.formatters = {'@date': 'datetime'}
show(plot)