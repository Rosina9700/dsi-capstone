import pandas as pd
import numpy as np

class Baseline_average(object):
    def __init__(self):
        self.averages = None
        self.freq = 'H'
        self.score_ = None
        self.y_train = None

    def fit(self, y):
        self.y_train = y
        y['dayofweek'] = y.index.dayofweek
        y['hour'] = y.index.hour
        self.averages = y.groupby(['dayofweek','hour'])['power_all'].mean()
        y.drop(['dayofweek','hour'],axis=1, inplace=True)
        self.score_ = self.score(y)
        return self

    def predict(self, start, periods):
        date_index = pd.date_range(start,periods=periods,freq=self.freq)
        predictions = pd.Series(date_index).apply(lambda x: self.averages[x.dayofweek][x.hour])
        return predictions

    def score(self, y):
        predictions = self.predict(y.index.min().strftime('%Y-%m-%d %H:%M:00'), len(y)).values
        true = y.values
        rmse = np.sqrt(((true - predictions)**2).mean())
        return rmse

    def forecast(self,steps):
        start_date = self.y_train.index.max()
        forecasts = np.zeros(steps)
        for i in xrange(1,steps+1):
            next_step = start_date + pd.Timedelta(hours=i)
            dayofweek = next_step.dayofweek
            hour = next_step.hour
            pred = self.averages[dayofweek][hour]
            forecasts[i-1] = pred
        return forecasts


class Baseline_previous(object):
    def __init__(self):
        self.y_train = None

    def fit(self, y):
        self.y_train = y
        return self

    def forecast(self, steps):
        forecasts = np.zeros(steps)
        for i in xrange(0,steps):
            forecasts[i-1] = self.y_train.values[-1]
        return forecasts


#
# def baseline_previous(y):
#     cv_data = get_cross_val_data(y,5)
#     diff_sq = []
#     for i in range(len(cv_data)-1):
#         if len(cv_data[i+1]) != 0:
#             y_true = cv_data[i+1].values[0]
#             y_pred = cv_data[i].values[-1]
#             d = (y_true - y_pred)**2
#             diff_sq.append(d)
#     rmse = np.sqrt(np.asarray(diff_sq).mean())
#     return rmse

def baseline_previous(y):
    y_pred = y.shift()[1:]
    rmse = np.sqrt(((y_pred - y[1:]) ** 2).mean())
    return rmse
