import warnings
import itertools
import pandas as pd
import numpy as np
from datetime import datetime

class Results_data(object):
    def __init__(self, project_name):
        self.project_name = project_name
        self.df = None

    def get_order(self, param_string):
        split = param_string.split("'order': (")
        p = int(split[1][0])
        d = int(split[1][3])
        q = int(split[1][6])
        return (p,d,q)

    def get_seasonal_order(self, param_string):
        split = param_string.split("'seasonal_order': (")
        p = int(split[1][0])
        d = int(split[1][3])
        q = int(split[1][6])
        s = int(split[1][9])
        return (p,d,q,s)

    def get_data(self):
        filelocation = 'output_{}_daily.csv'.format(self.project_name)
        df = pd.read_csv(filelocation,sep=';')
        df['order'] = df['sarimax_params'].apply(self.get_order)
        df['seasonal_order'] = df['sarimax_params'].apply(self.get_seasonal_order)
        self.df = df
        return self

    def get_params(self):
        temp = self.df
        sarima_params = (temp.iloc[0,7],temp.iloc[0,8])
        sarimaX_params = (temp.iloc[2,7],temp.iloc[2,8])
        return sarima_params, sarimaX_params


class Data_preparation (object):
    def __init__(self, project_name, freq):
        self.project_name = project_name
        self.freq = freq
        self.df = None

    def get_data(self):
        '''
        Read in the featurized data for the given project_name
        PARAMETERS:
        -----------
        project_name: String
        RETURNS:
        -----------
        df: Pandas DataFrame with DatetimeIndex
        '''
        filelocation='{}_featurized.csv'.format(self.project_name)
        df = pd.read_csv(filelocation)
        df['t'] = pd.to_datetime(df['t'], format='%Y-%m-%d %H:%M:%S')
        df.set_index('t',inplace=True)
        self.df = df
        return self.df

    def get_ready_for_sarima(self, agg, feature):
        '''
        Calculate total power for that site
        PARAMETERS:
        -----------
        df: Pandas DataFrame with DatetimeIndex
        feature: String
        freq: String following panda resample frequency nomenclature
        RETURNS:
        -----------
        y: Pandas DataFrame with DatetimeIndex
        '''
        y = self.df[feature]
        y = y.fillna(y.bfill())
        ignore_last=False
        ignore_first=True

        if self.freq=='H':
            if y.index.max().minute != 0:
                ignore_last=True
            if y.index.min().minute != 0:
                ignore_first=True
        elif self.freq=='D':
            if y.index.max().hour < 23:
                ignore_last=True
            if y.index.min().hour > 0:
                ignore_first=True

        if agg == 'sum':
            y = y.resample(self.freq).sum()

        elif agg == 'mean':
            y = y.resample(self.freq).mean()
        if ignore_last ==True:
            y = y[:-1]
        if ignore_first ==True:
            y = y[1:]

        return pd.DataFrame(y)

    def create_variable(self, agg, feature):
        y = self.get_ready_for_sarima(agg, feature)
        y = pd.DataFrame(y)
        y = self.add_exogs(y)
        return y

    def add_exogs(self, y):
        exog = self.get_ready_for_sarima(agg='mean', feature='T')
        y['T-1'] = exog['T'].shift(1)
        y = y.fillna(y.bfill())
        y['weekday'] = y.index.dayofweek
        y['weekday'] = y['weekday'].apply(lambda x: 1 if x < 5 else 0)
        return y
