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
        filelocation = '{}_fbeta.csv'.format(self.project_name)
        df1 = pd.read_csv(filelocation,sep=';')
        df1['order'] = df1['sarimax_params'].apply(self.get_order)
        df1['seasonal_order'] = df1['sarimax_params'].apply(self.get_seasonal_order)
        df1['beta_var'] = 0
        filelocation = '{}_vbeta.csv'.format(self.project_name)
        df2 = pd.read_csv(filelocation,sep=';')
        df2['order'] = df2['sarimax_params'].apply(self.get_order)
        df2['seasonal_order'] = df2['sarimax_params'].apply(self.get_seasonal_order)
        df2['beta_var'] = 1
        df = df1.append(df2)
        self.df = df
        return self

    def get_params(self):
        temp = self.df
        sarima_params = (temp.iloc[0,7],temp.iloc[0,8])
        sarimaX_params = (temp.iloc[2,7],temp.iloc[2,8])
        return sarima_params, sarimaX_params


class Data_preparation (object):
    def __init__(self, project_name, freq, T=True):
        '''
        Initialises the data preparation class for a given project and
        specified temporal frequency
        PARAMETERS
        ----------
        project_name: String
        freq: String according to pandas resample() nomenclature
        '''
        self.project_name = project_name
        self.freq = freq
        self.df = None
        self.T = T

    def get_data(self):
        '''
        Read in the featurized data for the given project_name and saves the
        dataframe to the class for further processing
        '''
        filelocation='{}_featurized.csv'.format(self.project_name)
        df = pd.read_csv(filelocation)
        df['t'] = pd.to_datetime(df['t'], format='%Y-%m-%d %H:%M:%S')
        df.set_index('t',inplace=True)
        self.df = df
        return self

    def get_ready_for_sarima(self, agg, feature):
        '''
        Selects a feature from the original dataframe, resamples it to the
        desired frequency and handles missing data and incomplete aggregates
        at the edges.
        PARAMETERS:
        -----------
        agg: String
        feature: String
        RETURNS:
        -----------
        y: Pandas DataFrame
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
        '''
        Creates the Y-variable you want to forecast and adds the exogenous variables
        PARAMETERS:
        -----------
        agg: String ('sum' or 'mean')
        features: String
        RETURNS:
        -----------
        y: Pandas DataFrame
        '''
        y = self.get_ready_for_sarima(agg, feature)
        y = pd.DataFrame(y)
        y = self.add_exogs(y)
        return y

    def add_exogs(self, y):
        '''
        Add exogenous variables to the design matrix. If T==True, temperature will be added.
        PARAMETERS:
        -----------
        y: Pandas DataFrame (target variable)
        T: Boolean
        RETURNS:
        -----------
        y: Pandas DataFrame
        '''
        if self.T:
            exog = self.get_ready_for_sarima(agg='mean', feature='T')
            y['T-1'] = exog['T'].shift(1)
            y = y.fillna(y.bfill())
        y['weekday'] = y.index.dayofweek
        y['weekday'] = y['weekday'].apply(lambda x: 1 if x < 5 else 0)
        return y
