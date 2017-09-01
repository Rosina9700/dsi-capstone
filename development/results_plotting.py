import warnings
import itertools
import pandas as pd
import numpy as np
import csv
import statsmodels.api as sm
from datetime import datetime
from baseline_models import Baseline_average, Baseline_previous, baseline_rolling_predictions, baseline_cross_val_score
from data_wrangling import Results_data, Data_preparation
import sys
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


def fit_sarimaX(y, arima_params, s_params):
    '''
    Fit a SARIMA model to data with given parameters
    -----------
    y: Pandas Series
    arima_params: Tuple
    s_params: Tuple
    RETURNS:
    -----------
    results: SARIMAResults Class Object
    '''
    print 'SARIMAX: {} x {}'.format(arima_params, s_params)

    mod = sm.tsa.statespace.SARIMAX(y[0], y[1],
                                    order=arima_params,
                                    seasonal_order=s_params,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)

    results = mod.fit()
    return results

def fit_sarima(y, arima_params, s_params):
    '''
    Fit a SARIMA model to data with given parameters
    -----------
    y: Pandas Series
    arima_params: Tuple
    s_params: Tuple
    RETURNS:
    -----------
    results: SARIMAResults Class Object
    '''
    print 'SARIMAX: {} x {}'.format(arima_params, s_params)
    mod = sm.tsa.statespace.SARIMAX(y,
                                    order=arima_params,
                                    seasonal_order=s_params,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = mod.fit()
    return results


def rolling_predictions_sarima(y,end,window,params,types=1):
    '''
    Calculating the one-step ahead forecast and rmse for
    a given dataset with both the SARIMA and SARIMAX models.
    -----------
    y: Pandas Series
    end: integer
    window: integer
    params: Tuple
    RETURNS:
    -----------
    forecast: Numpy array
    rmse: float
    model: SARIMAResults Class Object
    '''
    forecast_s = np.zeros(window)
    forecast_sX = np.zeros(window)
    endog = y.ix[:,0]
    exog = sm.add_constant(y.ix[:,1:])
    results = dict()
    for i in xrange(window):
        endog_temp = endog.ix[:end+i]
        exog_temp =  exog.ix[:end+i,:]
        print 'length of cross validation data {}'.format(len(endog_temp))
        model_s, model_sX = None, None
        if types <= 1:
            try:
                model_s = fit_sarima(endog_temp, params[0], params[1])
                forecast_s[i]= model_s.forecast(steps=1).values[0]
            except:
                print 'SKIPPED SARIMA {}-{}'.format(params[0], params[1])


        if types >=1:
            try:
                model_sX = fit_sarimaX((endog_temp, exog_temp), params[0], params[1])
                forecast_sX[i]= model_sX.forecast(steps=1,exog=exog.ix[end+i,:].values.reshape(1,exog.shape[1])).values[0]
            except:
                print 'SKIPPED SARIMAX {}-{}'.format(params[0], params[1])


        # print exog.ix[end+i,:].values.reshape(1,exog.shape[1])
    true = endog[end:end+window].values
    rmse_s = np.sqrt(((true-forecast_s)**2).mean())
    rmse_sX = np.sqrt(((true-forecast_sX)**2).mean())
    results['sarima'] = (forecast_s, rmse_s, model_s)
    results['sarimaX'] = (forecast_sX, rmse_sX, model_sX)
    return results

def baseline_forecasts(y, window, f):
    print '\nbaseline - previous...'
    b_previous = Baseline_previous()
    forecast_b1, b1_test_rmse, model = baseline_rolling_predictions(b_previous, pd.DataFrame(y.ix[:,0]),len(y)-window,window)
    print 'Baseline-previous test RMSE {}'.format(b1_test_rmse)

    b2_test_rmse = None
    forecast_b2 = np.empty(season)
    if f=='H':
        print 'baseline - averages....'
        b_average = Baseline_average()
        forecast_b2, b2_test_rmse, model = baseline_rolling_predictions(b_average, pd.DataFrame(y.ix[:,0]),len(y)-window,window)
        print 'Baseline-averages test RMSE {}'.format(b2_test_rmse)

    return forecast_b1, forecast_b2

def line_plot_predictions(forecasts, true):
    x_axis = true.index.values
    fig, axes = plt.subplots(len(forecasts), sharex=True, sharey=True, figsize=(20,9))
    fig.suptitle('Model performance on test data')
    fig.text(0.5, 0.02, 'Time', ha='center')
    fig.text(0.04, 0.5, 'Energy demand (Wh)', va='center', rotation='vertical')
    # plt.figure(figsize=(20,8))
    # ax[0].plot(x_axis, true.ix[:,0].values, label='measured')
    color = ['m','b','g','o']
    counter = 0
    for ax, f in zip(axes,forecasts):
        ax.plot(x_axis, true.ix[:,0].values,label='measured',color='r' ,alpha=0.75)
        ax.plot(x_axis, f[1], '--',label=f[0], color=color[counter], alpha=0.75)
        ax.legend(loc=1)
        # ax.set_xlabel('Time')
        # ax.set_ylabel('Energy Demand (Wh)')
        counter += 1
    pass

def scatter_plot_predictions(forecasts, true):
    x_axis = true.index.values
    fig, axes = plt.subplots(1,len(forecasts), sharex=True, sharey=True, figsize=(30,5))
    fig.suptitle('Model performance on test data')
    fig.text(0.5, 0.02, 'Measured energy demand values', ha='center')
    fig.text(0.04, 0.5, 'Forecasted energy demand (Wh)', va='center', rotation='vertical')
    # plt.figure(figsize=(20,8))
    # ax[0].plot(x_axis, true.ix[:,0].values, label='measured')
    color = ['m','b','g','o']
    counter = 0
    for ax, f in zip(axes,forecasts):
        ax.scatter(true.ix[:,0].values, f[1],label=f[0],color=color[counter] ,alpha=0.75)
        ax.legend(loc=1)
        # ax.set_xlabel('Time')
        # ax.set_ylabel('Energy Demand (Wh)')
        counter += 1
    pass


if __name__== '__main__':
    project_name, f, season, location = sys.argv[1],sys.argv[2],int(sys.argv[3]),sys.argv[4]
    if sys.argv[5] == 'True':
        T_dependant = True
    else:
        T_dependant = False

    if location == 'local':
        p = '../../capstone_data/Azimuth/clean/{}'.format(project_name)
    else:
        p = project_name

    print'get data for {}....'.format(p)
    dp = Data_preparation(p,f,T_dependant)
    df = dp.get_data()
    y = dp.create_variable(agg='sum',feature='power_all')
    tuned_results = Results_data(project_name)
    params_s, params_sX = tuned_results.get_data().get_params()

    cv_folds = 25

    y_train = y[:-3*season]
    y_test = y[-3*season:]
    forecasts = []

    forecast_b1, forecast_b2 = baseline_forecasts(y,3*season,f)
    forecasts.append(['forecast_b1',forecast_b1])
    if season == 'H':
        forecasts.append(['forecast_b2',forecast_b2])

    print '\nFitting Sarima-X models...'

    # For Sarima model
    results_s = rolling_predictions_sarima(y,len(y_train),3*season,params_s,types=0)
    test_rmse_s = results_s['sarima'][1]
    forecast_s = results_s['sarima'][0]
    forecasts.append(['sarima',forecast_s])
    print'Sarima test RMSE {}'.format(test_rmse_s)

    # For SarimaX model
    results_sX = rolling_predictions_sarima(y,len(y_train),3*season,params_sX,types=2)
    test_rmse_sX = results_sX['sarimaX'][1]
    forecast_sX = results_sX['sarimaX'][0]
    forecasts.append(['sarimax',forecast_sX])
    print'SarimaX test RMSE {}'.format(test_rmse_sX)
    #
    # line_plot_predictions(forecasts, y_test)
    scatter_plot_predictions(forecasts, y_test)
    plt.show()
