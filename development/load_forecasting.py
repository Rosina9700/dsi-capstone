import warnings
import itertools
import pandas as pd
import numpy as np
import csv
import statsmodels.api as sm
from datetime import datetime
from baseline_models import Baseline_average, Baseline_previous, baseline_rolling_predictions, baseline_cross_val_score
import sys

def get_data(project_name):
    '''
    Read in the featurized data for the given project_name
    PARAMETERS:
    -----------
    project_name: String
    RETURNS:
    -----------
    df: Pandas DataFrame with DatetimeIndex
    '''
    filelocation='{}_featurized.csv'.format(project_name)
    df = pd.read_csv(filelocation)
    df['t'] = pd.to_datetime(df['t'], format='%Y-%m-%d %H:%M:%S')
    df.set_index('t',inplace=True)
    if 'power_all_old' in df.columns:
        return df
    else:
        df = calculate_power(df)


    return df

def calculate_power(df):
    '''
    Calculate total power for that site
    PARAMETERS:
    -----------
    df: Pandas DataFrame with DatetimeIndex
    RETURNS:
    -----------
    df: Pandas DataFrame with DatetimeIndex
    '''
    df['power_1'] = df['load_v1rms'] * df['load_i1rms']
    df['power_2'] = df['load_v2rms'] * df['load_i2rms']
    df['power_3'] = df['load_v3rms'] * df['laod_i3rms']
    df['power_all'] = ( df['power_1'] +df['power_2']+df['power_3'] ) * 5./12
    return df

def get_ready_for_sarima(df, agg, feature, freq='H'):
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
    y = df[feature]
    y = y.fillna(y.bfill())
    ignore_last=False
    ignore_first=True

    if freq=='H':
        if y.index.max().minute != 0:
            ignore_last=True
        if y.index.min().minute != 0:
            ignore_first=True
    elif freq=='D':
        if y.index.max().hour < 23:
            ignore_last=True
        if y.index.min().hour > 0:
            ignore_first=True

    if agg == 'sum':
        y = y.resample(freq).sum()

    elif agg == 'mean':
        y = y.resample(freq).mean()
    if ignore_last ==True:
        y = y[:-1]
    if ignore_first ==True:
        y = y[1:]

    return pd.DataFrame(y)

def add_exogs(df, y, freq):
    exog = get_ready_for_sarima(df,agg='mean', freq=f, feature='T')
    y['T-1'] = exog['T'].shift(1)
    y = y.fillna(y.bfill())
    y['weekday'] = y.index.dayofweek
    y['weekday'] = y['weekday'].apply(lambda x: 1 if x < 5 else 0)
    return y

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


def cross_val_score(y, params, chunks, window=1):
    '''
    Break a training set into chunks and calcualtes the average
    rmse from forecasts. The training set gradually grow by size chunk at
    each iteration.
    -----------
    y: Pandas Series
    params: Tuple
    chunks: integer
    window: integer
    RETURNS:
    -----------
    rmse: float
    model: SARIMAResults Class Object
    '''

    length = len(y.ix[:,0])-window
    chunks = min(chunks, length/2)
    chunk_size = (length/2)/chunks
    rmses_s = []
    rmses_sX = []
    for i in xrange(chunks+1):
        end_index = (length/2) + (i)*chunk_size
        print 'data length: {} chunks size: {}'.format(length, end_index)
        results = rolling_predictions_sarima(y,end_index,window, params)
        rmses_s.append(results['sarima'][1])
        rmses_sX.append(results['sarimaX'][1])
    return (np.asarray(rmses_s).mean(), results['sarima'][2]), (np.asarray(rmses_sX).mean(), results['sarimaX'][2])

def grid_search_sarima(y, pdq, seasonal_pdq, k):
    '''
    For the pdq's and seasonal_pdq's provided, fit every possible model
    and cross validate with a k chunks.
    -----------
    y: Pandas Series
    param: List of Tuples
    param_seasonal: List of Tuples
    k: Integer
    RETURNS:
    -----------
    results: List of Tuples
    '''
    print 'number of models {}'.format(len(pdq)*len(seasonal_pdq))
    results_s = []
    results_sX = []
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            res_s, res_sX = cross_val_score(y, (param, param_seasonal), k)
            results_s.append(res_s)
            results_sX.append(res_sX)
    return results_s, results_sX

def find_best_sarima(y, params, season, k=10):
    '''
    Grid search over every possible combination of p,d,q and season provided. In
    the cross validation, use k chunks to calculate rmse.
    -----------
    y: Pandas Series
    param: Tuple
    season: Inter
    k: Integer
    RETURNS:
    -----------
    results: SARIMAXResults Object, float
    '''
    pdq = list(itertools.product(params[0], params[1], params[2]))
    s_pdq = list(itertools.product(range(0,2), range(0,2), range(0,2)))
    if season == 7:
        seasonal_pdq = [(x[0], x[1], x[2], season) for x in s_pdq]
    else:
        seasonal_pdq = [(x[0], 0, x[2], season) for x in s_pdq]
    warnings.filterwarnings("ignore") # specify to ignore warning messages
    results_s, results_sX = grid_search_sarima(y, pdq, seasonal_pdq, k)
    # results_s, results_sX = grid_search_sarima(y, [(0,1,1)], [(1,0,1,season)], k)
    top_ind_s = np.nanargmin(np.array([r[0] for r in results_s]))
    top_ind_sX = np.nanargmin(np.array([r[0] for r in results_sX]))
    return (results_s[top_ind_s][1], results_s[top_ind_s][0]), (results_sX[top_ind_sX][1], results_sX[top_ind_sX][0])

if __name__== '__main__':
    project_name = sys.argv[1]
    f = sys.argv[2]
    season = int(sys.argv[3])
    location = sys.argv[4]
    if location == 'local':
        p = '../../capstone_data/Azimuth/clean/{}'.format(project_name)
    else:
        p = project_name

    print'get data for {}....'.format(p)
    df = get_data(p)

    y = get_ready_for_sarima(df,agg='sum',freq=f, feature='power_all')
    y = add_exogs(df, y, freq=f)
    cv_folds = 25

    y_train = y[:-2*season]
    y_test = y[-2*season:]

    print '\nbaseline - previous...'
    b_previous = Baseline_previous()
    b1_train_rmse, model = baseline_cross_val_score(b_previous, pd.DataFrame(y_train.ix[:,0]), cv_folds, window=1)
    forecast, b1_test_rmse, model = baseline_rolling_predictions(b_previous, pd.DataFrame(y.ix[:,0]),len(y_train),2*season)
    print 'Baseline-previous train RMSE {}'.format(b1_train_rmse)
    print 'Baseline-previous test RMSE {}'.format(b1_test_rmse)

    b2_train_rmse = None
    b2_test_rmse = None
    if f=='H':
        print 'baseline - averages....'
        b_average = Baseline_average()
        b2_train_rmse, model = baseline_cross_val_score(b_average, pd.DataFrame(y_train.ix[:,0]), cv_folds, window=1)
        forecast, b2_test_rmse, model = baseline_rolling_predictions(b_average, pd.DataFrame(y.ix[:,0]),len(y_train),2*season)
        print 'Baseline-averages train RMSE {}'.format(b2_train_rmse)
        print 'Baseline-averages test RMSE {}'.format(b2_test_rmse)

    #
    print '\nfind best sarima...'
    y_train = y[:-2*season]
    y_test = y[-2*season:]
    p = range(0,3)
    q = range(1,3)
    d = range(0,2)
    params = (p,d,q)
    results_s, results_sX = find_best_sarima(y_train, params, season, k=cv_folds)

    # For Sarima model
    model_s = results_s[0]
    train_rmse_s = results_s[1]

    best_params_s = model_s.specification
    params = (best_params_s['order'],best_params_s['seasonal_order'])
    print('SARIMA{}x{}{} - AIC:{}'.format(params[0], params[1],
                                     best_params_s['seasonal_periods'],model_s.aic))

    results_s = rolling_predictions_sarima(y,len(y_train),2*season,params,types=0)
    test_rmse_s = results_s['sarima'][1]
    print 'Sarima training cross validation RMSE: {}'.format(train_rmse_s)
    print'Sarima test RMSE {}'.format(test_rmse_s)

    # For SarimaX model
    model_sX = results_sX[0]
    train_rmse_sX = results_sX[1]

    best_params_sX = model_sX.specification
    params = (best_params_sX['order'],best_params_sX['seasonal_order'])
    print('SARIMA{}x{}{} - AIC:{}'.format(params[0], params[1],
                                     best_params_sX['seasonal_periods'],model_sX.aic))

    results_sX = rolling_predictions_sarima(y,len(y_train),2*season,params,types=2)
    test_rmse_sX = results_sX['sarimaX'][1]
    print 'SarimaX training cross validation RMSE: {}'.format(train_rmse_sX)
    print'SarimaX test RMSE {}'.format(test_rmse_sX)

    now = datetime.now().strftime('%m_%d_%H_%M_%S')
    filename = 'output_{}_{}.csv'.format(project_name,now)
    test = 1
    train = 0
    model_name_s = 'sarima'
    model_name_sX = 'sarimaX'
    train_results_s = '{};{};{};{};{};{};{}'.format(project_name, model_name_s, train, b1_train_rmse, b2_train_rmse, train_rmse_s, best_params_s)
    test_results_s = '{};{};{};{};{};{};{}'.format(project_name, model_name_s, test, b1_test_rmse, b2_test_rmse, test_rmse_s, best_params_s)
    train_results_sX = '{};{};{};{};{};{};{}'.format(project_name, model_name_sX, train, b1_train_rmse, b2_train_rmse, train_rmse_sX, best_params_sX)
    test_results_sX = '{};{};{};{};{};{};{}'.format(project_name, model_name_sX, test, b1_test_rmse, b2_test_rmse, test_rmse_sX, best_params_sX)
    header = 'project;model;test;baseline_previous;baseline_averages;sarimax;sarimax_params'
    to_print = [header,train_results_s, test_results_s, train_results_sX, test_results_sX]
    with open(filename, "wb") as file:
        for r in to_print:
            file.write(r)
            file.write('\n')
