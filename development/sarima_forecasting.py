import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm

def get_data(project_name):
    filelocation='{}_featurized.csv'.format(project_name)
    df = pd.read_csv(filelocation)
    df['t'] = pd.to_datetime(df['t'], format='%Y-%m-%d %H:%M:%S')
    df.set_index('t',inplace=True)
    return df

def calculate_power(df):
    df['power_1'] = df['load_v1rms'] * df['load_i1rms']
    df['power_2'] = df['load_v2rms'] * df['load_i2rms']
    df['power_3'] = df['load_v3rms'] * df['laod_i3rms']
    df['power_all'] = df['power_1'] +df['power_2']+df['power_3']
    return df

def get_ready_for_arima(df, feature, freq='H'):
    y = pd.DataFrame(df[feature])
    y = y[feature].resample(freq).mean()
    y = y.fillna(y.bfill())
    return pd.DataFrame(y)

class Baseline (object):
    def __init__(self):
        self.averages = None
        self.freq = 'H'
        self.score_ = None

    def fit(self, y):
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

def fit_sarima(y, arima_params, s_params):
    print 'SARIMAX: {} x {}'.format(arima_params, s_params)
    mod = sm.tsa.statespace.SARIMAX(y,
                                    order=arima_params,
                                    seasonal_order=s_params,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)

    results = mod.fit()
    return results

def test_all_params(y, pdq, seasonal_pdq):
    print 'number of models {}'.format(len(pdq)*len(seasonal_pdq))
    model_list = []
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                model = fit_sarima(y,param,param_seasonal)
                model_list.append(model)
                print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, model.aic))
            except:
                continue
    return model_list

def find_best_sarima(y, range_ind, season):
    p = d = q = range(range_ind[0],range_ind[1])
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], season) for x in list(itertools.product(p, d, q))]
    warnings.filterwarnings("ignore") # specify to ignore warning messages
    model_list = test_all_params(y, pdq, seasonal_pdq)
    top_ind = np.array([m.aic for m in model_list]).argmin()
    return model_list[top_ind]

def evaluate_sarima_predict(model, y):
    start_date = y.index.min().strftime('%Y-%m-%d %H:%M:00')
    pred = model.get_prediction(start=pd.to_datetime(start_date), dynamic=False)
    y_forecasted = pred.predicted_mean
    y_forecasted = y_forecasted.values.reshape(len(y_forecasted),1)
    rmse = np.sqrt(((y_forecasted - y) ** 2).mean())
    return rmse


if __name__== '__main__':
    projects = ['project_5526']
    for p in projects:
        print'get data....'
        project_name = p
        df = get_data(project_name)
        df = calculate_power(df)
        y = get_ready_for_arima(df,freq='H', feature='power_all')

        print'baseline model....'
        baseline = Baseline()
        length_data = len(y)
        y_train = y[:int(length_data*(5./6))]
        y_test = y[int(length_data*(5./6)):]
        model = baseline.fit(y_train)
        print 'Training rmse: {}'.format(model.score_)
        b_test_score = model.score(y_test)
        print 'Testing rmse: {}'.format(b_test_score)

        print 'find best sarima...'
        model = find_best_sarima(y,(0,2),24)
        best_params = model.specification
        print('ARIMA{}x{}{} - AIC:{}'.format(best_params['order'], best_params['seasonal_order'],
                                             best_params['seasonal_periods'],model.aic))
        print 'specifications:\n{}'.format(best_params)

        test_score = evaluate_sarima_predict(model, y_test)
        print('The Mean Squared Error of our forecasts is {}'.format(round(test_score, 2)))

        filename = 'output_{}.txt'.format(project_name)
        with open(filename, "w") as text_file:
            text_file.write("{}\nBaseline RMSE: {}\nBest parameters: \n{}\n SARIMA AIC: {}\n RMSE: {}".format(project_name,b_test_score,best_params,model.aic,test_score))
