import numpy as np
import pandas as pd

def get_clean_data(filename):
    df = pd.read_csv('../capstone_data/Azimuth/clean/{}'.format(filename))
    df['t'] = pd.to_datetime(df['t'], format='%Y-%m-%d %H:%M:%S')
    df.set_index('t',inplace=True)
    return df

def check_time_shift(timestamps, delta, expected_shift=5):
    '''
    Creates a column which flags if the time shift creates time delta which are
    not as expected. Flags data points where continuity is broken
    PARAMETERS
    ----------
    timestamps: Pandas DatetimeIndex
    delta: unit time shift to be applied
    expected_shift: Expected time shift in minutes
    RETURNS
    --------
    time_diff_col_name: Column name specific to this time shift
    time_diff.values: Pandas Series of flag values
    '''
    time_diff = (pd.Series(timestamps) - pd.Series(timestamps).shift(delta))
    time_string = '0 days 00:0{}:00'.format(delta*expected_shift)
    time_diff = time_diff.apply(lambda x: None if x!= pd.to_timedelta(time_string) else 1)
    time_diff_col_name = 't_diff-{}'.format(delta)
    return time_diff_col_name, time_diff.values

def shift_features(df, features, deltas):
    '''
    Create time shifted features for each data point.
    PARAMETERS
    ----------
    df: Pandas DataFrame with columns to shift
    features: List of column names to shift
    deltas: List of time shifts to implement
    RETURNS
    --------
    new_df: Pandas DataFrame with all shifted features added
    '''
    shifted_dfs = [df]
    for d in deltas:
        # Shift features and create new column names
        shifted = df[features].shift(d)
        col_names = []
        for f in features:
            name = '{}-{}'.format(f,d)
            col_names.append(name)
        shifted.columns = col_names
        # Check for time shift continuity
        col_name, values = check_time_shift(df.index, d)
        shifted[col_name] = values
        # Append shifted df to list of dfs
        shifted_dfs.append(shifted)
    # Concatenate all new shifted dfs to original df
    print 'num of shifted columns {}'.format(len(shifted_dfs))
    new_df = pd.concat(shifted_dfs,axis=1)
    print new_df.index
    # Drop columns with Na's (where continuity has been broken)
    new_df.dropna(inplace=True)
    return new_df

def create_dummies(df, columns):
    for c in columns:
        df = pd.get_dummies(df, columns=[c])
    return df

def get_weather_data():
    weather_16 = pd.read_csv('merra_data/accra_2016/weather_data_Accra_2016.csv')
    weather_17 = pd.read_csv('merra_data/accra_2017/weather_data_Accra_2017.csv')
    weather = weather_16.append(weather_17, ignore_index=True)
    weather['timestamp'] = pd.to_datetime(weather['timestamp'], format='%Y-%m-%d %H:%M:%S')
    weather.sort_values('timestamp',inplace=True)
    weather = weather.set_index('timestamp')
    weather['year'] = weather.index.year
    weather['month'] = weather.index.month
    weather['day'] = weather.index.day
    weather['hour'] = weather.index.hour
    weather = pd.DataFrame(weather.groupby(['year','month','day','hour']).mean()).reset_index()
    return weather

def add_weather_data(df, weather):
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['index'] = df.index
    new_df = df.merge(weather,how='right',on=['year','month','day','hour']).set_index('index')
    new_df.dropna(inplace=True)
    return new_df

if __name__=='__main__':
    project_name = 'project_4229'
    # read in data
    print 'reading clean data...'
    filename = '{}_clean.csv'.format(project_name)
    df = get_clean_data(filename)
    # created shifted features
    print 'creating shifted features...'
    df = shift_features(df, ['load_v1rms','load_v2rms','load_v3rms',
                                       'load_i1rms','load_i2rms','laod_i3rms',
                                   'relay_est'],[1,2,3,4])
    # # dummify categorical data
    df2 = create_dummies(df, ['month','dayofweek'])
    # # weather here?
    print 'adding weather data...'
    weather = get_weather_data()
    df3 = add_weather_data(df2, weather)
    print 'writing to csv...'
    filelocation='../capstone_data/Azimuth/clean/{}_clean_weather.csv'.format(project_name)
    df3.to_csv(filelocation)
    # drop continuous outage points
    # df2 = df[~((df['relay_est']==1)&(df['relay_est-1']==1))]
