import os
import json
import pandas as pd
import numpy as np
from hurst import compute_Hc
from sklearn.linear_model import LinearRegression
import yfinance as yf


# defining strategy parameters
params_1 = [{'strategy': 'MomTrading', 'Components': None, 'interval': 'D'}]

Strategy_On_params = {'Adaniports':params_1 , 'AXIS':params_1 , 'TCS':params_1 , 'Reliance':params_1 , 'MM':params_1 , 'CIPLA':params_1 , 'JSWSTEEL':params_1}

# equal weights
Wi = 1/len(Strategy_On_params)

Weight = {'AXIS':Wi  , 'TCS':Wi , 'Reliance':Wi , 'MM':Wi , 'Adaniports':Wi , 'CIPLA':Wi , 'JSWSTEEL':Wi}


def load_csv(symbol ,drop_date=None):
    file_path = os.path.join('database_fx', symbol, f'{symbol}.csv')
    if os.path.exists(file_path):
       d = pd.read_csv(file_path , index_col=0 , parse_dates=True)
       return d.loc[d.index.normalize()!=pd.Timestamp(drop_date)]
    return None

def get_ensemble_n(strategy_name):
    file_name = f'{strategy_name}_params'
    path = os.path.join('Models',strategy_name, file_name)
    with open(path, 'r') as f:
        params = json.load(f)
        params , sl_params , Vol_Params = params[:-2] , params[-2]['sl_params'] , params[-1]['Vol_Params']

    return params , sl_params , Vol_Params , len(params)

# indicator function
def TrendIntensityIndex(dt , window):
    ma = dt.rolling(window=window).mean()
    day_above_average = (dt>ma).rolling(window=window).sum()
    return 100* day_above_average/window

def calculate_MOM_Burst( dt ,  lookback  ):
#    calculating the indicator mom burst
    candle_range = dt['high']-dt['low']
    mean_range = candle_range.rolling(window  = lookback).mean()
    std_range = candle_range.rolling(window = lookback).std()
    mom_burst = (candle_range-mean_range)/std_range

    return mom_burst  , mean_range

def HedgeRatio(y , x , window):
    hr = np.full_like(y.values , np.nan)

    x = x.values.reshape(-1 , 1)
    y = y.values.reshape(-1 , 1)

    for i in range(window , y.shape[0]):
        X_ , Y_ = x[i-window:i] , y[i-window:i]
        m = LinearRegression().fit(X_,  Y_)
        hr[i] = m.coef_[0][0]
    return hr

def Z_score(dt,length):
    mean = dt.ewm(span=length).mean()
    std = dt.ewm(span=length).std()
    return (dt-mean)/std

def kaufMan_EF_Ratio(dt , n):
    absolute_change = abs(dt - dt .shift(n))
    total_change = dt.diff().abs().rolling(window=n).sum()
    return absolute_change / total_change

def ComputeRegime_HURST(dt , window):
    rolling_hurst_exponents = dt.rolling(window=window).apply(lambda X: compute_Hc(X)[0] , raw =False)
    regime = pd.Series(rolling_hurst_exponents , rolling_hurst_exponents.index  , name = 'HurstRegime')
    return regime

def GetHistory(symbol):
     return get_delta_historical_data(symbol)


def get_delta_historical_data(symbol , interval='1d' , period='1y') :
    symbol_ID={'AXIS' : 'AXISBANK.NS' , 'TCS' : 'TCS.NS' , 'Adaniports':'ADANIPORTS.NS' , 'MM':'M&M.NS' , 'Reliance':'RELIANCE.NS' , 'CIPLA':'CIPLA.NS' , 'JSWSTEEL':'JSWSTEEL.NS'}

    try:
        data = yf.download(symbol_ID[symbol] , period=period , interval=interval , multi_level_index=False)
        data.columns = ['close' , 'high' , 'low' ,  'open' , 'volume']
    except:
        print('{} Unable to Download:'.format(symbol))

    return data