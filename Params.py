import os
import json
import pandas as pd
import numpy as np
from hurst import compute_Hc
from sklearn.linear_model import LinearRegression
from fyersApi import  fetch_ohlcv


# defining strategy parameters
params_1 = [{'strategy': 'MomTrading', 'Components': None, 'interval': 'D'}]

Strategy_On_params = {'NIFTYBANK':params_1 , 'NIFTY50':params_1}


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


def GetHistory(market , symbol) :
    symbol_ID={'NIFTYBANK' : 'NSE:NIFTYBANK-INDEX' , 'NIFTY50' : 'NSE:NIFTY50-INDEX'}

    ohlcv= fetch_ohlcv(market , symbol_ID[symbol]  , limit=365)
    return ohlcv