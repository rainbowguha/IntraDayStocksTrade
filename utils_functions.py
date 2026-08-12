import os
import json
import pandas as pd
import pickle as pk
import numpy as np
import pytz
from fyersApi import fetch_ohlcv


time_zone = pytz.timezone('Asia/Kolkata')
DECIMAL_POINTS = 5

def load_csv(symbol ,drop_date=None):
    file_path = os.path.join('database_fx', symbol, f'{symbol}')
    if os.path.exists(file_path):
       d = pd.read_csv(file_path , index_col=0 , parse_dates=True)
       return d.loc[d.index.normalize()!=pd.Timestamp(drop_date)]
    return None

def load_str_params(strategy_name):
    file_name = 'strategy_params'
    _PATH_ = os.path.join('Models',strategy_name, file_name)
    with open(_PATH_, 'r') as f:
        params = json.load(f)
        params , market_regime_params , SL_COMM  , trained_upto = params[:-3][0] , params[-3] , params[-2]['SL_COMM'] , params[-1]['trained_upto']

    return params , market_regime_params , SL_COMM , trained_upto , len(params)

def load_models(n , strategy_name):
    file_name = 'base_model_{}.pkl'.format(n)
    _PATH_=os.path.join('Models' , strategy_name , file_name)
    with open(_PATH_ , 'rb') as f:
        mod = pk.load(f)
    return mod

def load_meta_models(strategy_name):
    file_name = 'meta_model.pkl'
    _PATH_=os.path.join('Models' , strategy_name , file_name)
    with open(_PATH_ , 'rb') as f:
        mod = pk.load(f)
    return mod

def load_dist_extractor(strategy_name):
    file_name = 'dist_extractor.pkl'
    _PATH_=os.path.join('Models' , strategy_name , file_name)
    with open(_PATH_ , 'rb') as f:
        mod = pk.load(f)
    return [m for m in mod]

def load_trade_logs(strategy_name):
    file_name = 'trade_logs.csv'
    _PATH_=os.path.join('Models' , strategy_name , file_name)
    return pd.read_csv(_PATH_,index_col=0 , parse_dates=True)


def GetHistory(market , symbol ,limit=365):
    data = pd.DataFrame()

    ID_y = {'NIFTYBANK' : 'NSE:NIFTYBANK-INDEX' , 'NIFTY50' : 'NSE:NIFTY50-INDEX'}

    __SYMBOL__ = ID_y[symbol]

    try:
        data = fetch_ohlcv(market , __SYMBOL__  , limit=limit)
    except:
        print('{} Unable to Download:'.format(symbol))

    return data


def rolling_spread(y, x, window):

    # Rolling means
    mean_x = x.rolling(window).mean()
    mean_y = y.rolling(window).mean()

    # Rolling covariance and variance
    cov_xy = y.rolling(window).cov(x)
    var_x = x.rolling(window).var()

    # Rolling beta
    hedge_ratio = cov_xy / var_x

    # Rolling alpha
    alpha = mean_y - hedge_ratio * mean_x

    # Spread (residual)
    spread = y - (alpha + hedge_ratio * x)

    return spread, hedge_ratio

def kaufMan_EF_Ratio(dt , n) :
    absolute_change=abs(dt-dt.shift(n))
    total_change=dt.diff().abs().rolling(window=n).sum()
    return absolute_change / total_change

def compute_rolling_sharpe(returns , window: int = 20) -> pd.Series :
    if not isinstance(returns , pd.Series) :
        returns=pd.Series(returns)

    roll_mean=returns.rolling(window).mean()
    roll_std=returns.rolling(window).std().replace(0 , np.nan)
    sharpe=(roll_mean / roll_std) * np.sqrt(252)
    return sharpe.fillna(0)