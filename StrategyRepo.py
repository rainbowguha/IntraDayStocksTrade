from utils_functions import *
import pandas_ta as ta
import pandas as pd


def MomTrading(DT , CMP_DT , **kwargs):
        
        # creating variable
        features = pd.DataFrame()
        lags = 5
        lookback = kwargs['lookback']
        window = kwargs['window']

        # calculating indicators
        DailyChange = DT['close'].pct_change()
        volatility_1 = DT['close'].ewm(span=window).std()
        rsi = ta.rsi(DT['close'], lookback)
        adx = ta.adx(DT['high'], DT['low'], DT['close'], lookback).iloc[:, 0]
        candle_Range = DT['high'] - DT['low']
        ibs = (DT['close'] - DT['low']) / candle_Range
        ma = DT['close'].ewm(span=lookback).mean()
        corr = DT['close'].rolling(window=window).corr(ma)
        spr = DT['close'] - ma
        slope = ta.slope(ma, window)

        mean = 100 * DT['close'].pct_change().rolling(window=lookback).mean()
        std = 100 * DT['close'].pct_change().rolling(window=lookback).std()
        r = 100 * DT['close'].pct_change()
        z_score = (r - mean) / std

        hh = z_score.rolling(window=window).max()
        ll = z_score.rolling(window=window).min()
        mid = z_score.rolling(window=window).quantile(0.5)
        z_slope = ta.slope(z_score, window)

        # setting features
        features['daily_change'] = DailyChange
        features['rsi'] = rsi
        features['adx'] = adx
        features['ibs'] = ibs
        features['spr'] = spr
        features['slope'] = slope
        features['zscore'] = z_score
        features['UP_Z'] = z_score - hh
        features['DN_Z'] = ll - z_score
        features['MID_Z'] = z_score - mid
        features['corr'] = corr
        features['daily_Range'] = candle_Range
        features['Volatility'] = volatility_1
        features['z_slope'] = z_slope

        return features

def CointTrading(DT , CMP_DT , **kwargs):
    # creating variable
    features=pd.DataFrame()
    lags = 5
    lookback = kwargs['lookback']
    window = kwargs['window']
    y = DT
    x = CMP_DT

    # calculating indicators
    spread , hedge_ratio = rolling_spread(y['close'] , x['close'] , lookback)
    smoothPrice = y['close'].rolling(window = lookback).mean()
    corr = y['close'].pct_change().rolling(window).corr(x['close'].pct_change())
    
    features['corr'] = corr
    features['spread'] = spread
    features['hedge_ratio'] = hedge_ratio
    features['z_spread'] = (spread - spread.rolling(window).mean()) / spread.rolling(window).std()
    features['hedge_ratio_std'] = hedge_ratio.rolling(window).std()  # Stability measure
    features['spread_velocity'] = spread.diff()
    features['spread_acceleration'] = spread.diff().diff()
    features['spread_volatility'] = spread.rolling(window = window).std()
    features['Slope'] = smoothPrice.diff(window) / window
    features['Price_Vs_SmoothPrice'] = (y['close']-smoothPrice)/smoothPrice

    return features
