import os
import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime
import pytz
import pickle as pk
from scipy.stats import skew , kurtosis
from Params import (load_csv , get_ensemble_n , kaufMan_EF_Ratio , ComputeRegime_HURST
                    )

import warnings as ws
ws.simplefilter('ignore')


class PredictorEngine:
    TICKER = None

    def __init__(self,name,ticker,Components,interval):
        self.time_zone = pytz.timezone('Asia/Kolkata')
        self.strategy_name = name
        self.symbol = ticker
        self.Components = Components
        self.interval = interval
        self.data = None
        self.cmp_data = None
        self.params = None
        self.sl_params = None
        self.Vol_Params = None
        self.ensemble_model = None
        self.base_ml = [None]
        self.series_idx = None
        self.Vol_Model = None
        self.Hurst_Model = None
        self.load_model()

    def load_model(self):
        self.params , self.sl_params , self.Vol_Params , ensemble=get_ensemble_n(self.strategy_name)
        for n in range(1,ensemble+1):
            self.base_ml.append(self.get_model(n, 'base_model'))

        self.ensemble_model = self.get_model(1,'ENSEM')
        self.Vol_Model = self.get_model(1 , 'Vol_Model')
        self.Hurst_Model = self.get_model(1 , 'Hurst_Model')

    def get_model(self, n, model):
        file_name = f'{self.strategy_name}_{n}' if model == 'base_model' else f'{self.strategy_name}_{model}'
        file_path = os.path.join('Models', self.strategy_name, f'{file_name}.pkl')
        with open(file_path, 'rb') as file:
            loaded_model = pk.load(file)
        return loaded_model

    def load_history(self):
        common_index = None
        today=datetime.now(self.time_zone).date()
        symbol_list = [self.symbol , self.Components] if self.Components else [self.symbol]
        history = [load_csv(s , today) for s in symbol_list]
        if len(symbol_list)>1:
            for h in history:
                common_index = h.index if common_index is None else common_index.intersection(h.index)
            self.data = history[0].loc[common_index]
            self.cmp_data = history[1].loc[common_index]
        else:
            self.data = history[0]

    def Normalization(self , features , normal_window=10 , skip_col: list = ['_VOL_RAW_REG_' , '_HURST_RAW_REG_'] , normalization=True , winsorization=False) :

        # preprocessing features
        features=features.dropna(axis=0)
        skip_col=[col for col in skip_col if col in features.columns]

        # rolling normalization
        def normalize(x , window):
            z = (x - x.rolling(window=window).median())/x.rolling(window = window).std()
            return z

        # tackling extreme value(* outliers)
        if winsorization :
            columns=[col for col in features.columns if col not in skip_col]
            Q_Range=features[columns].quantile([0.01 , 0.85])
            FEAT=features[columns].to_numpy()
            features[columns]=np.clip(FEAT , Q_Range.iloc[0].to_numpy() , Q_Range.iloc[1].to_numpy())

            # Normalization
        if normalization :
            NORM_DATASET = features.drop(columns = skip_col)
            OTHER = features[skip_col]
            NORM_FEAT = normalize(NORM_DATASET , normal_window)
            standardized_features = pd.concat([NORM_FEAT , OTHER] , axis=1)
        else :
            standardized_features=features

        return standardized_features.dropna(axis=0)

    def VolatilityRegimer(self , x  , lookback ,DoubleSmoothingWindow) :
        DailyChange=x.rename('DailyChange')
        Volatility=DailyChange.rolling(window=lookback).std().rename('Volatility')
        if DoubleSmoothingWindow:
            Volatility = Volatility.rolling(window=DoubleSmoothingWindow).mean()

        FEAT=pd.concat([DailyChange , Volatility] , axis=1).dropna()
        regimes=pd.Series(self.Vol_Model.predict(FEAT) , index=FEAT.index , name='_VOL_RAW_REG_')

        return regimes

    def Hurst_regimer(self , x , hurst) :
        DailyChange=x.rename('DailyChange')
        FEAT=pd.concat([DailyChange , hurst] , axis=1).dropna()
        regimes=pd.Series(self.Hurst_Model.predict(FEAT) , index=FEAT.index , name='_HURST_RAW_REG_')
        return regimes

    def GetMarketRegime(self) :
        daily_change=(self.data['close']-self.data['close'].shift(1)) / self.data['close'].shift(1)
        HurstRegime=ComputeRegime_HURST(self.data['close'] , window=100)
        _VOL_RAW_REG_ =self.VolatilityRegimer(daily_change , **self.Vol_Params)
        _Hurst_RAW_REG_=self.Hurst_regimer(daily_change , HurstRegime)

        REGIME_FEAT={}
        for w in [10 , 30 , 60 , 100] :
            REGIME_FEAT[f'skew_{w}']=daily_change.rolling(window=w).apply(lambda x : skew(x , bias=False) , raw=True)
            REGIME_FEAT[f'kurt_{w}']=daily_change.rolling(window=w).apply(lambda x : kurtosis(x , bias=False) ,
                                                                          raw=True)
            REGIME_FEAT[f'Vol_Regime_{w}']=daily_change.ewm(span=w , min_periods=1).std()
            REGIME_FEAT[f'Kuafman_ER_{w}']=kaufMan_EF_Ratio(self.data['close'] , w)

        # joining the Market regime feature set
        return pd.concat([HurstRegime , pd.DataFrame(REGIME_FEAT) , _VOL_RAW_REG_ , _Hurst_RAW_REG_] , axis = 1)

    def MomTrading(self , window , lookback , lags=5 , normal_window=10 , X_col=None) :
        # creating variable
        features=pd.DataFrame()

        # calculating indicators
        volatility_1=self.data['close'].ewm(span=window).std()
        rsi= ta.rsi(self.data['close'] , lookback)
        adx= ta.adx(self.data['high'] , self.data['low'] , self.data['close'] , lookback).iloc[: , 0]
        candle_Range= self.data['high']-self.data['low']
        ibs= (self.data['close']-self.data['low']) / candle_Range
        ma= self.data['close'].ewm(span=lookback).mean()
        corr= self.data['close'].rolling(window=window).corr(ma)
        spr= self.data['close']-ma
        slope= ta.slope(ma , window)

        mean=100 * self.data['close'].pct_change().rolling(window=lookback).mean()
        std=100 * self.data['close'].pct_change().rolling(window=lookback).std()
        r=100 * self.data['close'].pct_change()
        z_score=(r-mean) / std

        hh=z_score.rolling(window=window).max()
        ll=z_score.rolling(window=window).min()
        mid=z_score.rolling(window=window).quantile(0.5)
        z_slope = ta.slope(z_score , window)

        # setting features
        features['daily_change']=self.data['close'].pct_change()
        features['rsi']=rsi
        features['adx']=adx
        features['ibs']=ibs
        features['spr']=spr
        features['slope']=slope
        features['zscore']=z_score
        features['UP_Z']=z_score-hh
        features['DN_Z']=ll-z_score
        features['MID_Z']=z_score-mid
        features['corr']=corr
        features['daily_Range']=candle_Range
        features['Volatility']=volatility_1
        features['z_slope'] = z_slope

        #  calculating lagged features
        if lags :
            lagged_features=[features.shift(lag).add_suffix(f'_{lag}') for lag in range(1 , lags+1)]
            features=pd.concat([features]+lagged_features , axis=1)

        #  Adding  Regimes
        market_regime = self.GetMarketRegime().dropna()
        common_index=features.index.intersection(market_regime.index)
        features=features.loc[common_index]

        # adding Market Regimes
        features=pd.concat([features , market_regime] , axis=1)

        normalized_features=self.Normalization(features[X_col] , normal_window)

        return normalized_features

    def generate_features(self, params):
        normalize_features = None
        if self.strategy_name=='MomTrading_{}'.format(self.symbol) or self.strategy_name=='MeanTrading_{}'.format(self.symbol):
            normalize_features = self.MomTrading(**params)

        return normalize_features.tail(5)

    def GetPrediction(self) :
        self.load_history()
        probability=[self.base_ml[i].predict_proba(self.generate_features(param)) for i , param in enumerate(self.params , start=1)]
        proba=np.column_stack(probability)
        signal = self.ensemble_model.predict(proba)[-1]
        return int(signal) , self.get_sl(signal , **self.sl_params)

    def get_sl(self , signal ,  lookback , quantiles):
        x =self.data
        var = 0

#       days
        pos_candle=x['open'] < x['close']
        neg_candle=x['open'] > x['close']
        SL_RANG_POS=(x['open']-x['low']).rename('SL_RANG_POS')
        SL_RANG_NEG=(x['high']-x['open']).rename('SL_RANG_NEG')

        if signal > 0 :
    #       bullish days
            contra_move_bullish=SL_RANG_POS[pos_candle]
            contra_move_bullish=contra_move_bullish.reindex(x.index , method='ffill')
            MAX_STOP_POS=contra_move_bullish.rolling(window=lookback , min_periods=2).quantile(quantiles).fillna(0.0).rename('MAX_STOP_POS')
            var=MAX_STOP_POS.iloc[-1]
        elif signal < 0 :
    #       bearish days
            contra_move_bearish=SL_RANG_NEG[neg_candle]
            contra_move_bearish=contra_move_bearish.reindex(x.index , method='ffill')
            MAX_STOP_NEG=contra_move_bearish.rolling(window=lookback , min_periods=2).quantile(quantiles).fillna(0.0).rename('MAX_STOP_NEG')
            var=MAX_STOP_NEG.iloc[-1]

        return var