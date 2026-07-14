import pandas as pd
from utils_classes import RiskAdjustmentEngine
from utils_functions import *
from StrategyRepo import *
from market_regime import compute_market_regime
from datetime import datetime , timedelta
import bottleneck as bn

class PredictorEngine:
    def __init__(self , str_name ,  symbol):
        self.symbol = symbol
        self.__NAME__ = '{}_{}'.format(str_name , symbol)
        self.str_params , self.mkt_reg_params  , self.SL_COMM , trained_upto , n_models = load_str_params(self.__NAME__)
        self.__BS_MOD__ = [load_models(m , self.__NAME__) for m in range(1 , n_models + 1)]
        self.__START_DATE__ = datetime.strptime(trained_upto , "%Y-%m-%d") + timedelta(days=1)
        self.data = None
        self.t_returns = None

    def load_history(self):
        t_day = datetime.now(time_zone).date()
        self.data = load_csv(self.symbol , t_day)
        self.t_returns=((self.data['close']-self.data['open']) / self.data['open'])

    def GenFeatures(self , X_col  , **kwargs):
        lags = 5
        features = pd.DataFrame()

        if 'MomTrading' in self.__NAME__:
            features=MomTrading(self.data , None , **kwargs)

        # adding gaps & overnight features
        eps=1e-8

        gap=100 * (self.data['open']-self.data['close'].shift()) / (self.data['close'].shift()+eps)
        intraday_range=100 * (self.data['high']-self.data['low']) / (self.data['close'].shift()+eps)
        gap_volatility=gap.rolling(kwargs['window']).std()

        features['gap']=gap
        features['gap_vs_intraday_range']=gap / (intraday_range+eps)
        features['gap_ratio']=gap / (gap.shift()+eps)
        features['gap_vs_volatility']=gap / (gap_volatility+eps)
        features['gap_fill_ratio']=100 * (self.data['close']-self.data['open']) / (
                    (self.data['open']-self.data['close'].shift())+eps)
        features['gaps_binary']=np.sign(gap).rolling(window=kwargs['window'] , min_periods=2).sum()

        # calculating lagged features
        if lags:
            lagged_features = [features.shift(lag).add_suffix(f'_{lag}') for lag in range(1, lags+1)]
            features = pd.concat([features] + lagged_features, axis=1)

        # Adding  Regimes & Normalization
        market_regime=compute_market_regime(self.data.copy() , **self.mkt_reg_params).dropna()
        common_index=features.index.intersection(market_regime.index)
        features=features.loc[common_index]

        features=pd.concat([features , market_regime] , axis=1)

        NORM_FEAT=self.Normalization(features[X_col] , kwargs['normal_window'])

        return NORM_FEAT[self.__START_DATE__:]

    def Normalization(self , features , normal_window ,
                      skip_patterns: list = ['_VOL_RAW_REG_' , '_HURST_RAW_REG_' , 'gaps_binary'] ,
                      normalization=True) :

        epsilon=1e-8

        # preprocessing features
        features=features.dropna(axis=0)
        skip_columns=[col for col in features.columns if any(pattern in col for pattern in skip_patterns)]

        def normalize(x: pd.DataFrame , window: int , vol_window: int | None = None ,
                      min_periods_frac: float = 0.5) -> pd.DataFrame :

            vol_window=vol_window or window
            min_periods=max(2 , int(vol_window * min_periods_frac))

            ranked=bn.move_rank(x.values , window=window , axis=0)
            norm=pd.DataFrame((ranked-0.5) * 2 , index=x.index , columns=x.columns)

            rolling_vol=(
                x.rolling(window=vol_window , min_periods=min_periods)
                .std()
                .shift(1))

            vol_floor=(
                rolling_vol
                .expanding(min_periods=min_periods)
                .quantile(0.01)
                .shift(1))

            rolling_vol=rolling_vol.clip(lower=vol_floor)

            vol_scaled=norm / rolling_vol

            return vol_scaled.clip(-1.0 , 1.0)

        # Volatility based Normalization
        if normalization :
            NORM_DATASET=features.drop(columns=skip_columns)
            OTHER=features[skip_columns]
            NORM_FEAT=normalize(NORM_DATASET , normal_window)
            standardized_features=pd.concat([NORM_FEAT , OTHER] , axis=1)
        else :
            standardized_features=features

        standardized_features=standardized_features.replace([np.inf , -np.inf] , np.nan)

        return standardized_features

    def __GET_SIG__(self):
        self.load_history()
        __RAW_FEAT__= [self.GenFeatures(**self.str_params[i]) for i in range(len(self.str_params))]

        # today prediction
        try:
            today_prediction = [self.__BS_MOD__[i].predict(__RAW_FEAT__[i].iloc[-1].values.reshape(1, -1)) for i in range(len(__RAW_FEAT__))]
            today_prediction = np.sign(np.sum(today_prediction))
        except :
            today_prediction = 0

        # logs ////
        # print(self.symbol , self.sl_points , today_prediction)

        return today_prediction , self.SL_COMM['sl_params']