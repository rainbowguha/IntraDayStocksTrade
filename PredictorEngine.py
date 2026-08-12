import pandas as pd
from utils_functions import *
from StrategyRepo import *
from market_regime import compute_market_regime , ComputeVol_Z
from datetime import datetime , timedelta
import bottleneck as bn

class PredictorEngine:
    def __init__(self , str_name ,  symbol):
        self.symbol = symbol
        self.__NAME__ = '{}_{}'.format(str_name , symbol)
        self.str_params , self.mkt_reg_params  , self.SL_COMM , trained_upto , n_models = load_str_params(self.__NAME__)
        self.__BS_MOD__=[load_models(m , self.__NAME__) for m in range(1 , n_models+1)]
        self.__META_MOD__=load_meta_models(self.__NAME__)
        self.__DIST_MOD__=load_dist_extractor(self.__NAME__)
        self.__TRADE_LOGS__=load_trade_logs(self.__NAME__)
        self.__START_DATE__ = datetime.strptime(trained_upto , "%Y-%m-%d") + timedelta(days=1)
        self.data = None
        self.t_returns = None
        self.market_regime = None
        self.Vol_Z = None

    def load_history(self):
        t_day = datetime.now(time_zone).date()
        self.data = load_csv(self.symbol , t_day)
        self.t_returns=((self.data['close']-self.data['open']) / self.data['open'])

        # MarketRegime
        self.market_regime=compute_market_regime(self.data.copy() , **self.mkt_reg_params).dropna()
        self.Vol_Z = ComputeVol_Z(self.data.copy() , **self.mkt_reg_params)

    def GenFeatures(self , X_col , **kwargs) :
        lags=5
        features=pd.DataFrame()

        if 'MomTrading' in self.__NAME__ :
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
        if lags :
            lagged_features=[features.shift(lag).add_suffix(f'_{lag}') for lag in range(1 , lags+1)]
            features=pd.concat([features]+lagged_features , axis=1)

        # Adding  Regimes & Normalization
        common_index=features.index.intersection(self.market_regime.index)
        features=features.loc[common_index]

        features=pd.concat([features , self.market_regime] , axis=1)

        NORM_FEAT=self.Normalization(features[X_col].join(self.Vol_Z , how='inner') , kwargs['normal_window'])
        NORM_FEAT['Vol_Z']=NORM_FEAT.pop('Vol_Z')

        return NORM_FEAT[self.__START_DATE__ :]

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
            return norm

        # Volatility based Normalization
        if normalization :
            NORM_DATASET=features.drop(columns=skip_columns)
            OTHER=features[skip_columns]
            NORM_FEAT=normalize(NORM_DATASET , normal_window)
            standardized_features=pd.concat([NORM_FEAT , OTHER] , axis=1)
        else :
            standardized_features=features

        standardized_features=standardized_features.replace([np.inf , -np.inf] , np.nan)

        return round(standardized_features , DECIMAL_POINTS)

    def _GEN_META_features(self , X , short_window=5 , long_window=10):
        eps=1e-5
        ComSlip=self.SL_COMM['COMM']
        meta_features = pd.DataFrame()
        trades = []

        # past records
        _hX = [x.shift(1).dropna() for x in X]

        common_index =_hX[0].index
        for x in _hX[1:]:
            common_index = common_index.intersection(x.index)

        INPUT_DT = [x.loc[common_index] for x in _hX]
        hard_predictions = [self.__BS_MOD__[i].predict(INPUT_DT[i].iloc[: , :-1] , INPUT_DT[i].iloc[: , -1]) for i in range(len(self.__BS_MOD__))]

        eval_returns_asset = self.t_returns.loc[common_index]

        for i in range(len(self.__BS_MOD__)):
            rets = (hard_predictions[i] * eval_returns_asset) - (np.abs(hard_predictions[i]) * ComSlip / 100)
            active_trade_mask=(hard_predictions[i] != 0)
            active_trade_rets=rets[active_trade_mask]

            trade_logs = self.__TRADE_LOGS__[f'm_{i}']
            from_date = trade_logs.index[-1] + timedelta(days=1)
            to_dates = common_index[-1]
            trades.append(pd.concat([trade_logs[trade_logs!=0.0] , active_trade_rets[from_date:]])[:to_dates].iloc[-long_window:])

        if np.any([len(trade)<long_window for trade in trades]):
            return meta_features

        lst_indices = X[-1].index[-1]
        lst_X = [x.loc[[lst_indices]] for x in X]
        G_signal = [self.__BS_MOD__[i].predict(lst_X[i].iloc[:, :-1] , lst_X[i].iloc[: , -1]) for i in range(len(self.__BS_MOD__))]
        today_prediction = np.sign(np.sum(G_signal , axis=0))

        proba = [self.__BS_MOD__[i].predict_proba(lst_X[i].iloc[:, :-1] , lst_X[i].iloc[: , -1]) for i in range(len(self.__BS_MOD__))]
        dir_scores = np.array([p[: , -1] - p[: , 0] for p in proba])
        confidence_lvl = np.array([1 - p[: , 1] for p in proba])

        dist_ood_proba = [self.__DIST_MOD__[i].transform(lst_X[i].iloc[: , :-1]) for i in range(len(self.__BS_MOD__))]

        meta_features['signal'] = today_prediction
        meta_features['mean_dr_score'] = np.mean(dir_scores , axis=0)
        meta_features['mean_confi_lvl'] = np.mean(confidence_lvl , axis=0)

        for i ,trade_rets in enumerate(trades):

            # 1. Trade-level Sharpe Velocity
            t_fast_sharpe=compute_rolling_sharpe(trade_rets , window=short_window)
            t_slow_sharpe=compute_rolling_sharpe(trade_rets , window=long_window)
            trade_sharpe_velocity=t_fast_sharpe-t_slow_sharpe

            # 2. Trade-level Win Rate Ratio
            t_fast_win=(trade_rets > 0).astype(float).rolling(window=short_window).mean()
            t_slow_win=(trade_rets > 0).astype(float).rolling(window=long_window).mean()
            trade_winrate_ratio=(t_fast_win+eps) / (t_slow_win+eps)

            # 3. Trade-level Downside Volatility Ratio
            t_downside_rets=np.minimum(0 , trade_rets)
            t_fast_down_vol=np.sqrt((t_downside_rets ** 2).rolling(window=short_window).mean())
            t_slow_down_vol=np.sqrt((t_downside_rets ** 2).rolling(window=long_window).mean())
            trade_downside_vol_ratio=(t_fast_down_vol+eps) / (t_slow_down_vol+eps)

            meta_features[f'sharpe_velocity_m{i}']=trade_sharpe_velocity.iloc[-1]
            meta_features[f'winrate_ratio_m{i}']=trade_winrate_ratio.iloc[-1]
            meta_features[f'downside_vol_ratio_m{i}'] = trade_downside_vol_ratio.iloc[-1]

            # proba based features
            meta_features[f'dir_sc_m{i}'] = dir_scores[i]
            meta_features[f'confi_lvl_m{i}'] = confidence_lvl[i]
            meta_features[[f'dist_m{i}' , f'OOD_proba_m{i}']] = dist_ood_proba[i]

        return meta_features

    def __GET_SIG__(self , meta_threshold=0.5):
        t_pred = 0

        self.load_history()
        X = [self.GenFeatures(**self.str_params[i]) for i in range(len(self.str_params))]

        try:
            meta_feat = self._GEN_META_features(X)

            if not meta_feat.empty:
                meta_proba = self.__META_MOD__.predict_proba(meta_feat)[: , 1]
                base_signal = meta_feat['signal'].iloc[-1]
                t_pred = base_signal if meta_proba>=meta_threshold else 0
        except:
            print('UNABLE TO PROCESS SIGNAL:{}'.format(self.symbol))
        return t_pred , self.SL_COMM['sl_params']