from utils_functions import *
from scipy.stats import skew , kurtosis
import pandas as pd
from arch.unitroot import VarianceRatio

def rolling_variance_ratio(dt , window , lags) :
    log_series=np.log(dt['close'])

    def compute_vt(x) :
        return VarianceRatio(x , lags=lags , robust=True , overlap=True).vr

    VR_Series=log_series.rolling(window=window).apply(compute_vt)

    return VR_Series

def HurstRegime(DT , lags , lookback , regime_center) :
    regime_center_lower , regime_center_upper=regime_center

    HurstSeries=rolling_variance_ratio(DT.copy() , window=lookback , lags=lags).rename('HurstSeries')

    regime_lbs=pd.Series(0 , index=HurstSeries.index , name='_HURST_RAW_REG_' , dtype='int64')
    regime_confi_lvl=pd.Series(0 , index=HurstSeries.index , name='regime_confi_lvl' , dtype='float64')

    tr_mask=HurstSeries >= regime_center_upper
    mr_mask=HurstSeries <= regime_center_lower

    regime_lbs.loc[tr_mask]=1
    regime_lbs.loc[mr_mask]=-1

    regime_confi_lvl.loc[tr_mask]=HurstSeries-regime_center_upper
    regime_confi_lvl.loc[mr_mask]=HurstSeries-regime_center_lower

    return regime_lbs , regime_confi_lvl

def compute_regime_age(regimes , regime_window):
    is_same = (regimes == regimes.shift(1)).astype(int)
    r_age = is_same.groupby((is_same == 0).cumsum()).cumsum().astype(int)

    switched = (is_same == 0).astype(int)
    regime_switch_freq = switched.rolling(window=regime_window , min_periods=1).sum().astype(int)

    return np.log1p(r_age).rename('Regime_AGE')  ,   regime_switch_freq.rename('regime_switch_freq')

def compute_market_indicators(DT , args) :
    daily_change = ((DT['close']-DT['close'].shift()) / DT['close'].shift())

    REGIME_FEAT={}
    for p in args['Skewness'] :
        REGIME_FEAT[f'Skewness_{p}']=daily_change.rolling(window=p).apply(lambda x : skew(x , bias=False) ,raw=True)

    for p in args['Kurtosis'] :
        REGIME_FEAT[f'Kurtosis_{p}']=daily_change.rolling(window=p).apply(lambda x : kurtosis(x , bias=False) ,raw=True)

    for p in args['DailyVolatility'] :
        REGIME_FEAT[f'DailyVolatility_{p}']=daily_change.ewm(span=p , min_periods=1).std()

    for p in args['Kuafman_ER_'] :
        REGIME_FEAT[f'Kuafman_ER_{p}']=kaufMan_EF_Ratio(DT['close'] , p)

    return pd.DataFrame(REGIME_FEAT)


def compute_market_regime(DT , **kwargs):

    # HurstRegime
    args = {**kwargs['Hurst_Params'] , 'regime_center':kwargs['regime_center']}
    _HURST_RAW_REG_ , regime_confi_lvl = HurstRegime(DT , **args)

    # Indicators
    args = kwargs['Indicator_params']
    indicators = compute_market_indicators(DT ,args)

    # RegimeAge & Freq
    regime_age , regime_switch_freq = compute_regime_age(_HURST_RAW_REG_ , kwargs['RegimeWindow'])

    market_regime = pd.concat([indicators , regime_confi_lvl , regime_age.rename('Regime_AGE_BS') , regime_switch_freq ,
                                     _HURST_RAW_REG_.rename('_HURST_RAW_REG_BS')] , axis=1)

    return market_regime