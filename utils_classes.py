import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone ,TransformerMixin
import pandas_ta as ta

class BaggingBootstrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None, n_estimators=3, sigma=0.05, label_noise=0.05,
                 model_regime=None, skip_patterns=("_VOL_RAW_REG_", "_HURST_RAW_REG_", "gaps_binary"),
                 random_state=42):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.sigma = sigma
        self.label_noise = label_noise
        self.model_regime = model_regime
        self.skip_patterns = skip_patterns if skip_patterns is not None else []
        self.random_state = random_state
        self.estimators_ = []
        self.classes_ = None

    def _noise_col_indices(self, X):
        if isinstance(X, pd.DataFrame):
            skip = frozenset(col for col in X.columns if any(p in col for p in self.skip_patterns))
            return [i for i, col in enumerate(X.columns) if col not in skip]
        return list(range(X.shape[1]))

    def _fit_one(self, seed, X_arr, one_hot_y, noise_col_idx, sw_array, in_regime_mask):
        rng = np.random.default_rng(seed)
        n_samples, n_classes = X_arr.shape[0], one_hot_y.shape[1]

        clf = clone(self.base_estimator)
        if hasattr(clf, "random_state"):
            clf.set_params(random_state=int(rng.integers(0, 100_000)))

        idx = rng.choice(n_samples, size=n_samples, replace=True)
        X_sample = X_arr[idx].copy()
        y_oh = one_hot_y[idx].copy()

        if self.sigma > 0.0 and noise_col_idx:
            X_sample[:, noise_col_idx] *= rng.normal(1.0, self.sigma, size=(n_samples, len(noise_col_idx)))

        if self.label_noise > 0.0 and n_classes > 1:
            mask = in_regime_mask[idx] if in_regime_mask is not None else np.ones(n_samples, dtype=bool)
            if mask.any():
                y_oh[mask] = y_oh[mask] * (1.0 - self.label_noise) + (self.label_noise / n_classes)
            cumprobs = np.cumsum(y_oh, axis=1)
            chosen = np.clip((rng.random(n_samples)[:, None] > cumprobs).sum(axis=1), 0, n_classes - 1)
            y_final = self.classes_[chosen]
        else:
            y_final = self.classes_[np.argmax(y_oh, axis=1)]

        clf.fit(X_sample, y_final, **({'sample_weight': sw_array[idx]} if sw_array is not None else {}))
        return clf

    def fit(self, X, y, sample_weight=None, regime_labels=None):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_indexed = np.fromiter((class_to_idx[c] for c in y), dtype=int, count=len(y))
        one_hot_y = np.eye(n_classes, dtype=np.float32)[y_indexed]

        noise_col_idx = self._noise_col_indices(X)
        X_arr = np.asarray(X, dtype=np.float64)
        sw_array = np.asarray(sample_weight) if sample_weight is not None else None
        in_regime_mask = (np.asarray(regime_labels) == self.model_regime
                          if regime_labels is not None and self.model_regime is not None else None)

        seeds = np.random.default_rng(self.random_state).integers(0, 2**31, size=self.n_estimators)
        self.estimators_ = [self._fit_one(int(s), X_arr, one_hot_y, noise_col_idx, sw_array, in_regime_mask)
                            for s in seeds]
        return self

    def predict_proba(self, X):
        X_arr = np.asarray(X)
        n_classes = len(self.classes_)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        out = np.zeros((X_arr.shape[0], n_classes), dtype=np.float64)
        for clf in self.estimators_:
            proba = clf.predict_proba(X_arr)
            if len(clf.classes_) == n_classes and np.array_equal(clf.classes_, self.classes_):
                out += proba
            else:
                for j, cls in enumerate(clf.classes_):
                    out[:, class_to_idx[cls]] += proba[:, j]
        out /= len(self.estimators_)
        return out

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

class RiskAdjustmentEngine:
    def __init__(self):
        pass

    def GetRiskData(self , x ,  lookback , multiplier):

        atr = ta.atr(x['high'], x['low'], x['close'], lookback)
        atr_dist_pct = (atr.shift(1) * multiplier) / x['open']
        
#       days
        SL_RANG_POS=((x['open']-x['low']) / x['open']).rename('SL_RANG_POS')
        SL_RANG_NEG=((x['high']-x['open']) / x['open']).rename('SL_RANG_NEG')
        
#       bullish days
        MAX_STOP_POS = atr_dist_pct.rename('MAX_STOP_POS')

#       bearish days
        MAX_STOP_NEG = atr_dist_pct.rename('MAX_STOP_NEG')

        # setting up the risk data
        RiskData=pd.concat([-SL_RANG_POS ,  -MAX_STOP_POS, -SL_RANG_NEG ,-MAX_STOP_NEG] , axis=1)

        return RiskData , (atr * multiplier).iloc[-1]

    def GetAdjustedReturns(self  ,t_returns , y_pred , RiskData):
        strategy_returns = (t_returns.copy() * y_pred).rename('strategy_returns')
        y_pred = pd.Series(y_pred , index=t_returns.index , name='predictions')

        x = pd.concat([RiskData  , strategy_returns  , y_pred] ,axis=1).dropna()

# #     trading conditions
        bull_sl =(x['SL_RANG_POS']<=x['MAX_STOP_POS']) & (x['predictions'] > 0) & (x['MAX_STOP_POS'] < 0)
        bear_sl= (x['SL_RANG_NEG']<=x['MAX_STOP_NEG']) & (x['predictions'] < 0) & (x['MAX_STOP_NEG'] < 0)
        rets=x['strategy_returns'].copy()

#       sl hit:
        rets[bull_sl] = x['MAX_STOP_POS'][bull_sl]
        rets[bear_sl] = x['MAX_STOP_NEG'][bear_sl]

        return rets



