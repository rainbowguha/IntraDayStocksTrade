import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone ,TransformerMixin
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from scipy.stats import chi2


class BaggingBootstrapper(BaseEstimator , ClassifierMixin) :
    def __init__(self , base_estimator=None , n_estimators=3 , sigma=0.05 , label_noise=0.05 ,
                 model_regime=None , skip_patterns=("_VOL_RAW_REG_" , "_HURST_RAW_REG_" , "gaps_binary") ,
                 random_state=42) :
        self.base_estimator=base_estimator
        self.n_estimators=n_estimators
        self.sigma=sigma
        self.label_noise=label_noise
        self.model_regime=model_regime
        self.skip_patterns=skip_patterns if skip_patterns is not None else []
        self.random_state=random_state
        self.estimators_=[]
        self.classes_=None
        self.bin_edges_=None

    def _bootstrap_indices(self , n_samples) :
        rng=np.random.default_rng(self.random_state)
        return [rng.choice(n_samples , size=n_samples , replace=True) for _ in range(self.n_estimators)]

    def _volatility_indices(self , Vflag) :
        bin_codes , bin_edges=pd.qcut(Vflag , q=self.n_estimators , labels=False , retbins=True , duplicates='drop')
        self.bin_edges_=bin_edges
        return [np.flatnonzero(bin_codes.values == b) for b in range(len(bin_edges)-1)]

    def _noise_col_indices(self , X) :
        if isinstance(X , pd.DataFrame) :
            skip=frozenset(col for col in X.columns if any(p in col for p in self.skip_patterns))
            return [i for i , col in enumerate(X.columns) if col not in skip]
        return list(range(X.shape[1]))

    def _fit_one(self , seed , X_arr , one_hot_y , noise_col_idx , sw_array , in_regime_mask , indices) :
        rng=np.random.default_rng(seed)
        n_samples , n_classes=len(indices) , one_hot_y.shape[1]

        clf=clone(self.base_estimator)
        if hasattr(clf , "random_state") :
            clf.set_params(random_state=int(rng.integers(0 , 100_000)))

        X_sample=X_arr[indices].copy()
        y_oh=one_hot_y[indices].copy()

        if self.sigma > 0.0 and noise_col_idx :
            X_sample[: , noise_col_idx]*=rng.normal(1.0 , self.sigma , size=(n_samples , len(noise_col_idx)))

        if self.label_noise > 0.0 and n_classes > 1 :
            mask=in_regime_mask[indices] if in_regime_mask is not None else np.ones(n_samples , dtype=bool)
            if mask.any() :
                y_oh[mask]=y_oh[mask] * (1.0-self.label_noise)+(self.label_noise / n_classes)
            cumprobs=np.cumsum(y_oh , axis=1)
            chosen=np.clip((rng.random(n_samples)[: , None] > cumprobs).sum(axis=1) , 0 , n_classes-1)
            y_final=self.classes_[chosen]
        else :
            y_final=self.classes_[np.argmax(y_oh , axis=1)]

        clf.fit(X_sample , y_final , **({'sample_weight' : sw_array[indices]} if sw_array is not None else {}))
        return clf

    def fit(self , X , y , sample_weight=None , Vflag=None , regime_labels=None) :
        self.classes_=np.unique(y)
        n_classes=len(self.classes_)
        class_to_idx={c : i for i , c in enumerate(self.classes_)}
        y_indexed=np.fromiter((class_to_idx[c] for c in y) , dtype=int , count=len(y))
        one_hot_y=np.eye(n_classes , dtype=np.float32)[y_indexed]

        if Vflag is not None :
            estimator_indices=self._volatility_indices(Vflag)
        else :
            estimator_indices=self._bootstrap_indices(len(X))

        noise_col_idx=self._noise_col_indices(X)
        X_arr=np.asarray(X , dtype=np.float64)
        sw_array=np.asarray(sample_weight) if sample_weight is not None else None
        in_regime_mask=(np.asarray(regime_labels) == self.model_regime
                        if regime_labels is not None and self.model_regime is not None else None)

        seeds=np.random.default_rng(self.random_state).integers(0 , 2 ** 31 , size=len(estimator_indices))
        self.estimators_=[
            self._fit_one(int(s) , X_arr , one_hot_y , noise_col_idx , sw_array , in_regime_mask , indices)
            for s , indices in zip(seeds , estimator_indices)]
        return self

    def predict_proba(self , X , Vflag=None) :
        X_arr=np.asarray(X)
        n_samples=X_arr.shape[0]
        n_classes=len(self.classes_)
        out=np.zeros((n_samples , n_classes) , dtype=np.float64)

        if Vflag is None or self.bin_edges_ is None :
            class_to_idx={c : i for i , c in enumerate(self.classes_)}
            for clf in self.estimators_ :
                proba=clf.predict_proba(X_arr)
                if len(clf.classes_) == n_classes and np.array_equal(clf.classes_ , self.classes_) :
                    out+=proba
                else :
                    for j , cls in enumerate(clf.classes_) :
                        out[: , class_to_idx[cls]]+=proba[: , j]
            out/=len(self.estimators_)
            return out

        # Route samples based on saved volatility bin edges
        v_arr=np.asarray(Vflag)
        bin_ids=np.digitize(v_arr , self.bin_edges_[1 :-1])

        for bin_idx , clf in enumerate(self.estimators_) :
            sample_mask=(bin_ids == bin_idx)
            if not np.any(sample_mask) :
                continue

            sub_X=X_arr[sample_mask]
            proba=clf.predict_proba(sub_X)

            if len(clf.classes_) == n_classes and np.array_equal(clf.classes_ , self.classes_) :
                out[sample_mask]=proba
            else :
                clf_class_to_idx={c : i for i , c in enumerate(clf.classes_)}
                for j , cls in enumerate(self.classes_) :
                    if cls in clf_class_to_idx :
                        out[sample_mask , j]=proba[: , clf_class_to_idx[cls]]

        return out

    def predict(self , X , Vflag=None) :
        return self.classes_[np.argmax(self.predict_proba(X , Vflag=Vflag) , axis=1)]


class MahalanobisMetaFeatureExtractor(BaseEstimator , TransformerMixin) :
    """Computes Mahalanobis Distance & Chi-Square OOD Probability with Shrinkage."""

    def __init__(self , use_shrinkage: bool = True ,
                 skip_patterns=("_VOL_RAW_REG_" , "_HURST_RAW_REG_" , "gaps_binary")) :
        self.use_shrinkage=use_shrinkage
        self.mean_=None
        self.inv_cov_=None
        self.n_features_=None
        self.skip_patterns=skip_patterns

    def _preprocess_X(self , x) :
        X=x.copy()
        X=X.loc[: , ~X.columns.duplicated()]
        Valid_col=[col for col in X.columns if not any(p in col for p in self.skip_patterns)]
        return np.asarray(X[Valid_col] , dtype=np.float64)

    def fit(self , x: list) :
        X=self._preprocess_X(x)
        self.n_features_=X.shape[1]
        self.mean_=np.mean(X , axis=0)

        cov=np.cov(X , rowvar=False)
        if self.use_shrinkage :
            shrink_factor=0.05
            trace=np.trace(cov) if cov.ndim == 2 else cov
            cov=(1-shrink_factor) * cov+shrink_factor * np.eye(self.n_features_) * trace / self.n_features_

        self.inv_cov_=np.linalg.pinv(cov)
        return self

    def transform(self , x: list) -> np.ndarray :
        X=self._preprocess_X(x)
        diff=X-self.mean_

        dist_sq=np.einsum('ij,jk,ik->i' , diff , self.inv_cov_ , diff)
        dist_sq=np.maximum(dist_sq , 0.0)

        distances=np.sqrt(dist_sq)
        ood_probs=chi2.cdf(dist_sq , df=self.n_features_)
        return np.column_stack([distances , ood_probs])






