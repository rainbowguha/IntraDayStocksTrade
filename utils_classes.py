import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone ,TransformerMixin
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from scipy.stats import chi2


class BaggingBootstrapper(BaseEstimator , ClassifierMixin) :
    def __init__(self , base_estimator , n_estimators , max_samples=0.80 , random_state=42) :
        self.base_estimator=base_estimator
        self.n_estimators=n_estimators
        self.max_samples=max_samples
        self.random_state=random_state
        self.estimators_=[]
        self.classes_=None

    def _bootstrap_indices(self , n_samples) :
        size=int(n_samples * self.max_samples)
        rng=np.random.default_rng(self.random_state)
        return [rng.choice(n_samples , size=size , replace=True) for _ in range(self.n_estimators)]

    def _volatility_indices(self , Vflag) :
        bin_codes , bin_edges=pd.qcut(Vflag , q=self.n_estimators , labels=False , retbins=True , duplicates='drop')
        self.bin_edges_=bin_edges
        return [np.flatnonzero(bin_codes.values == b) for b in range(len(bin_edges)-1)]

    def get_weights(self , x) :
        mask=(x > 0)
        w=pd.Series(0.0 , index=x.index)
        w[mask]=x[mask].rank(pct=True)
        w=w / w[mask].mean()
        return np.asarray(w , dtype=np.float64)

    def fit(self , X , y , sample_weight=None , Vflag=None , regime_labels=None) :
        X=np.asarray(X , dtype=np.float64)
        y=np.asarray(y , dtype=np.float64)
        self.classes_=np.unique(y)

        sample_weight=self.get_weights(sample_weight)

        if Vflag is None :
            estimator_indices=self._bootstrap_indices(len(X))
        else :
            estimator_indices=self._volatility_indices(Vflag)

        seeds=np.random.default_rng(self.random_state).integers(0 , 2 ** 31 , size=len(estimator_indices))

        for idx , rs in zip(estimator_indices , seeds) :
            model=clone(self.base_estimator)
            model.set_params(random_state=rs)

            X_boot=X[idx]
            y_boot=y[idx]
            sw_boot=sample_weight[idx]

            self.estimators_.append(
                model.fit(
                    X_boot ,
                    y_boot , sample_weight=sw_boot)
            )

        return self

    def predict_proba(self , X , Vflag=None) :
        X=np.asarray(X)
        n_samples=X.shape[0]
        n_classes=len(self.classes_)
        out=np.zeros((n_samples , n_classes) , dtype=np.float64)
        class_to_idx={c : i for i , c in enumerate(self.classes_)}

        for clf in self.estimators_ :
            proba=clf.predict_proba(X)
            if len(clf.classes_) == n_classes and np.array_equal(clf.classes_ , self.classes_) :
                out+=proba
            else :
                for j , cls in enumerate(clf.classes_) :
                    out[: , class_to_idx[cls]]+=proba[: , j]

        out/=len(self.estimators_)
        return out

    def predict(self , X , Vflag=None) :
        return self.classes_[np.argmax(self.predict_proba(X) , axis=1)]


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






