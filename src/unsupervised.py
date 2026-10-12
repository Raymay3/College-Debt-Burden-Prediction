"""Reusable preprocessing and clustering components for the unsupervised analysis.

Every step that learns from data (column filtering, imputation, scaling, PCA,
clustering) is a scikit-learn transformer, so the full pipeline can be fit
inside each training fold in the integration notebook without leakage.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Columns that never enter the clustering feature space.
# OPEID6 is an identifier, debt_burden is the outcome (examined only after
# clustering), CONTROL is the sector label the clusters are compared against,
# and REGION is a nominal geographic code with no meaningful numeric order.
EXCLUDED_FROM_CLUSTERING = ["OPEID6", "debt_burden", "CONTROL", "REGION"]

# Right-skewed dollar and count features that are log-transformed.
LOG_FEATURES = [
    "TUITIONFEE_IN",
    "TUITIONFEE_OUT",
    "NET_PRICE",
    "COST_OF_ATTENDANCE",
    "UGDS",
    "NUMBRANCH",
    "NUM_PROGRAMS",
]


def clustering_columns(df):
    """Return the candidate clustering columns in a modeling dataset."""
    return [c for c in df.columns if c not in EXCLUDED_FROM_CLUSTERING]


class InstitutionFeatureFilter(BaseEstimator, TransformerMixin):
    """Select, log-transform, and flag institution features for clustering.

    Fitting learns, from the training data only:
      1. which columns exceed the missing-value threshold,
      2. which columns have near-zero variance,
      3. which columns are redundant because of high pairwise correlation,
      4. which retained columns need a missing-value indicator.

    Missing values are left in place for the downstream imputer.
    """

    def __init__(
        self,
        max_missing=0.40,
        nzv_freq_ratio=19.0,
        nzv_unique_pct=0.10,
        max_abs_corr=0.90,
        indicator_min_missing=0.05,
        log_features=tuple(LOG_FEATURES),
    ):
        self.max_missing = max_missing
        self.nzv_freq_ratio = nzv_freq_ratio
        self.nzv_unique_pct = nzv_unique_pct
        self.max_abs_corr = max_abs_corr
        self.indicator_min_missing = indicator_min_missing
        self.log_features = log_features

    def _log(self, X):
        X = X.copy()
        for col in self.log_features:
            if col in X.columns:
                X[col] = np.log1p(X[col].clip(lower=0))
        return X

    def _is_near_zero_variance(self, s):
        counts = s.dropna().value_counts()
        if len(counts) <= 1:
            return True
        freq_ratio = counts.iloc[0] / counts.iloc[1]
        unique_pct = len(counts) / s.notna().sum()
        return freq_ratio > self.nzv_freq_ratio and unique_pct < self.nzv_unique_pct

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.feature_names_in_ = np.array(X.columns)
        self.dropped_ = {}

        # 1. Missing-value threshold
        missing = X.isna().mean()
        for col in missing[missing > self.max_missing].index:
            self.dropped_[col] = f"missing {missing[col]:.0%}"
        X = X.drop(columns=list(self.dropped_))

        X = self._log(X)

        # 2. Near-zero variance
        for col in X.columns:
            if self._is_near_zero_variance(X[col]):
                self.dropped_[col] = "near-zero variance"
        X = X.drop(columns=[c for c in X.columns if c in self.dropped_])

        # 3. Collinearity: for each pair above the threshold (strongest first),
        # drop the column that is more correlated with the remaining features.
        corr = X.corr().abs()
        mean_corr = (corr.sum() - 1) / (len(corr) - 1)
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
        for (a, b), r in upper[upper > self.max_abs_corr].sort_values(ascending=False).items():
            if a in self.dropped_ or b in self.dropped_:
                continue
            drop, keep = (a, b) if mean_corr[a] >= mean_corr[b] else (b, a)
            self.dropped_[drop] = f"|r| = {r:.3f} with {keep}"
        self.kept_columns_ = [c for c in X.columns if c not in self.dropped_]

        # 4. Missing-value indicators, one per distinct missingness pattern
        self.indicator_columns_ = []
        seen_patterns = []
        for col in self.kept_columns_:
            pattern = X[col].isna().to_numpy()
            if pattern.mean() < self.indicator_min_missing:
                continue
            if any(np.array_equal(pattern, p) for p in seen_patterns):
                continue
            seen_patterns.append(pattern)
            self.indicator_columns_.append(col)

        return self

    def transform(self, X):
        X = self._log(pd.DataFrame(X)[self.kept_columns_])
        for col in self.indicator_columns_:
            X[f"{col}_MISSING"] = X[col].isna().astype(float)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(
            [str(c) for c in self.kept_columns_]
            + [f"{c}_MISSING" for c in self.indicator_columns_],
            dtype=object,
        )


class ClusterFeatures(BaseEstimator, TransformerMixin):
    """Fit a clusterer and expose its output as supervised-model features.

    The transform returns one-hot cluster membership followed by the
    Euclidean distance from each row to every cluster centroid.
    """

    def __init__(self, clusterer=None):
        self.clusterer = clusterer

    def fit(self, X, y=None):
        self.clusterer_ = clone(self.clusterer if self.clusterer is not None else KMeans())
        self.clusterer_.fit(X)
        self.centroids_ = self.clusterer_.cluster_centers_
        self.n_clusters_ = len(self.centroids_)
        return self

    def predict(self, X):
        return self.clusterer_.predict(X)

    def transform(self, X):
        labels = self.predict(X)
        onehot = np.eye(self.n_clusters_)[labels]
        distances = np.linalg.norm(
            np.asarray(X)[:, None, :] - self.centroids_[None, :, :], axis=2
        )
        return np.hstack([onehot, distances])

    def get_feature_names_out(self, input_features=None):
        k = range(self.n_clusters_)
        return np.array(
            [f"cluster_{i}" for i in k] + [f"cluster_dist_{i}" for i in k],
            dtype=object,
        )


def make_preprocessor(pca_variance=0.85, random_state=42):
    """Filter -> impute -> scale -> PCA, all fit on the data passed to fit()."""
    return Pipeline(
        [
            ("filter", InstitutionFeatureFilter()),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=pca_variance, random_state=random_state)),
        ]
    )


def make_cluster_feature_pipeline(n_clusters, pca_variance=0.85, random_state=42):
    """Full unsupervised pipeline producing cluster labels and centroid distances.

    Fit it on a training fold's clustering columns, then transform both the
    training and validation rows of that fold.
    """
    steps = make_preprocessor(pca_variance, random_state).steps + [
        (
            "cluster",
            ClusterFeatures(
                KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
            ),
        )
    ]
    return Pipeline(steps)
