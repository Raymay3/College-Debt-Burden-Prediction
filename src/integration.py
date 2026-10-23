"""Pipelines that combine the supervised model with unsupervised features.

The unsupervised block (filter -> impute -> scale -> PCA -> k-means) sits
inside the model pipeline, so it is refit on the training part of every
cross-validation fold and never sees the validation rows.
"""

from functools import partial

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.unsupervised import make_cluster_feature_pipeline, make_preprocessor

# Which unsupervised outputs are appended to the supervised features.
UNSUPERVISED_BLOCKS = ("membership", "distances", "membership+distances", "pca")


def _select_columns(X, start, stop):
    return np.asarray(X)[:, start:stop]


def make_unsupervised_block(block, n_clusters=4, pca_variance=0.85, random_state=42):
    """Unsupervised feature pipeline producing the requested block of features."""
    if block not in UNSUPERVISED_BLOCKS:
        raise ValueError(f"block must be one of {UNSUPERVISED_BLOCKS}")
    if block == "pca":
        return make_preprocessor(pca_variance, random_state)

    pipeline = make_cluster_feature_pipeline(n_clusters, pca_variance, random_state)
    # ClusterFeatures returns k one-hot columns followed by k centroid distances.
    columns = {
        "membership": (0, n_clusters),
        "distances": (n_clusters, 2 * n_clusters),
        "membership+distances": (0, 2 * n_clusters),
    }[block]
    pipeline.steps.append(
        ("select", FunctionTransformer(partial(_select_columns, start=columns[0], stop=columns[1])))
    )
    return pipeline


def make_lgbm(hyperparameters, random_state=42):
    """LightGBM regressor with the tuned hyperparameters from 02_supervised."""
    return LGBMRegressor(
        subsample_freq=1, random_state=random_state, n_jobs=1, verbose=-1, **hyperparameters
    )


def make_integrated_model(
    model, features, cluster_columns, block=None, n_clusters=4, pca_variance=0.85, random_state=42
):
    """Supervised features (passed through) plus an optional unsupervised block.

    With block=None the pipeline is the supervised model alone, so both arms
    of a comparison are built the same way.
    """
    transformers = [("base", "passthrough", list(features))]
    if block is not None:
        transformers.append((
            "unsupervised",
            make_unsupervised_block(block, n_clusters, pca_variance, random_state),
            list(cluster_columns),
        ))
    return Pipeline([("features", ColumnTransformer(transformers)), ("model", model)])


class ClusterMeanRegressor(BaseEstimator, RegressorMixin):
    """Predict the training mean of y within k-means clusters.

    The out-of-sample counterpart of the sector baseline: clusters are fit on
    the training rows only, and validation rows are assigned to the nearest
    training centroid.
    """

    def __init__(self, n_clusters=4, pca_variance=0.85, random_state=42):
        self.n_clusters = n_clusters
        self.pca_variance = pca_variance
        self.random_state = random_state

    def fit(self, X, y):
        self.clusters_ = make_cluster_feature_pipeline(
            self.n_clusters, self.pca_variance, self.random_state
        ).fit(X)
        labels = self._labels(X)
        self.group_means_ = pd.Series(np.asarray(y)).groupby(labels).mean()
        self.global_mean_ = float(np.mean(y))
        return self

    def _labels(self, X):
        return self.clusters_.transform(X)[:, : self.n_clusters].argmax(axis=1)

    def predict(self, X):
        return pd.Series(self._labels(X)).map(self.group_means_).fillna(self.global_mean_).to_numpy()
