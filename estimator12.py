"""
@author: David Diaz Vico
"""

from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def instance_estimator(transformer, predictor):
    """ Instance an estimator. """
    if transformer in ("variance", "pca"):
        trans = {"pca": PCA(n_components=32), "variance": VarianceThreshold()}[transformer]
    else:
        trans = None
    if predictor in ("logistic", "kneighbors", "sgd"):
        pred = {"logistic": LogisticRegression(),
                "kneighbors": KNeighborsClassifier(),
                "sgd": Pipeline([("std", StandardScaler()), ("sgd", SGDClassifier(penalty="elasticnet", alpha=0.001))])
               }[predictor]
    else:
        pred = LogisticRegression()
    if trans is not None:
        return Pipeline([("transformer", trans), ("scaler", StandardScaler()), ("classifier", pred)])
    else:
        return Pipeline([("scaler", StandardScaler()), ("classifier", pred)])
