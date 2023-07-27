"""
@author: David Diaz Vico
"""

from sklearn.datasets import (load_breast_cancer, load_digits, load_iris,
                              load_wine)

def load_dataset(name):
    """ Load a dataset. """
    if name in ("breast_cancer", "iris", "digits", "wine"):
        loader = {"breast_cancer": load_breast_cancer, "iris": load_iris,
                  "digits": load_digits, "wine": load_wine}
        X, y = loader[name](return_X_y=True)
        X_test = y_test = None
    else:
        raise Exception("Dataset unavailable")
    return X, y, X_test, y_test
