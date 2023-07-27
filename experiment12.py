#! /usr/bin/env python

"""
@author: David Diaz Vico
"""

from argparse import ArgumentParser
import joblib
import numpy as np
from sklearn.model_selection import cross_validate

from dataset import load_dataset
from estimator import instance_estimator


def run(dataset, transformer, predictor):
    """ Run an experiment. """
    X, y, X_test, y_test = load_dataset(dataset)
    estimator = instance_estimator(transformer, predictor)
    if (X_test is not None) and (y_test is not None):
        # Test score
        estimator.fit(X, y)
        scores = estimator.score(X_test, y_test)
    else:
        # CV score
        scores = cross_validate(estimator, X, y=y)
        estimator.fit(X, y)
    with open("scores.pkl", "wb") as handler:
        joblib.dump(scores, handler)
    with open("estimator.pkl", "wb") as handler:
        joblib.dump(estimator, handler)
    print(f"{scores}")
    cv_score = np.mean(scores["test_score"])
    print(f"{cv_score}")
    return scores


if __name__ == "__main__":
    parser = ArgumentParser(description="Run an experiment.")
    parser.add_argument("-d", "--dataset", type=str, help="dataset")
    parser.add_argument("-t", "--transformer", type=str, help="transformer")
    parser.add_argument("-p", "--predictor", type=str, help="predictor")
    args = parser.parse_args()
    run(args.dataset, args.transformer, args.predictor)
