#!/usr/bin/env python3

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MAX_ITER = 10


def main():

    # Train
    data = np.loadtxt('data/train.csv', dtype='f8', delimiter=',', skiprows=1)
    X, y = data[:, 1:], data[:, 0]
    classifier = Pipeline([('transformer', StandardScaler()),
                           ('predictor', MLPClassifier(max_iter=MAX_ITER))])
    search_space = {'predictor__hidden_layer_sizes': [[32], [32, 32]]}
    estimator = GridSearchCV(classifier, search_space, cv=3)
    estimator.fit(X, y)
    print('Best params: ' + str(estimator.best_params_))
    print('Best score: ' + str(estimator.best_score_))

    # Test
    X_test = np.loadtxt('data/test.csv', dtype='f8', delimiter=',', skiprows=1)
    ids = np.arange(1, len(X_test) + 1)
    predictions = estimator.predict(X_test)
    np.savetxt('sklearn_submission.csv', np.transpose([ids, predictions]),
               fmt=('%d', '%d'), delimiter=',', header='ImageId,Label',
               comments='')


if __name__ == '__main__':
    main()
