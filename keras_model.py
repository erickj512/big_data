#!/usr/bin/env python3

import numpy as np
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


EPOCHS = 10


def make_model(output_dim, dense_layer_sizes=[40], filters=20, kernel_size=(3, 3),
               pool_size=(2, 2), dropout=0.25):
    model = Sequential()
    # Add pair of convolutional-pooling hidden layers
    model.add(Conv2D(filters, kernel_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    # Add the flatten layer before the dense ones
    model.add(Flatten())
    # Add the dense hidden layers
    for layer_size in dense_layer_sizes:
        model.add(Dense(layer_size, activation='relu'))
    # Add a dropout layer to avoid overfitting
    model.add(Dropout(dropout))
    # Add the output layer
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model


def main():

    # Train
    data = np.loadtxt('data/train.csv', dtype='f8', delimiter=',', skiprows=1)
    X, y = data[:, 1:], data[:, 0]
    input_shape = (28, 28, 1)
    X = X.reshape(X.shape[0], *input_shape).astype('float32') / 255
    num_classes = len(np.unique(y))
    classifier = KerasClassifier(make_model, output_dim=num_classes,
                                 epochs=EPOCHS)
    search_space = {'dense_layer_sizes': [[32], [32, 32]], 'filters': [8, 16]}
    estimator = GridSearchCV(classifier, search_space, cv=3)
    estimator.fit(X, y)
    print('Best params: ' + str(estimator.best_params_))
    print('Best score: ' + str(estimator.best_score_))

    # Test
    X_test = np.loadtxt('data/test.csv', dtype='f8', delimiter=',', skiprows=1)
    X_test = X_test.reshape(X_test.shape[0], *input_shape).astype('float32') / 255
    ids = np.arange(1, len(X_test) + 1)
    predictions = estimator.predict(X_test)
    np.savetxt('keras_submission.csv', np.transpose([ids, predictions]),
               fmt=('%d', '%d'), delimiter=',', header='ImageId,Label',
               comments='')


if __name__ == '__main__':
    main()
