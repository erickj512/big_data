
from joblib import dump
from sklearn.datasets import load_boston
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.neighbors import KNeighborsRegressor

data = load_boston()
print(f"{data.DESCR}")
print(f"{data.data}")
print(f"{data.target}")

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target,
                                                    test_size=0.33)

predictor = KNeighborsRegressor(n_neighbors=3)
#predictor.fit(X_train, y_train)
#score = predictor.score(X_test, y_test)
#print(f"{score}")
scores = cross_validate(predictor, X_train, y_train, cv=3)
print(f"{scores}")

predictor.fit(X_train, y_train)
mae = mean_absolute_error(y_test, predictor.predict(X_test))
print(f"{mae}")

dump(predictor, 'predictor.joblib')
 