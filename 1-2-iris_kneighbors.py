from joblib import load
from sklearn.datasets import load_boston

predictor = load('predictor.joblib')

X, _ = load_boston(return_X_y=True)
print(f"{predictor.predict(X)}")
