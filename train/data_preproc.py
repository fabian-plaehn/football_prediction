import pickle
import numpy as np
import pandas as pd

with open("../get_data/team_data.pickle", "rb") as f:
    team_data = pickle.load(f)

with open("../get_data/stock_data_wo_if.pickle", "rb") as f:
    stock_data = pickle.load(f)

stock_data[0] = stock_data[0].apply(lambda x: 1 if x > 0 else 0)  # binary target
data = pd.concat([stock_data, team_data], axis=1)
data.dropna(subset=[0], inplace=True)
data.drop(data.loc[data.index < '2021-01-01 01:00:00'].index, inplace=True)
data.fillna(0, inplace=True)

data = data.to_numpy()

y = data[:, 0]
X = data[:, 1:]

y_train, y_test, y_valid,  = y[:int(y.shape[0] * 0.6)], y[int(y.shape[0] * 0.6):int(y.shape[0] * 0.8)], y[int(y.shape[0] * 0.8):]
X_train, X_test, X_valid,  = X[:int(X.shape[0] * 0.6)], X[int(X.shape[0] * 0.6):int(X.shape[0] * 0.8)], X[int(X.shape[0] * 0.8):]

print(y_test.shape, y_valid.shape, y_train.shape,
      X_test.shape, X_valid.shape, X_train.shape)
print(np.sum(y_test), np.sum(y_train), np.sum(y_valid))