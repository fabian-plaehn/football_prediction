import random
import pprint
import numpy as np
import torch
import xgboost as xgb
import pickle
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from data_preproc import X_train, y_train, X_test, y_test

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# hyperparam tuning: subsample (0, 1] feature sampling
best_settings = None
best_value = 0

for i in range(100):
    param = {
        'colsample_bynode': random.random(),  # sample columns
        'learning_rate': 1,  # fixed
        'max_depth': random.randint(1, 15),  # search
        'num_parallel_tree': random.randint(50, 500),  # search
        'objective': 'binary:logistic',
        "sampling_method": "gradient_based",
        'subsample': random.random(),  # search (0, 1]
        'tree_method': 'gpu_hist',
        'nthread': 4, 'eval_metric': ['error'],
    }

    evallist = [(dtrain, 'train'), (dtest, 'eval')]

    bst = xgb.train(param, dtrain, 50, evallist, num_boost_round=10, early_stopping_rounds=15)

    y_pred = np.round(bst.predict(dtest, iteration_range=(0, bst.best_iteration+1)))
    #print(bst.best_iteration)
    acc = accuracy_score(y_test, y_pred)
    #print('XGBoost model accuracy score: {0:0.4f}'.format(acc))

    if acc > best_value:
        print(f"new best value: {acc}")
        print(f"with parameter:")
        param["best_acc"] = acc
        pprint.pprint(param)
        best_settings = param
        best_value = acc
        xgb.plot_importance(bst)
        plt.show()

        xgb.plot_tree(bst)
        plt.show()

        with open("best_boosted.pickle", "wb") as f:
            pickle.dump(bst, f)

        with open("best_boosted_param.pickle", "wb") as f:
            pickle.dump(best_settings, f)
        # xgb.to_graphviz(bst, num_trees=2)
