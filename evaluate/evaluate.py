import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
import xgboost as xgb
from ..train.train_network import run_training, CustomSet, Model
import torch
import torch.nn as nn
from ..train.data_preproc import X_valid, y_valid, X_train, y_train, X_test, y_test
import pickle
import pprint


def eval_network():
    train_set = CustomSet(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    test_set = CustomSet(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32).unsqueeze(1))

    n_epochs = 180
    loss = nn.BCELoss()

    with open("../train/best_network_param.pickle", "rb") as f:
        network_param = pickle.load(f)

    pprint.pprint(network_param)

    train_loader = DataLoader(dataset=train_set, batch_size=network_param["batch_size"], shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=network_param["batch_size"], shuffle=True)
    model_dict, new_acc = run_training(params=network_param,
                                       loss_function=loss,
                                       num_epochs=n_epochs,
                                       train_dataloader=train_loader,
                                       val_dataloader=test_loader,
                                       verbose=True)

def eval_tree():
    with open("best_boosted_param.pickle", "rb") as f:
        network_param = pickle.load(f)

    print(network_param)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    print(network_param)
    evallist = [(dtrain, 'train'), (dtest, 'eval')]
    bst = xgb.train(network_param, dtrain, 50, evallist, num_boost_round=10, early_stopping_rounds=15)
    y_pred = np.round(bst.predict(dtest, iteration_range=(0, bst.best_iteration+1)))
    #print(bst.best_iteration)
    acc = accuracy_score(y_test, y_pred)

    print(acc)

def eval_together():
    with open("best_network_param.pickle", "rb") as f:
        network_param = pickle.load(f)
    pprint.pprint(network_param)
    X_valid_ = torch.tensor(X_valid, dtype=torch.float32)
    model = Model(58, 1, network_param["hidden_layers"])
    model.load_state_dict(torch.load("best_network.pt"))
    with torch.no_grad():
        y_pred_n = model(X_valid_).squeeze(1).numpy()

    acc = accuracy_score(y_valid, np.round(y_pred_n))
    print(f"acc network: {acc}")

    dtest = xgb.DMatrix(X_valid, label=y_valid)
    with open("best_boosted.pickle", "rb") as f:
        bst = pickle.load(f)

    with open("best_boosted_param.pickle", "rb") as f:
        boosted_param = pickle.load(f)

    print(boosted_param)

    y_pred_t = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))
    acc = accuracy_score(y_valid, np.round(y_pred_t))
    print(f"acc trees: {acc}")

    #mean
    acc = accuracy_score(y_valid, np.round((y_pred_t + y_pred_n)/2))
    print(f"acc mean: {acc}")

    right_pos = np.sum((y_pred_t > 0.5) & (y_pred_n > 0.5) & (y_valid == 1))
    right_neg = np.sum((y_pred_t < 0.5) & (y_pred_n < 0.5) & (y_valid == 0))
    all_ = np.sum((y_pred_t < 0.5) & (y_pred_n < 0.5)) + np.sum((y_pred_t > 0.5) & (y_pred_n > 0.5))

    print(f"max_vote: {(right_pos + right_neg)/all_}")


if __name__ == "__main__":
    #eval_network()
    #eval_tree()
    eval_together()