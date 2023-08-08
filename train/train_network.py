import random
from copy import deepcopy
import pprint
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
import tqdm
import pickle
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import Dataset, DataLoader
from data_preproc import X_train, y_train, X_test, y_test


class Model(nn.Module):
    def __init__(self, in_feature, out_features, hidden_num_list):
        super(Model, self).__init__()
        self.module_list = nn.ModuleList()
        for number in hidden_num_list:
            self.module_list.append(nn.Linear(in_feature, number))
            self.module_list.append(nn.BatchNorm1d(num_features=number))
            self.module_list.append(nn.ReLU())
            in_feature = number
        self.module_list.append(nn.Linear(number, out_features))
        self.seq_module = nn.Sequential(*self.module_list)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.seq_module(x))


class CustomSet(Dataset):
    def __init__(self, x, y):
        self.data = x
        self.t = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):

        return self.data[item], self.t[item]


class Dummy_With:
    def __init__(self):
        ...

    def __enter__(self):
        return

    def __exit__(self, type, value, traceback):
        return


def run_through(loader, opti, model, loss_function, eval=False):
    epoch_loss = []
    model.eval() if eval else model.train()
    epoch_correct, epoch_total = 0, 0
    dummy_with = Dummy_With()
    for x, y in (tbar := tqdm.tqdm(loader, leave=False)):
        opti.zero_grad()
        x = x.cuda()
        y = y.cuda()
        with torch.no_grad() if eval else dummy_with:
            y_pred = model(x)
        if not eval:
            loss = loss_function(y_pred, y)
            loss.backward()
            opti.step()
        else:
            loss = loss_function(y_pred, y)

        epoch_correct += torch.sum(torch.eq(y.cpu(), torch.round(y_pred.cpu().detach())))
        epoch_total += len(y)

        tbar.set_description(f"Accuracy: {epoch_correct / epoch_total}")
        epoch_loss.append(loss.detach().cpu().item())

    return np.mean(epoch_loss), epoch_correct / epoch_total


def run_training(params, loss_function, num_epochs,
                 train_dataloader, test_dataloader, verbose=False):
    pbar = tqdm.trange(num_epochs)
    torch.manual_seed(42)
    model = Model(params["in_features"], 1, params["hidden_layers"])
    model = model.cuda()
    opt = optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_model = None
    best_acc = 0
    tries = 0
    max_tries = 15
    for epoch in pbar:
        epoch_train_loss, epoch_train_acc = run_through(train_dataloader, opt, model,
                                                        loss_function, eval=False)

        epoch_val_loss, epoch_val_acc = run_through(test_dataloader, opt,
                                                    model, loss_function,
                                                    eval=True)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accs.append(epoch_train_acc)
        val_accs.append(epoch_val_acc)
        if epoch_val_acc > best_acc:
            best_model = deepcopy(model.state_dict())
            best_acc = epoch_val_acc
            tries=0
        else:
            tries += 1

        if max_tries < tries:
            return best_model, best_acc

        if verbose:
            pbar.write(f'Train loss: {epoch_train_loss:.2f}, val loss: {epoch_val_loss:.2f}, train acc: {epoch_train_acc:.3f}, val acc {epoch_val_acc:.3f}')
            plot("Accuracy", "Epoch", train_accs, val_accs, yscale='linear')

    return best_model, best_acc


def plot(title, label, train_results, val_results, yscale='linear', save_path=None,
         extra_pt=None, extra_pt_label=None):
    epoch_array = np.arange(len(train_results)) + 1
    train_label, val_label = "Training " + label.lower(), "Validation " + label.lower()

    sns.set(style='ticks')

    plt.plot(epoch_array, train_results, epoch_array, val_results, linestyle='dashed', marker='o')
    legend = ['Train results', 'Validation results']

    if extra_pt:
        plt.plot(extra_pt[0], extra_pt[1], linestyle='dashed', marker='o', color='black')
        legend.append(extra_pt_label)

    plt.legend(legend)
    plt.xlabel('Epoch')
    plt.ylabel(label)
    plt.yscale(yscale)
    plt.title(title)

    sns.despine(trim=True, offset=5)
    plt.title(title, fontsize=15)
    if save_path:
        plt.savefig(str(save_path), bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print(X_train.shape)
    train_set = CustomSet(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    test_set = CustomSet(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32).unsqueeze(1))

    n_epochs = 180
    loss = nn.BCELoss()
    best_acc = 0
    for j in range(100):
        num_hidden = random.randint(1, 5)
        hidden_list = [4**random.randint(2, 5) for i in range(num_hidden)]
        param = {"lr": 10 ** (random.randint(-50, -30)/10), "weight_decay": 10 ** (random.randint(-100, -30)/10),
                 "batch_size": 2 ** random.randint(4, 6), "hidden_layers": hidden_list,
                 "in_features": X_train.shape[1]}

        train_loader = DataLoader(dataset=train_set, batch_size=param["batch_size"], shuffle=True)
        test_loader = DataLoader(dataset=test_set, batch_size=param["batch_size"], shuffle=True)
        model_dict, new_acc = run_training(params=param,
                                           loss_function=loss,
                                           num_epochs=n_epochs,
                                           train_dataloader=train_loader,
                                           test_dataloader=test_loader,
                                           verbose=False)
        if new_acc > best_acc:
            print(f"new best acc: {new_acc}")
            print("with param")
            param["best_acc"] = new_acc
            pprint.pprint(param)
            best_acc = new_acc
            torch.save(model_dict, "train/best_network.pt")
            with open("train/best_network_param.pickle", "wb") as f:
                pickle.dump(param, f)
