import time
import torch
import pandas as pd
import numpy as np
from sktime.datasets import load_from_tsfile_to_dataframe
import torch.nn as nn, torch.optim as optim
from AnyRocket import MiniRocket
import torch.nn.functional as F
from torch import autograd
from torchviz import make_dot
import copy
from torch.utils.data import random_split
from numba.typed import List
import numba
from numba import get_num_threads, njit, prange, set_num_threads, vectorize
from sklearn.model_selection import train_test_split


def predict(best_model):
    args = \
        {
            "score": True,
            "test_size": 30
        }
    predictions = []
    correct = 0
    total = 0

    X_test_transform, feature_para_dict = rocket.transform(X=X_test)
    X_testing = torch.tensor(np.array(X_test_transform), requires_grad=True)

    if (use_gpu):
        X_testing = X_testing.cuda()

    _predictions = best_model(X_testing).argmax(1)

    _predictions = _predictions.cpu().numpy()
    predictions.append(_predictions)
    print(predictions)
    total += len(X_test_transform)
    correct += (_predictions == y_test).sum()
    #return np.concatenate(predictions), correct / total

    return correct / total


from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline


def predict_plus(args, best_model, feature_para_dict, data):
    test_assembly_duration = 0
    test_duration = 0

    X, y, X_train, y_train, X_val, y_val, X_test, y_test, X_train_val, y_train_val = data

    predictions = []
    correct = 0
    total = 0

    _start_time = time.perf_counter()
    rocket = MiniRocket(num_kernels=args["num_kernels"])  # by default, ROCKET uses 10,000 kernels
    rocket.fit(X)
    dilations = rocket.named_parameters["dilations"]
    para_1 = list(best_model[0].named_parameters())
    alpha = para_1[1][1].data
    indexs = alpha.sort()[1].tolist()[int(-args["k"]):]
    feature_k_para = list([])

    for item in indexs:
        feature_k_para.append(feature_para_dict[item])
    para = convert_to_paras(num_dilation=int(dilations.size), feature_para_dict=feature_k_para)

    rocket.named_parameters["NN"] = para["NN"]
    rocket.named_parameters["biases"] = para["bias"]
    print("===============P1   NN.sum==============")
    print(para["NN"].sum())

    test_assembly_duration += time.perf_counter() - _start_time

    _start_time = time.perf_counter()
    X_test_transform, feature_para_dict = rocket.transform(X=X_test)
    X_train_transform, feature_para_dict = rocket.transform(X=X_train)
    X_train_val_transform, feature_para_dict = rocket.transform(X=X_train_val)
    X_testing = torch.tensor(np.array(X_test_transform), requires_grad=True)

    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    classifier.fit(X_train_val_transform, y_train_val)
    a = classifier.score(X_test_transform, y_test)
    test_duration += time.perf_counter() - _start_time

    result = {"predict_accuracy": a,
              "test_assembly_duration": test_assembly_duration,
              "test_duration": test_duration,
              }

    return result


def predict_2plus(args, best_model, feature_para_dict, data):
    X, y, X_train, y_train, X_val, y_val, X_test, y_test, X_train_val, y_train_val = data
    predictions = []
    correct = 0
    total = 0

    para_1 = list(best_model[0].named_parameters())
    pre_alpha = para_1[1][1].data
    indexs = pre_alpha.sort()[1].tolist()[-int(args["k"]):]

    now_alpha = torch.zeros_like(pre_alpha)
    now_alpha[indexs] = 1

    rocket = MiniRocket(num_kernels=args["num_kernels"])  # by default, ROCKET uses 10,000 kernels

    rocket.fit(X_train, y_train)
    X_test_transform, feature_para_dict = rocket.transform(X=X_test)

    X_testing = torch.tensor(np.array(X_test_transform), requires_grad=True)

    para_1[1][1].data = now_alpha
    print("=============new alpha============")
    print(now_alpha)

    _predictions = best_model(X_testing).argmax(1).cpu().numpy()
    para_1[1][1].data = pre_alpha

    predictions.append(_predictions)
    print(predictions)
    total += len(X_test_transform)
    correct += (_predictions == y_test).sum()

    return correct / total


def convert_to_paras(num_dilation, feature_para_dict):
    feature_para_dict = pd.DataFrame(feature_para_dict)
    feature_para_dict
    print("num_dilation")
    print(num_dilation)
    kernel_ = feature_para_dict.iloc[:, :-1].copy()
    kernel = kernel_.drop_duplicates(subset=[0, 1, 2], keep='first')

    bias = kernel.iloc[:, -1].copy()
    bias = np.array(bias, dtype=np.float32)
    print("bias")
    print(bias)

    NN_ = kernel.copy()
    NN_.iloc[:, 0] = NN_.iloc[:, 0] * 84 + NN_.iloc[:, 1].copy()
    NN__ = NN_.iloc[:, 0].copy()
    ans = NN__.value_counts()
    print("ans")
    print(ans)
    NN = np.zeros(84 * num_dilation, dtype=np.int32)
    #print(ans.size)
    print("NN.size")
    print(NN.size)

    for item in ans.index:
        NN[int(item)] = int(ans[item])

    named_para = {
        "bias": bias,
        "NN": NN,
    }
    return named_para
