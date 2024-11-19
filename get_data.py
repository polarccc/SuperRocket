import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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


def get_4data(args):
    path = args["path"]
    data_name = args["data_name"]
    X_part1, y_part1 = load_from_tsfile_to_dataframe(path + "_TRAIN.ts")
    X_part2, y_part2 = load_from_tsfile_to_dataframe(path + "_TEST.ts")

    X = X_part1
    y = y_part1

    X = X.reset_index(drop=True)

    y_part2int = y_part2.astype("int32")
    c_num = np.max(y_part2int)

    print("y")
    print(y)

    if (c_num == 1):
        c_num = 2
        y[y == "1"] = "2"
        y[y == "-1"] = "1"
        y[y == "0"] = "1"
        y_part2[y_part2 == "1"] = "2"
        y_part2[y_part2 == "-1"] = "1"
        y_part2[y_part2 == "0"] = "1"

    args["c_num"] = c_num
    print("c_num")
    print(c_num)
    X_val = 1
    X_train = 1
    y_val = 1
    y_train = 1
    # for [1,5]
    for item in range(1, args["c_num"] + 1):
        # X_samelabel X_s
        X_s = X.iloc[y == str(item)].reset_index(drop=True)
        if (X_s.size == 1 or X_s.size == 0):
            continue

        X_s_train_val = X_s
        y_s_train_val = np.full(X_s.size, str(item))

        # val train 0.5 0.5

        X_s_val, X_s_train, y_s_val, y_s_train = train_test_split(
            X_s_train_val, y_s_train_val, test_size=0.5, random_state=args["random_state"])
        # 0.5
        X_s_train = X_s_train.reset_index(drop=True)
        X_s_val = X_s_val.reset_index(drop=True)

        if (X_s_val.shape[0] > X_s_train.shape[0]):
            X_s_val = X_s_val.drop(index=X_s_val.index[-1])
            y_s_val = y_s_val[0:-1]
        elif (X_s_train.shape[0] > X_s_val.shape[0]):
            X_s_train = X_s_train.drop(index=X_s_train.index[-1])
            y_s_train = y_s_train[0:-1]

        # assert X_s_val.shape[0]==y_s_val.size
        # assert X_s_train.shape[0]==y_s_train.size

        if (type(X_val) == type(1)):
            X_val = X_s_val
            y_val = y_s_val
        else:
            X_val = X_val._append(X_s_val)
            y_val = np.append(y_val, y_s_val)

        if (type(X_train) == type(1)):
            X_train = X_s_train
            y_train = y_s_train
        else:
            X_train = X_train._append(X_s_train)
            y_train = np.append(y_train, y_s_train)
        print("y_train")
        print(y_train)

    X_train = X_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)

    X_test = X_part2
    y_test = y_part2

    X_train_val = X_train._append(X_val)
    X_train_val = X_train_val.reset_index(drop=True)
    y_train_val = np.append(y_train, y_val)

    # label [1,5]->[0,4] str->int
    y_train = y_train.astype("int32") - 1
    y_val = y_val.astype("int32") - 1
    y_test = y_test.astype("int32") - 1
    y_train_val = y_train_val.astype("int32") - 1
    y = y.astype("int32") - 1

    assert X_val.shape[0] == y_val.size
    print(X_train.shape[0])
    print(y_train.size)
    assert X_train.shape[0] == y_train.size
    print(X_train.shape)
    print(X_val.shape)
    assert X_train.shape == X_val.shape
    assert sum(y_train) != 0
    data = [X, y, X_train, y_train, X_val, y_val, X_test, y_test, X_train_val, y_train_val]

    return data, args


def get_3data(args):
    path = args["path"]
    data_name = args["data_name"]
    X_train, y_train = load_from_tsfile_to_dataframe(path + "_TRAIN.ts")
    X_test, y_test = load_from_tsfile_to_dataframe(path + "_TEST.ts")

    data = [X_train, y_train, X_test, y_test]

    return data, args
