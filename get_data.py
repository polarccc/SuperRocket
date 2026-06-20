import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import re
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


def _load_timeseries_file(file_path):
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".ts":
        return load_from_tsfile_to_dataframe(str(file_path))

    if suffix == ".txt":
        rows = []
        labels = []

        with file_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#") or line.startswith("@"):
                    continue

                parts = re.split(r"[\s,]+", line)
                if len(parts) < 2:
                    continue

                labels.append(parts[0])
                values = [float(value) for value in parts[1:] if value != ""]
                rows.append(pd.Series(values))

        if not rows:
            raise ValueError(f"No valid samples found in {file_path}")

        return pd.DataFrame({"dim_0": rows}), pd.Series(labels, dtype="object")

    raise ValueError(f"Unsupported dataset file type: {file_path}")


def _encode_labels(label_train, label_test):
    combined_labels = np.asarray(list(label_train) + list(label_test), dtype="object")
    unique_labels = pd.Index(combined_labels.astype(str)).unique().tolist()
    label_to_index = {label: index for index, label in enumerate(unique_labels)}

    encoded_train = np.asarray([label_to_index[str(label)] for label in label_train], dtype="int32")
    encoded_test = np.asarray([label_to_index[str(label)] for label in label_test], dtype="int32")
    return encoded_train, encoded_test, label_to_index


def _load_dataset_split(base_path, split_name):
    base_path = Path(base_path)
    candidates = [
        base_path.parent / f"{base_path.name}_{split_name}.ts",
        base_path.parent / f"{base_path.name}_{split_name}.txt",
        base_path.parent / f"{base_path.name}_{split_name}.TS",
        base_path.parent / f"{base_path.name}_{split_name}.TXT",
    ]

    for candidate in candidates:
        if candidate.exists():
            return _load_timeseries_file(candidate)

    raise FileNotFoundError(
        f"Could not find {split_name} file for {base_path}. "
        f"Tried: {', '.join(str(candidate) for candidate in candidates)}"
    )


def get_4data(args):
    path = args["path"]
    data_name = args["data_name"]
    X_part1, y_part1 = _load_dataset_split(path, "TRAIN")
    X_part2, y_part2 = _load_dataset_split(path, "TEST")

    y_train_encoded, y_test_encoded, label_to_index = _encode_labels(y_part1, y_part2)
    args["c_num"] = len(label_to_index)

    X = X_part1.reset_index(drop=True)
    y = y_train_encoded

    # print("c_num")
    # print(args["c_num"])

    X_val = None
    X_train = None
    y_val = None
    y_train = None

    for class_index in range(args["c_num"]):
        X_s = X.iloc[y == class_index].reset_index(drop=True)
        if X_s.shape[0] < 2:
            continue

        y_s = np.full(X_s.shape[0], class_index, dtype="int32")
        X_s_val, X_s_train, y_s_val, y_s_train = train_test_split(
            X_s, y_s, test_size=0.5, random_state=args["random_state"]
        )

        X_s_train = X_s_train.reset_index(drop=True)
        X_s_val = X_s_val.reset_index(drop=True)

        if X_s_val.shape[0] > X_s_train.shape[0]:
            X_s_val = X_s_val.drop(index=X_s_val.index[-1])
            y_s_val = y_s_val[:-1]
        elif X_s_train.shape[0] > X_s_val.shape[0]:
            X_s_train = X_s_train.drop(index=X_s_train.index[-1])
            y_s_train = y_s_train[:-1]

        if X_val is None:
            X_val = X_s_val
            y_val = y_s_val
        else:
            X_val = X_val._append(X_s_val)
            y_val = np.append(y_val, y_s_val)

        if X_train is None:
            X_train = X_s_train
            y_train = y_s_train
        else:
            X_train = X_train._append(X_s_train)
            y_train = np.append(y_train, y_s_train)

    X_train = X_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)

    X_test = X_part2.reset_index(drop=True)
    y_test = y_test_encoded

    X_train_val = X_train._append(X_val)
    X_train_val = X_train_val.reset_index(drop=True)
    y_train_val = np.append(y_train, y_val)

    assert X_val.shape[0] == y_val.size
    assert X_train.shape[0] == y_train.size
    assert X_train.shape == X_val.shape
    assert y_train.size > 0
    data = [X, y, X_train, y_train, X_val, y_val, X_test, y_test, X_train_val, y_train_val]

    return data, args


def get_3data(args):
    path = args["path"]
    data_name = args["data_name"]
    X_train, y_train = _load_dataset_split(path, "TRAIN")
    X_test, y_test = _load_dataset_split(path, "TEST")

    y_train, y_test, _ = _encode_labels(y_train, y_test)

    data = [X_train, y_train, X_test, y_test]

    return data, args
