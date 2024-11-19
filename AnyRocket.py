__author__ = "Angus Dempster"
__all__ = ["MiniRocket"]

import multiprocessing
from nntplib import NNTP

import numpy as np
import pandas as pd
from numba import get_num_threads, njit, prange, set_num_threads, vectorize
import numba
from sktime.transformations.base import _PanelToTabularTransformer
from sktime.utils.validation.panel import check_X
from numba.typed import List

from get_NN import get_NN, get_assign_array, get_feature_combination
import random


class MiniRocket(_PanelToTabularTransformer):
    _tags = {"univariate-only": True}

    def __init__(
            self,
            num_kernels=10_000,
            max_dilations_per_kernel=32,
            n_jobs=1,
            random_state=None,
            verbose=2,
            num_feature_combination=400,
    ):
        self.num_kernels = num_kernels
        super(MiniRocket, self).__init__()
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.n_jobs = n_jobs
        self.random_state = (
            np.int32(random_state) if isinstance(random_state, int) else None
        )
        self.verbose = verbose
        self.num_feature_combination = num_feature_combination

    def fit(self, X, y=None, max_dilations_per_kernel=32, seed=None,
            selected_kernel_index_int=17576888821917879161389045, selected_dilation_index_int=0, specificed_NN=0,
            NN=[0]):
        print("X =============")
        print(X)
        X = check_X(X, enforce_univariate=True, coerce_to_numpy=True).astype(np.float32)
        *_, n_timepoints = X.shape
        if n_timepoints < 9:
            raise ValueError(
                (
                    f"n_timepoints must be >= 9, but found {n_timepoints};"
                    " zero pad shorter series so that n_timepoints == 9"
                )
            )

        self.named_parameters = _fit_multi(
            X, self.num_kernels, self.max_dilations_per_kernel, self.random_state, selected_kernel_index_int,
            selected_dilation_index_int, specificed_NN, NN
        )
        self._is_fitted = True
        return self

    def transform(self, X, y=None):

        self.check_is_fitted()
        X = check_X(X, coerce_to_numpy=True).astype(np.float32)
        print("X_shape=", X.shape)

        prev_threads = get_num_threads()
        if self.n_jobs < 1 or self.n_jobs > multiprocessing.cpu_count():
            n_jobs = multiprocessing.cpu_count()
        else:
            n_jobs = self.n_jobs
        set_num_threads(n_jobs)
        set_num_threads(prev_threads)

        X_ = _transform_multi(X, tuple(self.named_parameters.values()))
        return pd.DataFrame(X_[0]), X_[1]


@njit(
    "float32[:](float32[:,:,:],int32[:],int32[:],int32[:],int32[:],float32[:],optional(int32))",  # noqa
    fastmath=True,
    parallel=False,
    cache=True,
)
def _fit_biases_multi(
        X,
        num_channels_per_combination,
        channel_indices,
        dilations,
        NN,
        quantiles,
        seed,
):
    if seed is not None:
        np.random.seed(seed)

    n_instances, n_columns, n_timepoints = X.shape

    indices = np.array(
        (
            0,
            1,
            2,
            0,
            1,
            3,
            0,
            1,
            4,
            0,
            1,
            5,
            0,
            1,
            6,
            0,
            1,
            7,
            0,
            1,
            8,
            0,
            2,
            3,
            0,
            2,
            4,
            0,
            2,
            5,
            0,
            2,
            6,
            0,
            2,
            7,
            0,
            2,
            8,
            0,
            3,
            4,
            0,
            3,
            5,
            0,
            3,
            6,
            0,
            3,
            7,
            0,
            3,
            8,
            0,
            4,
            5,
            0,
            4,
            6,
            0,
            4,
            7,
            0,
            4,
            8,
            0,
            5,
            6,
            0,
            5,
            7,
            0,
            5,
            8,
            0,
            6,
            7,
            0,
            6,
            8,
            0,
            7,
            8,
            1,
            2,
            3,
            1,
            2,
            4,
            1,
            2,
            5,
            1,
            2,
            6,
            1,
            2,
            7,
            1,
            2,
            8,
            1,
            3,
            4,
            1,
            3,
            5,
            1,
            3,
            6,
            1,
            3,
            7,
            1,
            3,
            8,
            1,
            4,
            5,
            1,
            4,
            6,
            1,
            4,
            7,
            1,
            4,
            8,
            1,
            5,
            6,
            1,
            5,
            7,
            1,
            5,
            8,
            1,
            6,
            7,
            1,
            6,
            8,
            1,
            7,
            8,
            2,
            3,
            4,
            2,
            3,
            5,
            2,
            3,
            6,
            2,
            3,
            7,
            2,
            3,
            8,
            2,
            4,
            5,
            2,
            4,
            6,
            2,
            4,
            7,
            2,
            4,
            8,
            2,
            5,
            6,
            2,
            5,
            7,
            2,
            5,
            8,
            2,
            6,
            7,
            2,
            6,
            8,
            2,
            7,
            8,
            3,
            4,
            5,
            3,
            4,
            6,
            3,
            4,
            7,
            3,
            4,
            8,
            3,
            5,
            6,
            3,
            5,
            7,
            3,
            5,
            8,
            3,
            6,
            7,
            3,
            6,
            8,
            3,
            7,
            8,
            4,
            5,
            6,
            4,
            5,
            7,
            4,
            5,
            8,
            4,
            6,
            7,
            4,
            6,
            8,
            4,
            7,
            8,
            5,
            6,
            7,
            5,
            6,
            8,
            5,
            7,
            8,
            6,
            7,
            8,
        ),
        dtype=np.int32,
    ).reshape(84, 3)

    num_kernels = len(indices)
    num_dilations = len(dilations)

    num_features = np.sum(NN)

    biases = np.zeros(num_features, dtype=np.float32)

    feature_index_start = 1
    combination_index = 0
    num_channels_start = 0

    for dilation_index in range(num_dilations):

        dilation = dilations[dilation_index]
        padding = ((9 - 1) * dilation) // 2

        for kernel_index in range(num_kernels):

            num_features_this_dilation = NN[dilation_index * 84 + kernel_index]
            feature_index_end = feature_index_start + num_features_this_dilation

            num_channels_this_combination = num_channels_per_combination[
                combination_index
            ]

            num_channels_end = num_channels_start + num_channels_this_combination

            channels_this_combination = channel_indices[
                                        num_channels_start:num_channels_end
                                        ]

            _X = X[np.random.randint(n_instances)][channels_this_combination]

            A = -_X  # A = alpha * X = -X
            G = _X + _X + _X  # G = gamma * X = 3X

            C_alpha = np.zeros(
                (num_channels_this_combination, n_timepoints), dtype=np.float32
            )
            C_alpha[:] = A

            C_gamma = np.zeros(
                (9, num_channels_this_combination, n_timepoints), dtype=np.float32
            )
            C_gamma[9 // 2] = G

            start = dilation
            end = n_timepoints - padding

            for gamma_index in range(9 // 2):
                C_alpha[:, -end:] = C_alpha[:, -end:] + A[:, :end]
                C_gamma[gamma_index, :, -end:] = G[:, :end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):
                C_alpha[:, :-start] = C_alpha[:, :-start] + A[:, start:]
                C_gamma[gamma_index, :, :-start] = G[:, start:]

                start += dilation

            index_0, index_1, index_2 = indices[kernel_index]

            C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]
            C = np.sum(C, axis=0)

            biases[feature_index_start:feature_index_end] = np.quantile(
                C, quantiles[feature_index_start:feature_index_end]
            )

            feature_index_start = feature_index_end

            combination_index += 1
            num_channels_start = num_channels_end
    return biases


def _fit_dilations(n_timepoints, num_features, max_dilations_per_kernel):
    num_kernels = 84
    print("num_features")
    print(num_features)
    num_features_per_kernel = num_features // num_kernels
    true_max_dilations_per_kernel = min(
        num_features_per_kernel, max_dilations_per_kernel
    )
    multiplier = num_features_per_kernel / true_max_dilations_per_kernel

    max_exponent = np.log2((n_timepoints - 1) / (9 - 1))
    dilations, num_features_per_dilation = np.unique(
        np.logspace(0, max_exponent, true_max_dilations_per_kernel, base=2).astype(
            np.int32
        ),
        return_counts=True,
    )
    num_features_per_dilation = (num_features_per_dilation * multiplier).astype(
        np.int32
    )  # this is a vector

    remainder = num_features_per_kernel - np.sum(num_features_per_dilation)
    i = 0
    while remainder > 0:
        num_features_per_dilation[i] += 1
        remainder -= 1
        i = (i + 1) % len(num_features_per_dilation)

    # added section no.1
    num_dilations = len(dilations)
    import random

    NN = np.random.randint(low=0, high=1000, size=num_dilations * 84)

    NN = NN * num_features / NN.sum()
    NN = np.float32(NN)
    NN = np.int32(NN)
    Rest = num_features - NN.sum()
    for i in range(84):
        for j in range(num_dilations):
            if (Rest == 0):
                break
            else:
                NN[j * 84 + i] = NN[j * 84 + i] + 1
                Rest = Rest - 1
    print(NN)
    print(sum(NN))

    return dilations, NN


def _quantiles(n):
    return np.array(
        [(_ * ((np.sqrt(5) + 1) / 2)) % 1 for _ in range(1, n + 1)], dtype=np.float32
    )


def _fit_multi(X, num_features=10_000, max_dilations_per_kernel=32, seed=None,
               selected_kernel_index_int=17576888821917879161389045, selected_dilation_index_int=0, specificed_NN=0,
               NN=[0]):
    _, n_columns, n_timepoints = X.shape

    num_kernels = 84

    dilations, NN = _fit_dilations(
        n_timepoints, num_features, max_dilations_per_kernel
    )
    num_dilations = len(dilations)
    if (specificed_NN == 1):
        NN = get_NN(num_features=num_features, num_dilations=num_dilations, selected_kernel_index_int=1000000000,
                    selected_dilation_index_int=1000)
    if (specificed_NN == 2):
        NN = get_assign_array(feature_num=num_features, dilation_num=num_dilations, order=selected_kernel_index_int)
    #print(NN)
    #print(sum(NN))
    quantiles = _quantiles(np.sum(NN))

    num_combinations = num_kernels * num_dilations

    max_num_channels = min(n_columns, 9)
    max_exponent = np.log2(max_num_channels + 1)

    num_channels_per_combination = (
            2 ** np.random.uniform(0, max_exponent, num_combinations)
    ).astype(np.int32)

    channel_indices = np.zeros(num_channels_per_combination.sum(), dtype=np.int32)

    num_channels_start = 0
    for combination_index in range(num_combinations):
        num_channels_this_combination = num_channels_per_combination[combination_index]
        num_channels_end = num_channels_start + num_channels_this_combination
        channel_indices[num_channels_start:num_channels_end] = np.random.choice(
            n_columns, num_channels_this_combination, replace=False
        )

        num_channels_start = num_channels_end

    biases = _fit_biases_multi(
        X,
        num_channels_per_combination,
        channel_indices,
        dilations,
        NN,
        quantiles,
        seed
    )

    named_para = {
        "num_channels_per_combination": num_channels_per_combination,
        "channel_indices": channel_indices,
        "dilations": dilations,
        "NN": NN,
        "biases": biases,
    }

    return named_para


@vectorize("float32(float32,float32)", nopython=True, cache=True)
def _PPV(a, b):
    if a > b:
        return 1
    else:
        return 0


@njit(
    "Tuple((float32[:,:],List(List(float64))))(float32[:,:,:],Tuple((int32[:],int32[:],int32[:],int32[:],float32[:])))",
    #noqa
    fastmath=True,
    parallel=False,
    cache=True,
)
def _transform_multi(X, parameter):
    n_instances, n_columns, n_timepoints = X.shape
    (
        num_channels_per_combination,
        channel_indices,
        dilations,
        NN,
        biases
    ) = parameter

    # equivalent to:
    # >>> from itertools import combinations
    # >>> indices = np.array([_ for _ in combinations(np.arange(9), 3)])
    indices = np.array(
        (
            0,
            1,
            2,
            0,
            1,
            3,
            0,
            1,
            4,
            0,
            1,
            5,
            0,
            1,
            6,
            0,
            1,
            7,
            0,
            1,
            8,
            0,
            2,
            3,
            0,
            2,
            4,
            0,
            2,
            5,
            0,
            2,
            6,
            0,
            2,
            7,
            0,
            2,
            8,
            0,
            3,
            4,
            0,
            3,
            5,
            0,
            3,
            6,
            0,
            3,
            7,
            0,
            3,
            8,
            0,
            4,
            5,
            0,
            4,
            6,
            0,
            4,
            7,
            0,
            4,
            8,
            0,
            5,
            6,
            0,
            5,
            7,
            0,
            5,
            8,
            0,
            6,
            7,
            0,
            6,
            8,
            0,
            7,
            8,
            1,
            2,
            3,
            1,
            2,
            4,
            1,
            2,
            5,
            1,
            2,
            6,
            1,
            2,
            7,
            1,
            2,
            8,
            1,
            3,
            4,
            1,
            3,
            5,
            1,
            3,
            6,
            1,
            3,
            7,
            1,
            3,
            8,
            1,
            4,
            5,
            1,
            4,
            6,
            1,
            4,
            7,
            1,
            4,
            8,
            1,
            5,
            6,
            1,
            5,
            7,
            1,
            5,
            8,
            1,
            6,
            7,
            1,
            6,
            8,
            1,
            7,
            8,
            2,
            3,
            4,
            2,
            3,
            5,
            2,
            3,
            6,
            2,
            3,
            7,
            2,
            3,
            8,
            2,
            4,
            5,
            2,
            4,
            6,
            2,
            4,
            7,
            2,
            4,
            8,
            2,
            5,
            6,
            2,
            5,
            7,
            2,
            5,
            8,
            2,
            6,
            7,
            2,
            6,
            8,
            2,
            7,
            8,
            3,
            4,
            5,
            3,
            4,
            6,
            3,
            4,
            7,
            3,
            4,
            8,
            3,
            5,
            6,
            3,
            5,
            7,
            3,
            5,
            8,
            3,
            6,
            7,
            3,
            6,
            8,
            3,
            7,
            8,
            4,
            5,
            6,
            4,
            5,
            7,
            4,
            5,
            8,
            4,
            6,
            7,
            4,
            6,
            8,
            4,
            7,
            8,
            5,
            6,
            7,
            5,
            6,
            8,
            5,
            7,
            8,
            6,
            7,
            8,
        ),
        dtype=np.int32,
    ).reshape(84, 3)

    num_kernels = len(indices)
    num_dilations = len(dilations)

    num_features = 4 * np.sum(NN)

    n_instances = X.shape[0]
    features = np.zeros((n_instances, num_features), dtype=np.float32)
    feature_para_dict = []

    for example_index in prange(n_instances):
        _X = X[example_index]

        A = -_X  # A = alpha * X = -X
        G = _X + _X + _X  # G = gamma * X = 3X

        feature_index_start = int(0)
        real_feature_index_start = 0
        #feature_index_start=feature_index_start.astype(np.int32)
        combination_index = 0
        num_channels_start = 0

        for dilation_index in range(num_dilations):

            _padding0 = dilation_index % 2

            dilation = dilations[dilation_index]
            padding = ((9 - 1) * dilation) // 2

            C_alpha = np.zeros((n_columns, n_timepoints), dtype=np.float32)
            C_alpha[:] = A

            C_gamma = np.zeros((9, n_columns, n_timepoints), dtype=np.float32)
            C_gamma[9 // 2] = G

            start = dilation
            end = n_timepoints - padding

            for gamma_index in range(9 // 2):
                C_alpha[:, -end:] = C_alpha[:, -end:] + A[:, :end]
                C_gamma[gamma_index, :, -end:] = G[:, :end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):
                C_alpha[:, :-start] = C_alpha[:, :-start] + A[:, start:]
                C_gamma[gamma_index, :, :-start] = G[:, start:]

                start += dilation

            for kernel_index in range(num_kernels):

                num_features_this_dilation = NN[dilation_index * 84 + kernel_index]

                feature_index_end = feature_index_start + num_features_this_dilation
                num_channels_this_combination = num_channels_per_combination[
                    combination_index
                ]

                num_channels_end = num_channels_start + num_channels_this_combination

                channels_this_combination = channel_indices[
                                            num_channels_start:num_channels_end
                                            ]

                _padding1 = (_padding0 + kernel_index) % 2

                index_0, index_1, index_2 = indices[kernel_index]

                C = (
                        C_alpha[channels_this_combination]
                        + C_gamma[index_0][channels_this_combination]
                        + C_gamma[index_1][channels_this_combination]
                        + C_gamma[index_2][channels_this_combination]
                )
                C = np.sum(C, axis=0)

                if _padding1 == 0:
                    # 这里的feature_count实际上是 dialited_kernel数 是哪个10000
                    for feature_count in np.arange(num_features_this_dilation, dtype=np.int32):
                        feature_index = feature_index_start + feature_count
                        _bias = biases[int(feature_index)]
                        ppv = 0
                        last_val = 0
                        max_stretch = 0.0
                        mean_index = 0
                        mean = 0

                        for j in range(C.shape[0]):
                            if C[j] > _bias:
                                ppv += 1
                                mean_index += j
                                mean += C[j] + _bias
                            elif C[j] < _bias:
                                stretch = j - last_val
                                if stretch > max_stretch:
                                    max_stretch = stretch
                                last_val = j
                        stretch = C.shape[0] - 1 - last_val
                        if stretch > max_stretch:
                            max_stretch = stretch

                        C = C - _bias
                        real_feature_index_end = real_feature_index_start + len([23, 22, 7, 4])
                        _pool_layer = [23, 22, 7, 4]
                        for feature_ID_count in range(len([23, 22, 7, 4])):
                            feature_ID = _pool_layer[feature_ID_count]

                            real_feature_index = real_feature_index_start + feature_ID_count
                            if (example_index == 1):
                                feature_para_dict.append([dilation_index, kernel_index, _bias, feature_ID])
                            if (feature_ID == 23):
                                features[example_index, real_feature_index] = ppv / C.shape[0]
                            elif (feature_ID == 22):
                                features[example_index, real_feature_index] = max_stretch
                            elif (feature_ID == 24):
                                features[example_index, real_feature_index] = np.max(C)
                            elif (feature_ID == 25):
                                features[example_index, real_feature_index] = np.mean(C)
                            elif (feature_ID == 17):
                                features[example_index, real_feature_index] = 0  #catch22_all(C)['values'][feature_ID]
                            elif (feature_ID == 4):
                                features[example_index, real_feature_index] = mean_index / ppv if ppv > 0 else -1
                            elif (feature_ID == 7):
                                features[example_index, real_feature_index] = mean / ppv if ppv > 0 else 0

                        real_feature_index_start = real_feature_index_end

                        #print("real_feature_index_start")
                        #print(real_feature_index_start)
                        end = end + num_features

                else:
                    _c = C[padding:-padding]
                    for feature_count in range(num_features_this_dilation):
                        feature_index = feature_index_start + feature_count
                        _bias = biases[feature_index]

                        ppv = 0
                        last_val = 0
                        max_stretch = 0.0
                        mean_index = 0
                        mean = 0

                        for j in range(_c.shape[0]):
                            if _c[j] > _bias:
                                ppv += 1
                                mean_index += j
                                mean += _c[j] + _bias
                            elif _c[j] < _bias:
                                stretch = j - last_val
                                if stretch > max_stretch:
                                    max_stretch = stretch
                                last_val = j
                        stretch = _c.shape[0] - 1 - last_val
                        if stretch > max_stretch:
                            max_stretch = stretch

                        _c = _c - _bias

                        _pool_layer = [23, 22, 7, 4]
                        real_feature_index_end = real_feature_index_start + len(_pool_layer)
                        for feature_ID_count in range(len(_pool_layer)):
                            feature_ID = _pool_layer[feature_ID_count]

                            real_feature_index = real_feature_index_start + feature_ID_count
                            if (example_index == 1):
                                feature_para_dict.append([dilation_index, kernel_index, _bias, feature_ID])
                            if (feature_ID == 23):
                                features[example_index, real_feature_index] = ppv / _c.shape[0]
                            elif (feature_ID == 22):
                                features[example_index, real_feature_index] = max_stretch
                            elif (feature_ID == 24):
                                features[example_index, real_feature_index] = np.max(_c)
                            elif (feature_ID == 25):
                                features[example_index, real_feature_index] = np.mean(_c)
                            elif (feature_ID == 17):
                                features[example_index, real_feature_index] = 0  #catch22_all(_c)['values'][feature_ID]
                            elif (feature_ID == 4):
                                features[example_index, real_feature_index] = mean_index / ppv if ppv > 0 else -1
                            elif (feature_ID == 7):
                                features[example_index, real_feature_index] = mean / ppv if ppv > 0 else 0

                        real_feature_index_start = real_feature_index_end

                        #print("real_feature_index_start")
                        #print(real_feature_index_start)
                        end = end + num_features
                feature_index_start = feature_index_end
                combination_index += 1
                num_channels_start = num_channels_end

    #print(numba.typeof((X, parameter,pool_layer)))
    #print(numba.typeof((features,feature_para_dict)))
    return features, feature_para_dict


def index_int2array(index_int):
    index_str = bin(index_int).replace("0b", "")
    index_array = np.array([], dtype=int)
    for i in range(len(index_str)):
        if (index_str[i] == '1'):
            index_array = np.append(index_array, i)
    return index_array
