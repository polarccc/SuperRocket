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


class My_sigmoid(nn.Module):
    def __init__(self, input_features, bias=False, c=30):
        super(My_sigmoid, self).__init__()
        self.input_features = input_features
        self.gamma = torch.empty((input_features), dtype=torch.float32).uniform_(-1, 1)
        self.gamma = nn.Parameter(torch.tensor(self.gamma, requires_grad=True))
        self.alpha = nn.Parameter(1 / (1 + torch.exp(self.gamma * c)))  #*(-30)
        self.alpha = nn.Parameter(torch.tensor(self.alpha, requires_grad=True))

    def forward(self, input):
        return input.mul(self.alpha)


class TotalLossFn(nn.Module):
    def __init__(self):
        super(TotalLossFn, self).__init__()

    def forward(self, gamma, alpha, grad_Loss_val_alpha, grad_Loss_val_weight,
                mid, grad_alpha_gamma, beta):
        return total_loss_fn.apply(gamma, alpha, grad_Loss_val_alpha, grad_Loss_val_weight,
                                   mid, grad_alpha_gamma, beta)


class total_loss_fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gamma, alpha, grad_Loss_val_alpha, grad_Loss_val_weight,
                mid, grad_alpha_gamma, beta):
        ctx.save_for_backward(gamma, alpha, grad_Loss_val_alpha, grad_Loss_val_weight,
                              mid, grad_alpha_gamma, beta)
        return alpha

    @staticmethod
    def backward(ctx, grad_output):
        gamma, alpha, grad_Loss_val_alpha, grad_Loss_val_weight, mid, grad_alpha_gamma, beta = ctx.saved_tensors
        mid_1 = mid
        mid_2 = torch.tensor(10) * mid_1
        mid_3 = grad_Loss_val_alpha - mid_2
        mid_appendix = 10 * (1 / gamma.mul(gamma))
        print("mid+appendix=")
        print(mid_appendix.size)
        print(mid_appendix)
        grad_alpha = mid_3 + beta + mid_appendix  #10=args[lamda]
        grad_gamma = grad_alpha * grad_alpha_gamma

        print("-----------------grad_gamma--------------------")
        print(grad_gamma)
        return grad_gamma, grad_alpha, grad_Loss_val_alpha, grad_Loss_val_weight, mid, grad_alpha_gamma, beta
