from predict_all import predict, predict_plus, predict_2plus, convert_to_paras
from Mylayer import My_sigmoid, TotalLossFn
from get_data import get_4data
from torch.utils.tensorboard import SummaryWriter
import psutil
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import matplotlib.pyplot as plt
import torch
import pandas as pd
import numpy as np
import torch.nn as nn, torch.optim as optim
from AnyRocket import MiniRocket
from torchviz import make_dot
import numba
from numba.typed import List
import config

# from numba import get_num_threads, njit, prange, set_num_threads, vectorize
# from sklearn.model_selection import train_test_split
# import math
# import copy
# from torch.utils.data import random_split
# import torch.nn.functional as F
# from torch import autograd
# from sktime.datasets import load_from_tsfile_to_dataframe

device = torch.device("cuda")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = \
    {
        "num_kernels": 100000,
        "num_features": 400000,
        "k": 50000,
        "lr_weight": 100,
        "lr_beta": 0.01,
        "lr_gamma": 0.01,
        "minibatch_size": 256,

        "max_epochs": 200,
        "max_iteration": 200,

        "Print_process": True,
        "re_lr": 100,
        "random_state": 55,
        "lamda": 10,

        "c": -30,

        "path": ".\\Beef\\Beef",
        "data_name": "Beef",
        "c_num": 100,

        "num_threads": 1,
        "best_result": 0.9,
    }

def main(data, args):
    train_duration = 0
    test_duration = 0
    result = {}
    start_time = time.perf_counter()

    def ViewV(VV):
        g = make_dot(VV)
        g.view()

    def View(VV, model):
        g = make_dot(VV, params=dict(model.named_parameters()))
        g.view()

    def PrintPara(model):
        print("===========gamma===============")
        print("===========alpha=====================")
        return

    def Sigmoidd(x):
        return 1 / (1 + torch.exp(x * (-1)))

    # log for print
    log_model = []
    log_accuracy = []
    log_2accuracy = []
    log_3accuracy = []
    log_k = []
    log_lossval = []
    log_2loss = []

    writer = SummaryWriter('runs/' + str(args["data_name"] + "_Test1"))

    X, y, X_train, y_train, X_val, y_val, X_test, y_test, X_train_val, y_train_val = data

    print("y_train")
    print(y_train)
    rocket = MiniRocket(num_kernels=args["num_kernels"])  # by default, ROCKET uses 10,000 kernels
    rocket.fit(X_train, y_train)
    NN = rocket.named_parameters["NN"]

    X_train_transform, feature_para_dict = rocket.transform(X_train)
    X_val_transform, feature_para_dict = rocket.transform(X_val)

    args["num_features"] = X_train_transform.shape[1]

    y_train = np.array(y_train, dtype="long")
    y_val = np.array(y_val, dtype="long")
    y_test = np.array(y_test, dtype="long")

    model_main = nn.Sequential(My_sigmoid(args["num_features"], c=args["c"]),
                               nn.Linear(args["num_features"], args["c_num"], bias=False))

    mm = model_main[0]
    para___ = mm.named_parameters()
    para = list(para___)
    #list(model_main[1].parameters())
    a = para[1]

    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device)

    num_dialtion = rocket.named_parameters["dilations"].size

    optimizer_gamma = optim.Adam(model_main[0].parameters(), lr=args["lr_gamma"])
    optimizer_weight = optim.Adam(model_main[1].parameters(), lr=args["lr_weight"])

    scheduler_gamma = torch.optim.lr_scheduler.StepLR(optimizer_gamma, step_size=20, gamma=0.95)
    scheduler_weight = torch.optim.lr_scheduler.StepLR(optimizer_weight, step_size=20, gamma=0.95)

    beta = 0.0
    beta = torch.tensor(beta)

    #------------Added batch training----------------
    ans = 0
    model_main = model_main.to(device)

    iteration = 0

    for epoch in range(args["max_epochs"]):
        #---------------One epoch (trained once)----------

        #Completed the shuffle of all training data
        minibatches = torch.randperm(len(X_train_transform)).split(args["minibatch_size"])
        for minibatch_index, minibatch in enumerate(minibatches):
            #--------------One iteration (training with one minibatch)----------
            if minibatch_index > 0 and len(minibatch) < args["minibatch_size"]:
                break
            #---------------Training Dataset-------------------> loss
            print("------------Step1--------------")
            # X_training represents the minibatch data during training
            # X_train_transform represents the entire training data set

            torch.cuda.empty_cache()
            torch.cuda.empty_cache()

            iteration = iteration + 1
            # DataFrame -> 2d narray ->tensor
            X_training = torch.tensor(np.array(X_train_transform.iloc[minibatch]), requires_grad=True)
            y_training = torch.tensor(y_train[minibatch])

            X_training = torch.tensor(X_training, requires_grad=True)

            # use_gpu
            X_training = X_training.to(device)
            print("X_TRAINING IS CUDA? ")
            print(X_training.is_cuda)
            print("=====================")
            _y_training = model_main(X_training)

            y_training = y_training.type(torch.LongTensor)
            # use_gpu
            y_training = y_training.to(device)

            loss_fn.to(device)
            loss_train = loss_fn(_y_training, y_training)

            print("y_train")
            print(y_train)

            optimizer_gamma.zero_grad()
            para_1 = list(model_main[0].named_parameters())
            para_2 = list(model_main[1].named_parameters())

            grad_Loss_train_alpha = torch.autograd.grad(loss_train, para_1[1][1],
                                                        grad_outputs=torch.ones_like(loss_train),
                                                        create_graph=True,
                                                        retain_graph=True,
                                                        )[0]
            print("------------------grad_Loss_train_alpha-----------------")
            print(grad_Loss_train_alpha)

            optimizer_gamma.zero_grad()
            grad_Loss_train_weight = torch.autograd.grad(loss_train, para_2[0][1],
                                                         grad_outputs=torch.ones_like(loss_train),
                                                         create_graph=True,
                                                         retain_graph=True,
                                                         )[0]
            print("------------------grad_Loss_train_weight-----------------")
            print(grad_Loss_train_weight)

            weight_row = para_2[0][1].data
            weight_prime = para_2[0][1].data - args["lr_weight"] * grad_Loss_train_weight

            #-----------------Validation Dataset-------------------->Loss Function
            optimizer_gamma.zero_grad()
            print("minibatch")
            print(minibatch)
            print("val_shape=")
            print(X_val_transform.shape)
            print("train_shape=")
            print(X_train_transform.shape)
            # minibatch 1~len
            X_valing = torch.tensor(np.array(X_val_transform.iloc[minibatch]), requires_grad=True)
            y_valing = torch.tensor(y_val[minibatch])
            # NumPy indexes start at 0
            X_valing = torch.tensor(X_valing, requires_grad=True)
            # use_ gpu

            X_valing = X_valing.to(device)
            _y_valing = model_main(X_valing)

            y_valing = y_valing.type(torch.LongTensor)

            # usegpu
            y_valing = y_valing.to(device)
            loss_val = loss_fn(_y_valing, y_valing)
            log_lossval.append(loss_val.data)

            #Seek the derivative of the validation data set
            optimizer_gamma.zero_grad()
            para_1 = list(model_main[0].named_parameters())
            para_2 = list(model_main[1].named_parameters())
            grad_Loss_val_alpha = torch.autograd.grad(loss_val, para_1[1][1],
                                                      grad_outputs=torch.ones_like(loss_val),
                                                      create_graph=True,
                                                      retain_graph=True,
                                                      )[0]

            print("------------------grad_Loss_val_alpha-----------------")
            print(grad_Loss_val_alpha)

            optimizer_gamma.zero_grad()
            grad_Loss_val_weight = torch.autograd.grad(loss_val, para_2[0][1],
                                                       grad_outputs=torch.ones_like(loss_val),
                                                       create_graph=True,
                                                       retain_graph=True,
                                                       )[0]
            print("------------------grad_Loss_val_weight-----------------")
            print(grad_Loss_val_weight)

            print("=================||grad_Loss_val_weight||_2===============")
            print(torch.norm(grad_Loss_val_weight))
            eplision = 0.01 / (torch.norm(grad_Loss_val_weight) + 1e-6)

            print("=================eplision===============")
            print(eplision)

            #weight_prime
            #weight_row
            #weight_plus
            weight_plus = weight_row + eplision * grad_Loss_val_weight
            #weight_sub
            weight_sub = weight_row - eplision * grad_Loss_val_weight

            #grad_Loss_train_alpha_for_weight_plus
            para_2[0][1].data = weight_plus
            _y_training = model_main(X_training)
            y_training = y_training.type(torch.LongTensor)

            y_training = y_training.to(device)

            loss_train = loss_fn(_y_training, y_training)
            grad_Loss_train_alpha_for_weight_plus = torch.autograd.grad(loss_train, para_1[1][1],
                                                                        grad_outputs=torch.ones_like(
                                                                            torch.tensor(loss_train)),
                                                                        create_graph=True,
                                                                        retain_graph=True,
                                                                        )[0]
            print("------------------grad_Loss_train_alpha_for_weight_plus-----------------")
            print(grad_Loss_train_alpha_for_weight_plus)
            #grad_Loss_train_alpha_for_weight_sub
            para_2[0][1].data = weight_sub
            _y_training = model_main(X_training)
            y_training = y_training.type(torch.LongTensor)

            y_training = y_training.to(device)

            loss_train = loss_fn(_y_training, y_training)
            grad_Loss_train_alpha_for_weight_sub = torch.autograd.grad(loss_train, para_1[1][1],
                                                                       grad_outputs=torch.ones_like(
                                                                           torch.tensor(loss_train)),
                                                                       create_graph=True,
                                                                       retain_graph=True,
                                                                       )[0]
            #restart
            para_2[0][1].data = weight_prime
            # mid
            mid = (grad_Loss_train_alpha_for_weight_plus - grad_Loss_train_alpha_for_weight_sub) / (2 * eplision)
            print("------------------mid-----------------")
            print(mid)

            # Find the derivative of the total loss
            optimizer_gamma.zero_grad()

            #  model -> cpu
            #model_main=model_main.cpu()
            gamma = para_1[0][1]
            alpha = 1 / (1 + torch.exp(gamma * (args["c"])))
            #ViewV(alpha)
            grad_alpha_gamma = torch.autograd.grad(alpha, gamma,
                                                   grad_outputs=torch.ones_like(alpha),
                                                   create_graph=True,
                                                   retain_graph=True
                                                   )[0]

            optimizer_gamma.zero_grad()
            TotalLossfn = TotalLossFn();

            totalloss = TotalLossfn(para_1[0][1], para_1[1][1],
                                    grad_Loss_val_alpha, grad_Loss_val_weight,
                                    mid,
                                    grad_alpha_gamma,
                                    beta,
                                    )
            #View(totalloss,model_main)
            totalloss.backward(torch.ones_like(para_1[0][1]))

            # ----------------Freeze parameters, update gamma alpha beta-------------------->end
            para_2[0][1].requires_grad = False
            para_1[1][1].requires_grad = False
            optimizer_gamma.step()
            para_2[0][1].requires_grad = True
            para_1[1][1].requires_grad = True
            print("-------------Updating gamma------------")

            Sigmoid_fn = torch.nn.Sigmoid()
            para_1[1][1].data = Sigmoid_fn(para_1[0][1].data * args["c"] * (-1))
            #assert math.isnan(para_1[1][1].data.sum().item())==False

            print("-------------Updating alpha---------------")
            PrintPara(model_main)
            print("===========================================")

            print("beta")
            print(beta)
            print("k")
            print(args["k"])

            print("============================================")
            beta = beta + args["lr_beta"] * (para_1[1][1].data.sum().item() - args["k"])
            log_2loss.append(para_1[1][1].data.sum().item() - args["k"])
            #----------------step2----------------
            optimizer_weight.zero_grad()
            print("-------------------Step 2-------------")
            X_training = torch.tensor(np.array(X_train_transform.iloc[minibatch]), requires_grad=True)
            y_training = torch.tensor(y_train[minibatch])

            X_training = torch.tensor(X_training, requires_grad=True)

            X_training = X_training.to(device)
            y_training = y_training.to(device)
            loss_fn = loss_fn.to(device)

            _y_training = model_main(X_training)
            #View(_y_training,model_main)

            y_training = y_training.type(torch.LongTensor)

            y_training = y_training.to(device)
            loss_train = loss_fn(_y_training, y_training)

            #PrintPara(model_main)
            grad_Loss_train_weight = torch.autograd.grad(loss_train, para_2[0][1],
                                                         grad_outputs=torch.ones_like(loss_train),
                                                         create_graph=True,
                                                         retain_graph=True,
                                                         )[0]

            re_loss = 0
            for para in model_main[1].parameters():
                re_loss += torch.sum(torch.abs(para))

            loss_train = loss_fn(_y_training, y_training)
            loss = loss_train + args["re_lr"] * re_loss

            optimizer_weight.zero_grad()
            loss.backward()

            para_1[1][1].requires_grad = False
            para_1[0][1].requires_grad = False

            weight_row = para_2[0][1].data
            optimizer_weight.step()
            weight_prime = para_2[0][1].data

            para_1[1][1].requires_grad = True
            para_1[0][1].requires_grad = True
            print("Updating weight")

            log_k.append(para_1[1][1].data.sum().item())

            scheduler_gamma.step()
            scheduler_weight.step()

            # log with tensorboard
            print("=======log_k==========")
            print(log_k)
            # best_model = copy.deepcopy(model_main)
            plt.plot(log_k)
            writer.add_scalar('k', para_1[1][1].data.sum().item(), iteration)
            model_main = model_main.cpu()
            alpha__ = list(model_main[0].named_parameters())[1][1].detach().numpy()
            model_main = model_main.to(device)
            alpha_1d = alpha__.flatten()
            try:
                writer.add_histogram(tag="alpha hist", values=alpha_1d, global_step=iteration)
            except:
                pass
            finally:
                alpha_sort = np.sort(alpha_1d)
                stt_time = time.perf_counter()
                last_test_duration = 0
                if (iteration % 3 == 0):
                    predict_result = predict_plus(args=args, best_model=model_main, feature_para_dict=feature_para_dict,
                                                  data=data)
                    writer.add_scalar('predict_accuracy', predict_result["predict_accuracy"], iteration)
                    ans = max(ans, predict_result["predict_accuracy"])
                last_test_duration = time.perf_counter() - stt_time
                test_duration += last_test_duration

                writer.add_scalar('alpha[k]', alpha_sort[int(args["k"])], iteration)
                writer.add_scalar('alpha[-k]', alpha_sort[int(-args["k"])], iteration)

                writer.add_scalar('loss_val', loss_val, iteration)
                writer.add_scalar('loss_train', loss_train, iteration)
                writer.add_scalar('sum(alpha)-k', para_1[1][1].data.sum().item() - args["k"], iteration)

                print("===========epoch:==========")
                print(epoch)
                print("===========iteration:===========")
                print(iteration)
                if (iteration > args["max_iteration"]):
                    break
            if (iteration > args["max_iteration"]):
                break

            model_main = model_main.cpu()
            torch.cuda.empty_cache()
            model_main = model_main.to(device)

    model_main.cpu()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()

    train_duration += time.perf_counter() - start_time
    predict_result["predict_accuracy"] = ans
    result["predict_result"] = predict_result
    result["train_duration"] = train_duration - test_duration
    result["test_duration"] = last_test_duration
    writer.add_scalar('predict_accuracy', predict_result["predict_accuracy"], iteration)

    return result


import pytz
from datetime import datetime
import platform
import socket
import time


def Run(args, path_, data_name, best_result, output_dir):
    args["best_result"] = best_result
    path = path_ + data_name + "/" + data_name
    args["path"] = path
    args["data_name"] = data_name
    print('data_name:', data_name)

    _start_time = time.perf_counter()
    data, args = get_4data(args)
    dataload_duration = time.perf_counter() - _start_time

    physical_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)
    cpu_freq = psutil.cpu_freq()
    max_freq = cpu_freq.max
    min_freq = cpu_freq.min
    memory = np.round(psutil.virtual_memory().total / 1e9)

    args["data_name"] = data_name

    if args["num_threads"] > 0:
        numba.set_num_threads(args["num_threads"])
    result = main(data, args)

    df_metrics = pd.DataFrame(data=np.zeros((1, 21), dtype=np.cfloat), index=[0],
                              columns=['timestamp', 'itr', 'classifier',
                                       'num_kernels',
                                       'num_features',
                                       'k',
                                       'lr_weight',
                                       'lr_beta',
                                       'lr_gamma',
                                       'dataset',
                                       'physical_cores',
                                       "train_duration", "test_duration", "test_assembly_duration",
                                       "predict_accuracy",
                                       "epoch",
                                       "itr",
                                       "logical_cores",
                                       'max_freq', 'min_freq', 'memory'])
    df_metrics["timestamp"] = datetime.utcnow().replace(tzinfo=pytz.utc).strftime("%Y-%m-%d %H:%M:%S")
    df_metrics["itr"] = args["max_epochs"]
    df_metrics["num_kernels"] = args["num_kernels"]
    df_metrics["num_features"] = args["num_features"]
    df_metrics["k"] = args["k"]
    df_metrics["lr_weight"] = args["lr_weight"]
    df_metrics["lr_beta"] = args["lr_beta"]
    df_metrics["lr_gamma"] = args["lr_gamma"]
    df_metrics["dataset"] = args["data_name"]

    df_metrics["train_duration"] = result["train_duration"]
    df_metrics["test_duration"] = result["predict_result"]["test_duration"]
    df_metrics["test_assembly_duration"] = result["predict_result"]["test_assembly_duration"]
    df_metrics["predict_accuracy"] = result["predict_result"]["predict_accuracy"]
    df_metrics["physical_cores"] = physical_cores
    df_metrics["logical_cores"] = logical_cores
    df_metrics["max_freq"] = max_freq
    df_metrics["min_freq"] = min_freq
    df_metrics["memory"] = memory
    df_metrics["epoch"] = args["max_epochs"]
    df_metrics["itr"] = args["max_iteration"]
    df_metrics.to_csv(output_dir + args["data_name"] + '.csv', index=False)


path_ = './data/'
datasets = [
    ['SmoothSubspace', 1.0],
]
output_dir = "./output/"

for data in datasets:
    Run(args, path_, data_name=data[0], best_result=data[1], output_dir=output_dir)
