from predict_all import predict, predict_plus, predict_2plus, convert_to_paras
from Mylayer import My_sigmoid, TotalLossFn
from get_data import get_4data
from torch.utils.tensorboard import SummaryWriter
import psutil
import os

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
from pathlib import Path


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
        "minibatch_size": 32,

        "max_epochs": 10,
        "max_iteration": False,

        "Print_process": False,
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
            # print("------------Step1--------------")
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
            _y_training = model_main(X_training)

            y_training = y_training.type(torch.LongTensor)
            # use_gpu
            y_training = y_training.to(device)

            loss_fn.to(device)
            loss_train = loss_fn(_y_training, y_training)
            optimizer_gamma.zero_grad()
            para_1 = list(model_main[0].named_parameters())
            para_2 = list(model_main[1].named_parameters())

            grad_Loss_train_alpha = torch.autograd.grad(loss_train, para_1[1][1],
                                                        grad_outputs=torch.ones_like(loss_train),
                                                        create_graph=True,
                                                        retain_graph=True,
                                                        )[0]
            # print("------------------grad_Loss_train_alpha-----------------")
            # print(grad_Loss_train_alpha)

            optimizer_gamma.zero_grad()
            grad_Loss_train_weight = torch.autograd.grad(loss_train, para_2[0][1],
                                                         grad_outputs=torch.ones_like(loss_train),
                                                         create_graph=True,
                                                         retain_graph=True,
                                                         )[0]
            # print("------------------grad_Loss_train_weight-----------------")
            # print(grad_Loss_train_weight)

            weight_row = para_2[0][1].data
            weight_prime = para_2[0][1].data - args["lr_weight"] * grad_Loss_train_weight

            #-----------------Validation Dataset-------------------->Loss Function
            optimizer_gamma.zero_grad()
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

            # print("------------------grad_Loss_val_alpha-----------------")
            # print(grad_Loss_val_alpha)

            optimizer_gamma.zero_grad()
            grad_Loss_val_weight = torch.autograd.grad(loss_val, para_2[0][1],
                                                       grad_outputs=torch.ones_like(loss_val),
                                                       create_graph=True,
                                                       retain_graph=True,
                                                       )[0]
            # print("------------------grad_Loss_val_weight-----------------")
            # print(grad_Loss_val_weight)

            # print("=================||grad_Loss_val_weight||_2===============")
            # print(torch.norm(grad_Loss_val_weight))
            eplision = 0.01 / (torch.norm(grad_Loss_val_weight) + 1e-6)

            # print("=================eplision===============")
            # print(eplision)

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
            # print("-------------Updating gamma------------")

            Sigmoid_fn = torch.nn.Sigmoid()
            para_1[1][1].data = Sigmoid_fn(para_1[0][1].data * args["c"] * (-1))
            #assert math.isnan(para_1[1][1].data.sum().item())==False

            # print("============================================")
            beta = beta + args["lr_beta"] * (para_1[1][1].data.sum().item() - args["k"])
            log_2loss.append(para_1[1][1].data.sum().item() - args["k"])
            #----------------step2----------------
            optimizer_weight.zero_grad()
            # print("-------------------Step 2-------------")
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
            # print("Updating weight")

            log_k.append(para_1[1][1].data.sum().item())

            scheduler_gamma.step()
            scheduler_weight.step()

            # log with tensorboard
            # print("=======log_k==========")
            # print(log_k)
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

                # print("===========epoch:==========")
                # print(epoch)
                # print("===========iteration:===========")
                # print(iteration)
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
    result["test_duration"] = test_duration
    writer.add_scalar('predict_accuracy', predict_result["predict_accuracy"], iteration)

    return result


import pytz
from datetime import datetime
import platform
import socket
import time


RESULT_SUMMARY_COLUMNS = [
    "timestamp",
    "dataset",
    "status",
    "error_message",
    "train_duration",
    "test_time",
    "test_duration",
    "test_transform_time",
    "test_core_duration",
    "train_classifier_time",
    "train_transform_time",
    "rocket_fit_time",
    "feature_selection_time",
    "predict_accuracy",
    "num_kernels",
    "num_features",
    "k",
    "physical_cores",
    "logical_cores",
    "max_freq",
    "min_freq",
    "memory",
]


def build_result_row(dataset, df_metrics=None, status="success", error_message=""):
    row = {column: None for column in RESULT_SUMMARY_COLUMNS}
    row["timestamp"] = datetime.utcnow().replace(tzinfo=pytz.utc).strftime("%Y-%m-%d %H:%M:%S")
    row["dataset"] = dataset
    row["status"] = status
    row["error_message"] = error_message

    if df_metrics is not None:
        row["timestamp"] = df_metrics.loc[0, "timestamp"]
        row["train_duration"] = df_metrics.loc[0, "train_duration"]
        row["test_time"] = df_metrics.loc[0, "test_duration"]
        row["test_duration"] = df_metrics.loc[0, "test_duration"]
        row["test_transform_time"] = df_metrics.loc[0, "test_transform_time"]
        row["test_core_duration"] = df_metrics.loc[0, "test_core_duration"]
        row["train_classifier_time"] = df_metrics.loc[0, "train_classifier_time"]
        row["train_transform_time"] = df_metrics.loc[0, "train_transform_time"]
        row["rocket_fit_time"] = df_metrics.loc[0, "rocket_fit_time"]
        row["feature_selection_time"] = df_metrics.loc[0, "feature_selection_time"]
        row["predict_accuracy"] = df_metrics.loc[0, "predict_accuracy"]
        row["num_kernels"] = df_metrics.loc[0, "num_kernels"]
        row["num_features"] = df_metrics.loc[0, "num_features"]
        row["k"] = df_metrics.loc[0, "k"]
        row["physical_cores"] = df_metrics.loc[0, "physical_cores"]
        row["logical_cores"] = df_metrics.loc[0, "logical_cores"]
        row["max_freq"] = df_metrics.loc[0, "max_freq"]
        row["min_freq"] = df_metrics.loc[0, "min_freq"]
        row["memory"] = df_metrics.loc[0, "memory"]

    return row


def Run(args, path_, data_name, best_result, output_dir):
    args["best_result"] = best_result
    path = path_ + data_name + "/" + data_name
    args["path"] = path
    args["data_name"] = data_name
    print('data_name:', data_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

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

    df_metrics = pd.DataFrame(data=np.zeros((1, len(RESULT_SUMMARY_COLUMNS)), dtype=np.cfloat), index=[0],
                              columns=RESULT_SUMMARY_COLUMNS)
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
    df_metrics["test_duration"] = result["test_duration"]
    df_metrics["test_transform_time"] = result["predict_result"]["test_transform_time"]
    df_metrics["test_core_duration"] = result["predict_result"]["test_core_duration"]
    df_metrics["train_classifier_time"] = result["predict_result"]["train_classifier_time"]
    df_metrics["train_transform_time"] = result["predict_result"]["train_transform_time"]
    df_metrics["rocket_fit_time"] = result["predict_result"]["rocket_fit_time"]
    df_metrics["feature_selection_time"] = result["predict_result"]["feature_selection_time"]
    df_metrics["predict_accuracy"] = result["predict_result"]["predict_accuracy"]
    df_metrics["physical_cores"] = physical_cores
    df_metrics["logical_cores"] = logical_cores
    df_metrics["max_freq"] = max_freq
    df_metrics["min_freq"] = min_freq
    df_metrics["memory"] = memory
    df_metrics["epoch"] = args["max_epochs"]
    df_metrics["itr"] = args["max_iteration"]
    df_metrics.to_csv(Path(output_dir) / f"{args['data_name']}.csv", index=False)
    return df_metrics


def append_metrics_row(csv_path, row):
    csv_file = Path(csv_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    df_row = pd.DataFrame([row], columns=RESULT_SUMMARY_COLUMNS)
    df_row.to_csv(csv_file, mode="a", header=not csv_file.exists(), index=False)


def initialize_result_csv(csv_path):
    csv_file = Path(csv_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    if not csv_file.exists() or csv_file.stat().st_size == 0:
        pd.DataFrame(columns=RESULT_SUMMARY_COLUMNS).to_csv(csv_file, index=False)


def discover_datasets(path_root):
    dataset_names = []
    patterns = ["*/*_TRAIN.ts", "*/*_TRAIN.txt", "*/*_TRAIN.TS", "*/*_TRAIN.TXT"]
    seen = set()
    for pattern in patterns:
        for train_file in sorted(Path(path_root).glob(pattern)):
            dataset_name = train_file.stem[:-6]
            if dataset_name in seen:
                continue
            test_candidates = [
                train_file.with_name(f"{dataset_name}_TEST.ts"),
                train_file.with_name(f"{dataset_name}_TEST.txt"),
                train_file.with_name(f"{dataset_name}_TEST.TS"),
                train_file.with_name(f"{dataset_name}_TEST.TXT"),
            ]
            if any(test_file.exists() for test_file in test_candidates):
                seen.add(dataset_name)
                dataset_names.append(dataset_name)
    return dataset_names


if __name__ == "__main__":
    path_ = "./data/"
    datasets = ['ACSF1']
    output_dir = "./output/"
    result_csv = str(Path(output_dir) / "superrocket_inference_times.csv")
    initialize_result_csv(result_csv)

    for data in datasets:
        try:
            df_metrics = Run(args, path_, data_name=data[0], best_result=data[1], output_dir=output_dir)
            append_metrics_row(result_csv, build_result_row(data[0], df_metrics, status="success"))
        except Exception as exc:
            append_metrics_row(result_csv, build_result_row(data[0], status="failed", error_message=f"{type(exc).__name__}: {exc}"))
            print(f"[{data[0]}] failed: {exc}")
