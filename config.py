args = {
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

path_ = "/mnt/data/user_liangzhiyu/lichen/MultiRocket/Univariate_ts/"
output_dir = "/mnt/data/user_liangzhiyu/lichen/MultiRocket/"

datasets = [
    ['ScreenType', 0.7],
]
