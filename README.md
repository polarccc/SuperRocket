# SuperRocket

## Requirements
* numba == 0.50.1
* numpy == 1.18.5
* pandas == 1.0.5
* scikit_learn >= 0.23.1
* sktime == 0.4.3
* torch==1.11.0+cu113
  
Also you can just follow the steps below to install environment dependencies.
```bash
# using Anaconda to create a virtual environment
conda create --name SuperRocket python=3.8
conda activate SuperRocket

# install dependencies (might be tricky for some packages)
pip install -r requirements.txt

```

## Datasets
We use the 109 UCR datasets in this study.
Please download the datasets from [timeseriesclassification.com](https://timeseriesclassification.com/dataset.php). 
Then, place the downloaded offline dataset in the following path:

```bash
.
├── data
│     ├── SmoothSubspace
│     └── ...
├── output
│     ├── SmoothSubspace
│     └── ...
├── SuperRocket.py   
└── ...
```


## Usage - SuperRocket

here are the defualt arguments, and you can modify it at any time.
```bash
args = \
    {
        "num_kernels": 100000,
        "num_features": 400000,
        "k": 50000,
        "lr_weight": 100,
        "lr_beta": 0.01,
        "lr_gamma": 0.01,
        "minibatch_size": 256,

        ...
    }

```
Use the command to evaluate the UCR datasets:
```bash
cd ./SuperRocket
python SuperRocket.py
```
