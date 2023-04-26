# ml_lab


## Setup
1. Create python environment
    ```
    conda create -n ml_lab python=3.10
    conda activate ml_lab
    poetry install
    ```
    **NOTE:** If you have an issue installing editdistspacy, you may need to do the following:
    ```
    sudo apt-get update
    sudo apt-get install gcc
    sudo apt-get install g++
    ```
2. Create a `.env` file in the root of the repo that has 1 variable "ROOT_DIR" with its value equal to the absolute path to the repo's directory, for example:
    ```
    ROOT_DIR="/Users/jillvillany/Github/ml_lab"
    ```
3. Run `python setup.py`

## Train/ Test the Neural Net Model
```
python ml_lab/train/train_nn.py
python tests/eval_model.py -model nn # evaluate performnace
python tests/unit/test_4138_nn.py # unit testing
```

## Train/ Test the Decision Tree Model
```
python ml_lab/train/train_decision_tree.py
python tests/eval_model.py -model dt # evaluate performance
python tests/unit/test_4138_dec_tree.py # unit testing
```
