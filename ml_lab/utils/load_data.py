from typing import Tuple, Union
import pandas as pd
import argparse
from ml_lab.models.form_4138_decision_tree import Form4138DecisionTree
from ml_lab.models.form_4138_nn import Form4138NN


def load_y_truth_y_pedictions(model:Union[Form4138DecisionTree, Form4138NN], data_split:str) -> Tuple[list, list]:
    """Load data from train or test split, run model batch predict, and return the truth, predictions

    Args:
        model (Union[Form4138DecisionTree, Form4138NN]): A model instance of one of the following types that has already been trained
        data_split (str): train or test

    Returns:
        Tuple[list, list]: y_truth, y_predictions
    """    
    print(f"Loading y_truth, y_predictions for {data_split} data...")
    data = pd.read_csv(f"data/{data_split}.csv")
    y_truth = data["TRUTH"].to_list()

    model.load()
    y_pred = model.predict_batch(data["TEXT"].to_list())
    print(f"Loaded y_truth, y_predictions for {len(y_truth)} {data_split} cases")

    return y_truth, y_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", default="dt", choices=["dt", "nn"], help="dt = decision tree and nn = neural net")
    parser.add_argument("-data", default="test", choices=["train", "test"])
    args = parser.parse_args()

    if args.model == "dt":
        model = Form4138DecisionTree()
    else:
        model = Form4138NN()

    y_truth, y_pred = load_y_truth_y_pedictions(model, args.data)
