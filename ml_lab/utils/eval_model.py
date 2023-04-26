import pandas as pd
import argparse
from typing import Union
from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score
from ml_lab.models.form_4138_decision_tree import Form4138DecisionTree
from ml_lab.models.form_4138_nn import Form4138NN
from ml_lab.utils.load_data import load_y_truth_y_pedictions


def eval_model(model:Union[Form4138DecisionTree, Form4138NN]) -> pd.DataFrame:
    """Get the accuracy, precision, recall, and f1 of the train and test set for a given model and save the results in a dataframe

    Args:
        model (Union[Form4138DecisionTree, Form4138NN]): A model instance of one of the following types that has already been trained

    Returns:
        pd.DataFrame: dataframe containing the following columns: "data",
        "accuracy", "precision", "recall", "f1". The data column refers to whether the metric was taken on the train or test split.
    """    
    model.load()

    results = {
        "data": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": []
    }

    for data_split in ["train", "test"]:
        # load data
        y_truth, y_pred = load_y_truth_y_pedictions(model, data_split)

        # calculate accuracy, precision, recall, f1
        accuracy = round(accuracy_score(y_truth, y_pred),4)
        precision = round(precision_score(y_truth, y_pred, average='weighted'),4)
        recall = round(recall_score(y_truth, y_pred, average='weighted'),4)
        f1 = round(f1_score(y_truth, y_pred, average='weighted'),4)

        # add metrics to results
        results["data"].append(data_split)
        results["accuracy"].append(accuracy)
        results["precision"].append(precision)
        results["recall"].append(recall)
        results["f1"].append(f1)

    # create results dataframe
    df = pd.DataFrame(results)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", default="dt", choices=["dt", "nn"], help="dt = decision tree and nn = neural net")
    args = parser.parse_args()

    if args.model == "dt":
        model = Form4138DecisionTree()
    else:
        model = Form4138NN()
    
    df = eval_model(model)

    print(f"\nEval Metrics for the {type(model).__name__} model")
    print(df)