import os
import pickle
import pandas as pd
import numpy as np
from typing import Union

from ml_lab.models.form_4138_decision_tree import Form4138DecisionTree


def load_data() -> Union[np.ndarray,list]:
    """Load training data

    Returns:
        Union[np.ndarray,list]: x_train, y_train
    """
    decision_tree = Form4138DecisionTree()
    
    print("\nLoading data...")
    train = pd.read_csv("data/train.csv")

    docs = train["TEXT"]
    x_train = decision_tree.custom_vectorizer.transform(docs)

    train["TRUTH"] = train["TRUTH"].apply(decision_tree.label_mapper.get_label_id) 
    y_train = train["TRUTH"].to_list()
    print("Data Loaded!")
    
    return x_train, y_train


def train_decision_tree(config:dict) -> None:
    """Train decision tree and save model to pkl file in data_artifacts folder

    Args:
        config (dict): hyperparameters to use for decision tree model
    """
    decision_tree = Form4138DecisionTree(config=config)
    model = decision_tree.model
    x_train, y_train = load_data()

    print("\nTraining Model...")
    model.fit(x_train, y_train)

    os.makedirs(decision_tree.models_dir, exist_ok=True)
    pickle.dump(model, open(decision_tree.model_fp, 'wb'))
    print(f"\nSaved model to {decision_tree.model_fp}")


if __name__ == "__main__":
    config = {'max_depth': 50, 'min_samples_split': 2}
    train_decision_tree(config)
