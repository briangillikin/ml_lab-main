import os
import json
import numpy as np
import pandas as pd
from collections import Counter
from imblearn.over_sampling import RandomOverSampler


def oversample_data(df:pd.DataFrame) -> pd.DataFrame:
    """Take the train split of the data and oversample the minority class so that all classes have the same number of observations

    Args:
        df (pd.DataFrame): Train split of the data

    Returns:
        pd.DataFrame: Oversampled train split of the data
    """    
    print(f"Len df before oversampling: {len(df)}")
    
    # Format data as needed for oversample fit
    x = df["TEXT"].to_list()
    y = df["TRUTH"].to_list()
    x = np.array(x).reshape((len(x), 1))

    # # summarize class distribution
    # print(f"\nInitial class distribution:\n{Counter(y)}")

    # define oversampling strategy
    oversample = RandomOverSampler()
    # fit and apply the transform
    x_over, y_over = oversample.fit_resample(x, y)

    # # summarize class distribution
    # print(f"\nOversampled class distribution:\n{Counter(y_over)}")

    # Format data back to original 
    x_over = list(np.array(x_over).reshape((-1)))
    # print(x_over[:5])

    # create new df
    new_df = {
        "TEXT": x_over,
        "TRUTH": y_over
    }
    new_df = pd.DataFrame.from_dict(new_df)
    print(f"Len df after oversampling: {len(new_df)}")

    return new_df


def create_data_artifacts(oversample:bool=True) -> None:
    """Create the following data artifacts:
    1. Label map
    2. Train Split
    3. Test Split
    Save them to the data folder.

    Args:
        oversample (bool, optional): Whether or not to oversample the train split. Defaults to True.
    """
    # define fps
    train_data_fp = "data/train.csv"
    test_data_fp = "data/test.csv"
    label_map_fp = "data/label_map.json"

    # read in all data
    df = pd.read_csv("data/form_4138_data.csv")
    df = df.dropna()
    print(df.head())
    
    # save label map
    label_map = {}
    labels = list(df["TRUTH"].unique())
    for i, label in enumerate(labels):
        label_map[label] = i

    with open(label_map_fp, 'w') as f:
        json.dump(label_map, f, indent=4)
    print(f"\nWrote {label_map_fp}")

    # create train/ test split 
    df_train = df.sample(frac=0.8, random_state=1)
    df_test = df.drop(df_train.index)
    
    if oversample:
        # oversample to handle class imbalance
        # NOTE: IMP to run this after the train/test to prevent the same data points from ending up in both the training and test sets
        print("\nOversample train data...")
        df_train = oversample_data(df_train)
        
    # save train/test data to csv
    df_train.to_csv(train_data_fp, index=False)
    print(f"\nWrote {train_data_fp} with {len(df_train)} records")
    
    df_test.to_csv(test_data_fp, index=False)
    print(f"Wrote {test_data_fp} with {len(df_test)} records")


if __name__ == "__main__":
    create_data_artifacts(oversample=True)