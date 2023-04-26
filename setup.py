import pandas as pd

from ml_lab.utils.create_data_artifacts import create_data_artifacts
from ml_lab.train.generate_spellcheck_dict import generate_spellcheck_dict
from ml_lab.train.fit_custom_vectorizer import fit_custom_vectorizer


def setup():
    # create train/test split and label_map.json
    create_data_artifacts(oversample=True)

    # create spellcheck dictionary
    generate_spellcheck_dict()

    # fit custom_vectorizer
    df = pd.read_csv("data/form_4138_data.csv")
    docs = list(df["TEXT"])
    fit_custom_vectorizer(docs)

    print("\nSetup complete!")


if __name__ == "__main__":
    setup()
