import os
import pickle
import json
import pandas as pd
from typing import Union
from scipy.sparse import csr_matrix

from ml_lab.utils.keyword_extractor import KeywordExtractor


def fit_custom_vectorizer(docs:list) -> Union[csr_matrix, dict]:
    """Fit custom vectorizer on full corpus of text

    Args:
        docs (list): list of all the text documents in the corpus

    Returns:
        Union[csr_matrix, dict]: term document matrix, vocab
    """    
    keyword_extractor = KeywordExtractor()

    print("Fitting Custom Vectorizer")
    ## build term document matrix (tdm)
    indptr = [0]
    indices = []
    data = []
    vocab = {}
    for i, doc in enumerate(docs):
        print(f"Fitting on doc {i+1} of {len(docs)}")
        keywords = keyword_extractor.extract_keywords(doc)
        for keyword in keywords:
            index = vocab.setdefault(keyword, len(vocab))
            indices.append(index)
            data.append(1)
        indptr.append(len(indices))
    
    if len(data) == 0:
        # Create an empty matrix
        tdm = csr_matrix((len(docs), 0))
    else:
        tdm = csr_matrix((data, indices, indptr), dtype=int)

    os.makedirs("model_artifacts", exist_ok=True)
    
    # save model artifacts
    tdm_fp = "model_artifacts/custom_vectorizer_tdm.pkl"
    with open(tdm_fp, 'wb') as f:
        pickle.dump(tdm, f)
    
    vocab_fp = "model_artifacts/custom_vectorizer_vocab.json"
    with open(vocab_fp, 'w') as f:
        json.dump(vocab, f)
    
    return tdm, vocab


if __name__ == "__main__":
    df = pd.read_csv("data/form_4138_data.csv")
    docs = list(df["TEXT"])
    tdm, vocab = fit_custom_vectorizer(docs)