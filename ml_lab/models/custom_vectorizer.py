import os
import pickle
import json
import numpy as np
from typing import Union

from ml_lab.utils.keyword_extractor import KeywordExtractor


class CustomVectorizer():
    def __init__(self) -> None:
        self.tdm_fp = "model_artifacts/custom_vectorizer_tdm.pkl"
        self.vocab_fp = "model_artifacts/custom_vectorizer_vocab.json"
        self.train_fp = "data/form_4138_data.csv"
        self.keyword_extractor = KeywordExtractor()

        if os.path.exists(self.tdm_fp) and os.path.exists(self.vocab_fp):
            with open(self.tdm_fp, 'rb') as f:
                self.tdm = pickle.load(f)
            with open(self.vocab_fp, 'r') as f:
                self.vocab = json.load(f)
        else:
            raise Exception("You must first fit the custom vectorizer by running `python ml_lab/train/fit_custom_vectorizer.py`")
    
    
    def transform(self, docs:Union[str,list]) -> np.ndarray:
        """Vectorize one or multiple documents

        Args:
            docs (Union[str,list]): text or list of text docs to vectorize

        Returns:
            np.ndarray: sparse matrix
        """        
        # just given 1 doc to transform
        if isinstance(docs, str):
            docs = [docs]

        X = np.zeros((len(docs), len(self.vocab)), dtype=int)

        for i, doc in enumerate(docs):
            keywords = self.keyword_extractor.extract_keywords(doc)

            for keyword in keywords:
                if keyword in self.vocab:
                    x = self.vocab[keyword]
                    X[i, x] += 1 

        return X


if __name__ == "__main__":
    custom_vectorizer = CustomVectorizer()
    
    # test transform with 1 doc
    doc = "I need to reschedule my exam"
    docs = custom_vectorizer.transform(doc)
    print(f"\n1 Doc Ex:\nDoc Text: {doc}\nDocs Shape: {docs.shape}\nDoc:{docs[0]}")

    word_indices = [i for i,val in enumerate(docs[0]) if val == 1]
    print(f"Word indices: {word_indices}")

    words = []
    for indx in word_indices:
        word = list(custom_vectorizer.vocab.keys())[list(custom_vectorizer.vocab.values()).index(indx)]
        words.append(word)
    print(f"Words: {words}")

    # test tranform with list of docs
    docs = ["I need to reschedule my exam", "I request a personal hearing"]
    docs = custom_vectorizer.transform(docs)
    print(f"\n2 Doc Ex:\nDocs Shape: {docs.shape}")