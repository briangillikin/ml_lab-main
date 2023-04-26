import os
import spacy
import pickle
import pandas as pd
from ml_lab.models.spellchecker import SpellChecker

nlp = spacy.blank("en")

def generate_base_dict(keep_proportion: float = 1) -> dict[str, int]:
    """ Generates the proportion of the spellcheck dictionary that comes from scraping the training data and intersecting it with the known vocab.

        Args:
            keep_proportion (float): text to preprocess

        Returns:
            dict[str, int]: a dictionary with word, count key value pairs containing the words occurring in both the training data and the known vocab.
    """        
    with open("data/known_vocab.pkl", "rb") as f:
        known_vocab = pickle.load(f)


    data_4138 = pd.read_csv("data/form_4138_data.csv")

    keep_chars = ["-", "/"]
    delete_chars = ["'", "="]

    vocab_dict = {}
    for i, row in data_4138.iterrows():
        text = row["TEXT"]
        fixedup_text = SpellChecker.fixup_text(text.lower(), keep_punct=keep_chars, delete_punct=delete_chars)
        alpha_tokens = [token.text for token in nlp.tokenizer(fixedup_text) if token.is_alpha]
        for alpha_token in alpha_tokens:
            if alpha_token not in vocab_dict:
                vocab_dict[alpha_token] = 1
            else:
                vocab_dict[alpha_token] += 1

    vocab_dict = {word: count for word, count in sorted(vocab_dict.items(), key=lambda x: (-x[1], x[0])) if word in known_vocab}

    if keep_proportion != 1:
        cutoff_index = int(len(vocab_dict) * keep_proportion)
        vocab_dict = {word: count for word, count in vocab_dict[:cutoff_index].items()}

    return vocab_dict

        
def supplement_with_feature_words(vocab_dict: dict[str, int]) -> dict[str, int]:
    """ Supplements the existing vocab_dict with all words that we use as features with their counts set to 1.

        Args:
            vocab_dict (dict[str, int]): the 'vocab_dict' outputted by 'generate_base_dict' containing word, count pairs 
            that come from the training data and 'known_vocab'.

        Returns:
            dict[str, int]: a dictionary with word, count key value pairs containing the words occurring in both the
            training data and the known vocab.
    """ 
    dictionary_files = os.listdir("dictionaries")
    for dictionary_file in dictionary_files:
        with open(f"dictionaries/{dictionary_file}") as f:
            lines = f.read().split("\n")
        for line in lines:
            split_line = line.split(",")
            for term in split_line:
                alpha_term_tokens = [token.lower_ for token in nlp.tokenizer(term) if token.is_alpha]
                for alpha_term_token in alpha_term_tokens:
                    if alpha_term_token not in vocab_dict:
                        vocab_dict[alpha_term_token] = 1
    return vocab_dict


def generate_spellcheck_dict(keep_proportion: float = 1) -> None:
    """ Combines the vocab_dicts created in 'generate_base_dict' and 'supplement_with_feature_words' into the final vocab_dict which is then made into a text file to be read into the SymSpell object.

        Args:
            keep_proportion (float): text to preprocess

        Returns:
            dict[str, int]: a dictionary with word, count key value pairs containing the words occurring in both the training data and the known vocab and supplemented with our feature words if not already in vocab_dict
    """    
    vocab_dict = generate_base_dict(keep_proportion)

    vocab_dict = supplement_with_feature_words(vocab_dict)

    os.makedirs("model_artifacts", exist_ok=True)
    spellcheck_dict_fp = "model_artifacts/4138_spellcheck_dict.txt"
    with open(spellcheck_dict_fp, "w") as f:
        for word, count in vocab_dict.items():
            f.write(f"{word} {count}\n")

    print(f"Wrote a dictionary of length {len(vocab_dict)} to {spellcheck_dict_fp}")


if __name__ == "__main__":
    generate_spellcheck_dict()