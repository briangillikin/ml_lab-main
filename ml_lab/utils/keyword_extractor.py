import spacy
import json
import argparse
from collections import defaultdict
from spacy.matcher import Matcher

from ml_lab.utils.text_preprocessor import TextPreprocessor


class KeywordExtractor():
    def __init__(self) -> None:
        self.text_preprocessor = TextPreprocessor()
        self.nlp = spacy.blank("en")
        self.matcher = Matcher(self.nlp.vocab)
        self.dict_dir = "dictionaries"
        self.load_patterns()

    def load_patterns(self):
        """Loop through each file and add each term to the spaCy Matcher with a label equal to the first word on the line (for keyword normalization)
        """
        fps = ["dictionaries/4138_terms.csv", "dictionaries/medical_conditions.csv", "dictionaries/medical_procedures.csv", "dictionaries/medical_bodypart.csv"]
        for fp in fps:
            with open(fp, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    terms = line.strip().split(",")
                    for term in terms:
                        pattern = []
                        for token in self.nlp.tokenizer(term): 
                            pattern.append({"LOWER": token.text})
                        self.matcher.add(terms[0], [pattern])

    def extract_keywords(self, text:str, normalize:bool=True) -> list:    
        """Extract keywords from text

        Args:
            text (str): text to extract keyword from
            normalize (bool, optional): Whether to return the text picked up or the normalized term. Defaults to True.

        Returns:
            list: list of keywords
        """
        text = self.text_preprocessor.preprocess_text(text)
        doc = self.nlp(text)
        matches = self.matcher(doc)
        keywords = []
        for match_id, start, end in matches:
            string_id = self.nlp.vocab.strings[match_id]  # Get string representation (label)
            span = doc[start:end]  # The matched span
            # print(string_id, start, end, span.text)
            if normalize:
                keywords.append(string_id)
            else:
                keywords.append(span)
        
        return keywords
    
    def get_keyword_counts(self, text:str, normalize=True) -> dict:
        """Get keyword counts from text

        Args:
            text (str): text to extract keyword counts from
            normalize (bool, optional): Whether to return the text picked up or the normalized term. Defaults to True.

        Returns:
            dict: dictionary with each keyword found as a key and the number of occurences as the value
        """        
        text = self.text_preprocessor.preprocess_text(text)
        doc = self.nlp(text)
        matches = self.matcher(doc)
        keyword_counts = defaultdict(int)
        for match_id, start, end in matches:
            string_id = self.nlp.vocab.strings[match_id]  # Get string representation (label)
            span = doc[start:end]  # The matched span
            # print(string_id, start, end, span.text)
            if normalize:
                keyword_counts[string_id] += 1
            else:
                keyword_counts[span] += 1
        
        return keyword_counts


if __name__ == "__main__":
    text = "THE BOARD OF VETERANS APPEALS HAS FORMALLY PLACED VETERANS APPEAL ON THE BOARDS DOCKET. THE CHANGE WE WOULD LIKE TO MAKE IS AN ADDRESS CHANGE WHICH IS John A Doe 1111 Washington Drive, Arling VA 22203"
    parser = argparse.ArgumentParser()
    parser.add_argument("-text", default=text)
    args = parser.parse_args()

    kw_extractor = KeywordExtractor()

    print(f"\nText:\n{args.text}")
    keywords = kw_extractor.extract_keywords(args.text)
    print(f"\nKeywords:\n{keywords}\n")

    keyword_counts = kw_extractor.get_keyword_counts(args.text)
    print(f"\nKeyword Counts:\n{json.dumps(keyword_counts, indent=4)}\n")
