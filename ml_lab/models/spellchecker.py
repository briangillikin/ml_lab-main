import os
import spacy
from spacy.tokens import Token
import argparse
from symspellpy import SymSpell, Verbosity
import re

class SpellChecker:
    def __init__(self, edit_distance=2):
        self.nlp = spacy.blank("en")
        self.spellcheck_dict_fp = "model_artifacts/4138_spellcheck_dict.txt"
        self.edit_distance = edit_distance
        self.prefix_length = 3
        self.speller = self.set_up_spellchecker()

    def set_up_spellchecker(self):
        """
        This method initializes the spellchecker with the pre-created dictionary.
        """
        if not os.path.isfile(self.spellcheck_dict_fp):
            raise Exception("You must first generate the spellcheck dictionary by running `python ml_lab/train/generate_spellcheck_dict.py`")

        speller = SymSpell(max_dictionary_edit_distance=self.edit_distance, prefix_length=self.prefix_length, count_threshold=1)
        speller.load_dictionary(self.spellcheck_dict_fp, 0, 1)

        return speller
    
    @staticmethod
    def fixup_text(text:str, keep_punct:list=[], remove_punct:list=[], delete_punct:list=[]) -> str:
        """Perform text preprocessing needed for spellcheck

        Args:
            text (str): text to preprocess
            keep_punct (list, optional): Punctuation to keep. Defaults to [].
            remove_punct (list, optional): Punctuation to remove and replace with a space. Defaults to [].
            delete_punct (list, optional): Punctuation to delete. Defaults to [].

        Returns:
            str: preprocessed text
        """        
        text = text.strip()

        exclude_punct = list(set(['-',',','(', ')',"'", '"', '•', '/', ':', ';', '_', '*', '!', '|','\\','[',']','.']).union(remove_punct) - set(keep_punct))

        # Punctuation not part of the features
        keep = list()
        chars = list(text)
        for ch in chars:
            if ch in delete_punct:
                #Don't add a space for that character
                continue
            elif ch in exclude_punct:
                keep.append(' ')
            else:
                keep.append(ch)

        text = ''.join(keep)
        # print("TEXT AFTER REMOVING PUNCT:", text)

        # Strip other punctuation
        words = text.split()
        keep = list()
        for word in words:
            word = word.strip('-')
            # word = word.strip('.')
            keep.append(word)

        text = ' '.join(keep)
        text = re.sub(r' +', ' ', text)

        return text
    
    def lookup_word(self, token:Token) -> str:
        """This method attempts to spellcheck 'token'. If there is a word within the specified edit distance of 'token' it returns the best option. Otherwise, it just returns the word.

        Args:
            token (Token): spacy token (one word within the larger text)

        Returns:
            str: input or corrected text (if applicable)
        """        
        auto_corrected = self.speller.lookup(token.text, Verbosity.TOP, max_edit_distance=self.edit_distance)
        if len(auto_corrected) == 1:
            return auto_corrected[0].term
        else:
            return token.text

    def correct_field(self, field_text:str) -> str:
        """This method takes in 'field_text' and spell corrects it returning the spell corrected field. 

        Args:
            field_text (str): text to correct

        Returns:
            str: corrected text
        """        
        corrected_field = ""
        field_doc = self.nlp.tokenizer(field_text)
        for token in field_doc:
            if token.is_alpha and len(token.text) > 3:
                corrected_word = self.lookup_word(token)
                corrected_field += corrected_word
                corrected_field += token.whitespace_
            else:
                corrected_field += token.text_with_ws
        return corrected_field

    def correct_field_test(self, field_text:str) -> str:
        """This method is the same as 'correct_field', however it also outputs the words that were spell corrected 
        for testing purposes. 

        Args:
            field_text (str): text to correct

        Returns:
            str: corrected text
        """        
        corrected_field = ""
        corrected_words = []
        field_doc = self.nlp.tokenizer(field_text)
        for token in field_doc:
            if token.is_alpha and len(token.text) > 3:
                corrected_word = self.lookup_word(token)
                if corrected_word != token.text:
                    corrected_words.append((token.text, corrected_word))
                corrected_field += corrected_word
                corrected_field += token.whitespace_
            else:
                corrected_field += token.text_with_ws
        return corrected_field, corrected_words
        
if __name__ == "__main__":
    text = "I need to add a depandant to my claim"
    parser = argparse.ArgumentParser()
    parser.add_argument("-text", default=text)
    args = parser.parse_args()

    spellchecker = SpellChecker()
    print(f"\nRaw Text:\n{args.text}")
    corrected_field, corrected_words = spellchecker.correct_field_test(args.text)
    print(f"\nSpellchecked Text:\n{corrected_field}\n\nCorrected Words:\n{corrected_words}")