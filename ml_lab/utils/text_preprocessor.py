import argparse
from ml_lab.models.spellchecker import SpellChecker


class TextPreprocessor():
    def __init__(self) -> None:
        self.spellcheck = SpellChecker()

    def preprocess_text(self, text:str) -> str:
        """Perform text preprocesing

        Args:
            text (str): text to process

        Returns:
            str: preprocessed text
        """        
        text = str(text)
        # remove symbols/ punctuation
        keep_chars = ["-", "/"]
        delete_chars = ["'", "="]
        fixedup_text = SpellChecker.fixup_text(text.lower(), keep_punct=keep_chars, delete_punct=delete_chars)
        # spellcheck
        spellchecked_text = self.spellcheck.correct_field(fixedup_text)

        return spellchecked_text

if __name__ == "__main__":
    text = "I need to add a depandant to my claim"
    parser = argparse.ArgumentParser()
    parser.add_argument("-text", default=text)
    args = parser.parse_args()

    text_preprocessor = TextPreprocessor()

    print(f"\nRaw text:\n{args.text}")
    preprocessed_text = text_preprocessor.preprocess_text(args.text)
    print(f"\nPreprocessed text:\n{preprocessed_text}\n")