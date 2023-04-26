import pandas as pd
import pytest

from ml_lab.utils.text_preprocessor import TextPreprocessor

preprocessor = TextPreprocessor()

### read in test data and configure tests ###
df = pd.read_csv('data/preprocessing_test_data.csv')

# delete rows where at least 1 element is missing
df.dropna()

testData = list()
testNames = list()
for i, row in df.iterrows():
    excel_row = str(i + 2)
    excel_row = "0" * (3 - len(excel_row)) + excel_row
    test_name = 'row-' + str(excel_row)
    data = (test_name, row['original_word'], row['true_word'])

    testNames.append(test_name)
    testData.append(data)


@pytest.mark.parametrize("test_name, original_word, true_word", testData, ids = testNames)
def test_preprocessing(test_name, original_word, true_word):
    preprocessed_word = preprocessor.preprocess_text(original_word)

    print(f"{'Original Word:':18s} {original_word}")
    print(f"Preprocessed Word: {preprocessed_word}")
    print(f"{'True Word:':18s} {true_word}")

    assert true_word == preprocessed_word


if __name__ == "__main__":
    """
    In order to run a specific test case in debug mode, edit the row variable
    """
    row = 2

     # filter the testData on the item corresponding to the excel row
    test_name =  testData[row - 2][0]
    original_word = testData[row - 2][1]
    true_word = testData[row - 2][2]

    test_preprocessing(test_name, original_word, true_word)

