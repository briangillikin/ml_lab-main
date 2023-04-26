import argparse
from ml_lab.models.form_4138_decision_tree import Form4138DecisionTree


def test(text):
    model = Form4138DecisionTree()
    model.load()
    prediction = model.predict(text)

    return prediction


if __name__ == "__main__":
    text = "I request a personal hearing"
    parser = argparse.ArgumentParser()
    parser.add_argument("-text", default=text)
    args = parser.parse_args()

    prediction = test(text)
    print(f"\nPrediction: {prediction}")
