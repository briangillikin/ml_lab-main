import argparse
from ml_lab.models.form_4138_nn import Form4138NN


def test(text):
    model = Form4138NN()
    model.load()
    prediction, conf = model.predict(text)

    return prediction, conf


if __name__ == "__main__":
    text = "I request a personal hearing"
    parser = argparse.ArgumentParser()
    parser.add_argument("-text", default=text)
    args = parser.parse_args()

    prediction, conf = test(text)
    print(f"\nPrediction: {prediction}")
    print(f"Confidence: {conf}")
