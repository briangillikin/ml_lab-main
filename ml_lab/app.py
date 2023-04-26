import json
from flask import Flask, render_template, request
from ml_lab.models.form_4138_nn import Form4138NN
from ml_lab.utils.keyword_extractor import KeywordExtractor


app = Flask(__name__)
model = Form4138NN()
model.load()


@app.route("/")
@app.route("/home", methods = ['GET'])
def home():
    return render_template('home.html')


@app.route("/classification", methods = ['POST'])
def show_classification():
    input_text = request.form['inputText']
    print("\nINPUT_TEXT:", input_text)

    # Get keywords
    keyword_extractor = KeywordExtractor()
    keyword_counts = keyword_extractor.get_keyword_counts(input_text)
    print(f"\nKEYWORD COUNTS: {json.dumps(keyword_counts, indent=4)}")

    # Classify
    classification, conf = model.predict(input_text)
    conf = f"{conf*100:.2f}%"
    print(f"CLASSIFICATION: {classification}")
    print(f"CONFIDENCE: {conf}")

    # Display result
    return render_template('classification.html', input_text=input_text, keywords=keyword_counts, classification=classification, confidence=conf)


if __name__ == '__main__':
    app.run(debug=True)
