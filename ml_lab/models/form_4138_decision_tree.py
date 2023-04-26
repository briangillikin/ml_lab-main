import os
import pickle
from sklearn.tree import DecisionTreeClassifier

from ml_lab.utils.data_loader import LabelMapper
from ml_lab.models.custom_vectorizer import CustomVectorizer


class Form4138DecisionTree():
    def __init__(self, config={'max_depth': 30, 'min_samples_split': 2}) -> None:
        self.models_dir = "model_artifacts"
        self.model_fp = f"{self.models_dir}/4138_decision_tree.pkl"
        self.custom_vectorizer = CustomVectorizer()
        self.label_mapper = LabelMapper()
        self.labels = list(self.label_mapper.label_to_id.keys())
        self.num_labels = self.label_mapper.num_labels

        self.model = DecisionTreeClassifier(max_depth=config["max_depth"], min_samples_split=config["min_samples_split"], random_state=1)

    def load(self):
        """Load trained model

        Raises:
            Exception: You must train model if not already trained
        """        
        if not os.path.exists(self.model_fp):
            raise Exception("You must first train the model by running `python ml_lab/train/train_decision_tree.py`")
        with open(self.model_fp, 'rb') as f:
            self.model = pickle.load(f)

    def predict(self, doc:str) -> str:
        """Predict label classification of given text

        Args:
            doc (str): text to classify

        Returns:
            str: label predicted
        """        
        doc = self.custom_vectorizer.transform(doc)
        prediction = self.model.predict(doc)
        label_name = self.label_mapper.get_label_name(prediction[0])
        
        return label_name
    
    def predict_batch(self, docs:list) -> list:
        """Predict label classification on a list of text docs

        Args:
            docs (list): list of text docs to classify

        Returns:
            list: labels predicted
        """
        predictions = []
        for doc in docs:
            prediction = self.predict(doc)
            predictions.append(prediction)

        return predictions


if __name__ == "__main__":
    model = Form4138DecisionTree()
    print(model.model)