import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ml_lab.utils.data_loader import LabelMapper
from ml_lab.models.custom_vectorizer import CustomVectorizer


class Form4138NN(nn.Module):
    def __init__(self, hidden_dim=110):
        super(Form4138NN, self).__init__()
        self.models_dir = "model_artifacts"
        self.model_fp = f"{self.models_dir}/4138_nn.pkl"
        self.custom_vectorizer = CustomVectorizer()
        self.label_mapper = LabelMapper()

        input_dim = len(self.custom_vectorizer.vocab)
        self.labels = list(self.label_mapper.label_to_id.keys())
        self.num_labels = self.label_mapper.num_labels

        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, self.num_labels)
   
    def forward(self, x:torch.tensor) -> torch.tensor:
        """Compute output tensors from input tensors

        Args:
            x (torch.tensor): Input tensor (transformed text data)

        Returns:
            torch.tensor: Output tensor
        """
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))

        return F.softmax(x, dim=1)
    
    def load(self):
        """Load trained model

        Raises:
            Exception: You must train model if not already trained
        """    
        if not os.path.exists(self.model_fp):
            raise Exception("You must first train the model by running `python ml_lab/train/train_nn.py`")
        self.load_state_dict(torch.load(self.model_fp))
        self.eval()

    def predict(self, doc:str) -> str:
        """Predict label classification of given text

        Args:
            doc (str): text to classify

        Returns:
            str: label predicted
        """    
        doc = self.custom_vectorizer.transform(doc)
        doc = torch.from_numpy(doc.astype(np.float32))
        prediction = self(doc)
        probs = F.softmax(prediction, dim=1)
        conf, label_id = torch.max(probs, 1)

        label_name = self.label_mapper.get_label_name(label_id)
        conf = conf.item()

        return label_name, conf
    
    def predict_batch(self, docs:list) -> list:
        """Predict label classification on a list of text docs

        Args:
            docs (list): list of text docs to classify

        Returns:
            list: labels predicted
        """
        predictions = []
        for doc in docs:
            prediction, conf = self.predict(doc)
            predictions.append(prediction)
        
        return predictions

       
if __name__ == "__main__":
    model = Form4138NN()
    print(model)
