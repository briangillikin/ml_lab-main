import os
import json
import torch
import argparse
from typing import Union


class LabelMapper():
    """Translate between numerically encoded label and text version of label
    """    
    def __init__(self) -> None:
        fp = "data/label_map.json"
        if not os.path.exists(fp):
            raise Exception("You must first create the data artifacts by running `python ml_lab/utils/create_data_artifacts.py`")
        with open(fp, "r") as f:
            self.label_to_id = json.load(f)
        
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        self.num_labels = len(self.label_to_id)

    def get_label_id(self, label:str) -> int:
        """Get label id from label text

        Args:
            label (str): label text

        Returns:
            int: numeric encoding for label text
        """        
        return self.label_to_id[label]
    
    def get_label_name(self, label_id:Union[int, torch.tensor]) -> str:   
        """Get label text from label id

        Args:
            label_id (Union[int, torch.tensor]): label id as integer, vectorized torch tensor or torch tensor of length 1 resulting from argmax

        Returns:
            str: label text
        """        
        if torch.is_tensor(label_id):
            if len(label_id) == 1:
                # argmax tensor with label_id inside
                label_id = label_id.item()
            else:
                # tensor with probabilities for each class
                label_id = torch.argmax(label_id).item()
        return self.id_to_label[label_id]


if __name__ == "__main__":
    label_id = 0
    parser = argparse.ArgumentParser()
    parser.add_argument("-id", default=label_id)
    args = parser.parse_args()

    label_mapper = LabelMapper()

    print(f"\nLabel ID: {args.id}")
    label_name = label_mapper.get_label_name(int(args.id))
    print(f"Label Name: {label_name}")