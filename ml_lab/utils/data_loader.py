import pandas as pd
import numpy as np
import torch
from typing import Tuple
from torch.utils.data import Dataset, DataLoader

from ml_lab.models.custom_vectorizer import CustomVectorizer
from ml_lab.utils.label_mapper import LabelMapper


class CustomDataset(Dataset):
    """Inherits Pytorch Dataset class. Used to load the train/ test data in batches for the pytorch neural net model.
    """      
    def __init__(self, fp:str):
        """Initialize custom dataset

        Args:
            fp (str): dataset filepath
        """        
        self.custom_vectorizer = CustomVectorizer()
        self.data = []

        df = pd.read_csv(fp)
        for _, row in df.iterrows():
            label = row["TRUTH"]
            text = row["TEXT"]
            self.data.append([label, text])

        self.label_mapper = LabelMapper()

    def __len__(self) -> int:
        """Internal function definition so len() can be called on dataset

        Returns:
            int: Length of dataset
        """        
        return len(self.data)

    def __getitem__(self, idx:int) -> Tuple[torch.tensor, torch.tensor]:   
        """Internal function to retrieve a sample with list indexing and perform necessary preprocessing. Used when loading the data in batches, to get indices from the sampler until the desired batch_size is reached
        
        Args:
            idx (int): Index of dataset to retrieve

        Returns:
            Tuple[torch.tensor, torch.tensor]:
                text: vectorized text converted to a torch tensor
                label: vectorized label converted to a torch tensor
        """
        label, text = self.data[idx]
        # create label tensor
        # convert label to list of zeros with 1 in index of its label mapping
        label_id = self.label_mapper.get_label_id(label)
        label_array = np.zeros((self.label_mapper.num_labels), dtype=np.float32) # float since output is float
        label_array[label_id] = 1
        label = torch.from_numpy(label_array)

        # create text tensor
        # extract/transform keywords using custom vectorizer
        text = self.custom_vectorizer.transform(text)[0]
        text = torch.from_numpy(text.astype(np.float32))

        return text, label


if __name__ == "__main__":
    # test CustomDataset
    train_data = CustomDataset("data/train.csv")
    print(f"\nLen training data: {len(train_data)}")

    doc, label = train_data[72]
    label_name = train_data.label_mapper.get_label_name(label)
    print(f"Label name: {label_name}")
    print(f"Label Tensor: {label}\nDoc Tensor: {doc}")

    # Instantiate train and test data
    batch_size = 64
    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    test_data = CustomDataset("data/test.csv")
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    # Check it's working
    for batch, (X, y) in enumerate(train_dataloader):
        print(f"Batch: {batch+1}")
        print(f"X\n  shape: {X.shape}\n  dtype: {X.dtype}")
        print(f"y\n  shape: {y.shape}\n  dtype: {y.dtype}\n")
        break
