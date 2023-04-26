import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from datetime import datetime
from ray.air import session

from ml_lab.utils.data_loader import CustomDataset
from ml_lab.models.form_4138_nn import Form4138NN

from dotenv import load_dotenv
load_dotenv()


def train_nn(config:dict) -> None:
    """Train neural network and save model to pkl file in data_artifacts folder

    Args:
        config (dict): parameters to use for training the neural network
    """    
    # set root_dir for when working with ray tune
    root_dir = os.getenv("ROOT_DIR")
    os.chdir(root_dir)

    # seed everything for reproducibility
    torch.manual_seed(1)

    # init model
    model = Form4138NN(hidden_dim=config["hidden_dim"])

    # init data/ data loaders
    train_data = CustomDataset('data/train.csv')
    train_dataloader = DataLoader(dataset=train_data, batch_size=config["batch_size"], shuffle=True)

    test_data = CustomDataset('data/test.csv')
    test_dataloader = DataLoader(dataset=test_data, batch_size=config["batch_size"], shuffle=True)

    # define train variables
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    loss_values = []
    
    start = datetime.now()
    for i in range(config["num_epochs"]):
        elapsed_time = (datetime.now() - start).total_seconds()
        print(f"\nEpoch {i + 1} of {config['num_epochs']} elapsed time = {elapsed_time} s")
        train_samples = 0
        train_loss = 0
        train_correct = 0
        ## Train
        for X, y in train_dataloader: 
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            predictions = model(X)
            loss = loss_fn(predictions, y)
            loss_values.append(loss.item())
            loss.backward()
            optimizer.step()

            # calculate batch level metrics
            train_samples += y.size(0)
            train_loss += loss.item() * train_samples
            train_correct += (torch.argmax(predictions, 1) == torch.argmax(y, 1)).float().sum()
            
        # display train metrics for epoch
        trainTemplate = "epoch: {} train loss: {:.3f} train accuracy: {:.3f}"
        print(trainTemplate.format(i + 1, (train_loss / train_samples),
            (train_correct / train_samples)))

        ## Eval
        test_samples = 0
        test_loss = 0
        test_correct = 0
        with torch.no_grad():
            # NOTE: We're not training so we don't need to calculate the gradients for our outputs
            for X, y in test_dataloader:
                predictions = model(X)
                loss = loss_fn(predictions, y)

                # calculate batch level metrics
                test_samples += y.size(0)
                test_loss += loss.item() * test_samples
                test_correct += (torch.argmax(predictions, 1) == torch.argmax(y, 1)).float().sum()
        
        # display test metrics for epoch
        trainTemplate = "epoch: {} test loss: {:.3f} test accuracy: {:.3f}"
        print(trainTemplate.format(i + 1, (test_loss / test_samples),
            (test_correct / test_samples)))

        if config['tune_session']:
            session.report({"loss": test_loss, "accuracy": test_correct / test_samples})

    if not config['tune_session']:
        # save trained model
        os.makedirs(model.models_dir, exist_ok=True)
        torch.save(model.state_dict(), model.model_fp)
        print(f"\nSaved model to {model.model_fp}")


if __name__ == "__main__":
    config = {
        'tune_session': False,
        'hidden_dim': 110,
        'batch_size': 200,
        'num_epochs': 50,
        'lr': 0.001,
        'weight_decay': 0.0001
        }
    train_nn(config)