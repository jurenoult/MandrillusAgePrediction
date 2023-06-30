from torch.utils.data import DataLoader, random_split
import os
import torch
from .dataset import AugmentedDataset
import numpy as np

def split_dataset(dataset, train_ratio, batch_size, augment=True, num_workers=0):
    # Split the dataset into training and validation subsets
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    if augment:
        train_dataset = AugmentedDataset(train_dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, train_dataset, val_dataset

def save(model, prefix, exp_name):
    torch.save(model.state_dict(), f"models/{prefix}_{exp_name}.h5")
    
def load(model, prefix, exp_name):
    path = f"models/{prefix}_{exp_name}.h5"
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
    else:
        print(f"Could not find path at : {path}")
    return model

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)