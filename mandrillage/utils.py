import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Subset

from .dataset import AugmentedDataset


def get_logger(name=__name__) -> logging.Logger:
    logger = logging.getLogger(name)
    return logger


log = get_logger(__name__)


def create_kfold_dataset(dataset, k, fold_index):
    """
    Create a k-fold dataset using PyTorch.

    Parameters:
        dataset (Dataset): The PyTorch dataset you want to split.
        k (int): Number of folds.
        fold_index (int): The index of the fold to use for validation (0 to k-1).
        batch_size (int, optional): Batch size for data loaders (default is 32).
        num_workers (int, optional): Number of workers for data loading (default is 0).
        pin_memory (bool, optional): Whether to use pinned memory for data loaders (default is False).
        shuffle (bool, optional): Whether to shuffle the data before creating folds (default is True).

    Returns:
        DataLoader, DataLoader: Training DataLoader and Validation DataLoader.
    """
    assert 0 <= fold_index < k, "Invalid fold_index. Should be in the range 0 to k-1."

    indices = torch.arange(len(dataset))

    fold_size = len(dataset) // k
    val_indices = indices[fold_index * fold_size : (fold_index + 1) * fold_size]

    train_indices = torch.cat(
        [indices[: fold_index * fold_size], indices[(fold_index + 1) * fold_size :]]
    )

    return train_indices, val_indices


def create_loader(dataset, train_indices, val_indices, batch_size, num_workers=0):
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, train_dataset, val_dataset


def split_dataset(dataset, train_ratio, batch_size, augment=True, num_workers=0):
    # Split the dataset into training and validation subsets
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    if augment:
        train_dataset = AugmentedDataset(train_dataset)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
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
