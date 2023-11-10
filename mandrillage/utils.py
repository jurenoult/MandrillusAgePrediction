import logging
import os

import numpy as np
import torch

DAYS_IN_YEAR = 365.25


def get_logger(name=__name__) -> logging.Logger:
    logger = logging.getLogger(name)
    return logger


log = get_logger(__name__)


def create_kfold_data(dataset, k, fold_index):
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

    ids = np.unique(dataset["id"].values)
    indices = ids

    # Shuffle indices (but always the same)
    np.random.seed(5)
    np.random.shuffle(indices)

    fold_size = len(ids) // k
    val_indices = indices[fold_index * fold_size : (fold_index + 1) * fold_size]

    train_indices = np.concatenate(
        [indices[: fold_index * fold_size], indices[(fold_index + 1) * fold_size :]]
    )

    print(f"Train indices: {train_indices}")
    print(f"Val indices: {val_indices}")
    return train_indices, val_indices


def split_indices(data, train_ratio):
    val_split = 1.0 - train_ratio
    k = int(1.0 // val_split)
    # val_k = k - 1
    val_k = 0

    return create_kfold_data(data, k, val_k)


def save(model, prefix, exp_name, output_dir):
    save_dir = os.path.join(output_dir, "checkpoints")

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    path = os.path.join(save_dir, f"{prefix}_{exp_name}.h5")
    torch.save(model.state_dict(), path)


def load(model, prefix, exp_name, output_dir):
    path = os.path.join(output_dir, "checkpoints")
    path = os.path.join(path, f"{prefix}_{exp_name}.h5")
    if os.path.exists(path):
        log.info(f"Loading weight: {path}")
        model.load_state_dict(torch.load(path))
    else:
        log.warning(f"Could not find path at : {path}")
    return model


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
