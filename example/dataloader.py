from typing import Tuple

import numpy as np
import cupy as cp
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from custom_types import Backend, DataArray


def get_mnist_dataloader(
    backend: Backend = np,
    test_size: float = 0.2,
    random_state: int = 2999,
    smoothing: float = 1e-15,
) -> Tuple[DataArray, ...]:
    """ 
    Download MNIST dataset.
    
    Args:
        backend (Backend): numpy or cupy.
        test_size (float): The test data size of dataset.
        random_state (int): The random seed.
        smoothing (float): A small number to avoid the denominator is zero.

    Returns:
        Tuple[DataArray, ...]: (train_x, test_x, train_y, test_y)
    """
    # Load data
    mnist = fetch_openml("mnist_784")
    x = mnist["data"].to_numpy()
    y = mnist["target"].astype(int).to_numpy()

    # Normalization
    mu = np.mean(x)
    sigma = np.std(x)
    x = (x - mu) / (sigma + smoothing)

    # Split
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=test_size, random_state=random_state)

    # OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)
    train_y = encoder.fit_transform(train_y.reshape(-1, 1))
    test_y = encoder.fit_transform(test_y.reshape(-1, 1))

    # Convert into backend data type
    train_x = backend.asarray(train_x)
    test_x = backend.asarray(test_x)
    train_y = backend.asarray(train_y)
    test_y = backend.asarray(test_y)

    return train_x, test_x, train_y, test_y

