from typing import Tuple
import numpy as np
import cupy as cp
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def get_mnist_dataloader(backend = cp) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mnist = fetch_openml("mnist_784")

    x = mnist["data"].to_numpy()
    y = mnist["target"].astype(int).to_numpy()

    mu = np.mean(x)
    sigma = np.std(x)
    x = (x - mu) / sigma

    train_x, test_x = train_test_split(x, test_size=0.2, random_state=2999)
    train_y, test_y = train_test_split(y, test_size=0.2, random_state=2999)

    # OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)
    train_y = encoder.fit_transform(train_y.reshape(-1, 1))
    test_y = encoder.fit_transform(test_y.reshape(-1, 1))

    train_x = backend.asarray(train_x)
    test_x = backend.asarray(test_x)
    train_y = backend.asarray(train_y)
    test_y = backend.asarray(test_y)

    return train_x, test_x, train_y, test_y

