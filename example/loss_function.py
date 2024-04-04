from typing import Union
import numpy as np
import cupy as cp
from utils import get_backend


DataArray = Union[np.ndarray, cp.ndarray]


def cross_entropy_loss(y_true: DataArray, y_pred: DataArray, backend=np) -> DataArray:
    backend = get_backend(y_true)
    # Note: y_true must be one-hot encoding format
    # y_true's shape is (batch_size, classes_num)
    # y_pred's shape is (batch_size, classes_num), it's a logits
    batch_size = y_true.shape[0]

    smoothing = 1e-15
    loss = -1 / batch_size * backend.sum(y_true * backend.log(y_pred + smoothing))

    return loss

