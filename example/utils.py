from typing import TypeVar, Union
import cupy as cp
import numpy as np


DataArray = TypeVar("DataArray", np.ndarray, cp.ndarray)


def get_backend(x: DataArray):
    return np if isinstance(x, np.ndarray) else cp


def softmax(x: DataArray) -> DataArray:
    backend = get_backend(x)
    exps = backend.exp(x - backend.max(x, axis=-1, keepdims=True))
    return exps / backend.sum(exps, axis=-1, keepdims=True)


def ReLU(x: DataArray) -> DataArray:
    backend = get_backend(x)
    return backend.maximum(0, x)


class AverageMeter:
    """Computes and stores the average and current value of losses"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset all attributes"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, count_num: int = 1) -> None:
        """Update the loss value"""
        self.val = val
        self.sum += val * count_num
        self.count += count_num
        self.avg = self.sum / self.count



