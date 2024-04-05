import cupy as cp
import numpy as np

from custom_types import Backend, DataArray


def get_backend(x: DataArray) -> Backend:
    return np if isinstance(x, np.ndarray) else cp


def softmax(x: DataArray) -> DataArray:
    """
    Softmax function.
    ```
    s(x_i)=\frac{e^{x_i}}{\sum_{j=1}^{n}e^{x_j}}
    ```

    Args:
        x (DataArray): The input data that is an array.
    Returns:
        DataArray: The output data after softmax.
    """
    backend = get_backend(x)
    exps = backend.exp(x - backend.max(x, axis=-1, keepdims=True))
    return exps / backend.sum(exps, axis=-1, keepdims=True)


def ReLU(x: DataArray) -> DataArray:
    backend = get_backend(x)
    return backend.maximum(0, x)


class AverageMeter:
    """Computes and stores the average and current value of losses."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset all attributes."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, count_num: int = 1) -> None:
        """
        Update the loss value.
        
        Args:
            val (float): The current update value.
            count_num (int): The number of updated data.
        """
        self.val = val
        self.sum += val * count_num
        self.count += count_num
        self.avg = self.sum / self.count
