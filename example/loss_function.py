from custom_types import DataArray
from utils import get_backend


def cross_entropy_loss(
    y_pred: DataArray,
    y_true: DataArray,
    smoothing: float = 1e-15,
) -> DataArray:
    """
    The cross-entropy loss function.
    

    Args:
        y_pred (DataArray): The model prediction. Shape is (batch_size, classes_num), They're probabilities.
        y_true (DataArray): The ground truth data. Shape is (batch_size, classes_num), They're one-hot encoding format.
        smoothing (float): Clip for numerical stability
    Returns:
        DataArray: The calculated loss scores.
    """
    backend = get_backend(y_pred)

    if get_backend(y_true) != backend:
        raise TypeError("The Backend type of `y_pred` needs to be the same with `y_true`.")

    batch_size = y_true.shape[0]

    total_loss = -1 * backend.sum(y_true * backend.log(y_pred + smoothing))
    average_loss = total_loss / batch_size

    return average_loss

