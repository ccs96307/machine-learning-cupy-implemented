from typing import TypeVar, Union
import cupy as cp
import numpy as np


DataArray = TypeVar("DataArray", np.ndarray, cp.ndarray)
Backend = Union[type(np), type(cp)]
